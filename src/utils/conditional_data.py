"""Conditional Stage-B data: the common48w prompt / thinking / response package.

One row becomes two streams:

    input_ids     [prompt tokens | response tokens] in one `max_length` window,
                  with `cond_seq_mask` marking the prompt. The prompt is a clean
                  prefix: it is never noised, never predicted, and is excluded
                  from the loss, exactly as the upstream conditional path does.
    plan tokens   the gold thinking, which `plan_stream` compresses into the
                  variable-K plan target. Training only -- generation never sees
                  it.

Length policy (`condition_max_tokens` + `max_length`) is deliberate, not
incidental. With the shipped 1024 / 2048 pair the response can never be
truncated, because the package's longest response is 1006 tokens and the prompt
can never take more than half the window. Rows that still overflow are counted
and reported rather than silently trimmed.
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from utils.logging_utils import log_for_0
from utils.thinking_tokenization import thinking_token_ids

SEED_STREAMS = ("response_noise", "token_time", "branch", "plan_noise", "plan_time")
SEED_KEYS = tuple(f"{stream}_seed" for stream in SEED_STREAMS)
PLAN_GROUP_SIZE = 4


def _sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_seed(master_seed: int, presentation_index: int, example_id: str, stream: str) -> int:
    """Per-sample RNG seed. A pure function, so the four groups agree by construction."""
    payload = f"conditional-v1\0{master_seed}\0{presentation_index}\0{example_id}\0{stream}"
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16) % (2 ** 63 - 1)


class ConditionalPairedDataset(torch.utils.data.Dataset):
    """Parquet prompt/thinking/response rows, verified against the shipped manifest.

    The whole package is ~500 MB and the rows are read in a shuffled order, so it
    is held in memory as one Arrow table: shard-at-a-time caching would thrash,
    and per-row Parquet seeks are far slower than the training step they feed.
    """

    COLUMNS = ("example_id", "source", "prompt", "thinking", "response")

    def __init__(self, manifest_path, expected_sha256: Optional[str] = None,
                 verify_shards: bool = True):
        import pyarrow.parquet as pq

        self.manifest_path = str(Path(manifest_path).resolve())
        if expected_sha256 and _sha256_file(manifest_path) != expected_sha256:
            raise ValueError("conditional train manifest SHA256 mismatch")
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not manifest.get("complete"):
            raise ValueError("conditional train manifest is not complete")
        if tuple(manifest.get("schema", ())) != self.COLUMNS:
            raise ValueError(f"unexpected conditional schema: {manifest.get('schema')}")

        root = Path(manifest_path).parent
        tables, rows = [], 0
        for shard in manifest["shards"]:
            path = root / shard["name"]
            if not path.is_file() or path.stat().st_size != int(shard["bytes"]):
                raise ValueError(f"missing or size-mismatched conditional shard: {path}")
            if verify_shards and _sha256_file(path) != shard["sha256"]:
                raise ValueError(f"conditional shard SHA256 mismatch: {path}")
            table = pq.read_table(path, columns=list(self.COLUMNS))
            if table.num_rows != int(shard["rows"]):
                raise ValueError(f"conditional shard row mismatch: {path}")
            tables.append(table)
            rows += table.num_rows
        if rows != int(manifest["rows"]):
            raise ValueError("conditional total row count mismatch")

        import pyarrow as pa
        self.table = pa.concat_tables(tables)
        self.manifest = manifest
        self._columns = {name: self.table.column(name) for name in self.COLUMNS}

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, index):
        return {name: column[index].as_py() for name, column in self._columns.items()}

    def example_ids(self):
        return self._columns["example_id"].to_pylist()


class ConditionalSchedule(torch.utils.data.Dataset):
    """A fixed presentation order plus per-sample RNG seeds, shared by all groups.

    The unconditional protocol shipped this as an 80k-line JSONL artifact guarded
    by a SHA gate. It does not need to be an artifact: the order is
    `randperm(len(base), seed)[:rows]` and the seeds are a hash of
    (master seed, position, example id, stream), so every group derives the same
    schedule from the same two numbers. `fingerprint()` is what a run logs to
    prove it, instead of a file everyone has to copy around.
    """

    def __init__(self, base: ConditionalPairedDataset, rows: int, master_seed: int = 42):
        if rows > len(base):
            raise ValueError(f"schedule wants {rows} rows, dataset has {len(base)}")
        self.base = base
        self.rows = int(rows)
        self.master_seed = int(master_seed)
        generator = torch.Generator().manual_seed(self.master_seed)
        self.order = torch.randperm(len(base), generator=generator)[:rows].tolist()
        ids = base.example_ids()
        self.example_ids = [ids[i] for i in self.order]
        if len(set(self.example_ids)) != self.rows:
            raise ValueError("schedule contains duplicate example ids")

    def __len__(self):
        return self.rows

    def __getitem__(self, index):
        row = self.base[self.order[index]]
        example_id = row["example_id"]
        row["presentation_index"] = index
        for stream in SEED_STREAMS:
            row[f"{stream}_seed"] = derive_seed(self.master_seed, index, example_id, stream)
        return row

    def fingerprint(self) -> str:
        """Identity of this presentation order, for the run log and cross-group checks."""
        digest = hashlib.sha256(f"conditional-v1\0{self.master_seed}\0{self.rows}".encode())
        for index, example_id in enumerate(self.example_ids):
            digest.update(f"{index}\0{example_id}\0".encode())
        return digest.hexdigest()


class ConditionalCollator:
    """[prompt | response] in one window, plus the gold-thinking plan tokens."""

    def __init__(self, tokenizer, max_length: int = 2048, condition_max_tokens: int = 1024,
                 max_plan_slots: int = 255, pad_token_id: Optional[int] = None,
                 plan_token_capacity: Optional[int] = None):
        if condition_max_tokens >= max_length:
            raise ValueError("condition_max_tokens must leave room for a response")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.condition_max_tokens = int(condition_max_tokens)
        self.max_plan_slots = int(max_plan_slots)
        # How many gold-thinking tokens the plan target may consume. The legacy
        # 4-to-1 MLP path derives it from the slot count; the span-VAE path must
        # pass it explicitly (SPAN * K_MAX = 1024), because 4 * 64 = 256 silently
        # cut every thinking to its first quarter and capped the budget at 16 slots.
        self.plan_token_capacity = int(plan_token_capacity if plan_token_capacity is not None
                                       else PLAN_GROUP_SIZE * self.max_plan_slots)
        self.pad_token_id = int(pad_token_id if pad_token_id is not None
                                else tokenizer.pad_token_id)

    # Truncation is reported per row inside the batch, not accumulated on `self`:
    # with num_workers > 0 the collator that runs is a forked copy, so counters
    # kept here would stay at zero in exactly the runs that matter.
    TRUNCATION_KEYS = ("prompt_truncated", "response_truncated", "thinking_truncated")

    def _encode(self, rows):
        prompts = [thinking_token_ids(self.tokenizer, row["prompt"], add_special_tokens=False)
                   for row in rows]
        responses = [thinking_token_ids(self.tokenizer, row["response"], add_special_tokens=True)
                     for row in rows]
        thinkings = [thinking_token_ids(self.tokenizer, row["thinking"], add_special_tokens=True)
                     for row in rows]
        return prompts, responses, thinkings

    def __call__(self, rows):
        batch = len(rows)
        prompts, responses, thinkings = self._encode(rows)
        plan_width = max(1, max(len(t) for t in thinkings))
        plan_width = min(plan_width, self.plan_token_capacity)

        input_ids = np.full((batch, self.max_length), self.pad_token_id, dtype=np.int64)
        plan_ids = np.full((batch, plan_width), self.pad_token_id, dtype=np.int64)
        plan_mask = np.zeros((batch, plan_width), dtype=bool)
        cond_lengths = np.zeros(batch, dtype=np.int64)
        total_lengths = np.zeros(batch, dtype=np.int64)

        cut = {key: np.zeros(batch, dtype=np.int64) for key in self.TRUNCATION_KEYS}
        for index, (prompt, response, thinking) in enumerate(zip(prompts, responses, thinkings)):
            if len(prompt) > self.condition_max_tokens:
                prompt = prompt[:self.condition_max_tokens]
                cut["prompt_truncated"][index] = 1
            room = self.max_length - len(prompt)
            if len(response) > room:
                response = response[:room]
                cut["response_truncated"][index] = 1
            if not response:
                raise ValueError(f"no room for a response: {rows[index]['example_id']}")
            if len(thinking) > plan_width:
                thinking = thinking[:plan_width]
                cut["thinking_truncated"][index] = 1

            sequence = prompt + response
            input_ids[index, :len(sequence)] = sequence
            cond_lengths[index] = len(prompt)
            total_lengths[index] = len(sequence)
            plan_ids[index, :len(thinking)] = thinking
            plan_mask[index, :len(thinking)] = True

        positions = np.arange(self.max_length)[None, :]
        is_cond = positions < cond_lengths[:, None]
        is_valid = positions < total_lengths[:, None]
        # A 2-D validity mask, not the upstream [B,S,S] form: the asymmetric
        # prompt/response visibility is applied by `encode_conditional_x0`, which
        # needs no transformers version that still accepts 3-D T5 masks.
        result = {
            "input_ids": torch.from_numpy(input_ids),
            "encoder_attention_mask": torch.from_numpy(is_valid.astype(np.float32)),
            "attention_mask": torch.from_numpy(is_valid.astype(np.float32)),
            "cond_seq_mask": torch.from_numpy(is_cond.astype(np.float32)),
            "plan_input_ids": torch.from_numpy(plan_ids),
            "plan_attention_mask": torch.from_numpy(plan_mask),
            "example_id": [row["example_id"] for row in rows],
            "source": [row["source"] for row in rows],
        }
        result.update({key: torch.from_numpy(value) for key, value in cut.items()})
        if "presentation_index" in rows[0]:
            result["presentation_index"] = torch.tensor(
                [int(row["presentation_index"]) for row in rows], dtype=torch.long)
        for key in SEED_KEYS:
            if key in rows[0]:
                result[key] = torch.tensor([int(row[key]) for row in rows], dtype=torch.long)
        return result


def plan_slots_for(token_count: int) -> int:
    return math.ceil(token_count / PLAN_GROUP_SIZE)


def plan_token_capacity_for(config) -> int:
    """Gold-thinking tokens the configured plan source can absorb."""
    slots = int(getattr(config, "max_plan_slots", None) or config.num_plan_slots or 255)
    if getattr(config, "plan_source", None) == "span_vae":
        from modules.plan_vae import K_MAX, SPAN
        if slots != K_MAX:
            raise ValueError(f"span_vae plan source needs max_plan_slots == {K_MAX}, got {slots}")
        return SPAN * K_MAX
    return PLAN_GROUP_SIZE * slots


def get_conditional_dataloader(dataset, tokenizer, config, batch_size: int,
                               num_workers: int = 0, distributed: bool = False):
    """Sequential loader over a fixed schedule: the order IS the experiment."""
    capacity = plan_token_capacity_for(config)
    collator = ConditionalCollator(
        tokenizer, max_length=config.max_length,
        condition_max_tokens=config.condition_max_tokens,
        max_plan_slots=int(getattr(config, "max_plan_slots", None) or config.num_plan_slots or 255),
        plan_token_capacity=capacity,
    )
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=False)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler, shuffle=False,
        num_workers=num_workers, collate_fn=collator, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    log_for_0(f"Conditional loader: {len(dataset)} rows, max_length={config.max_length}, "
              f"condition_max_tokens={config.condition_max_tokens}, plan_token_capacity={capacity}")
    return loader, collator


def truncation_metrics(batch):
    """Per-batch truncation counts, for the training log."""
    return {key: int(batch[key].sum()) for key in ConditionalCollator.TRUNCATION_KEYS
            if key in batch}
