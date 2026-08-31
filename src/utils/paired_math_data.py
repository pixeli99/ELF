"""Paired math data: prompt, reasoning, short answer.

Built by `tools/build_paired_math.py`. Each row carries the reasoning and the final
answer separately, so the three arms differ in what the model has to emit:

    answer      the no-reasoning arm and the plan arm: the window holds prompt + answer
    cot_answer  the explicit-reasoning arm: the window holds prompt + reasoning + answer

Measured on the package: prompt + answer fits 512 positions for 99.7% of rows, while
prompt + reasoning + answer needs 1024 for 92.8%. A diffusion model denoises the whole
canvas at every step, so that window difference is the cost axis.

Padding follows the ELF conditional recipe: the tail after the target is filled with EOS
and a short band of it stays in the loss, which is what teaches the model to stop.
"""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow.parquet as pq
import torch

from utils.data_utils import get_pad_token_id
from utils.logging_utils import log_for_0
from utils.thinking_tokenization import thinking_token_ids

TARGETS = ("answer", "cot_answer")


class PairedMathDataset(torch.utils.data.Dataset):
    """Parquet-backed rows of example_id / source / prompt / thinking / answer."""

    COLUMNS = ("example_id", "source", "prompt", "thinking", "answer")

    def __init__(self, data_dir: str, split: str = "train"):
        root = Path(data_dir)
        manifest = json.loads((root / "manifest.json").read_text())
        if split == "train":
            names = manifest["shards"]
        elif split == "heldout":
            names = [manifest["heldout_shard"]]
        else:
            raise ValueError(f"unknown split: {split}")
        tables = [pq.read_table(root / "data" / n, columns=list(self.COLUMNS)) for n in names]
        self._columns = {c: pa_concat(tables, c) for c in self.COLUMNS}
        self._length = len(self._columns["example_id"])
        log_for_0(f"Paired math {split}: {self._length} rows from {len(names)} shard(s) in {root}")

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        return {c: self._columns[c][index].as_py() for c in self.COLUMNS}


def pa_concat(tables, column):
    import pyarrow as pa
    return pa.concat_arrays([t.column(column).combine_chunks() for t in tables])


class PairedMathCollator:
    """[prompt | target] in one window, prompt as a clean prefix that carries no loss."""

    TRUNCATION_KEYS = ("prompt_truncated", "target_truncated")

    def __init__(self, tokenizer, max_length: int = 512, condition_max_tokens: int = 384,
                 target: str = "answer", plan_token_capacity: int = 0,
                 pad_token_id: Optional[int] = None):
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}")
        if condition_max_tokens >= max_length:
            raise ValueError("condition_max_tokens must leave room for a target")
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.condition_max_tokens = int(condition_max_tokens)
        self.target = target
        self.plan_token_capacity = int(plan_token_capacity)
        self.pad_token_id = int(pad_token_id if pad_token_id is not None else tokenizer.pad_token_id)

    def _target_text(self, row):
        if self.target == "answer":
            return row["answer"]
        return f"{row['thinking']}\nThe answer is {row['answer']}"

    def __call__(self, rows):
        batch = len(rows)
        prompts = [thinking_token_ids(self.tokenizer, r["prompt"], add_special_tokens=False)
                   for r in rows]
        targets = [thinking_token_ids(self.tokenizer, self._target_text(r), add_special_tokens=True)
                   for r in rows]

        input_ids = np.full((batch, self.max_length), self.pad_token_id, dtype=np.int64)
        cond_lengths = np.zeros(batch, dtype=np.int64)
        total_lengths = np.zeros(batch, dtype=np.int64)
        cut = {k: np.zeros(batch, dtype=np.int64) for k in self.TRUNCATION_KEYS}

        for index, (prompt, target) in enumerate(zip(prompts, targets)):
            if len(prompt) > self.condition_max_tokens:
                prompt = prompt[:self.condition_max_tokens]
                cut["prompt_truncated"][index] = 1
            room = self.max_length - len(prompt)
            if len(target) > room:
                target = target[:room]
                cut["target_truncated"][index] = 1
            if not target:
                raise ValueError(f"no room for a target: {rows[index]['example_id']}")
            sequence = prompt + target
            input_ids[index, :len(sequence)] = sequence
            cond_lengths[index] = len(prompt)
            total_lengths[index] = len(sequence)

        positions = np.arange(self.max_length)[None, :]
        is_cond = positions < cond_lengths[:, None]
        is_valid = positions < total_lengths[:, None]
        result = {
            "input_ids": torch.from_numpy(input_ids),
            "encoder_attention_mask": torch.from_numpy(is_valid.astype(np.float32)),
            "attention_mask": torch.from_numpy(is_valid.astype(np.float32)),
            "cond_seq_mask": torch.from_numpy(is_cond.astype(np.float32)),
            "example_id": [r["example_id"] for r in rows],
            "source": [r["source"] for r in rows],
        }
        result.update({k: torch.from_numpy(v) for k, v in cut.items()})

        if self.plan_token_capacity > 0:
            width = self.plan_token_capacity
            plan_ids = np.full((batch, width), self.pad_token_id, dtype=np.int64)
            plan_mask = np.zeros((batch, width), dtype=bool)
            for index, r in enumerate(rows):
                ids = thinking_token_ids(self.tokenizer, r["thinking"], add_special_tokens=True)[:width]
                plan_ids[index, :len(ids)] = ids
                plan_mask[index, :len(ids)] = True
            result["plan_input_ids"] = torch.from_numpy(plan_ids)
            result["plan_attention_mask"] = torch.from_numpy(plan_mask)
        return result


def get_paired_math_dataloader(dataset, tokenizer, config, batch_size: int,
                               num_workers: int = 0, distributed: bool = False,
                               shuffle: bool = True):
    capacity = 0
    if int(getattr(config, "num_plan_slots", 0) or 0) > 0:
        from modules.plan_vae import K_MAX, SPAN
        capacity = SPAN * K_MAX
    collator = PairedMathCollator(
        tokenizer, max_length=config.max_length,
        condition_max_tokens=config.condition_max_tokens,
        target=getattr(config, "paired_math_target", "answer"),
        plan_token_capacity=capacity,
        pad_token_id=get_pad_token_id(tokenizer, getattr(config, "pad_token", "pad")),
    )
    sampler = None
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        shuffle=(shuffle and sampler is None),
        num_workers=num_workers, collate_fn=collator, drop_last=True,
        persistent_workers=num_workers > 0,
    )
    log_for_0(f"Paired math loader: target={collator.target}, max_length={config.max_length}, "
              f"condition_max_tokens={config.condition_max_tokens}, "
              f"plan_token_capacity={capacity}, pad_token={getattr(config, 'pad_token', 'pad')}")
    return loader, collator
