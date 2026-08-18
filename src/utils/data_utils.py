import json
import hashlib
import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from utils.encoder_utils import build_self_attn_cond_masks
from utils.logging_utils import log_for_0
from utils.thinking_tokenization import thinking_token_ids


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FormalStageBPairedDataset(torch.utils.data.Dataset):
    """Lazy, verified train-only thinking/response pairs for formal Stage-B."""

    def __init__(self, manifest_path, tokenizer, max_length=1024):
        self.manifest_path = str(Path(manifest_path).resolve())
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not manifest.get("complete") or not manifest.get("training_ready"):
            raise ValueError("formal Stage-B manifest is not complete/training_ready")
        if not manifest.get("train_only") or manifest.get("validation_manifest") is not None \
                or manifest.get("test_manifest") is not None:
            raise ValueError("formal Stage-B requires a train-only manifest without val/test")
        split = manifest.get("splits", {}).get("train") or {}
        if manifest.get("paired_source_pool_manifest"):
            self._index_paired_source_pool(manifest, split)
            self.manifest = manifest
            return
        files = split.get("files") or []
        if not files:
            raise ValueError("formal Stage-B manifest has no train shard inventory")
        self.rows = []
        seen = set()
        for entry in files:
            path = Path(entry["path"])
            if not path.is_file() or path.stat().st_size != int(entry["bytes"]):
                raise ValueError(f"missing or size-mismatched Stage-B shard: {path}")
            if _sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Stage-B shard SHA256 mismatch: {path}")
            rows = 0
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    row = json.loads(line)
                    sample_id = str(row.get("sample_id") or row.get("source_id") or "")
                    source_id = str(row.get("source_id") or sample_id)
                    if not sample_id or sample_id in seen:
                        raise ValueError(f"missing/duplicate sample_id: {sample_id!r}")
                    if not isinstance(row.get("thinking"), str) or not row["thinking"].strip():
                        raise ValueError(f"missing thinking: {sample_id}")
                    if not isinstance(row.get("response"), str) or not row["response"].strip():
                        raise ValueError(f"missing response: {sample_id}")
                    thinking_tokens = int(row.get("thinking_tokens", 0))
                    response_tokens = int(row.get("response_tokens", 0))
                    if not 0 < thinking_tokens <= self.max_length:
                        raise ValueError(f"invalid thinking length for {sample_id}: {thinking_tokens}")
                    if not 0 < response_tokens <= self.max_length:
                        raise ValueError(f"invalid response length for {sample_id}: {response_tokens}")
                    seen.add(sample_id)
                    self.rows.append((str(path), offset, sample_id, source_id,
                                      thinking_tokens, response_tokens))
                    rows += 1
            if rows != int(entry["rows"]):
                raise ValueError(f"Stage-B shard row mismatch: {path}: {rows} != {entry['rows']}")
        if len(self.rows) != int(split.get("rows", manifest.get("rows", -1))):
            raise ValueError("formal Stage-B total row count mismatch")
        self.manifest = manifest

    def _index_paired_source_pool(self, manifest, split):
        selection_files = split.get("selection_files") or []
        wanted, selection_meta = [], {}
        for entry in selection_files:
            path = Path(entry["path"])
            if _sha256_file(path) != entry["sha256"]:
                raise ValueError(f"Stage-B selection shard SHA256 mismatch: {path}")
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    sid = str(row.get("sample_id") or row.get("source_id") or "")
                    if not sid or sid in selection_meta:
                        raise ValueError(f"missing/duplicate selected sample_id: {sid!r}")
                    selection_meta[sid] = {
                        "source_id": str(row.get("source_id") or sid),
                        "thinking_tokens": int(row["thinking_tokens"]),
                        "pair_hash": row.get("pair_hash"),
                    }
                    wanted.append(sid)
        if len(wanted) != int(split.get("rows", -1)):
            raise ValueError("Stage-B selection row count mismatch")
        pool_manifest_path = Path(manifest["paired_source_pool_manifest"])
        if _sha256_file(pool_manifest_path) != manifest["paired_source_pool_manifest_sha256"]:
            raise ValueError("paired source pool manifest SHA256 mismatch")
        pool_manifest = json.loads(pool_manifest_path.read_text())
        pool_dir = pool_manifest_path.parent
        found = {}
        for entry in pool_manifest["shards"]:
            path = pool_dir / entry["name"]
            if path.stat().st_size != int(entry["bytes"]) or _sha256_file(path) != entry["sha256"]:
                raise ValueError(f"paired source pool shard mismatch: {path}")
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell(); line = handle.readline()
                    if not line: break
                    row = json.loads(line)
                    sid = str(row.get("sample_id") or row.get("source_id") or "")
                    if sid not in selection_meta: continue
                    meta = selection_meta[sid]
                    source_pair_hash = row.get("exact_record_hash") or row.get("pair_hash")
                    if sid in found or source_pair_hash != meta["pair_hash"]:
                        raise ValueError(f"paired source identity/hash mismatch: {sid}")
                    thinking_tokens = int(row.get("thinking_tokens", 0))
                    response_tokens = int(row.get("response_tokens", 0))
                    if thinking_tokens != meta["thinking_tokens"]:
                        raise ValueError(f"paired thinking token mismatch: {sid}")
                    if not 0 < response_tokens <= self.max_length:
                        raise ValueError(f"invalid paired response length: {sid}")
                    found[sid] = (str(path), offset, sid, meta["source_id"],
                                  thinking_tokens, response_tokens)
        missing = set(wanted) - set(found)
        if missing:
            raise ValueError(f"paired source pool is missing {len(missing)} selected rows")
        self.rows = [found[sid] for sid in wanted]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path, offset, expected_id, source_id, expected_thinking, expected_response = self.rows[index]
        with open(path, "rb") as handle:
            handle.seek(offset)
            row = json.loads(handle.readline())
        if str(row.get("sample_id") or row.get("source_id")) != expected_id:
            raise RuntimeError("formal Stage-B shard changed after indexing")
        thinking_ids = thinking_token_ids(
            self.tokenizer, row["thinking"], add_special_tokens=True, truncation=False,
        )
        response_ids = thinking_token_ids(
            self.tokenizer, row["response"], add_special_tokens=True, truncation=False,
        )
        if len(thinking_ids) != expected_thinking or len(response_ids) != expected_response:
            raise ValueError(f"stored token count changed for {expected_id}")
        if len(thinking_ids) > self.max_length or len(response_ids) > self.max_length:
            raise ValueError(f"formal Stage-B sequence exceeds {self.max_length}: {expected_id}")
        return {
            "sample_id": expected_id, "source_id": source_id,
            "thinking_input_ids": thinking_ids, "response_input_ids": response_ids,
        }


class FormalStageBCollator:
    """Dynamic plan-token padding plus fixed-width Ordered response padding."""

    def __init__(self, pad_token_id, response_width=1024):
        self.pad_token_id = int(pad_token_id)
        self.response_width = int(response_width)

    def __call__(self, rows):
        batch = len(rows)
        plan_width = max(len(row["thinking_input_ids"]) for row in rows)
        plan_ids = torch.full((batch, plan_width), self.pad_token_id, dtype=torch.long)
        plan_mask = torch.zeros((batch, plan_width), dtype=torch.bool)
        response_ids = torch.full(
            (batch, self.response_width), self.pad_token_id, dtype=torch.long,
        )
        response_mask = torch.zeros((batch, self.response_width), dtype=torch.bool)
        for index, row in enumerate(rows):
            p = torch.as_tensor(row["thinking_input_ids"], dtype=torch.long)
            r = torch.as_tensor(row["response_input_ids"], dtype=torch.long)
            if p.numel() > 4 * 255 or r.numel() > self.response_width:
                raise ValueError(f"Stage-B sample exceeds configured width: {row['sample_id']}")
            plan_ids[index, :p.numel()] = p
            plan_mask[index, :p.numel()] = True
            response_ids[index, :r.numel()] = r
            response_mask[index, :r.numel()] = True
        result = {
            "sample_id": [row["sample_id"] for row in rows],
            "source_id": [row["source_id"] for row in rows],
            "input_ids": response_ids,
            "encoder_attention_mask": response_mask.float(),
            "attention_mask": response_mask.float(),
            "cond_seq_mask": torch.zeros_like(response_mask, dtype=torch.float32),
            "plan_input_ids": plan_ids,
            "plan_attention_mask": plan_mask,
        }
        for key in FormalStageBScheduleDataset.SEEDS:
            if key in rows[0]:result[key]=torch.tensor([int(row[key]) for row in rows],dtype=torch.long)
        if "presentation_index" in rows[0]:result["presentation_index"]=torch.tensor([int(row["presentation_index"]) for row in rows],dtype=torch.long)
        return result


class FormalStageBScheduleDataset(torch.utils.data.Dataset):
    """Sequential metadata schedule over a verified FormalStageBPairedDataset."""
    SEEDS = ("response_noise_seed", "token_time_seed", "branch_seed",
             "plan_noise_seed", "plan_time_seed")
    def __init__(self, base, schedule_path, expected_sha256=None):
        self.base=base;self.schedule_path=str(Path(schedule_path).resolve())
        if expected_sha256 and _sha256_file(schedule_path)!=expected_sha256:
            raise ValueError("formal Stage-B schedule SHA256 mismatch")
        by_id={row[2]:index for index,row in enumerate(base.rows)}
        self.rows=[];seen=set()
        with open(schedule_path) as handle:
            for index,line in enumerate(handle):
                row=json.loads(line);sid=str(row["sample_id"])
                if int(row["presentation_index"])!=index or int(row["optimizer_step"])!=index//8+1 or int(row["within_step"])!=index%8:
                    raise ValueError(f"schedule index/group mismatch at {index}")
                if sid in seen or sid not in by_id:raise ValueError(f"schedule missing/duplicate sample: {sid}")
                source=base.rows[by_id[sid]]
                if str(row["source_shard_path"])!=source[0] or int(row["source_byte_offset"])!=source[1] or int(row["thinking_tokens"])!=source[4] or int(row["response_tokens"])!=source[5]:
                    raise ValueError(f"schedule provenance mismatch: {sid}")
                if any(int(row[key])<0 for key in self.SEEDS):raise ValueError(f"invalid schedule RNG seed: {sid}")
                seen.add(sid);self.rows.append((by_id[sid],row))
        if len(self.rows) not in (160,80000):raise ValueError(f"schedule must contain 160 or 80000 rows, got {len(self.rows)}")
    def __len__(self):return len(self.rows)
    def __getitem__(self,index):
        base_index,meta=self.rows[index];row=self.base[base_index]
        row.update({key:int(meta[key]) for key in self.SEEDS})
        row.update({"presentation_index":int(meta["presentation_index"]),"pair_hash":meta["pair_hash"]})
        return row


def _process_count() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def _process_index() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def get_pad_token_id(tokenizer, pad_token: str = "pad") -> int:
    """Resolve the token id used for padding, optionally using EOS as pad."""
    token_id = tokenizer.eos_token_id if pad_token == "eos" else tokenizer.pad_token_id
    if token_id is None:
        raise ValueError("Tokenizer has no pad_token_id or eos_token_id.")
    return token_id


def prepare_batch(batch: Dict, config, generator: torch.Generator) -> Dict:
    """Convert numpy batch to torch tensors and sample label-drop decisions."""
    result = {}
    for k, v in batch.items():
        if isinstance(v, np.ndarray):
            result[k] = torch.from_numpy(v)
        elif isinstance(v, torch.Tensor):
            result[k] = v
        else:
            result[k] = v

    batch_size = result["input_ids"].shape[0]
    label_drop_mask = torch.zeros((batch_size,), dtype=torch.bool)
    if config.label_drop_prob > 0:
        u = torch.rand((batch_size,), generator=generator)
        label_drop_mask = u < config.label_drop_prob
    result["label_drop_mask"] = label_drop_mask
    return result


def pad_and_truncate(ids_list, target_len, pad_token_id):
    """Pad or truncate sequences to target_len, return stacked array and lengths."""
    padded, lengths = [], []
    for ids in ids_list:
        orig_len = min(len(ids), target_len)
        ids = ids[:target_len]
        if orig_len < target_len:
            ids = np.concatenate([ids, np.full(target_len - orig_len, pad_token_id, dtype=ids.dtype)])
        padded.append(ids)
        lengths.append(orig_len)
    return np.stack(padded), np.array(lengths)


def get_dataloader(
    dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = True,
    max_seq_length: int = 512,
    pad_token_id: int = 0,
    max_input_seq_length: Optional[int] = None,
    distributed: bool = True,
):
    """Create a DataLoader."""

    def collate_fn(batch_list):
        input_ids_list = [np.array(item["input_ids"]) for item in batch_list]

        if "condition_input_ids" in batch_list[0]:
            seq_list, cond_lens = [], []
            for item in batch_list:
                cond = np.array(item["condition_input_ids"])[:max_input_seq_length]
                inp = np.array(item["input_ids"])
                seq_list.append(np.concatenate([cond, inp]))
                cond_lens.append(len(cond))
            cond_lens = np.array(cond_lens)
        else:
            seq_list = input_ids_list
            cond_lens = np.zeros(len(input_ids_list), dtype=np.int32)

        ids, total_lens = pad_and_truncate(seq_list, max_seq_length, pad_token_id)
        pos = np.arange(max_seq_length)[None, :]
        is_cond = pos < cond_lens[:, None]
        is_valid = pos < total_lens[:, None]
        encoder_attn, attn, pred = build_self_attn_cond_masks(is_cond, is_valid, xp=np)
        result = {
            "input_ids": ids,
            "encoder_attention_mask": encoder_attn,
            "attention_mask": attn,
            "cond_seq_mask": pred,
        }
        for key in ("index", "input", "target"):
            if key in batch_list[0]:
                result[key] = [item[key] for item in batch_list]
        return result

    common = dict(
        batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
        drop_last=drop_last, persistent_workers=num_workers > 0,
        pin_memory=True,
    )
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=_process_count(), rank=_process_index(),
            shuffle=shuffle, drop_last=drop_last,
        )
        return DataLoader(dataset, sampler=sampler, **common)
    return DataLoader(dataset, shuffle=shuffle, **common)


def load_jsonl_dataset(path, tokenizer, input_key="input", output_key="output"):
    """Load a JSONL eval set (one `{input, output}` example per line)."""
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            examples.append({
                "index": i,
                "input": data[input_key],
                "target": data[output_key],
                "condition_input_ids": tokenizer(data[input_key], add_special_tokens=False)["input_ids"],
                "input_ids": tokenizer(data[output_key], add_special_tokens=False)["input_ids"],
            })
    return examples


def split_thinking_documents(rows, seed: int = 42, split_mode: str = "80_10_10"):
    """Deterministically split whole thinking documents without leakage."""
    if len(rows) < 10:
        raise ValueError("thinking plan training requires at least 10 documents")
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    if split_mode == "80_10_10":
        train_end = int(0.8 * len(rows))
        val_end = train_end + int(0.1 * len(rows))
    elif split_mode == "90_10":
        train_end = int(0.9 * len(rows))
        val_end = len(rows)
    else:
        raise ValueError(f"Unknown thinking split mode: {split_mode}")
    result = {}
    for name, indices in (
        ("train", order[:train_end]), ("val", order[train_end:val_end]),
        ("test", order[val_end:]),
    ):
        result[name] = [{**rows[index], "document_index": index} for index in indices]
    return result


def load_thinking_jsonl_splits(path: str, seed: int = 42, split_mode: str = "80_10_10"):
    with open(path, "r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for index, row in enumerate(rows):
        if not row.get("thinking_text", "").strip():
            raise ValueError(f"thinking_text is empty at document {index}")
        response = row.get("target_text") or row.get("source_context_text")
        if not response:
            raise ValueError(f"target_text/source_context_text is empty at document {index}")
    return split_thinking_documents(rows, seed=seed, split_mode=split_mode)


def collate_thinking_documents(
    batch_list, tokenizer, max_length: int, plan_add_special_tokens: bool = True,
):
    """Tokenize response and thinking separately; plan token length stays dynamic."""
    pad_token_id = get_pad_token_id(tokenizer)
    responses = [item.get("target_text") or item["source_context_text"] for item in batch_list]
    thinking = [item["thinking_text"] for item in batch_list]
    response_ids = [
        np.asarray(tokenizer(text, add_special_tokens=True)["input_ids"], dtype=np.int64)
        for text in responses
    ]
    ids, lengths = pad_and_truncate(response_ids, max_length, pad_token_id)
    position = np.arange(max_length)[None, :]
    is_valid = position < lengths[:, None]
    is_cond = np.zeros_like(is_valid)
    # Stage-B response has no condition prefix. T5 expects a [B,L] padding mask;
    # ELF receives its separate [B,L] valid-token mask below.
    encoder_attn = is_valid.astype(np.float32)
    attn = is_valid.astype(np.float32)
    pred = is_cond.astype(np.float32)

    plan_ids = [
        np.asarray(
            thinking_token_ids(
                tokenizer, text, add_special_tokens=plan_add_special_tokens,
            ),
            dtype=np.int64,
        )
        for text in thinking
    ]
    max_plan_tokens = max(map(len, plan_ids))
    plan_padded, plan_mask = [], []
    for values in plan_ids:
        width = len(values)
        plan_padded.append(np.pad(values, (0, max_plan_tokens - width), constant_values=pad_token_id))
        plan_mask.append(np.arange(max_plan_tokens) < width)
    return {
        "input_ids": torch.from_numpy(ids),
        "encoder_attention_mask": torch.from_numpy(encoder_attn),
        "attention_mask": torch.from_numpy(attn),
        "cond_seq_mask": torch.from_numpy(pred),
        "plan_input_ids": torch.from_numpy(np.stack(plan_padded)),
        "plan_attention_mask": torch.from_numpy(np.stack(plan_mask)).bool(),
        "document_index": torch.tensor([item["document_index"] for item in batch_list]),
        "source_hash": [item.get("source_hash", "") for item in batch_list],
    }


def get_thinking_dataloader(
    dataset, tokenizer, batch_size: int, max_length: int, shuffle: bool,
    num_workers: int = 0, drop_last: bool = True, distributed: bool = True,
    plan_add_special_tokens: bool = True,
):
    collate_fn = lambda rows: collate_thinking_documents(
        rows, tokenizer, max_length, plan_add_special_tokens=plan_add_special_tokens,
    )
    common = dict(
        batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn,
        drop_last=drop_last, persistent_workers=num_workers > 0, pin_memory=True,
    )
    if distributed:
        sampler = DistributedSampler(
            dataset, num_replicas=_process_count(), rank=_process_index(),
            shuffle=shuffle, drop_last=drop_last,
        )
        return DataLoader(dataset, sampler=sampler, **common)
    return DataLoader(dataset, shuffle=shuffle, **common)


# ============================================
# Dataset loading
# ============================================

def _looks_like_save_to_disk_arrow(ds) -> bool:
    """Detect HF datasets uploaded via `save_to_disk` (returns 1-row of metadata)."""
    return (
        len(ds) == 1
        and any(c.startswith("_") for c in ds.column_names)
        and not any(not c.startswith("_") for c in ds.column_names)
    )


def load_dataset_split(path: str, dataset_cache_dir=None):
    """Load a dataset. Tries HuggingFace Hub first; falls back to local on-disk Arrow."""
    from datasets import DatasetDict, load_dataset as hf_load_dataset, load_from_disk
    ds = None
    try:
        ds = hf_load_dataset(path, cache_dir=dataset_cache_dir)
    except Exception:
        ds = load_from_disk(path)

    if isinstance(ds, DatasetDict):
        splits = list(ds.keys())
        if len(splits) != 1:
            raise ValueError(f"Expected dataset at {path!r} to have a single split, got {splits}.")
        ds = ds[splits[0]]

    if _looks_like_save_to_disk_arrow(ds):
        from huggingface_hub import snapshot_download
        log_for_0(
            f"Dataset at {path!r} looks like a save_to_disk-format HF repo; "
            f"re-downloading via snapshot_download + load_from_disk."
        )
        local_dir = snapshot_download(repo_id=path, repo_type="dataset", cache_dir=dataset_cache_dir)
        ds = load_from_disk(local_dir)
        if isinstance(ds, DatasetDict):
            splits = list(ds.keys())
            if len(splits) != 1:
                raise ValueError(f"Expected dataset at {path!r} to have a single split, got {splits}.")
            ds = ds[splits[0]]

    ds.set_format(type="numpy", columns=ds.column_names)
    return ds


def load_dataset(config, dataset_cache_dir=None):
    """Resolve config.data_path / config.eval_data_path into train/eval datasets."""
    log_for_0(f"Loading dataset from {config.data_path}...")
    train_dataset = load_dataset_split(config.data_path, dataset_cache_dir)
    log_for_0(f"Train size: {len(train_dataset)}")

    eval_dataset = None
    if config.eval_data_path:
        eval_dataset = load_dataset_split(config.eval_data_path, dataset_cache_dir)
        log_for_0(f"Eval size: {len(eval_dataset)}")
    else:
        log_for_0("No eval dataset")
    return train_dataset, eval_dataset
