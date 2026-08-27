"""Streaming Dolma pretraining data: jsonl.zst shards -> fixed 1024-token windows.

Backbone pretraining only. Documents are tokenized with the frozen T5 tokenizer,
truncated to `max_length`, padded in the collate; documents shorter than
`min_tokens` are skipped. There is no condition prefix and no plan tokens --
batches carry the plain unconditional schema train_step already understands.

Sharding: the file list is split round-robin over (rank, dataloader worker), so
DDP needs no DistributedSampler and every worker owns a disjoint set of files.
Each pass over the dataset draws a fresh shard order (seeded by a per-iterator
counter), so "epochs" see the corpus in different orders. The dataset declares a
nominal __len__ (`samples_per_epoch`) purely so the trainer's epoch arithmetic
works; the stream itself is effectively unbounded.
"""

import io
import json
import random
from pathlib import Path

import torch

from utils.logging_utils import log_for_0


def list_dolma_shards(data_dir):
    shards = sorted(str(p) for p in Path(data_dir).glob("*/*.jsonl.zst"))
    if not shards:
        raise ValueError(f"no jsonl.zst shards under {data_dir}")
    return shards


def _iter_documents(shard_path):
    import zstandard

    with open(shard_path, "rb") as raw:
        stream = zstandard.ZstdDecompressor().stream_reader(raw)
        for line in io.TextIOWrapper(stream, encoding="utf-8", errors="replace"):
            if not line.strip():
                continue
            try:
                text = json.loads(line).get("text") or ""
            except json.JSONDecodeError:
                continue
            if text:
                yield text


class DolmaStreamDataset(torch.utils.data.IterableDataset):
    def __init__(self, data_dir, tokenizer, max_length=1024, min_tokens=64,
                 samples_per_epoch=256000, seed=42, rank=0, world=1):
        super().__init__()
        self.shards = list_dolma_shards(data_dir)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.min_tokens = int(min_tokens)
        self.samples_per_epoch = int(samples_per_epoch)
        self.seed = int(seed)
        self.rank, self.world = int(rank), int(world)
        self._passes = 0
        log_for_0(f"Dolma stream: {len(self.shards)} shards, rank {rank}/{world}")

    def __len__(self):
        return self.samples_per_epoch

    def _my_shards(self):
        info = torch.utils.data.get_worker_info()
        worker, workers = (info.id, info.num_workers) if info else (0, 1)
        stride = self.world * workers
        offset = self.rank * workers + worker
        order = list(self.shards)
        random.Random(f"{self.seed}-{self._passes}").shuffle(order)
        return order[offset::stride]

    def __iter__(self):
        self._passes += 1
        buffer, BATCH = [], 64
        for shard in self._my_shards():
            for text in _iter_documents(shard):
                buffer.append(text)
                if len(buffer) < BATCH:
                    continue
                yield from self._tokenize(buffer)
                buffer = []
        if buffer:
            yield from self._tokenize(buffer)

    def _tokenize(self, texts):
        encoded = self.tokenizer(texts, add_special_tokens=True, truncation=True,
                                 max_length=self.max_length,
                                 return_attention_mask=False)["input_ids"]
        for ids in encoded:
            if len(ids) >= self.min_tokens:
                yield ids


class DolmaCollator:
    """Pad to a fixed width; unconditional masks (cond_seq_mask all zero)."""

    def __init__(self, pad_token_id, max_length=1024):
        self.pad_token_id = int(pad_token_id)
        self.max_length = int(max_length)

    def __call__(self, rows):
        batch = len(rows)
        input_ids = torch.full((batch, self.max_length), self.pad_token_id, dtype=torch.long)
        valid = torch.zeros((batch, self.max_length), dtype=torch.float32)
        for i, ids in enumerate(rows):
            n = min(len(ids), self.max_length)
            input_ids[i, :n] = torch.as_tensor(ids[:n], dtype=torch.long)
            valid[i, :n] = 1.0
        return {"input_ids": input_ids, "encoder_attention_mask": valid,
                "attention_mask": valid,
                "cond_seq_mask": torch.zeros_like(valid)}


def get_dolma_dataloader(dataset, pad_token_id, batch_size, max_length=1024,
                         num_workers=4):
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size,
        collate_fn=DolmaCollator(pad_token_id, max_length),
        num_workers=num_workers, drop_last=True, pin_memory=True,
        persistent_workers=num_workers > 0, prefetch_factor=4 if num_workers else None,
    )
