"""The Dolma streaming pretraining loader."""

import io
import json
import os
import sys
import unittest
from pathlib import Path

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils.dolma_data import DolmaCollator, DolmaStreamDataset, list_dolma_shards


class WordTokenizer:
    pad_token_id = 0

    def __call__(self, texts, add_special_tokens=True, truncation=True,
                 max_length=1024, return_attention_mask=False):
        out = []
        for text in texts:
            ids = [2 + (abs(hash(w)) % 60) for w in text.split()]
            if add_special_tokens:
                ids = ids + [1]
            out.append(ids[:max_length] if truncation else ids)
        return {"input_ids": out}


def _write_shards(root, n_shards=4, docs_per_shard=30):
    import zstandard

    for s in range(n_shards):
        d = Path(root) / f"source-{s:02d}"
        d.mkdir(parents=True, exist_ok=True)
        payload = "".join(
            json.dumps({"id": f"{s}-{i}",
                        "text": " ".join(f"w{s}x{i}n{j}" for j in range(5 + 7 * i))})
            + "\n" for i in range(docs_per_shard))
        (d / "shard_0.jsonl.zst").write_bytes(
            zstandard.ZstdCompressor().compress(payload.encode()))


class DolmaStreamTests(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.dir = tempfile.mkdtemp()
        _write_shards(self.dir)

    def _dataset(self, **kw):
        args = dict(tokenizer=WordTokenizer(), max_length=64, min_tokens=8,
                    samples_per_epoch=100, seed=1, rank=0, world=1)
        args.update(kw)
        return DolmaStreamDataset(self.dir, **args)

    def test_short_documents_are_skipped_and_long_ones_truncated(self):
        rows = list(self._dataset())
        self.assertTrue(all(8 <= len(r) <= 64 for r in rows))
        self.assertTrue(any(len(r) == 64 for r in rows))  # truncation happened

    def test_ranks_see_disjoint_shards(self):
        a = [tuple(r) for r in self._dataset(rank=0, world=2)]
        b = [tuple(r) for r in self._dataset(rank=1, world=2)]
        self.assertTrue(a and b)
        self.assertFalse(set(a) & set(b))

    def test_passes_reshuffle_but_cover_the_same_corpus(self):
        ds = self._dataset()
        first = [tuple(r) for r in ds]
        second = [tuple(r) for r in ds]
        self.assertEqual(sorted(first), sorted(second))
        self.assertNotEqual(first, second)

    def test_collator_masks(self):
        collate = DolmaCollator(pad_token_id=0, max_length=64)
        batch = collate([[5, 6, 7], list(range(2, 66))])
        self.assertEqual(tuple(batch["input_ids"].shape), (2, 64))
        self.assertEqual(batch["attention_mask"].sum(1).tolist(), [3.0, 64.0])
        self.assertEqual(float(batch["cond_seq_mask"].sum()), 0.0)
        self.assertEqual(int(batch["input_ids"][0, 3]), 0)

    def test_missing_dir_is_loud(self):
        with self.assertRaises(ValueError):
            list_dolma_shards(self.dir + "/nothing-here")


if __name__ == "__main__":
    unittest.main()
