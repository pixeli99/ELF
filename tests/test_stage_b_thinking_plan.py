import sys
from pathlib import Path
import unittest

import numpy as np
import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from modules.model import ELF
from modules.thinking_resampler import (
    AdjacentMLPResampler, build_adjacent_mlp_encoder, freeze_module,
)
from utils.data_utils import collate_thinking_documents, split_thinking_documents
from utils.plan_utils import (
    apply_plan_whitening, build_thinking_plan_target, masked_plan_mse,
)
from utils.sampling_utils import frozen_pool_plan_target


class FakeTokenizer:
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        return {"input_ids": [ord(char) % 31 + 1 for char in text] + ([1] if add_special_tokens else [])}


class ThinkingDataTest(unittest.TestCase):
    def test_variable_plan_lengths_are_padded_with_mask(self):
        batch = [
            {"thinking_text": "abc", "target_text": "response", "document_index": 0},
            {"thinking_text": "abcdefgh", "target_text": "target", "document_index": 1},
        ]
        output = collate_thinking_documents(batch, FakeTokenizer(), max_length=16)
        self.assertEqual(output["plan_input_ids"].shape, (2, 9))
        self.assertEqual(output["plan_attention_mask"].sum(dim=1).tolist(), [4, 9])

    def test_document_split_has_no_overlap(self):
        rows = [{"source_hash": str(index)} for index in range(100)]
        splits = split_thinking_documents(rows, seed=42)
        self.assertEqual([len(splits[key]) for key in ("train", "val", "test")], [80, 10, 10])
        hashes = [{row["source_hash"] for row in splits[key]} for key in ("train", "val", "test")]
        self.assertFalse(hashes[0] & hashes[1] or hashes[0] & hashes[2] or hashes[1] & hashes[2])

    def test_plan_tokens_do_not_add_eos(self):
        batch = [{"thinking_text": "abcd", "target_text": "response", "document_index": 0}]
        output = collate_thinking_documents(
            batch, FakeTokenizer(), max_length=16, plan_add_special_tokens=False,
        )
        self.assertEqual(output["plan_attention_mask"].sum().item(), 4)


class DynamicPlanModelTest(unittest.TestCase):
    def make_model(self):
        return ELF(
            text_encoder_dim=8, max_length=6, hidden_size=16, depth=1, num_heads=4,
            bottleneck_dim=4, num_time_tokens=1, num_self_cond_cfg_tokens=0,
            num_model_mode_tokens=0, vocab_size=20, num_plan_slots=8,
            num_plan_time_tokens=1, plan_whiten="zscore",
        ).eval()

    def test_runtime_k_slices_plan_slots_and_outputs(self):
        model = self.make_model()
        x = torch.randn(2, 5, 8)
        plan = torch.randn(2, 3, 8)
        mask = torch.tensor([[1, 1, 1], [1, 1, 0]]).bool()
        _, _, plan_output = model(x, torch.ones(2), x_plan=plan, t_plan=torch.ones(2), plan_mask=mask)
        self.assertEqual(plan_output.shape, (2, 3, 8))

    def test_plan_padding_cannot_affect_token_output(self):
        model = self.make_model()
        x = torch.randn(1, 5, 8)
        plan_a = torch.randn(1, 3, 8)
        plan_b = plan_a.clone()
        plan_b[:, 2] = 1000
        mask = torch.tensor([[1, 1, 0]]).bool()
        out_a = model(x, torch.ones(1), x_plan=plan_a, t_plan=torch.ones(1), plan_mask=mask)[0]
        out_b = model(x, torch.ones(1), x_plan=plan_b, t_plan=torch.ones(1), plan_mask=mask)[0]
        torch.testing.assert_close(out_a, out_b)

    def test_max_plan_slots_overflow_is_explicit(self):
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, "max_plan_slots"):
            model(torch.randn(1, 5, 8), torch.ones(1), x_plan=torch.randn(1, 9, 8))


class ThinkingTargetTest(unittest.TestCase):
    def test_different_document_lengths_produce_dynamic_slot_mask(self):
        latents = torch.randn(2, 9, 8)
        token_mask = torch.tensor([
            [1, 1, 1, 1, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]).bool()
        encoder = build_adjacent_mlp_encoder(8, 4, 16)
        plans, plan_mask = build_thinking_plan_target(latents, token_mask, encoder, 3)
        self.assertEqual(plans.shape, (2, 3, 8))
        self.assertTrue(torch.equal(
            plan_mask, torch.tensor([[1, 1, 0], [1, 1, 1]]).bool(),
        ))

    def test_compressed_plan_overflow_is_explicit(self):
        encoder = build_adjacent_mlp_encoder(8, 4, 16)
        with self.assertRaisesRegex(ValueError, "max_plan_slots=2"):
            build_thinking_plan_target(
                torch.randn(1, 9, 8), torch.ones(1, 9).bool(), encoder, 2,
            )


class MaskedPlanObjectiveTest(unittest.TestCase):
    def test_padding_does_not_enter_loss_or_whitening(self):
        target = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]]])
        prediction = target.clone()
        prediction[:, 2] = -100
        mask = torch.tensor([[1, 1, 0]]).bool()
        self.assertEqual(float(masked_plan_mse(prediction, target, mask)), 0.0)

        holder = type("Holder", (), {})()
        holder.plan_whiten = "zscore"
        holder.plan_target_mean = torch.tensor([1.0, 1.0])
        holder.plan_target_std = torch.tensor([2.0, 2.0])
        whitened = apply_plan_whitening(holder, target, mask)
        self.assertTrue(torch.equal(whitened[:, 2], torch.zeros_like(whitened[:, 2])))

    def test_t5_and_stage_a_encoder_can_be_frozen(self):
        t5 = freeze_module(torch.nn.Linear(8, 8))
        resampler = AdjacentMLPResampler(embedding_dim=8, group_size=4, hidden_dim=16)
        freeze_module(resampler.encoder)
        self.assertTrue(all(not p.requires_grad for p in t5.parameters()))
        self.assertTrue(all(not p.requires_grad for p in resampler.encoder.parameters()))

if __name__ == "__main__":
    unittest.main()
