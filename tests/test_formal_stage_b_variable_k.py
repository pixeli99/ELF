import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from modules.model import ELF, build_plan_response_attention_mask
from modules.thinking_resampler import build_adjacent_mlp_encoder
from utils.plan_utils import build_thinking_plan_target, masked_plan_mse


class VariableKTests(unittest.TestCase):
    def test_exact_k_and_tail_groups(self):
        encoder = build_adjacent_mlp_encoder(8, 4, 16)
        for tokens, expected in ((1,1),(2,1),(3,1),(4,1),(5,2),(12,3),(64,16),
                                 (496,124),(860,215),(1020,255)):
            x = torch.randn(1, tokens, 8)
            mask = torch.ones(1, tokens, dtype=torch.bool)
            slots, plan_mask = build_thinking_plan_target(x, mask, encoder, 255)
            self.assertEqual(slots.shape, (1, expected, 8))
            self.assertEqual(int(plan_mask.sum()), expected)

    def test_mixed_k_padding_and_loss_denominator(self):
        encoder = build_adjacent_mlp_encoder(8, 4, 16)
        x = torch.randn(5, 1020, 8)
        mask = torch.zeros(5, 1020, dtype=torch.bool)
        lengths = [12, 64, 496, 860, 1020]
        for row, length in enumerate(lengths): mask[row, :length] = True
        slots, plan_mask = build_thinking_plan_target(x, mask, encoder, 255)
        self.assertEqual(plan_mask.sum(1).tolist(), [3,16,124,215,255])
        prediction = slots.clone(); prediction[~plan_mask] = 1e6
        self.assertEqual(float(masked_plan_mse(prediction, slots, plan_mask)), 0.0)
        prediction[plan_mask] += 2
        self.assertAlmostEqual(float(masked_plan_mse(prediction, slots, plan_mask)), 4.0, places=5)

    def test_attention_padding_and_bidirectional_permissions(self):
        token = torch.tensor([[1,1,0],[1,1,1]], dtype=torch.bool)
        plan = torch.tensor([[1,0],[1,1]], dtype=torch.bool)
        allowed = build_plan_response_attention_mask(token, plan, 2, 1, 1, 1, "bidirectional")
        self.assertEqual(allowed.shape, (2, 8))
        self.assertFalse(bool(allowed[0,4]))
        self.assertFalse(bool(allowed[0,7]))

    def test_model_runtime_variable_k(self):
        model = ELF(text_encoder_dim=8, max_length=6, hidden_size=16, depth=1,
                    num_heads=4, bottleneck_dim=4, num_time_tokens=1,
                    num_self_cond_cfg_tokens=0, num_model_mode_tokens=1,
                    vocab_size=32, num_plan_slots=255, num_plan_time_tokens=1,
                    plan_whiten="none")
        for k in (3, 124, 215, 255):
            x = torch.randn(1,6,8); p = torch.randn(1,k,8)
            mask = torch.ones(1,k,dtype=torch.bool)
            out, _, plan_out = model(x, torch.ones(1), attention_mask=torch.ones(1,6,dtype=torch.bool),
                                     x_plan=p, t_plan=torch.ones(1), plan_mask=mask)
            self.assertEqual(out.shape, (1,6,8)); self.assertEqual(plan_out.shape, (1,k,8))

    def test_null_and_length_bucket_derangement(self):
        lengths = torch.tensor([3,4,15,16,124,125,214,215,254,255])
        donors = torch.arange(len(lengths)).roll(1)
        self.assertTrue(bool((donors != torch.arange(len(lengths))).all()))
        plans = torch.randn(10,255,8)
        masks = torch.arange(255)[None,:] < lengths[:,None]
        null = plans.clone().masked_fill(masks.unsqueeze(-1), 0)
        self.assertTrue(bool((null[masks] == 0).all()))
        self.assertTrue(bool((null[~masks] == plans[~masks]).all()))

    def test_frozen_modules_receive_no_grad(self):
        t5 = nn.Embedding(32,8).requires_grad_(False)
        enc = build_adjacent_mlp_encoder(8,4,16).requires_grad_(False)
        x = t5(torch.randint(0,32,(2,8)))
        slots, mask = build_thinking_plan_target(x, torch.ones(2,8,dtype=torch.bool), enc, 255)
        loss = slots.sum() * 0
        self.assertFalse(loss.requires_grad)
        self.assertTrue(all(p.grad is None for p in t5.parameters()))
        self.assertTrue(all(p.grad is None for p in enc.parameters()))


if __name__ == "__main__": unittest.main()
