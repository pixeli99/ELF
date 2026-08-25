"""The four-group contract, stated once and checked directly.

`utils/plan_stream.build_plan_stream` is the single place a group's plan tensors
come from, so this is the file that says what each group means.
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from configs.config import Config, load_config_from_yaml
from modules.model import ELF
from modules.thinking_resampler import build_adjacent_mlp_encoder
from utils.plan_stream import (GROUP_PROTOCOL, assert_group_protocol, build_plan_stream,
                               plan_slot_lengths, resolve_group)

WIDTH, SEQ, VOCAB, SLOTS = 8, 8, 64, 8
CONFIG_DIR = os.path.join(ROOT, "src/configs/training_configs")


class StubT5(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, WIDTH)
        self.embedding.weight.data.normal_(0.0, 1.0)
        self.requires_grad_(False)

    def forward(self, input_ids, attention_mask=None, deterministic=True):
        return self.embedding(input_ids)


def _config(mode):
    c = Config()
    c.latent_mean, c.latent_std = 0.0, 0.2
    c.max_length = SEQ
    c.plan_source = "thinking_mlp_4to1"
    c.plan_whiten = "none"
    c.plan_time_schedule = "uniform"
    c.denoiser_noise_scale = 1.0
    c.use_bf16 = False
    c.group_mode = mode
    c.num_plan_slots = c.max_plan_slots = 0 if mode == "vanilla" else SLOTS
    c.plan_register_only = mode == "register"
    c.plan_done_frac = 0.15 if mode == "ordered" else 0.0
    c.plan_diag_frac = 1.0 if mode == "diagonal" else 0.0
    c.plan_loss_weight = 0.0 if mode == "register" else 1.0
    return c


def _stream(mode, decoder_rows, thinking_lengths=(8, 4)):
    """Build one group's plan stream. `decoder_rows` is the per-row branch gate."""
    config = _config(mode)
    torch.manual_seed(0)
    model = ELF(text_encoder_dim=WIDTH, max_length=SEQ, hidden_size=16, depth=1,
                num_heads=4, bottleneck_dim=4, num_time_tokens=1,
                num_self_cond_cfg_tokens=0, num_model_mode_tokens=1, vocab_size=VOCAB,
                num_plan_slots=config.num_plan_slots, num_plan_time_tokens=1,
                plan_whiten="none")
    batch_size = len(thinking_lengths)
    plan_mask = torch.zeros(batch_size, SEQ, dtype=torch.bool)
    for row, length in enumerate(thinking_lengths):
        plan_mask[row, :length] = True
    batch = {
        "plan_input_ids": torch.randint(0, VOCAB, (batch_size, SEQ)),
        "plan_attention_mask": plan_mask,
    }
    t = torch.rand(batch_size)
    stream = build_plan_stream(
        config=config, group=resolve_group(config), batch=batch, model=model,
        encoder=StubT5(), plan_encoder=build_adjacent_mlp_encoder(WIDTH, 4, 16),
        t=t, decoder_step_active=torch.tensor(decoder_rows, dtype=torch.float32),
        x0=torch.randn(batch_size, SEQ, WIDTH), loss_mask=torch.ones(batch_size, SEQ),
        schedule_seed=lambda *a, **k: None,
    )
    return stream, t


class GroupContractTests(unittest.TestCase):
    def test_vanilla_contributes_nothing(self):
        stream, _ = _stream("vanilla", [0.0, 0.0])
        self.assertIsNone(stream.x_plan_input)
        self.assertIsNone(stream.t_plan_input)
        self.assertIsNone(stream.plan_mask)
        self.assertFalse(stream.supervised)

    def test_register_is_noise_at_clock_zero_on_every_row(self):
        stream, _ = _stream("register", [1.0, 0.0])
        self.assertFalse(stream.supervised)
        self.assertIsNone(stream.x0_plan)
        torch.testing.assert_close(stream.t_plan_input, torch.zeros(2))
        # Same compute budget as ordered: K follows the thinking length.
        self.assertEqual(stream.plan_mask.sum(1).tolist(), [2, 1])
        # Padding slots carry no content.
        self.assertEqual(float(stream.x_plan_input[~stream.plan_mask].abs().max()), 0.0)

    def test_ordered_decoder_rows_get_a_finished_plan(self):
        stream, _ = _stream("ordered", [1.0, 0.0])
        self.assertTrue(stream.supervised)
        self.assertEqual(float(stream.t_plan_input[0]), 1.0)
        torch.testing.assert_close(stream.x_plan_input[0], stream.x0_plan[0])
        # Denoiser rows see the noised plan at their own clock instead.
        torch.testing.assert_close(stream.t_plan_input[1], stream.plan_t[1])
        torch.testing.assert_close(stream.x_plan_input[1], stream.plan_z[1])

    def test_diagonal_pins_the_plan_clock_to_the_token_clock(self):
        for gate in ([0.0, 0.0], [1.0, 1.0], [1.0, 0.0]):
            with self.subTest(decoder_rows=gate):
                stream, t = _stream("diagonal", gate)
                token_clock = torch.where(torch.tensor(gate) > 0, torch.ones_like(t), t)
                torch.testing.assert_close(stream.t_plan_input, token_clock)

    def test_ordered_and_register_agree_on_k(self):
        ordered, _ = _stream("ordered", [0.0, 0.0])
        register, _ = _stream("register", [0.0, 0.0])
        self.assertEqual(ordered.plan_mask.tolist(), register.plan_mask.tolist())

    def test_slot_lengths_round_up(self):
        mask = torch.zeros(5, 16, dtype=torch.bool)
        for row, length in enumerate((1, 4, 5, 12, 16)):
            mask[row, :length] = True
        self.assertEqual(plan_slot_lengths(mask).tolist(), [1, 1, 2, 3, 4])


class GroupValidationTests(unittest.TestCase):
    def test_shipped_overlays_match_their_labels(self):
        for mode in ("ordered", "diagonal", "register", "vanilla"):
            with self.subTest(mode=mode):
                config = load_config_from_yaml(
                    os.path.join(CONFIG_DIR, f"train_stage_b_common80k_{mode}_10k_v1.yml"))
                group = assert_group_protocol(config)
                self.assertEqual(group.mode, mode)
                self.assertEqual(group.plan_enabled, mode != "vanilla")
                self.assertEqual(group.supervised, mode in ("ordered", "diagonal"))

    def test_mislabelled_group_is_rejected(self):
        config = _config("diagonal")
        config.plan_diag_frac = 0.0          # a diagonal run that is not diagonal
        with self.assertRaises(ValueError):
            assert_group_protocol(config)

    def test_vanilla_with_slots_is_rejected(self):
        config = _config("vanilla")
        config.num_plan_slots = 8
        with self.assertRaises(ValueError):
            resolve_group(config)

    def test_unknown_group_is_rejected(self):
        config = _config("ordered")
        config.group_mode = "planned"
        with self.assertRaises(ValueError):
            resolve_group(config)

    def test_protocol_table_covers_every_plan_group(self):
        self.assertEqual(set(GROUP_PROTOCOL), {"ordered", "diagonal", "register"})

    def test_exploratory_sweeps_are_not_blocked_by_the_structural_check(self):
        """resolve_group must stay permissive; only the launch path is strict."""
        config = _config("ordered")
        config.plan_done_frac = 0.5
        self.assertTrue(resolve_group(config).supervised)
        with self.assertRaises(ValueError):
            assert_group_protocol(config)


if __name__ == "__main__":
    unittest.main()
