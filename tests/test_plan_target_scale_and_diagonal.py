"""Regression tests for the Stage-B plan target.

1. Stage B must hand the frozen Stage-A MLP the SAME normalized latents the MLP
   was trained on. Feeding raw T5 output shrinks the input by 1/latent_std and
   the compressed slots leave the manifold the Stage-A decoder was fitted to.
2. The diagonal group must satisfy t_plan == t_tok on every row, decoder rows
   included, because generation always decodes at (t_tok=1, t_plan=1).
"""

import os
import sys
import unittest

import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from configs.config import Config
from modules.model import ELF
from modules.thinking_resampler import build_adjacent_mlp_encoder
from train_step import train_step
from utils.encoder_utils import encode_thinking_x0
from utils.train_utils import TrainState

LATENT_STD = 0.2
WIDTH = 8


class StubT5(nn.Module):
    """Deterministic stand-in for the frozen T5 encoder."""

    def __init__(self, width=WIDTH):
        super().__init__()
        self.embedding = nn.Embedding(64, width)
        self.embedding.weight.data.normal_(0.0, 1.0)
        self.requires_grad_(False)

    def forward(self, input_ids, attention_mask=None, deterministic=True):
        return self.embedding(input_ids)


class ThinkingLatentScaleTests(unittest.TestCase):
    def test_stage_b_encode_applies_stage_a_normalization(self):
        encoder = StubT5()
        ids = torch.randint(0, 64, (3, 12))
        mask = torch.ones_like(ids, dtype=torch.bool)
        raw = encoder(input_ids=ids, attention_mask=mask).float()
        got = encode_thinking_x0(ids, mask, encoder, 0.0, LATENT_STD, use_bf16=False)
        torch.testing.assert_close(got, raw / LATENT_STD)
        self.assertGreater((got - raw).abs().max().item(), 0.0)

    def test_matches_the_canonical_stage_a_path(self):
        from src.utils.formal_thinking_mlp import canonical_t5_x0

        encoder = StubT5()
        ids = torch.randint(0, 64, (2, 16))
        mask = torch.ones_like(ids, dtype=torch.bool)
        torch.testing.assert_close(
            encode_thinking_x0(ids, mask, encoder, 0.0, LATENT_STD, use_bf16=False),
            canonical_t5_x0(encoder, ids, mask, 0.0, LATENT_STD).float(),
        )


def _tiny_config(group_mode, decoder_prob, latent_std=LATENT_STD):
    config = Config()
    config.latent_mean, config.latent_std = 0.0, latent_std
    config.max_length = 8
    config.num_plan_slots = config.max_plan_slots = 4
    config.num_plan_time_tokens = 1
    config.num_time_tokens = 1
    config.num_self_cond_cfg_tokens = 0
    config.num_model_mode_tokens = 1
    config.plan_source = "thinking_mlp_4to1"
    config.plan_resampler = "frozen_pool"
    config.plan_whiten = "none"
    config.plan_register_only = False
    config.plan_time_schedule = "uniform"
    config.plan_done_frac = 0.0
    config.plan_diag_frac = 1.0 if group_mode == "diagonal" else 0.0
    config.plan_loss_weight = 1.0
    config.group_mode = group_mode
    config.decoder_prob = decoder_prob
    config.self_cond_prob = 0.0
    config.label_drop_prob = 0.0
    config.denoiser_noise_scale = 1.0
    config.use_bf16 = False
    config.grad_accum_steps = 1
    return config


def _run_and_capture(group_mode, decoder_prob, latent_std=LATENT_STD):
    """Run one step and return the (t_plan, t_token) actually fed to the model."""
    torch.manual_seed(0)
    model = ELF(
        text_encoder_dim=WIDTH, max_length=8, hidden_size=16, depth=1, num_heads=4,
        bottleneck_dim=4, num_time_tokens=1, num_self_cond_cfg_tokens=0,
        num_model_mode_tokens=1, vocab_size=64, num_plan_slots=4,
        num_plan_time_tokens=1, plan_whiten="none",
    )
    captured = {}
    inner_forward = model.forward

    def recording_forward(*args, **kwargs):
        captured.setdefault("t", args[1] if len(args) > 1 else kwargs["t"])
        captured.setdefault("t_plan", kwargs.get("t_plan"))
        captured.setdefault("x_plan", kwargs.get("x_plan"))
        return inner_forward(*args, **kwargs)

    model.forward = recording_forward
    state = TrainState(
        model=model, optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
        dropout_generator=torch.Generator().manual_seed(0),
    )
    batch_size = 4
    batch = {
        "input_ids": torch.randint(0, 64, (batch_size, 8)),
        "encoder_attention_mask": torch.ones(batch_size, 8),
        "attention_mask": torch.ones(batch_size, 8),
        "cond_seq_mask": torch.zeros(batch_size, 8),
        "plan_input_ids": torch.randint(0, 64, (batch_size, 8)),
        "plan_attention_mask": torch.ones(batch_size, 8, dtype=torch.bool),
    }
    train_step(
        state, StubT5(), batch, _tiny_config(group_mode, decoder_prob, latent_std),
        plan_encoder=build_adjacent_mlp_encoder(WIDTH, 4, 16).requires_grad_(False),
        update_model=False,
    )
    return captured


class PlanTargetUsesNormalizedLatentsTests(unittest.TestCase):
    def test_train_step_plan_target_depends_on_latent_std(self):
        """A raw-latent plan path would ignore latent_std and produce the same target."""
        scaled = _run_and_capture("ordered", decoder_prob=1.0, latent_std=LATENT_STD)
        unscaled = _run_and_capture("ordered", decoder_prob=1.0, latent_std=1.0)
        self.assertGreater(
            (scaled["x_plan"] - unscaled["x_plan"]).abs().max().item(), 1e-4,
        )


class DiagonalClockTests(unittest.TestCase):
    def test_diagonal_decoder_rows_stay_on_the_diagonal(self):
        captured = _run_and_capture("diagonal", decoder_prob=1.0)
        torch.testing.assert_close(captured["t"], torch.ones_like(captured["t"]))
        torch.testing.assert_close(captured["t_plan"], captured["t"])

    def test_diagonal_denoiser_rows_stay_on_the_diagonal(self):
        captured = _run_and_capture("diagonal", decoder_prob=0.0)
        torch.testing.assert_close(captured["t_plan"], captured["t"])

    def test_ordered_decoder_rows_see_a_finished_plan(self):
        captured = _run_and_capture("ordered", decoder_prob=1.0)
        torch.testing.assert_close(captured["t_plan"], torch.ones_like(captured["t_plan"]))


if __name__ == "__main__":
    unittest.main()
