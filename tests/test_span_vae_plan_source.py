"""The span_vae plan source: generated-null dynamic budget, no mask inputs."""

import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from configs.config import Config, load_config_from_yaml
from modules.model import ELF
from modules.plan_vae import K_MAX, Z_DIM, PlanVAE, span_pool
from train_step import train_step
from utils.plan_stream import assert_group_protocol, build_vae_plan_target, resolve_group
from utils.train_utils import TrainState

WIDTH, VOCAB = 512, 64
CONFIG_DIR = os.path.join(ROOT, "src/configs/training_configs")


class StubT5(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, WIDTH)
        self.embedding.weight.data.normal_(0.0, 1.0)
        self.requires_grad_(False)

    def forward(self, input_ids, attention_mask=None, deterministic=True):
        return self.embedding(input_ids)


def _config(mode="ordered"):
    c = Config()
    c.latent_mean, c.latent_std = 0.0, 0.2
    c.max_length = 64
    c.plan_source = "span_vae"
    c.plan_whiten = "external"
    c.plan_target_dim = Z_DIM
    c.plan_time_schedule = "uniform"
    c.num_time_tokens, c.num_plan_time_tokens = 1, 1
    c.num_self_cond_cfg_tokens, c.num_model_mode_tokens = 0, 1
    c.decoder_prob = 0.0
    c.self_cond_prob = 0.0
    c.label_drop_prob = 0.0
    c.denoiser_noise_scale = 1.0
    c.use_bf16 = False
    c.grad_accum_steps = 1
    c.group_mode = mode
    c.num_plan_slots = c.max_plan_slots = 0 if mode == "vanilla" else K_MAX
    c.plan_register_only = mode == "register"
    c.plan_done_frac = 0.15 if mode == "ordered" else 0.0
    c.plan_diag_frac = 1.0 if mode == "diagonal" else 0.0
    c.plan_loss_weight = 0.0 if mode == "register" else 1.0
    return c


class SpanPoolTests(unittest.TestCase):
    def test_active_follows_thinking_length(self):
        x = torch.randn(2, 40, WIDTH)
        mask = torch.zeros(2, 40, dtype=torch.bool)
        mask[0, :37], mask[1, :3] = True, True
        _, active = span_pool(x, mask)
        self.assertEqual(active.sum(1).tolist(), [3, 1])   # ceil(37/16), ceil(3/16)

    def test_pooling_ignores_padded_tokens(self):
        x = torch.randn(1, 40, WIDTH)
        mask = torch.zeros(1, 40, dtype=torch.bool)
        mask[0, :10] = True
        pooled, _ = span_pool(x, mask)
        torch.testing.assert_close(pooled[0, 0], x[0, :10].mean(0))


class GeneratedNullTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.vae = PlanVAE().eval().requires_grad_(False)

    def test_x0_plan_tail_is_exact_zero_and_mask_is_all_ones(self):
        config = _config()
        ids = torch.randint(0, VOCAB, (2, 40))
        mask = torch.zeros(2, 40, dtype=torch.bool)
        mask[0, :37], mask[1, :12] = True, True
        x0_plan, plan_mask, active = build_vae_plan_target(
            ids, mask, StubT5(), self.vae, config)
        self.assertEqual(tuple(x0_plan.shape), (2, K_MAX, Z_DIM))
        self.assertTrue(bool(plan_mask.all()), "nulls are content, not padding")
        self.assertEqual(active.sum(1).tolist(), [3, 1])
        self.assertEqual(float(x0_plan[0, 3:].abs().max()), 0.0)
        self.assertEqual(float(x0_plan[1, 1:].abs().max()), 0.0)
        self.assertGreater(float(x0_plan[0, :3].abs().max()), 0.0)

    def test_budget_is_a_pure_function_of_valid_length(self):
        config = _config()
        ids = torch.randint(0, VOCAB, (1, 64))
        for tokens, slots in ((1, 1), (16, 1), (17, 2), (33, 3), (64, 4)):
            mask = torch.zeros(1, 64, dtype=torch.bool)
            mask[0, :tokens] = True
            _, _, active = build_vae_plan_target(ids, mask, StubT5(), self.vae, config)
            self.assertEqual(int(active.sum()), slots, f"{tokens} tokens")


class SpanVaeTrainStepTests(unittest.TestCase):
    def _run(self, mode):
        config = _config(mode)
        torch.manual_seed(0)
        model = ELF(text_encoder_dim=WIDTH, max_length=64, hidden_size=16, depth=1,
                    num_heads=4, bottleneck_dim=4, num_time_tokens=1,
                    num_self_cond_cfg_tokens=0, num_model_mode_tokens=1, vocab_size=VOCAB,
                    num_plan_slots=config.num_plan_slots, num_plan_time_tokens=1,
                    plan_whiten=config.plan_whiten, plan_target_dim=config.plan_target_dim)
        state = TrainState(model=model,
                           optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
                           dropout_generator=torch.Generator().manual_seed(0))
        batch = {
            "input_ids": torch.randint(0, VOCAB, (2, 64)),
            "encoder_attention_mask": torch.ones(2, 64),
            "attention_mask": torch.ones(2, 64),
            "cond_seq_mask": torch.zeros(2, 64),
            "plan_input_ids": torch.randint(0, VOCAB, (2, 40)),
            "plan_attention_mask": torch.ones(2, 40, dtype=torch.bool),
        }
        vae = PlanVAE().eval().requires_grad_(False) if mode in ("ordered", "diagonal") else None
        _, metrics = train_step(state, StubT5(), batch, config,
                                plan_encoder=vae, update_model=False)
        return metrics

    def test_all_groups_run_with_span_vae(self):
        for mode in ("ordered", "diagonal", "register", "vanilla"):
            with self.subTest(mode=mode):
                metrics = self._run(mode)
                self.assertTrue(np.isfinite(float(metrics["loss"])))
                if mode == "vanilla":
                    self.assertFalse(metrics["plan_present"])
                else:
                    # full K_MAX capacity, always: the mask never varies
                    self.assertEqual(int(metrics["plan_capacity_slots"]), 2 * K_MAX)
                    self.assertEqual(int(metrics["plan_valid_slots"]), 2 * K_MAX)

    def test_external_whiten_model_has_no_whitener_buffers(self):
        model = ELF(text_encoder_dim=WIDTH, max_length=8, hidden_size=16, depth=1,
                    num_heads=4, bottleneck_dim=4, num_time_tokens=1,
                    num_self_cond_cfg_tokens=0, num_model_mode_tokens=1, vocab_size=8,
                    num_plan_slots=K_MAX, num_plan_time_tokens=1,
                    plan_whiten="external", plan_target_dim=Z_DIM)
        self.assertEqual(model.plan_latent_dim, Z_DIM)
        self.assertFalse(hasattr(model, "plan_target_mean"))
        self.assertFalse(hasattr(model, "plan_whiten_ready"))


class SpanVaeConfigTests(unittest.TestCase):
    def test_shipped_configs_use_the_vae_source(self):
        for mode in ("ordered", "diagonal", "register", "vanilla"):
            with self.subTest(mode=mode):
                config = load_config_from_yaml(os.path.join(
                    CONFIG_DIR, f"train_conditional_common48w_{mode}_10k_v1.yml"))
                assert_group_protocol(config)
                if mode == "vanilla":
                    self.assertEqual(config.num_plan_slots, 0)
                else:
                    self.assertEqual(config.plan_source, "span_vae")
                    self.assertEqual(config.num_plan_slots, K_MAX)
                    self.assertEqual(config.plan_target_dim, Z_DIM)
                    self.assertEqual(config.plan_whiten, "external")
                    self.assertTrue(config.plan_vae_artifact_sha256)


if __name__ == "__main__":
    unittest.main()
