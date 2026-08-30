"""Conditional Stage-B: the prompt window, the schedule, and the joint step."""

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
from modules.thinking_resampler import build_adjacent_mlp_encoder
from train_step import (
    _gate_plan_mediation_by_clock, _sample_plan_mediation_mask, train_step,
)
from utils.conditional_data import (ConditionalCollator, ConditionalSchedule,
                                    derive_seed, truncation_metrics)
from utils.plan_stream import PlanStream, assert_group_protocol, plan_loss
from utils.train_utils import TrainState

WIDTH, VOCAB = 8, 64
CONFIG_DIR = os.path.join(ROOT, "src/configs/training_configs")


class WordTokenizer:
    """Whitespace tokenizer with T5's EOS contract, enough to exercise the collator."""

    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=True, **kwargs):
        ids = [2 + (abs(hash(word)) % (VOCAB - 3)) for word in text.split()]
        if add_special_tokens:
            ids = ids + [self.eos_token_id]
        return {"input_ids": ids}


def _rows(n=3, prompt_words=5, thinking_words=9, response_words=4):
    return [{
        "example_id": f"id-{i}",
        "source": "unit-test",
        "prompt": " ".join(f"p{i}w{j}" for j in range(prompt_words)),
        "thinking": " ".join(f"t{i}w{j}" for j in range(thinking_words)),
        "response": " ".join(f"r{i}w{j}" for j in range(response_words)),
    } for i in range(n)]


class CollatorTests(unittest.TestCase):
    def setUp(self):
        self.collator = ConditionalCollator(WordTokenizer(), max_length=64,
                                            condition_max_tokens=32, max_plan_slots=255)

    def test_prompt_is_a_clean_prefix_of_the_window(self):
        batch = self.collator(_rows())
        for row in range(batch["input_ids"].shape[0]):
            cond = batch["cond_seq_mask"][row]
            valid = batch["attention_mask"][row]
            self.assertTrue(bool((cond[1:] <= cond[:-1]).all()), "cond mask must be a prefix")
            self.assertTrue(bool(((valid - cond) >= 0).all()), "cond must lie inside valid")
            self.assertEqual(int(cond.sum()), 5)      # prompt words, no EOS
            self.assertEqual(int(valid.sum()), 5 + 5)  # + response words + EOS

    def test_masks_are_two_dimensional(self):
        """3-D T5 masks stopped working at transformers 4.45; we never emit them."""
        batch = self.collator(_rows())
        self.assertEqual(batch["encoder_attention_mask"].shape, (3, 64))
        self.assertEqual(batch["attention_mask"].shape, (3, 64))
        self.assertEqual(batch["plan_input_ids"].shape[0], 3)
        self.assertEqual(batch["plan_attention_mask"].sum(1).tolist(), [10, 10, 10])

    def test_long_prompt_is_capped_and_reported(self):
        rows = _rows(n=1, prompt_words=100)
        batch = self.collator(rows)
        self.assertEqual(int(batch["cond_seq_mask"].sum()), 32)
        self.assertEqual(truncation_metrics(batch)["prompt_truncated"], 1)
        # The response still fits, which is the whole point of capping the prompt.
        self.assertEqual(truncation_metrics(batch)["response_truncated"], 0)

    def test_truncation_flags_ride_in_the_batch(self):
        """Worker processes cannot report through the parent's collator object."""
        batch = self.collator(_rows())
        for key in ConditionalCollator.TRUNCATION_KEYS:
            self.assertIn(key, batch)
            self.assertEqual(batch[key].shape, (3,))

    def test_condition_cap_must_leave_room(self):
        with self.assertRaises(ValueError):
            ConditionalCollator(WordTokenizer(), max_length=64, condition_max_tokens=64)


class StubAttentiveT5(nn.Module):
    """A stand-in that actually mixes across visible positions, so masking shows up."""

    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, WIDTH)
        self.embedding.weight.data.normal_(0.0, 1.0)
        self.requires_grad_(False)

    def forward(self, input_ids, attention_mask=None, deterministic=True):
        hidden = self.embedding(input_ids)
        if attention_mask is None:
            return hidden
        weights = attention_mask.to(hidden.dtype)
        pooled = (hidden * weights.unsqueeze(-1)).sum(1) / weights.sum(1).clamp_min(1).unsqueeze(-1)
        return hidden + pooled.unsqueeze(1)


class _Pool(torch.utils.data.Dataset):
    def __init__(self, n=64):
        self.rows = _rows(n)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return dict(self.rows[index])

    def example_ids(self):
        return [row["example_id"] for row in self.rows]


class ScheduleTests(unittest.TestCase):
    def test_schedule_is_a_function_of_seed_and_size(self):
        a = ConditionalSchedule(_Pool(), rows=32, master_seed=42)
        b = ConditionalSchedule(_Pool(), rows=32, master_seed=42)
        self.assertEqual(a.fingerprint(), b.fingerprint())
        self.assertEqual(a.example_ids, b.example_ids)

    def test_a_different_seed_is_a_different_schedule(self):
        a = ConditionalSchedule(_Pool(), rows=32, master_seed=42)
        c = ConditionalSchedule(_Pool(), rows=32, master_seed=43)
        self.assertNotEqual(a.fingerprint(), c.fingerprint())

    def test_rows_carry_their_own_rng_seeds(self):
        schedule = ConditionalSchedule(_Pool(), rows=8, master_seed=42)
        row = schedule[3]
        self.assertEqual(row["presentation_index"], 3)
        self.assertEqual(row["plan_time_seed"],
                         derive_seed(42, 3, row["example_id"], "plan_time"))
        self.assertNotEqual(row["plan_time_seed"], row["plan_noise_seed"])

    def test_schedule_cannot_exceed_the_pool(self):
        with self.assertRaises(ValueError):
            ConditionalSchedule(_Pool(n=8), rows=16)


def _tiny_config(mode, max_length=64):
    config = Config()
    config.latent_mean, config.latent_std = 0.0, 0.2
    config.max_length = max_length
    config.condition_max_tokens = 32
    config.plan_source = "thinking_mlp_4to1"
    config.plan_resampler = "frozen_pool"
    config.plan_whiten = "none"
    config.plan_time_schedule = "uniform"
    config.num_time_tokens, config.num_plan_time_tokens = 1, 1
    config.num_self_cond_cfg_tokens, config.num_model_mode_tokens = 0, 1
    config.decoder_prob = 0.0
    config.self_cond_prob = 0.0
    config.label_drop_prob = 0.0
    config.denoiser_noise_scale = 1.0
    config.use_bf16 = False
    config.grad_accum_steps = 1
    config.group_mode = mode
    config.num_plan_slots = config.max_plan_slots = 0 if mode == "vanilla" else 16
    config.plan_register_only = mode == "register"
    config.plan_done_frac = 0.15 if mode == "ordered" else 0.0
    config.plan_diag_frac = 1.0 if mode == "diagonal" else 0.0
    config.plan_loss_weight = 0.0 if mode == "register" else 1.0
    config.plan_response_attention = ("bidirectional" if mode == "vanilla"
                                      else "prompt_causal_bottleneck")
    return config


class StubT5(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, WIDTH)
        self.embedding.weight.data.normal_(0.0, 1.0)
        self.requires_grad_(False)

    def forward(self, input_ids, attention_mask=None, deterministic=True):
        return self.embedding(input_ids)


class PlanTokenCapacityTests(unittest.TestCase):
    def test_span_vae_capacity_keeps_the_whole_thinking(self):
        rows = _rows(n=1, thinking_words=600)
        legacy = ConditionalCollator(WordTokenizer(), max_length=64, condition_max_tokens=32,
                                     max_plan_slots=64)(rows)
        self.assertEqual(legacy["plan_input_ids"].shape[1], 256)
        self.assertEqual(int(legacy["thinking_truncated"].sum()), 1)
        vae = ConditionalCollator(WordTokenizer(), max_length=64, condition_max_tokens=32,
                                  max_plan_slots=64, plan_token_capacity=1024)(rows)
        self.assertEqual(vae["plan_input_ids"].shape[1], 601)   # 600 words + EOS
        self.assertEqual(int(vae["thinking_truncated"].sum()), 0)

    def test_capacity_follows_the_plan_source(self):
        from types import SimpleNamespace
        from utils.conditional_data import plan_token_capacity_for
        self.assertEqual(plan_token_capacity_for(SimpleNamespace(
            plan_source="span_vae", max_plan_slots=64, num_plan_slots=64)), 1024)
        self.assertEqual(plan_token_capacity_for(SimpleNamespace(
            plan_source="thinking_mlp_4to1", max_plan_slots=255, num_plan_slots=255)), 1020)
        with self.assertRaises(ValueError):
            plan_token_capacity_for(SimpleNamespace(plan_source="span_vae", max_plan_slots=16,
                                                    num_plan_slots=16))


class ConditionalStepTests(unittest.TestCase):
    def _run(self, mode, decoder_prob=0.0, plan_mediation_prob=0.0):
        config = _tiny_config(mode)
        config.decoder_prob = decoder_prob
        config.plan_mediation_prob = plan_mediation_prob
        batch = ConditionalCollator(WordTokenizer(), max_length=config.max_length,
                                    condition_max_tokens=32, max_plan_slots=16)(_rows(n=1))
        torch.manual_seed(0)
        model = ELF(text_encoder_dim=WIDTH, max_length=config.max_length, hidden_size=16,
                    depth=1, num_heads=4, bottleneck_dim=4, num_time_tokens=1,
                    num_self_cond_cfg_tokens=0, num_model_mode_tokens=1, vocab_size=VOCAB,
                    num_plan_slots=config.num_plan_slots, num_plan_time_tokens=1,
                    plan_whiten="none",
                    plan_response_attention=config.plan_response_attention)
        seen = {}
        inner = model.forward

        def recording(*args, **kwargs):
            seen.setdefault("x", args[0])
            seen.setdefault("plan_mask", kwargs.get("plan_mask"))
            seen.setdefault("condition_token_mask", kwargs.get("condition_token_mask"))
            seen.setdefault("plan_mediation_mask", kwargs.get("plan_mediation_mask"))
            return inner(*args, **kwargs)

        model.forward = recording
        state = TrainState(model=model,
                           optimizer=torch.optim.SGD(model.parameters(), lr=0.0),
                           dropout_generator=torch.Generator().manual_seed(0))
        torch.manual_seed(1)
        encoder = StubT5()
        _, metrics = train_step(
            state, encoder, batch, config,
            plan_encoder=build_adjacent_mlp_encoder(WIDTH, 4, 16).requires_grad_(False),
            update_model=False)
        seen["x0"] = encoder(batch["input_ids"]).float() / 0.2
        return batch, metrics, seen

    def test_loss_counts_response_tokens_only(self):
        batch, metrics, _ = self._run("ordered")
        response_tokens = int((batch["attention_mask"] - batch["cond_seq_mask"]).sum())
        self.assertEqual(int(metrics["response_valid_tokens"]), response_tokens)

    def test_prompt_positions_reach_denoiser_rows_unnoised(self):
        """The prompt is a clean prefix: denoiser rows must see x0 there."""
        batch, _, seen = self._run("ordered", decoder_prob=0.0)
        cond = batch["cond_seq_mask"][0].bool()
        torch.testing.assert_close(seen["x"][0][cond], seen["x0"][0][cond],
                                   rtol=1e-3, atol=1e-3)

    def test_decoder_rows_see_a_noised_prompt(self):
        """Documents inherited upstream behaviour, not an endorsement of it.

        The decoder branch builds its input without restoring the condition, so
        CE rows train on a noised prompt while generation always decodes from a
        prompt that `restore_cond` has kept clean. This is the upstream
        conditional path (WMT / XSum train the same way); it is recorded here so
        that changing it is a deliberate act with a failing test attached.
        """
        batch, _, seen = self._run("ordered", decoder_prob=1.0)
        cond = batch["cond_seq_mask"][0].bool()
        self.assertFalse(torch.allclose(seen["x"][0][cond], seen["x0"][0][cond],
                                        rtol=1e-3, atol=1e-3))

    def test_every_group_runs_on_a_conditional_batch(self):
        for mode in ("ordered", "diagonal", "register", "vanilla"):
            with self.subTest(mode=mode):
                _, metrics, seen = self._run(mode)
                self.assertTrue(np.isfinite(float(metrics["loss"])))
                self.assertEqual(metrics["group_mode"], mode)
                self.assertEqual(metrics["plan_present"], mode != "vanilla")
                if mode in ("ordered", "diagonal"):
                    self.assertGreater(float(metrics["plan_l2_loss"]), 0.0)
                else:
                    self.assertEqual(float(metrics["plan_l2_loss"]), 0.0)

    def test_plan_and_condition_coexist(self):
        batch, metrics, seen = self._run("ordered")
        self.assertEqual(int(metrics["plan_valid_slots"]),
                         int((batch["plan_attention_mask"].sum(1) + 3) // 4))
        self.assertTrue(torch.equal(
            seen["condition_token_mask"].cpu(), batch["cond_seq_mask"].bool(),
        ))

    def test_full_plan_mediation_reaches_every_training_forward(self):
        _, metrics, seen = self._run("ordered", plan_mediation_prob=1.0)
        self.assertEqual(int(metrics["plan_mediation_rows"]), 1)
        self.assertTrue(bool(seen["plan_mediation_mask"].all()))

    def test_formal_plan_mediation_sampling_is_resume_stable(self):
        batch = {"branch_seed": torch.arange(64, dtype=torch.long)}
        generator = torch.Generator().manual_seed(7)
        state_before = generator.get_state().clone()
        a = _sample_plan_mediation_mask(batch, 64, 0.5, generator, torch.device("cpu"))
        b = _sample_plan_mediation_mask(
            batch, 64, 0.5, torch.Generator().manual_seed(99), torch.device("cpu"),
        )
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(generator.get_state(), state_before))
        self.assertGreater(int(a.sum()), 0)
        self.assertLess(int(a.sum()), 64)
        self.assertIsNone(_sample_plan_mediation_mask(
            batch, 64, 0.0, generator, torch.device("cpu"),
        ))

    def test_plan_mediation_can_require_a_clean_plan_clock(self):
        sampled = torch.tensor([True, True, True, False])
        clock = torch.tensor([0.2, 0.75, 1.0, 1.0])
        gated = _gate_plan_mediation_by_clock(sampled, clock, 0.75)
        self.assertEqual(gated.tolist(), [False, True, True, False])
        with self.assertRaisesRegex(ValueError, "within"):
            _gate_plan_mediation_by_clock(sampled, clock, 1.1)


class ConditionalEncodingTests(unittest.TestCase):
    def test_prompt_latents_do_not_depend_on_the_response(self):
        """Generation only ever has the prompt, so training must encode it alone."""
        from utils.encoder_utils import encode_conditional_x0

        torch.manual_seed(0)
        encoder = StubAttentiveT5()
        collator = ConditionalCollator(WordTokenizer(), max_length=32,
                                       condition_max_tokens=16, max_plan_slots=16)
        rows = _rows(n=1)
        batch = collator(rows)
        other = dict(rows[0], response="completely different words here")
        batch_other = collator([other])

        args = dict(encoder=encoder, latent_mean=0.0, latent_std=0.2, use_bf16=False)
        a = encode_conditional_x0(batch["input_ids"], batch["attention_mask"],
                                  batch["cond_seq_mask"], **args)
        b = encode_conditional_x0(batch_other["input_ids"], batch_other["attention_mask"],
                                  batch_other["cond_seq_mask"], **args)
        cond = batch["cond_seq_mask"][0].bool()
        torch.testing.assert_close(a[0][cond], b[0][cond])
        # The response half must react to the response, or the encoder is inert.
        response = batch["attention_mask"][0].bool() & ~cond
        self.assertFalse(torch.allclose(a[0][response][:1], b[0][response][:1]))

    def test_dropping_the_condition_changes_only_the_response_half(self):
        from utils.encoder_utils import encode_conditional_x0

        torch.manual_seed(0)
        encoder = StubAttentiveT5()
        collator = ConditionalCollator(WordTokenizer(), max_length=32,
                                       condition_max_tokens=16, max_plan_slots=16)
        batch = collator(_rows(n=1))
        args = dict(encoder=encoder, latent_mean=0.0, latent_std=0.2, use_bf16=False)
        kept = encode_conditional_x0(batch["input_ids"], batch["attention_mask"],
                                     batch["cond_seq_mask"], **args)
        dropped = encode_conditional_x0(batch["input_ids"], batch["attention_mask"],
                                        batch["cond_seq_mask"],
                                        drop_condition=torch.tensor([True]), **args)
        cond = batch["cond_seq_mask"][0].bool()
        torch.testing.assert_close(kept[0][cond], dropped[0][cond])
        response = batch["attention_mask"][0].bool() & ~cond
        self.assertFalse(torch.allclose(kept[0][response], dropped[0][response]))


class ConditionalConfigTests(unittest.TestCase):
    def test_list_config_override_preserves_list_type(self):
        from configs.config import Config, apply_config_overrides

        config = Config()
        config.save_optimizer_steps = [1000, 2000]
        updated = apply_config_overrides(
            config, ["save_optimizer_steps=[500, 1000]"]
        )
        self.assertEqual(updated.save_optimizer_steps, [500, 1000])
        with self.assertRaisesRegex(ValueError, "must be a YAML list"):
            apply_config_overrides(config, ["save_optimizer_steps=500"])

    def test_low_time_plan_boost_survives_microbatch_one_normalization(self):
        prediction = torch.ones(1, 2, 3)
        stream = PlanStream(
            plan_mask=torch.ones(1, 2, dtype=torch.bool),
            x0_plan=torch.zeros_like(prediction),
            plan_t=torch.tensor([0.0]),
            supervised=True,
        )
        decoder = torch.zeros(1)
        self.assertEqual(float(plan_loss(prediction, stream, decoder)), 1.0)
        self.assertEqual(
            float(plan_loss(prediction, stream, decoder, low_t_boost=3.0)),
            4.0,
        )

    def test_shipped_conditional_overlays_are_consistent(self):
        for mode in ("ordered", "diagonal", "register", "vanilla"):
            with self.subTest(mode=mode):
                config = load_config_from_yaml(os.path.join(
                    CONFIG_DIR, f"train_conditional_common48w_{mode}_10k_v1.yml"))
                self.assertEqual(assert_group_protocol(config).mode, mode)
                self.assertEqual(config.max_length, 2048)
                # The response can never be truncated: the package's longest is 1006.
                self.assertLessEqual(config.condition_max_tokens, config.max_length - 1024)


if __name__ == "__main__":
    unittest.main()
