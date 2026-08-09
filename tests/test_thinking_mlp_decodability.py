import unittest
from types import SimpleNamespace

import torch

from src.modules.thinking_resampler import group_batched_thinking
from src.utils.formal_thinking_mlp import (
    adapt_to_fixed_decoder_width,
    mean_pool_reconstruction,
    resolve_decoder_response_width,
    restore_grouped_token_layout,
)
from src.utils.generation_utils import _dlm_decode_logits_batch


class ThinkingMLPDecodabilityTests(unittest.TestCase):
    def test_fixed_width_adapter_preserves_tokens(self):
        for length in (639, 1024):
            latent = torch.randn(1, length, 3)
            ids = torch.arange(length).view(1, -1)
            mask = torch.ones(1, length, dtype=torch.bool)
            padded, labels, padded_mask = adapt_to_fixed_decoder_width(
                latent, ids, mask, 1024, 0,
            )
            self.assertEqual(padded.shape, (1, 1024, 3))
            self.assertTrue(torch.equal(labels[0, :length], ids[0]))
            self.assertEqual(int(padded_mask.sum()), length)
            self.assertTrue((padded[:, length:] == 0).all())

    def test_over_width_rejected(self):
        latent = torch.randn(1, 1025, 2)
        ids = torch.zeros(1, 1025, dtype=torch.long)
        mask = torch.ones(1, 1025, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "1025"):
            adapt_to_fixed_decoder_width(latent, ids, mask, 1024, 0)

    def test_tail_group_padding_is_not_restored(self):
        latent = torch.randn(1, 7, 4)
        mask = torch.ones(1, 7, dtype=torch.bool)
        groups, reconstruction_mask, _ = group_batched_thinking(latent, mask, 4)
        mean_groups = mean_pool_reconstruction(groups, reconstruction_mask)
        restored = restore_grouped_token_layout(mean_groups, reconstruction_mask, mask)
        self.assertEqual(restored.shape, latent.shape)
        self.assertEqual(reconstruction_mask.flatten().tolist(), [True] * 7 + [False])

    def test_optional_decoder_mask_is_backward_compatible(self):
        class FakeModel:
            def __init__(self):
                self.calls = []

            def __call__(self, z, t, **kwargs):
                self.calls.append((z.clone(), kwargs))
                logits = torch.cat([z, z.new_zeros(z.shape[0], z.shape[1], 1)], -1)
                return z, logits, None

        config = SimpleNamespace(
            num_self_cond_cfg_tokens=0, self_cond_prob=0, use_bf16=False,
        )
        z = torch.randn(2, 9, 3)
        model = FakeModel()
        old = _dlm_decode_logits_batch(z, model, 1.0, config, 0.0)
        mask = torch.tensor([[1] * 7 + [0] * 2, [1] * 9], dtype=torch.bool)
        new = _dlm_decode_logits_batch(
            z, model, 1.0, config, 0.0, attention_mask=mask,
        )
        self.assertIsNone(model.calls[0][1]["attention_mask"])
        self.assertTrue(torch.equal(model.calls[1][1]["attention_mask"], mask))
        self.assertTrue(torch.equal(old, new))
        self.assertEqual(new.shape[:2], (2, 9))

    def test_response_width_is_model_config_contract(self):
        model = SimpleNamespace(max_length=1024)
        config = SimpleNamespace(max_length=1024)
        self.assertEqual(resolve_decoder_response_width(model, config), 1024)


if __name__ == "__main__":
    unittest.main()
