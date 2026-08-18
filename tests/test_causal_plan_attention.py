import copy
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modules.model import ELF, build_plan_response_attention_mask


def tiny(mode="bidirectional", max_slots=8):
    model = ELF(
        text_encoder_dim=8, max_length=6, hidden_size=16, depth=2, num_heads=4,
        bottleneck_dim=4, num_time_tokens=1, num_self_cond_cfg_tokens=1,
        num_model_mode_tokens=1, vocab_size=32, num_plan_slots=max_slots,
        num_plan_time_tokens=1, plan_whiten="none",
        plan_response_attention=mode,
    ).eval()
    # ELF output heads are intentionally zero-initialized; expose hidden-state dependence.
    with torch.no_grad():
        for module in (model.final_layer, model.plan_head):
            for parameter in module.parameters():
                parameter.normal_(mean=0.0, std=0.02)
    return model


class CausalPlanAttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 6, 8)
        self.plan = torch.randn(2, 5, 8)
        self.t = torch.tensor([.3, .7])
        self.tp = torch.tensor([.6, .9])
        self.token_mask = torch.ones(2, 6, dtype=torch.bool)
        self.plan_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool)

    def run_model(self, model, x=None, plan=None):
        return model(x if x is not None else self.x, self.t,
                     attention_mask=self.token_mask, deterministic=True,
                     self_cond_cfg_scale=torch.ones(2), decoder_step_active=False,
                     x_plan=plan if plan is not None else self.plan,
                     t_plan=self.tp, plan_mask=self.plan_mask)

    def test_bidirectional_default_is_elementwise_identical(self):
        old = tiny()
        explicit = tiny("bidirectional")
        explicit.load_state_dict(copy.deepcopy(old.state_dict()))
        for a, b in zip(self.run_model(old), self.run_model(explicit)):
            self.assertTrue(torch.equal(a, b))

    def test_causal_has_no_response_to_plan_leakage(self):
        model = tiny("causal_bottleneck")
        out_a = self.run_model(model)[2]
        out_b = self.run_model(model, x=self.x + torch.randn_like(self.x) * 3)[2]
        self.assertTrue(torch.allclose(out_a, out_b, atol=1e-6, rtol=1e-6))

    def test_plan_to_response_channel_remains_active(self):
        model = tiny("causal_bottleneck")
        out_a = self.run_model(model)[0]
        changed = self.plan.clone(); changed[:, :3] += 5
        out_b = self.run_model(model, plan=changed)[0]
        self.assertGreater((out_a - out_b).abs().max().item(), 1e-7)

    def test_bidirectional_positive_control(self):
        model = tiny("bidirectional")
        out_a = self.run_model(model)[2]
        out_b = self.run_model(model, x=self.x + torch.randn_like(self.x) * 3)[2]
        self.assertGreater((out_a - out_b).abs().max().item(), 1e-7)

    def test_dynamic_padding_and_max_k(self):
        model = tiny("causal_bottleneck", max_slots=281)
        plan = torch.randn(2, 281, 8)
        mask = torch.zeros(2, 281, dtype=torch.bool); mask[0, :3] = True; mask[1, :] = True
        args = dict(attention_mask=self.token_mask, deterministic=True,
                    self_cond_cfg_scale=torch.ones(2), decoder_step_active=False,
                    t_plan=self.tp, plan_mask=mask)
        a = model(self.x, self.t, x_plan=plan, **args)
        changed = plan.clone(); changed[0, 3:] = torch.randn_like(changed[0, 3:]) * 100
        b = model(self.x, self.t, x_plan=changed, **args)
        self.assertTrue(torch.allclose(a[0][0], b[0][0], atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.equal(a[2][0, 3:], torch.zeros_like(a[2][0, 3:])))

    def test_shared_mask_permissions_and_train_generation_parity(self):
        token = torch.tensor([[1, 1, 0]], dtype=torch.bool)
        plan = torch.tensor([[1, 0]], dtype=torch.bool)
        kwargs = dict(token_valid=token, plan_valid=plan, prefix_len=3, mode_len=1,
                      num_time_tokens=1, num_plan_time_tokens=1,
                      mode="causal_bottleneck")
        train_mask = build_plan_response_attention_mask(**kwargs)
        generation_mask = build_plan_response_attention_mask(**kwargs)
        self.assertTrue(torch.equal(train_mask, generation_mask))
        # layout: [token-time, plan-time, self-cfg, mode, plan0, plan1, response...]
        self.assertFalse(train_mask[0, 4, 0])
        self.assertTrue(train_mask[0, 4, 1])
        self.assertTrue(train_mask[0, 4, 4])
        self.assertFalse(train_mask[0, 4, 6])
        self.assertTrue(train_mask[0, 6, 4])
        self.assertTrue(train_mask[0, 6, 6])
        self.assertFalse(train_mask[0, :, 5].any())


if __name__ == "__main__":
    unittest.main()
