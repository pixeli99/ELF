import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import utils.generation_utils as generation_utils


def test_oracle_override_keeps_original_noise_as_plan_cfg_null(monkeypatch):
    captured = {}

    def fake_ode_step(*, z, z_plan, z_plan_null, **_kwargs):
        captured["conditioned"] = z_plan.detach().clone()
        captured["null"] = z_plan_null.detach().clone()
        return z, torch.zeros_like(z), z_plan

    monkeypatch.setattr(generation_utils, "_ode_step", fake_ode_step)
    config = SimpleNamespace(
        num_plan_slots=2,
        denoiser_noise_scale=2.0,
        use_bf16=False,
    )
    sampling = SimpleNamespace(
        sampling_method="ode",
        sde_gamma=0.0,
        plan_trajectory="planning_first",
        plan_lead_alpha=2.0,
        plan_cfg_scale=3.0,
    )
    response = torch.zeros(1, 3, 4)
    initial_noise = torch.full((1, 2, 4), -2.0)
    clean_plan = torch.full((1, 2, 4), 7.0)
    generation_utils._generate_samples_single_batch(
        model=SimpleNamespace(plan_latent_dim=4),
        generator=torch.Generator().manual_seed(0),
        z=response,
        t_steps=torch.tensor([0.0, 1.0]),
        cond_seq=torch.zeros_like(response),
        cond_seq_mask=torch.zeros(1, 3),
        config=config,
        sampling_config=sampling,
        cfg_scale=1.0,
        self_cond_cfg_scale=1.0,
        plan_override=[clean_plan, clean_plan],
        plan_override_t=1.0,
        freeze_plan_override=True,
        plan_mask=torch.ones(1, 2, dtype=torch.bool),
        initial_plan_noise=initial_noise,
    )
    torch.testing.assert_close(captured["conditioned"], clean_plan)
    torch.testing.assert_close(captured["null"], initial_noise)
