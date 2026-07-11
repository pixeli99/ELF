from typing import Optional

import torch
import torch.nn as nn

from configs.config import Config, SamplingConfig
from utils.sampling_utils import restore_cond, _ode_step, _sde_step


# ============================================
# Generation utilities
# ============================================

def mask_after_eos(predicted_ids: torch.Tensor, eos_token_id: int, pad_token_id: int) -> torch.Tensor:
    """Mask everything at/after first EOS token per sequence."""
    eos_mask = (predicted_ids == eos_token_id)
    keep_mask = (eos_mask.to(torch.int32).cumsum(dim=1) == 0)
    return torch.where(keep_mask, predicted_ids, torch.full_like(predicted_ids, pad_token_id))


def shift_left(x: torch.Tensor, shift_per_sample: torch.Tensor, pad_value=0, axis: int = 1) -> torch.Tensor:
    """Shift each sample left along the sequence axis; pad emptied positions."""
    if x.dim() < 2:
        raise ValueError("x must have at least batch and sequence dimensions")
    if axis < 0:
        axis = x.dim() + axis
    if axis == 0:
        raise ValueError("axis=0 is the batch axis and cannot be shifted")
    shift_per_sample = shift_per_sample.to(torch.long)
    if axis != 1:
        x = x.movedim(axis, 1)
    seq_len = x.shape[1]
    base_idx = torch.arange(seq_len, device=x.device)[None, :]
    gather_idx = shift_per_sample[:, None].to(x.device) + base_idx
    valid = gather_idx < seq_len
    gather_idx = gather_idx.clamp(0, seq_len - 1)
    if x.dim() == 2:
        shifted = torch.gather(x, 1, gather_idx)
        shifted = torch.where(valid, shifted, torch.full_like(shifted, pad_value))
    else:
        expand_shape = [-1, -1] + list(x.shape[2:])
        idx = gather_idx.view(*gather_idx.shape, *([1] * (x.dim() - 2))).expand(*expand_shape)
        valid_b = valid.view(*valid.shape, *([1] * (x.dim() - 2))).expand(*expand_shape)
        shifted = torch.gather(x, 1, idx)
        shifted = torch.where(valid_b, shifted, torch.full_like(shifted, pad_value))
    if axis != 1:
        shifted = shifted.movedim(1, axis)
    return shifted


# ============================================
# Single-batch sampling (PyTorch)
# ============================================

@torch.no_grad()
def _generate_samples_single_batch(
    model: nn.Module,
    generator: torch.Generator,
    z: torch.Tensor,
    t_steps: torch.Tensor,
    cond_seq: Optional[torch.Tensor],
    cond_seq_mask: Optional[torch.Tensor],
    config: Config,
    sampling_config: SamplingConfig,
    cfg_scale: float,
    self_cond_cfg_scale: float,
    record_plan: bool = False,
    plan_override: Optional[list] = None,
    plan_override_t: Optional[float] = None,
    freeze_plan_override: bool = False,
) -> torch.Tensor:
    """Generate samples for a single batch (PyTorch Euler / SDE rollout).

    record_plan: additionally return the list of plan latents the tokens saw — one entry
    per step plus the final decode plan (n entries for an n-point t_steps grid).
    plan_override: a list with the same layout that REPLACES the plan fed at each step
    (plan grafting / shuffle probes); the plan's own Euler updates are then discarded.
    plan_override_t: optional fixed clock for overridden plans. Oracle-plan eval uses 1.0.
    freeze_plan_override: when True, re-apply the overridden plan after every sampler update.
    """
    method = sampling_config.sampling_method
    batch_size, max_length, d_model = z.shape
    if cond_seq is None:
        cond_seq = torch.zeros((batch_size, max_length, d_model), dtype=z.dtype, device=z.device)
        cond_seq_mask = torch.zeros((batch_size, max_length), dtype=z.dtype, device=z.device)

    z = restore_cond(z, cond_seq, cond_seq_mask)
    x_pred = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)

    n = t_steps.shape[0]
    sde_gamma = getattr(sampling_config, "sde_gamma", 0.0)

    # Ordered ELF two-clock sampling: evolve a plan latent z_plan on its own clock t_plan.
    #   diagonal       -> t_plan == t_tok
    #   planning_first -> t_plan = min(1, alpha * t_tok); alpha > 1 leads, alpha < 1 LAGS
    #                     (the falsification arm)
    #   null           -> t_plan == 0 throughout (plan stays pure noise; register probe)
    plan_on = config.num_plan_slots > 0
    if plan_on:
        traj = getattr(sampling_config, "plan_trajectory", "diagonal")
        if traj == "null":
            alpha = 0.0
        elif traj == "planning_first":
            alpha = float(getattr(sampling_config, "plan_lead_alpha", 1.0))
        else:  # "diagonal"
            alpha = 1.0
        t_plan_steps = torch.clamp(t_steps * alpha, max=1.0)
        d_plan = getattr(model, "plan_latent_dim", d_model)
        if z.is_cuda:
            z_plan = torch.randn((batch_size, config.num_plan_slots, d_plan),
                                 dtype=z.dtype, device=z.device) * config.denoiser_noise_scale
        else:
            z_plan = (torch.randn((batch_size, config.num_plan_slots, d_plan),
                                  generator=generator, dtype=z.dtype) * config.denoiser_noise_scale).to(z.device)
    else:
        t_plan_steps = None
        z_plan = None

    # Plan grid-CFG null condition: the run's OWN initial noise plan at t_plan = 0
    # (deterministic, no extra RNG — "the plan as if it had never been denoised").
    plan_cfg_scale = float(getattr(sampling_config, "plan_cfg_scale", 1.0)) if plan_on else 1.0
    z_plan_null = z_plan.clone() if (plan_on and plan_cfg_scale != 1.0) else None

    step_kwargs = dict(
        model=model, config=config,
        cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        z_plan_null=z_plan_null, plan_cfg_scale=plan_cfg_scale,
    )

    def _tp(i):
        if plan_on and plan_override is not None and plan_override_t is not None:
            return float(plan_override_t)
        return t_plan_steps[i].item() if plan_on else None

    plan_traj = [] if record_plan else None

    def _override_at(i):
        nonlocal z_plan
        if plan_on and plan_override is not None:
            z_plan = plan_override[i].to(device=z.device, dtype=z.dtype)

    def _pre_step(i):
        """Override / record the plan latent fed at step i."""
        _override_at(i)
        if plan_on and record_plan:
            plan_traj.append(z_plan.detach().clone())

    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        for i in range(n - 2):
            _pre_step(i)
            t = t_steps[i].item()
            t_next = t_steps[i + 1].item()
            if method == "sde":
                z, x_pred, z_plan = _sde_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                    gamma=sde_gamma, generator=generator,
                    z_plan=z_plan, t_plan=_tp(i), t_plan_next=_tp(i + 1), **step_kwargs,
                )
                if freeze_plan_override:
                    _override_at(i)
            elif method == "ode":
                z, x_pred, z_plan = _ode_step(
                    z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
                    z_plan=z_plan, t_plan=_tp(i), t_plan_next=_tp(i + 1), **step_kwargs,
                )
                if freeze_plan_override:
                    _override_at(i)
            else:
                raise ValueError(f"Invalid sampling method: {method}")

        # Last step always with ODE.
        _pre_step(n - 2)
        t = t_steps[-2].item()
        t_next = t_steps[-1].item()
        z, x_pred, z_plan = _ode_step(
            z=z, t=t, t_next=t_next, x_pred_prev=x_pred,
            z_plan=z_plan, t_plan=_tp(n - 2), t_plan_next=_tp(n - 1), **step_kwargs,
        )
        if freeze_plan_override:
            _override_at(n - 2)
    # The final decode plan (fed to _dlm_decode_batch at t_plan = 1).
    if plan_on and plan_override is not None:
        z_plan = plan_override[n - 1].to(device=z.device, dtype=z.dtype)
    if plan_on and record_plan:
        plan_traj.append(z_plan.detach().clone())
    # Return the evolved plan latent (at t_plan=1) so decode can condition on it exactly as
    # training did (decoder rows always saw the clean plan). z_plan is None for a vanilla model.
    if record_plan:
        return z, z_plan, plan_traj
    return z, z_plan


@torch.no_grad()
def _dlm_decode_batch(z: torch.Tensor, model: nn.Module, t_final_val,
                      config, self_cond_cfg_scale: float, x_plan=None,
                      t_plan_decode_val: Optional[float] = None,
                      plan_trajectory: Optional[str] = None) -> torch.Tensor:
    """Decode z -> tokens with the DLM decoder head.

    Normal ordered trajectories (diagonal / planning_first / lagging) decode with
    t_plan=1 so the decoder sees a finished clean-plan condition. The strict null
    ablation is different: it keeps a pure-noise plan at t_plan=0 through sampling
    and decode, measuring an inference-time no-plan control rather than a path to
    the (t_tok=1, t_plan=1) endpoint. None for a vanilla model.
    """
    batch_size = z.shape[0]
    if isinstance(t_final_val, torch.Tensor) and t_final_val.dim() == 0:
        t_final = torch.full((batch_size,), t_final_val.item(), dtype=z.dtype, device=z.device)
    else:
        t_final = torch.full((batch_size,), float(t_final_val), dtype=z.dtype, device=z.device)
    if x_plan is None:
        t_plan = None
    else:
        if t_plan_decode_val is None:
            t_plan_decode_val = 0.0 if plan_trajectory == "null" else 1.0
        t_plan = torch.full((batch_size,), float(t_plan_decode_val), dtype=z.dtype, device=z.device)
    sc_batch = (
        torch.full((batch_size,), float(self_cond_cfg_scale), dtype=z.dtype, device=z.device)
        if config.num_self_cond_cfg_tokens > 0 else None
    )
    z_input = torch.cat([z, torch.zeros_like(z)], dim=-1) if config.self_cond_prob > 0 else z
    use_bf16 = bool(getattr(config, "use_bf16", True)) and z.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        _, decoder_logits, _ = model(
            z_input, t_final, deterministic=True,
            self_cond_cfg_scale=sc_batch,
            decoder_step_active=True,
            x_plan=x_plan, t_plan=t_plan,
        )
    return decoder_logits.argmax(dim=-1)


def _build_run_name(sampling_method, num_sampling_steps, cfg_scale, self_cond_cfg_scale,
                    time_schedule, sde_gamma, suffix, plan_trajectory=None, plan_lead_alpha=None,
                    plan_cfg_scale=None):
    ts_str = f"-ts_{time_schedule}"
    sccfg_str = f"-sccfg{self_cond_cfg_scale}" if self_cond_cfg_scale != 1.0 else ""
    sde_str = f"-gamma{sde_gamma}" if sampling_method == "sde" else ""
    # Distinguish plan trajectories (lead alpha, grid-CFG scale) so diagonal / planning-first
    # / alpha sweeps / plan-CFG sweeps write to separate dirs instead of overwriting each other.
    plan_str = ""
    if plan_trajectory:
        plan_str = f"-plan_{plan_trajectory}"
        if plan_trajectory == "planning_first" and plan_lead_alpha is not None:
            plan_str += f"_a{plan_lead_alpha}"
        if plan_cfg_scale is not None and plan_cfg_scale != 1.0:
            plan_str += f"_pcfg{plan_cfg_scale}"
    return f"{sampling_method}-steps{num_sampling_steps}-cfg{cfg_scale}{sccfg_str}{ts_str}{sde_str}{plan_str}-{suffix}"
