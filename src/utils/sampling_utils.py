from typing import Optional

import torch
import torch.nn.functional as F


# ============================================
# Noise Schedulers (how to compute z from x0 and noise)
# ============================================

def add_noise(x0, noise, t, config, cond_seq_mask=None):
    """Flow-matching interpolation z = t*x0 + (1-t)*noise*scale, preserving cond tokens."""
    t_expanded = t.reshape(-1, 1, 1)
    z = t_expanded * x0 + (1 - t_expanded) * noise * config.denoiser_noise_scale
    if cond_seq_mask is not None:
        z = cond_seq_mask * x0 + (1 - cond_seq_mask) * z
    return z


def frozen_pool_plan_target(x0, valid_mask, num_slots, return_slot_mask=False):
    """Build the plan-stream target x0_plan by masked mean-pooling x0 into K slots.

    x0: (N, S, C) clean encoder latents. valid_mask: (N, S), 1 for real target tokens.
    Positions [0, S) are split into `num_slots` near-equal contiguous segments; each slot
    is the mean of x0 over the valid positions in its segment (segments with no valid
    positions map to zeros). This is a *frozen* (parameter-free) resampler: x0_plan is a
    deterministic, lower-resolution view of x0 — it carries no information the tokens lack,
    so the planning claim is about ordering of resolution, not added information.

    Returns x0_plan: (N, K, C); with return_slot_mask also (N, K) 1.0 for slots whose
    segment contained at least one valid token (used by the whitener stats pass).
    """
    S = x0.shape[1]
    pos = torch.arange(S, device=x0.device)
    seg = (pos * num_slots) // S                      # (S,) segment id in [0, num_slots)
    onehot = F.one_hot(seg, num_slots).to(x0.dtype)   # (S, K)
    w = onehot.unsqueeze(0) * valid_mask.unsqueeze(-1).to(x0.dtype)  # (N, S, K)
    counts = w.sum(dim=1)                              # (N, K)
    denom = counts.unsqueeze(1).clamp(min=1.0)         # (N, 1, K)
    w = w / denom
    pooled = torch.einsum('nsk,nsc->nkc', w, x0)       # (N, K, C)
    if return_slot_mask:
        return pooled, (counts > 0).to(x0.dtype)
    return pooled


# ============================================
# Time Schedulers (how to sample t)
# ============================================

def sample_timesteps(
    batch_size: int,
    P_mean: float = -0.8,
    P_std: float = 0.8,
    time_schedule: str = 'logit_normal',
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
):
    """Sample timesteps using various time schedules.

    Args:
        batch_size: Number of samples
        P_mean: Mean for logit-normal distribution
        P_std: Std for logit-normal distribution
        time_schedule: 'logit_normal' or 'uniform'

    Returns:
        Sampled timesteps in [0, 1]
    """
    if time_schedule == 'logit_normal':
        z = torch.randn((batch_size,), dtype=dtype, device=device) * P_std + P_mean
        return torch.sigmoid(z)
    if time_schedule == 'uniform':
        return torch.rand((batch_size,), dtype=dtype, device=device)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


def get_sampling_steps(
    n_steps: int, time_schedule: str = "logit_normal",
    P_mean: float = -0.8, P_std: float = 0.8,
    device: Optional[torch.device] = None, dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a length-(n_steps+1) tensor of t values in [0, 1] for a sampling run.

    - "uniform": evenly-spaced linspace from 0 to 1 (deterministic).
    - "logit_normal": sorted logit-normal samples with 0 / 1 endpoints (random).
    """
    if time_schedule == "uniform":
        return torch.linspace(0.0, 1.0, n_steps + 1, dtype=dtype, device=device)
    if time_schedule == "logit_normal":
        steps = sample_timesteps(
            batch_size=n_steps - 1,
            P_mean=P_mean, P_std=P_std, time_schedule=time_schedule,
            device=device, dtype=dtype,
        )
        steps = torch.sort(steps).values
        endpoints_lo = torch.zeros((1,), dtype=dtype, device=steps.device)
        endpoints_hi = torch.ones((1,), dtype=dtype, device=steps.device)
        return torch.cat([endpoints_lo, steps, endpoints_hi], dim=0)
    raise ValueError(f"Unknown time_schedule: {time_schedule}")


# ============================================
# CFG Scale Sampling (how to sample cfg scale)
# ============================================

def sample_cfg_scale(batch_size, cfg_min=0.0, cfg_max=3.0,
                     dtype=torch.float32, device=None):
    """Sample CFG scale from log-uniform distribution in [cfg_min, cfg_max]."""
    u = torch.rand((batch_size,), dtype=dtype, device=device)
    a = float(1.0 + cfg_min)
    b = float(1.0 + cfg_max)
    log_ratio = torch.tensor(b / a, dtype=dtype, device=u.device).log()
    return a * torch.exp(u * log_ratio) - 1.0


# ============================================
# Conditioning helpers (preserve clean tokens during sampling)
# ============================================

def restore_cond(z_updated, cond_seq, cond_seq_mask):
    """Restore clean conditioning tokens in z after a denoising step."""
    mask = cond_seq_mask
    target_ndim = max(z_updated.dim(), cond_seq.dim())
    while mask.dim() < target_ndim:
        mask = mask.unsqueeze(-1)
    return torch.where(mask > 0, cond_seq, z_updated)


def restore_vx(v, x, cond_seq, cond_seq_mask):
    """Restore cond positions: x -> clean cond_seq, v -> 0 (cond tokens don't move)."""
    if cond_seq is not None:
        x = restore_cond(x, cond_seq, cond_seq_mask)
        v = restore_cond(v, torch.zeros_like(cond_seq), cond_seq_mask)
    return v, x


# ============================================
# Flow-matching forward passes (with optional self-cond / CFG)
# ============================================

def net_out_to_v_x(net_out, z, t, t_eps=5e-2):
    """Convert x_pred network output to v and x.

    When the model returns a tuple (denoised_output, decoder_logits),
    decoder logits are discarded here (used separately in training).
    """
    if isinstance(net_out, tuple):
        net_out = net_out[0]
    t_reshaped = t.reshape(-1, 1, 1)
    x = net_out
    denom = torch.clamp(1.0 - t_reshaped, min=t_eps)
    v = (x - z) / denom
    return v, x


def _plan_velocity(net_out, z_plan, t_plan_batch, t_eps):
    """Plan velocity from a model tuple output: (plan_output - z_plan)/clamp(1 - t_plan).

    Returns None when the model has no plan stream (2-tuple / plan_output None) or no plan
    latent is being tracked. z_plan is the current plan latent that was fed as x_plan.
    """
    plan_out = net_out[2] if (isinstance(net_out, tuple) and len(net_out) >= 3) else None
    if plan_out is None or z_plan is None or t_plan_batch is None:
        return None
    denom = torch.clamp(1.0 - t_plan_batch.reshape(-1, 1, 1), min=t_eps)
    return (plan_out - z_plan) / denom


def _forward_sample_self_cond(
    model, z, t_batch, x_pred_prev, config,
    self_cond_cfg_scale, cond_seq, cond_seq_mask,
    x_plan=None, t_plan_batch=None, plan_mask=None, nfe_counter=None,
    response_attention_mask=None,
):
    """Forward pass with self-conditioning. Returns (v, x, v_plan).

    v_plan is the plan-stream velocity from the primary (conditional) model call, or None
    when the model has no plan stream. The plan input x_plan / t_plan is passed to every
    model call (the token velocity is conditioned on the plan); the plan gets no separate CFG.
    """
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    pk = dict(x_plan=x_plan, t_plan=t_plan_batch, plan_mask=plan_mask)
    if cond_seq_mask is not None:
        condition_token_mask = cond_seq_mask
        while condition_token_mask.dim() > 2 and condition_token_mask.shape[-1] == 1:
            condition_token_mask = condition_token_mask.squeeze(-1)
        if tuple(condition_token_mask.shape) != tuple(z.shape[:2]):
            raise ValueError("cond_seq_mask must identify [B, S] prompt positions")
        pk["condition_token_mask"] = condition_token_mask.bool()
    if response_attention_mask is not None:
        pk["attention_mask"] = response_attention_mask

    def call_model(*args, **kwargs):
        if nfe_counter is not None:
            nfe_counter["model_forwards"] = nfe_counter.get("model_forwards", 0) + 1
        return model(*args, **kwargs)

    def _restore(v, x):
        return restore_vx(v, x, cond_seq=cond_seq, cond_seq_mask=cond_seq_mask)

    if config.num_self_cond_cfg_tokens > 0:
        if x_pred_prev is None:
            x_pred_prev = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
        self_cond_scale_batch = torch.full((z.shape[0],), float(self_cond_cfg_scale),
                                           dtype=z.dtype, device=z.device)
        net_out_cond = call_model(z_input_cond, t_batch, deterministic=True,
                                  self_cond_cfg_scale=self_cond_scale_batch, **pk)
        v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
        v_plan = _plan_velocity(net_out_cond, x_plan, t_plan_batch, t_eps)
        v_cond, x_cond = _restore(v_cond, x_cond)
        return v_cond, x_cond, v_plan

    # No self-conditioning
    if self_cond_prob == 0:
        net_out = call_model(z, t_batch, deterministic=True, **pk)
        v, x = net_out_to_v_x(net_out, z, t_batch, t_eps)
        v_plan = _plan_velocity(net_out, x_plan, t_plan_batch, t_eps)
        v, x = _restore(v, x)
        return v, x, v_plan

    # Combined unconditional and conditional forward pass
    v_uncond = x_uncond = v_plan = None
    if self_cond_cfg_scale != 1 or x_pred_prev is None:
        z_uncond = restore_cond(torch.zeros_like(z), cond_seq, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        net_out_uncond = call_model(z_input_uncond, t_batch, deterministic=True, **pk)
        v_uncond, x_uncond = net_out_to_v_x(net_out_uncond, z, t_batch, t_eps)
        v_plan = _plan_velocity(net_out_uncond, x_plan, t_plan_batch, t_eps)
        v_uncond, x_uncond = _restore(v_uncond, x_uncond)
        if self_cond_cfg_scale == 0.0 or x_pred_prev is None:
            return v_uncond, x_uncond, v_plan

    z_input_cond = torch.cat([z, x_pred_prev], dim=-1)
    net_out_cond = call_model(z_input_cond, t_batch, deterministic=True, **pk)
    v_cond, x_cond = net_out_to_v_x(net_out_cond, z, t_batch, t_eps)
    v_plan = _plan_velocity(net_out_cond, x_plan, t_plan_batch, t_eps)
    v_cond, x_cond = _restore(v_cond, x_cond)
    if self_cond_cfg_scale == 1:
        return v_cond, x_cond, v_plan

    v_out = v_uncond + self_cond_cfg_scale * (v_cond - v_uncond)
    x_out = x_uncond + self_cond_cfg_scale * (x_cond - x_uncond)
    v_out, x_out = _restore(v_out, x_out)
    return v_out, x_out, v_plan


def _forward_sample(
    model, z, t_batch, x_pred_prev, config,
    cfg_scale, self_cond_cfg_scale, cond_seq, cond_seq_mask,
    x_plan=None, t_plan_batch=None, plan_mask=None,
    x_plan_null=None, plan_cfg_scale=1.0, nfe_counter=None,
    response_attention_mask=None,
):
    """Forward pass with optional self-conditioning and CFG. Returns (v, x, v_plan).

    The plan velocity is taken from the conditional forward (the plan itself gets no CFG).
    plan_cfg_scale != 1 additionally applies *grid-CFG* to the token stream: the plan-
    conditioned prediction is extrapolated against a null forward at (x_plan = x_plan_null,
    t_plan = 0). The pure-noise plan at t_plan = 0 is a natural null condition — it needs no
    dropout training because the conditional-uniform plan clock covers that region.
    """
    v_cond, x_cond, v_plan = _forward_sample_self_cond(
        model, z, t_batch, x_pred_prev, config,
        self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        x_plan=x_plan, t_plan_batch=t_plan_batch,
        plan_mask=plan_mask, nfe_counter=nfe_counter,
        response_attention_mask=response_attention_mask,
    )
    if cfg_scale == 1.0:
        v_out, x_out = v_cond, x_cond
    else:
        # Unconditional forward: zero out cond prefix, no self-cond state, no restore
        z_uncond = restore_cond(z, torch.zeros_like(z), cond_seq_mask)
        x_pred_prev_uncond = (
            None if x_pred_prev is None
            else restore_cond(x_pred_prev, torch.zeros_like(x_pred_prev), cond_seq_mask)
        )
        v_uncond, x_uncond, _ = _forward_sample_self_cond(
            model, z_uncond, t_batch, x_pred_prev_uncond, config,
            self_cond_cfg_scale=self_cond_cfg_scale,
            cond_seq=torch.zeros_like(cond_seq), cond_seq_mask=cond_seq_mask,
            x_plan=x_plan, t_plan_batch=t_plan_batch,
            plan_mask=plan_mask, nfe_counter=nfe_counter,
            response_attention_mask=response_attention_mask,
        )
        v_out = v_uncond + cfg_scale * (v_cond - v_uncond)
        x_out = x_uncond + cfg_scale * (x_cond - x_uncond)
        v_out, x_out = restore_vx(v_out, x_out, cond_seq, cond_seq_mask)

    if plan_cfg_scale != 1.0 and x_plan is not None and x_plan_null is not None:
        t_plan_zero = torch.zeros_like(t_batch)
        v_null, x_null, _ = _forward_sample_self_cond(
            model, z, t_batch, x_pred_prev, config,
            self_cond_cfg_scale=self_cond_cfg_scale,
            cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
            x_plan=x_plan_null, t_plan_batch=t_plan_zero,
            plan_mask=plan_mask, nfe_counter=nfe_counter,
            response_attention_mask=response_attention_mask,
        )
        v_out = v_null + plan_cfg_scale * (v_out - v_null)
        x_out = x_null + plan_cfg_scale * (x_out - x_null)
        v_out, x_out = restore_vx(v_out, x_out, cond_seq, cond_seq_mask)
    return v_out, x_out, v_plan


def _ode_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask,
    z_plan=None, t_plan=None, t_plan_next=None,
    z_plan_null=None, plan_cfg_scale=1.0, plan_mask=None,
    plan_state_fn=None, plan_forward_trace=None, nfe_counter=None,
    sde_noise_observer=None,
    response_attention_mask=None,
):
    """Single ODE (Euler) step for sampling. Returns (z, x_pred, z_plan).

    When z_plan is provided (Ordered ELF), the plan latent is Euler-stepped on its own clock
    (t_plan -> t_plan_next) using the plan velocity from the same forward.
    """
    t_batch = torch.full((z.shape[0],), float(t), dtype=z.dtype, device=z.device)
    if plan_state_fn is not None:
        z_plan, t_plan = plan_state_fn(float(t))
        if plan_forward_trace is not None:
            plan_forward_trace.append((float(t), float(t_plan)))
    t_plan_batch = (None if z_plan is None
                    else torch.full((z.shape[0],), float(t_plan), dtype=z.dtype, device=z.device))
    v_pred, x_pred, v_plan = _forward_sample(
        model=model, z=z, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        x_plan=z_plan, t_plan_batch=t_plan_batch,
        plan_mask=plan_mask,
        x_plan_null=z_plan_null, plan_cfg_scale=plan_cfg_scale,
        nfe_counter=nfe_counter,
        response_attention_mask=response_attention_mask,
    )
    z_new = z + (t_next - t) * v_pred
    z_plan_new = z_plan if (z_plan is None or v_plan is None) else z_plan + (t_plan_next - t_plan) * v_plan
    return z_new, x_pred, z_plan_new


def _sde_step(
    model, z, t, t_next, x_pred_prev,
    config, cfg_scale, self_cond_cfg_scale,
    cond_seq, cond_seq_mask, gamma, generator,
    z_plan=None, t_plan=None, t_plan_next=None,
    z_plan_null=None, plan_cfg_scale=1.0, plan_mask=None,
    plan_state_fn=None, plan_forward_trace=None, nfe_counter=None,
    sde_noise_observer=None,
    response_attention_mask=None,
):
    """Per-step SDE-style sampler with hybrid (t-and-step) noise scaling. Returns (z, x_pred, z_plan).

    t_back = t * (1 - gamma * h), where h = t_next - t. alpha = 1 - gamma*h is the
    signal-preservation fraction, constant in t. gamma=0 degenerates to a plain ODE step.
    Uniform-N-step equivalence with old multiplicative gamma_old: gamma_hybrid = gamma_old * N.

    The plan latent does not receive SDE churn; it is Euler-stepped on its own clock using the
    plan velocity from the same forward (the token forward is at t_back, the plan at t_plan).
    """
    h = float(t_next - t)
    alpha = max(0.0, min(1.0, 1.0 - gamma * h))
    t_back = alpha * float(t)
    if z.is_cuda:
        eps = torch.randn(z.shape, dtype=z.dtype, device=z.device) * config.denoiser_noise_scale
    else:
        eps = torch.randn(z.shape, generator=generator, dtype=z.dtype) * config.denoiser_noise_scale
    if sde_noise_observer is not None:
        sde_noise_observer(eps, t_back)
    z_back = restore_cond(alpha * z + (1.0 - alpha) * eps, cond_seq, cond_seq_mask)
    t_batch = torch.full((z.shape[0],), t_back, dtype=z.dtype, device=z.device)
    if plan_state_fn is not None:
        z_plan, t_plan = plan_state_fn(t_back)
        if plan_forward_trace is not None:
            plan_forward_trace.append((t_back, float(t_plan)))
    t_plan_batch = (None if z_plan is None
                    else torch.full((z.shape[0],), float(t_plan), dtype=z.dtype, device=z.device))
    v_pred, x_pred, v_plan = _forward_sample(
        model=model, z=z_back, t_batch=t_batch, x_pred_prev=x_pred_prev,
        config=config, cfg_scale=cfg_scale, self_cond_cfg_scale=self_cond_cfg_scale,
        cond_seq=cond_seq, cond_seq_mask=cond_seq_mask,
        x_plan=z_plan, t_plan_batch=t_plan_batch,
        plan_mask=plan_mask,
        x_plan_null=z_plan_null, plan_cfg_scale=plan_cfg_scale,
        nfe_counter=nfe_counter,
        response_attention_mask=response_attention_mask,
    )
    z_new = z_back + (t_next - t_back) * v_pred
    z_plan_new = z_plan if (z_plan is None or v_plan is None) else z_plan + (t_plan_next - t_plan) * v_plan
    return z_new, x_pred, z_plan_new
