"""One mini-batch forward/backward for the ELF diffusion language model.

Each example in the batch independently picks the decoder (CE) or denoiser
(L2) branch via a Bernoulli draw at `decoder_prob`. A single forward consumes
a mixed input (decoder_z for decoder rows, denoiser_z for denoiser rows) and
both heads run; the CE / L2 losses are then masked to their respective rows
and combined with a single denominator. Self-conditioning + CFG guidance is
applied on the denoiser branch only.

Ordered ELF (gated by ``num_plan_slots > 0``): a second "planning" stream of K
slots is denoised jointly under its own clock ``t_plan``. The plan denoises
toward ``x0_plan`` (a frozen, whitened mean-pool of x0 — see
``ELF.build_plan_target``), following the same per-row mixing as the tokens
(decoder rows see a clean plan at t_plan=1, denoiser rows see a noised plan at
t_plan). The plan clock is sampled conditional-uniform with an atom at
t_plan=1 (science arm; see ``Config.plan_time_schedule``). The plan
x-prediction MSE (denoiser rows only) is added to the loss with weight
``plan_loss_weight``; with a whitened target, plan_l2 == 1.0 is exactly the
trivial predict-the-mean baseline.
``plan_register_only`` replaces the plan input with pure noise at t_plan=0 and
drops the loss (the ViT-registers control arm). When disabled the step is
byte-for-byte the original ELF (no extra RNG draws, no change to the loss graph).
"""

import contextlib
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train_utils import TrainState, ema_update, unwrap_model
from utils.encoder_utils import encode_text
from utils.sampling_utils import (
    sample_cfg_scale, add_noise, sample_timesteps,
    net_out_to_v_x, restore_cond,
)


def _trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]


def train_step(
    state: TrainState,
    encoder: nn.Module,
    batch: Dict[str, torch.Tensor],
    config,
) -> Tuple[TrainState, Dict[str, float]]:
    """Perform a single training step."""
    device = next(state.model.parameters()).device
    dtype = next(state.model.parameters()).dtype
    use_bf16 = bool(getattr(config, "use_bf16", True)) and device.type == "cuda"
    t_eps = config.t_eps
    self_cond_prob = config.self_cond_prob
    latent_mean, latent_std = config.latent_mean, config.latent_std
    decoder_prob = config.decoder_prob
    decoder_noise_scale = config.decoder_noise_scale

    plan_enabled = config.num_plan_slots > 0
    if plan_enabled and config.plan_resampler != "frozen_pool":
        raise NotImplementedError(
            f"plan_resampler={config.plan_resampler!r} is not implemented in v1; "
            "use 'frozen_pool' (learnable resampler is a Step-2 ablation)."
        )

    gen = state.dropout_generator

    # encoder_attention_mask: cond sees cond, x sees all
    input_ids = batch["input_ids"].to(device, non_blocking=True).long()
    encoder_attention_mask = batch["encoder_attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
    cond_seq_mask = batch["cond_seq_mask"].to(device, dtype=torch.float32, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
    label_drop_mask = batch.get("label_drop_mask",
                                torch.zeros((input_ids.shape[0],), dtype=torch.bool)).to(device, non_blocking=True)

    # Label drop before encoding: prevent target tokens from attending to
    # condition tokens so x0 is truly unconditional for dropped samples.
    if config.label_drop_prob > 0:
        drop = label_drop_mask.to(dtype=torch.float32).reshape(-1, 1, 1)  # (B, 1, 1)
        cond_mask = cond_seq_mask  # (B, S)
        # block_mask is 1 only at (non-cond row, cond col) — leaves cond↔cond unchanged
        block_mask = (1 - cond_mask).unsqueeze(-1) * cond_mask.unsqueeze(1)
        encoder_attention_mask = encoder_attention_mask * (1 - drop * block_mask)

    x0 = encode_text(
        input_ids=input_ids,
        attention_mask=encoder_attention_mask,
        encoder=encoder,
        latent_mean=latent_mean,
        latent_std=latent_std,
        use_bf16=use_bf16,
    ).to(dtype)

    batch_size, seq_length = x0.shape[0], x0.shape[1]

    t = sample_timesteps(
        batch_size,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
        device=device, dtype=dtype,
    )

    noise = torch.randn(x0.shape, dtype=dtype, device=device)

    if config.pad_token == "pad":
        loss_mask = attention_mask
    else:
        loss_mask = torch.ones_like(attention_mask)
    loss_mask = loss_mask * (1 - cond_seq_mask)  # (B, S), 1 = real target token

    cond_seq_mask = cond_seq_mask.unsqueeze(-1)  # (B, S, 1)

    denoiser_z = add_noise(x0, noise, t, config, cond_seq_mask=cond_seq_mask)

    drop = label_drop_mask.unsqueeze(1)  # (B, 1)
    if config.label_drop_prob > 0:
        denoiser_z = torch.where(drop.unsqueeze(-1) & (cond_seq_mask > 0), torch.zeros_like(denoiser_z), denoiser_z)
        x0 = torch.where(drop.unsqueeze(-1) & (cond_seq_mask > 0), torch.zeros_like(x0), x0)

    decoder_targets = input_ids  # (B, S)

    # Per-example branching: each example independently picks decoder (CE) vs.
    # denoiser (L2) instead of one scalar bernoulli per step. Smooths training
    decoder_step_active = torch.bernoulli(
        torch.full((batch_size,), decoder_prob, dtype=torch.float32),
        generator=gen,
    ).to(device=device, dtype=dtype)  # (B,) — 1.0 = decoder mode, 0.0 = denoiser
    decoder_mask_B11 = decoder_step_active.view(-1, 1, 1)
    decoder_mask_B1 = decoder_step_active.view(-1, 1)

    # Decoder-branch input: logit-normal-noised latent (decoder_z) at t=1
    decoder_z_vals = (
        torch.randn((batch_size * seq_length,), dtype=dtype, device=device)
        * config.decoder_p_std + config.decoder_p_mean
    )
    decoder_lambda_t = torch.sigmoid(decoder_z_vals).reshape(batch_size, seq_length, 1)
    decoder_noise = torch.randn(x0.shape, dtype=dtype, device=device) * decoder_noise_scale
    decoder_z = decoder_lambda_t * x0 + (1 - decoder_lambda_t) * decoder_noise

    t_expanded = t.reshape(-1, 1, 1)
    v_target = (x0 - denoiser_z) / torch.clamp(1 - t_expanded, min=t_eps)

    # ---- Planning stream: build target, sample its clock, noise it, per-row mix. ----
    # All plan RNG draws are guarded by plan_enabled, so the disabled path draws nothing
    # extra and the global RNG stream is identical to the original ELF.
    plan_z = plan_t = x0_plan = None
    x_plan_input = t_plan_input = None
    plan_x_aux = plan_t_aux = None  # plan inputs for the auxiliary (self-cond) forwards
    plan_supervised = plan_enabled and not config.plan_register_only
    if plan_enabled and config.plan_register_only:
        # True register control: slots are present but carry NO data-derived information
        # (pure noise at t_plan = 0 on every row, decoder rows included) and get no plan
        # loss. Isolates the ViT-registers effect from the planning effect.
        inner_model = unwrap_model(state.model)
        plan_noise = torch.randn(
            (batch_size, config.num_plan_slots, inner_model.plan_latent_dim),
            dtype=dtype, device=device,
        )
        x_plan_input = plan_noise * config.denoiser_noise_scale
        t_plan_input = torch.zeros_like(t)
        plan_x_aux, plan_t_aux = x_plan_input, t_plan_input
    elif plan_enabled:
        inner_model = unwrap_model(state.model)
        x0_plan = inner_model.build_plan_target(x0, loss_mask)  # (B, K, D), whitened
        # Plan clock. "uniform" (science arm): t_plan | t ~ U[0,1], so the training density
        # along ANY monotone trajectory equals f(t)·1 — diagonal / leading / lagging inference
        # paths are covered equally. The t_plan=1 atom covers the dwell segment of saturating
        # lead trajectories. "logit_normal" matches the token clock (systems arm).
        if config.plan_time_schedule == "uniform":
            plan_t_raw = torch.rand((batch_size,), dtype=dtype, device=device)
        elif config.plan_time_schedule == "logit_normal":
            plan_t_raw = sample_timesteps(
                batch_size,
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                time_schedule="logit_normal",
                device=device, dtype=dtype,
            )
        else:
            raise ValueError(f"Unknown plan_time_schedule: {config.plan_time_schedule!r}")
        if config.plan_done_frac > 0:
            done = torch.rand((batch_size,), dtype=dtype, device=device) < config.plan_done_frac
            plan_t_raw = torch.where(done, torch.ones_like(plan_t_raw), plan_t_raw)
        if config.plan_diag_frac > 0:
            on_diag = torch.rand((batch_size,), dtype=dtype, device=device) < config.plan_diag_frac
            plan_t = torch.where(on_diag, t, plan_t_raw)
        else:
            plan_t = plan_t_raw
        plan_noise = torch.randn(x0_plan.shape, dtype=dtype, device=device)
        plan_z = add_noise(x0_plan, plan_noise, plan_t, config)
        # Per-row mix (same as tokens): decoder rows -> clean plan at t_plan=1,
        # denoiser rows -> noised plan at t_plan.
        t_plan_input = decoder_step_active * torch.ones_like(t) + (1.0 - decoder_step_active) * plan_t
        x_plan_input = decoder_mask_B11 * x0_plan + (1.0 - decoder_mask_B11) * plan_z
        plan_x_aux, plan_t_aux = plan_z, plan_t

    if self_cond_prob > 0:
        use_self_cond_mask = (
            (torch.rand((batch_size,), dtype=dtype, device=device) < self_cond_prob)
            .reshape(-1, 1, 1).to(dtype)
        )
    else:
        use_self_cond_mask = None

    if config.num_self_cond_cfg_tokens > 0:
        self_cond_cfg_scale = sample_cfg_scale(
            batch_size,
            cfg_min=config.self_cond_cfg_min, cfg_max=config.self_cond_cfg_max,
            dtype=dtype, device=device,
        )
    else:
        self_cond_cfg_scale = None

    model = state.model

    def compute_shared_uncond(z, t_input, x_tokens):
        """Unconditional forward shared by self-cond-init and sc-cfg-uncond."""
        z_uncond = restore_cond(torch.zeros_like(z), x_tokens, cond_seq_mask)
        z_input_uncond = torch.cat([z, z_uncond], dim=-1)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            net_out_uncond = model(
                z_input_uncond, t_input,
                deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale,
                x_plan=plan_x_aux, t_plan=plan_t_aux,
            )
        return net_out_uncond

    def get_sc_cond_and_uncond(z, t_input, cond_mask, x_tokens, shared_net_out_uncond):
        if config.self_cond_prob == 0:
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                net_out_uncond = model(
                    z, t_input,
                    deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale,
                    x_plan=plan_x_aux, t_plan=plan_t_aux,
                )
            v_uncond, _ = net_out_to_v_x(net_out_uncond, z, t_input, t_eps)
            return v_uncond, v_uncond

        v_uncond, x_uncond = net_out_to_v_x(shared_net_out_uncond, z, t_input, t_eps)
        x_uncond = restore_cond(x_uncond, x_tokens, cond_mask)

        z_input_cond = torch.cat([z, x_uncond], dim=-1)
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
            net_out_cond = model(
                z_input_cond, t_input,
                deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale,
                x_plan=plan_x_aux, t_plan=plan_t_aux,
            )
        v_cond, _ = net_out_to_v_x(net_out_cond, z, t_input, t_eps)
        return v_cond, v_uncond

    def get_sc_guided_v(z, t_input, base_v_target, x_tokens, shared_net_out_uncond):
        """v target with self-conditioning guidance."""
        v_cond, v_uncond = get_sc_cond_and_uncond(
            z, t_input, cond_mask=cond_seq_mask, x_tokens=x_tokens,
            shared_net_out_uncond=shared_net_out_uncond,
        )
        sc_w = self_cond_cfg_scale.reshape(batch_size, 1, 1)
        sc_guidance = (1 - 1 / sc_w) * (v_cond - v_uncond)
        sc_guidance = torch.where(use_self_cond_mask.bool(), sc_guidance, torch.zeros_like(sc_guidance))
        return (base_v_target + sc_guidance).detach()

    def get_v_target(z, t_input, base_v_target, x_tokens, shared_net_out_uncond):
        """Compute final v target with self-conditioning guidance."""
        if config.num_self_cond_cfg_tokens > 0 and config.self_cond_prob > 0:
            return get_sc_guided_v(
                z, t_input, base_v_target=base_v_target, x_tokens=x_tokens,
                shared_net_out_uncond=shared_net_out_uncond,
            )
        return base_v_target

    model.train()

    # Per-example branching: build a mixed input (decoder_z for decoder-mode
    # rows, denoiser_z for denoiser-mode rows). One forward computes both
    # heads; we mask CE / L2 losses to their respective rows.
    denoiser_t = t
    decoder_t = torch.ones_like(t)
    t_mixed = decoder_step_active * decoder_t + (1.0 - decoder_step_active) * t  # (B,)
    z_mixed = decoder_mask_B11 * decoder_z + (1.0 - decoder_mask_B11) * denoiser_z

    # Self-cond shared forward (run on denoiser_z / t — only relevant for
    # denoiser-mode rows; decoder-mode rows zero out the self-cond half below).
    if self_cond_prob > 0 or config.num_self_cond_cfg_tokens > 0:
        shared_net_out_uncond = compute_shared_uncond(denoiser_z, denoiser_t, x0)
    else:
        shared_net_out_uncond = None

    if config.self_cond_prob > 0:
        _, x_pred_init = net_out_to_v_x(shared_net_out_uncond, denoiser_z, denoiser_t, t_eps)
        x_pred_init = restore_cond(x_pred_init, x0, cond_seq_mask)
        x_pred_cond = x_pred_init * use_self_cond_mask.to(dtype)
        x_pred_cond = restore_cond(x_pred_cond, x0, cond_seq_mask)
        # Zero the self-cond half for decoder-mode rows (matches the old
        # `cat([decoder_z, zeros], -1)` decoder-branch input).
        sc_half = x_pred_cond * (1.0 - decoder_mask_B11)
        model_input = torch.cat([z_mixed, sc_half], dim=-1)
    else:
        model_input = z_mixed

    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
        net_out, decoder_logits, plan_out = model(
            model_input, t_mixed,
            deterministic=False,
            self_cond_cfg_scale=self_cond_cfg_scale,
            decoder_step_active=decoder_step_active,  # (B,) tensor
            x_plan=x_plan_input, t_plan=t_plan_input,
        )

    # CE per-token (used on decoder-mode rows).
    log_probs = F.log_softmax(decoder_logits.to(torch.float32), dim=-1)
    ce_per_token = -log_probs.gather(-1, decoder_targets.unsqueeze(-1)).squeeze(-1)

    # L2 per-token (used on denoiser-mode rows). v_pred is extracted with
    # (denoiser_z, t) — meaningful only for denoiser rows; decoder rows are
    # masked out below.
    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, denoiser_t, t_eps)
    v_final_target = get_v_target(
        denoiser_z, denoiser_t, base_v_target=v_target, x_tokens=x0,
        shared_net_out_uncond=shared_net_out_uncond,
    )
    l2_per_token = ((v_pred - v_final_target) ** 2).mean(dim=-1)

    # Masks: each position is "alive" for exactly one branch.
    loss_mask_f = loss_mask.to(ce_per_token.dtype)
    ce_mask = loss_mask_f * decoder_mask_B1
    l2_mask = loss_mask_f * (1.0 - decoder_mask_B1)

    # Combined loss with a single denominator. In expectation this is
    # decoder_prob * mean_CE + (1 - decoder_prob) * mean_L2.
    total_sum = (ce_per_token * ce_mask).sum() + (l2_per_token * l2_mask).sum()
    loss = total_sum / torch.clamp(loss_mask_f.sum(), min=1.0)

    # Per-branch metrics: mean per-token within each branch.
    ce_loss_val = ((ce_per_token * ce_mask).sum()
                   / torch.clamp(ce_mask.sum(), min=1.0)).detach()
    l2_loss_val = ((l2_per_token * l2_mask).sum()
                   / torch.clamp(l2_mask.sum(), min=1.0)).detach()

    # Plan x-prediction MSE (denoiser-mode rows only), added to the loss. Deliberately
    # x-space rather than the tokens' v-space: (i) v-MSE = x-MSE / (1-t)^2 explodes by
    # 1/t_eps^2 on the t_plan=1 atom rows that the science-arm clock samples with finite
    # probability (the token clock never visits t≈1, so vanilla ELF never faced this);
    # (ii) with a whitened target (per-dim unit variance), plan_l2 == 1.0 is exactly the
    # trivial predict-the-mean baseline — the metric doubles as the kill criterion.
    # The sampler still derives the plan velocity from plan_out, unchanged.
    # When the plan stream is disabled (or in the register-only arm), `loss` is untouched.
    if plan_supervised:
        plan_l2_per_slot = ((plan_out - x0_plan) ** 2).mean(dim=-1)  # (B, K)
        plan_row_mask = (1.0 - decoder_mask_B1).to(plan_l2_per_slot.dtype)     # (B, 1)
        plan_denom = torch.clamp((plan_row_mask.expand_as(plan_l2_per_slot)).sum(), min=1.0)
        plan_l2 = (plan_l2_per_slot * plan_row_mask).sum() / plan_denom
        loss = loss + config.plan_loss_weight * plan_l2
        plan_l2_val = plan_l2.detach()
    else:
        plan_l2_val = torch.zeros((), device=device)

    accum_steps = max(config.grad_accum_steps, 1)
    state.step += 1
    is_optimizer_step = (state.step % accum_steps) == 0

    sync_ctx = model.no_sync() if (not is_optimizer_step and hasattr(model, 'no_sync')) else contextlib.nullcontext()
    with sync_ctx:
        (loss / accum_steps).backward()

    if is_optimizer_step:
        torch.nn.utils.clip_grad_norm_(_trainable_params(model), max_norm=1.0)
        state.optimizer.step()
        if state.lr_scheduler is not None:
            state.lr_scheduler.step()
        ema_update(state.ema_params1, state.model, config.ema_decay1)
        state.optimizer.zero_grad(set_to_none=True)

    metrics = {
        "loss": loss.detach(),
        "l2_loss": l2_loss_val,
        "ce_loss": ce_loss_val,
        "plan_l2_loss": plan_l2_val,
        "optimizer_step": is_optimizer_step,
    }
    return state, metrics
