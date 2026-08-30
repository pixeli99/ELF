"""One mini-batch forward/backward for the ELF diffusion language model.

Each example independently picks the decoder (CE) or denoiser (L2) branch via a
Bernoulli draw at `decoder_prob`. One forward consumes a mixed input (decoder_z
for decoder rows, denoiser_z for denoiser rows) and both heads run; the CE / L2
losses are masked to their respective rows and combined under a single
denominator. Self-conditioning and CFG guidance apply to the denoiser branch
only.

The planning stream is built entirely in `utils/plan_stream.py`, which owns the
four-group table (ordered / diagonal / register / vanilla), the single canonical
thinking -> plan target path, and the plan loss. This file only threads the
resulting tensors through the forward. When the group carries no plan slots the
step is byte-for-byte the original ELF: no extra RNG draws, no change to the
loss graph.
"""

import hashlib
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.train_utils import TrainState, ema_update, unwrap_model
from utils.encoder_utils import encode_x0
from utils.loss_utils import token_cross_entropy, token_feature_mse
from utils.plan_stream import build_plan_stream, plan_loss, resolve_group
from utils.sampling_utils import (
    sample_cfg_scale, add_noise, sample_timesteps, net_out_to_v_x, restore_cond,
)


def _trainable_params(model: nn.Module):
    return [p for p in model.parameters() if p.requires_grad]

def _diagnostic_tensor_hash(value):
    """Shape/dtype-tagged content hash, used by the cross-group smoke report."""
    if value is None:
        return None
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(
        str((array.shape, array.dtype)).encode() + array.tobytes()
    ).hexdigest()


def _sample_plan_mediation_mask(batch, batch_size, probability, generator, device):
    """Sample the training-only prompt-bypass intervention without perturbing formal RNG.

    Formal Stage-B rows carry a stable branch seed. Hashing that seed makes the
    intervention deterministic across resume while leaving every existing noise,
    branch, self-conditioning, and model RNG stream untouched. Legacy batches use
    the training generator only when the intervention is enabled.
    """
    probability = float(probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("plan_mediation_prob must be within [0, 1]")
    if probability == 0.0:
        return None
    if probability == 1.0:
        return torch.ones((batch_size,), dtype=torch.bool, device=device)
    if "branch_seed" in batch:
        values = batch["branch_seed"].reshape(-1)
        if values.numel() != batch_size:
            raise ValueError("branch_seed must contain one value per batch row")
        threshold = int(probability * (2 ** 64))
        selected = []
        for value in values.tolist():
            digest = hashlib.sha256(f"plan-mediation:{int(value)}".encode()).digest()
            selected.append(int.from_bytes(digest[:8], "little") < threshold)
        return torch.tensor(selected, dtype=torch.bool, device=device)
    return (
        torch.rand((batch_size,), generator=generator) < probability
    ).to(device=device)


def _gate_plan_mediation_by_clock(mask, t_plan_input, minimum):
    """Keep sampled mediation rows only when the model sees a sufficiently clean plan."""
    minimum = float(minimum)
    if not 0.0 <= minimum <= 1.0:
        raise ValueError("plan_mediation_min_t must be within [0, 1]")
    if mask is None:
        return None
    if t_plan_input is None:
        raise ValueError("plan mediation requires a runtime plan clock")
    if tuple(t_plan_input.shape) != tuple(mask.shape):
        raise ValueError("runtime plan clock must have shape [B]")
    return mask & (t_plan_input >= minimum)


def train_step(
    state: TrainState,
    encoder: nn.Module,
    batch: Dict[str, torch.Tensor],
    config,
    plan_encoder: nn.Module = None,
    update_model: bool = True,
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

    group = resolve_group(config)
    if (group.plan_enabled and config.plan_resampler != "frozen_pool"
            and getattr(config, "plan_source", "frozen_pool") == "frozen_pool"):
        raise NotImplementedError(
            f"plan_resampler={config.plan_resampler!r} is not implemented in v1; "
            "use 'frozen_pool' (learnable resampler is a Step-2 ablation)."
        )

    gen = state.dropout_generator

    def schedule_seed(name, offset=0):
        """Reseed from the schedule so the four groups see identical noise per sample.

        A no-op unless the batch carries schedule seeds, which is what keeps the
        legacy (non-formal) training path bit-identical to upstream ELF.
        """
        if name not in batch:
            return
        values = batch[name].reshape(-1)
        if values.numel() != 1:
            raise ValueError("common schedule training requires microbatch=1")
        seed = (int(values.item()) + int(offset)) % (2 ** 63 - 1)
        torch.manual_seed(seed)
        gen.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    # encoder_attention_mask: cond sees cond, x sees all
    input_ids = batch["input_ids"].to(device, non_blocking=True).long()
    encoder_attention_mask = batch["encoder_attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
    cond_seq_mask = batch["cond_seq_mask"].to(device, dtype=torch.float32, non_blocking=True)
    condition_token_mask = cond_seq_mask.bool()
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

    x0 = encode_x0(
        input_ids=input_ids,
        attention_mask=encoder_attention_mask,
        encoder=encoder,
        latent_mean=latent_mean,
        latent_std=latent_std,
        use_bf16=use_bf16,
        cond_mask=cond_seq_mask,
        drop_condition=label_drop_mask if config.label_drop_prob > 0 else None,
    ).to(dtype)

    batch_size, seq_length = x0.shape[0], x0.shape[1]

    plan_mediation_prob = float(getattr(config, "plan_mediation_prob", 0.0))
    plan_mediation_mask = _sample_plan_mediation_mask(
        batch, batch_size, plan_mediation_prob, gen, device,
    )
    if plan_mediation_mask is not None:
        attention_mode = getattr(unwrap_model(state.model), "plan_response_attention", None)
        if attention_mode != "prompt_causal_bottleneck":
            raise ValueError(
                "plan_mediation_prob > 0 requires "
                "plan_response_attention='prompt_causal_bottleneck'"
            )

    schedule_seed("token_time_seed")
    t = sample_timesteps(
        batch_size,
        P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
        time_schedule=config.time_schedule,
        device=device, dtype=dtype,
    )

    schedule_seed("response_noise_seed")
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
    schedule_seed("branch_seed")
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

    # ---- Planning stream. utils/plan_stream.py holds the four-group table. ----
    # Every draw in there is guarded by group.plan_enabled, so a plan-free run
    # consumes no extra RNG and its stream matches the original ELF exactly.
    plan = build_plan_stream(
        config=config, group=group, batch=batch, model=unwrap_model(state.model),
        encoder=encoder, plan_encoder=plan_encoder, t=t,
        decoder_step_active=decoder_step_active, x0=x0, loss_mask=loss_mask,
        schedule_seed=schedule_seed,
    )
    plan_mediation_mask = _gate_plan_mediation_by_clock(
        plan_mediation_mask, plan.t_plan_input,
        getattr(config, "plan_mediation_min_t", 0.0),
    )

    schedule_seed("branch_seed", offset=1)
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
                x_plan=plan.aux_x, t_plan=plan.aux_t,
                plan_mask=plan.plan_mask,
                condition_token_mask=condition_token_mask,
                plan_mediation_mask=plan_mediation_mask,
            )
        return net_out_uncond

    def get_sc_cond_and_uncond(z, t_input, cond_mask, x_tokens, shared_net_out_uncond):
        if config.self_cond_prob == 0:
            with torch.no_grad(), torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=use_bf16):
                net_out_uncond = model(
                    z, t_input,
                    deterministic=True, self_cond_cfg_scale=self_cond_cfg_scale,
                    x_plan=plan.aux_x, t_plan=plan.aux_t,
                    plan_mask=plan.plan_mask,
                    condition_token_mask=condition_token_mask,
                    plan_mediation_mask=plan_mediation_mask,
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
                x_plan=plan.aux_x, t_plan=plan.aux_t,
                plan_mask=plan.plan_mask,
                condition_token_mask=condition_token_mask,
                plan_mediation_mask=plan_mediation_mask,
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

    model.train(update_model)

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
            x_plan=plan.x_plan_input, t_plan=plan.t_plan_input,
            plan_mask=plan.plan_mask,
            condition_token_mask=condition_token_mask,
            plan_mediation_mask=plan_mediation_mask,
        )

    # CE per-token (used on decoder-mode rows).
    ce_per_token = token_cross_entropy(decoder_logits, decoder_targets)

    # L2 per-token (used on denoiser-mode rows). v_pred is extracted with
    # (denoiser_z, t) — meaningful only for denoiser rows; decoder rows are
    # masked out below.
    v_pred, _ = net_out_to_v_x(net_out, denoiser_z, denoiser_t, t_eps)
    v_final_target = get_v_target(
        denoiser_z, denoiser_t, base_v_target=v_target, x_tokens=x0,
        shared_net_out_uncond=shared_net_out_uncond,
    )
    l2_per_token = token_feature_mse(v_pred, v_final_target)

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
    if plan.supervised:
        plan_l2 = plan_loss(
            plan_out, plan, decoder_step_active,
            low_t_boost=float(getattr(config, "plan_low_t_loss_boost", 0.0)),
        )
        loss = loss + config.plan_loss_weight * plan_l2
        plan_l2_val = plan_l2.detach()
    else:
        plan_l2_val = torch.zeros((), device=device)

    grad_norm_val = torch.zeros((), device=device)
    if update_model:
        accum_steps = max(config.grad_accum_steps, 1)
        state.step += 1
        is_optimizer_step = (state.step % accum_steps) == 0

        # DDP synchronization is controlled by the caller, whose no_sync context
        # covers both forward and backward. Wrapping backward alone is ineffective.
        (loss / accum_steps).backward()

        if is_optimizer_step:
            grad_norm_val = torch.nn.utils.clip_grad_norm_(_trainable_params(model), max_norm=1.0).detach()
            state.optimizer.step()
            if state.lr_scheduler is not None:
                state.lr_scheduler.step()
            ema_update(state.ema_params1, state.model, config.ema_decay1)
            state.optimizer.zero_grad(set_to_none=True)
    else:
        is_optimizer_step = False

    metrics = {
        "loss": loss.detach(),
        "l2_loss": l2_loss_val,
        "ce_loss": ce_loss_val,
        "plan_l2_loss": plan_l2_val,
        "gradient_norm": grad_norm_val,
        "response_valid_tokens": loss_mask.sum().detach(),
        "plan_valid_slots": (plan.plan_mask.sum().detach() if plan.plan_mask is not None
                             else torch.zeros((), device=device)),
        "plan_capacity_slots": (torch.tensor(plan.plan_mask.numel(), device=device)
                                if plan.plan_mask is not None else torch.zeros((), device=device)),
        "plan_mediation_rows": (plan_mediation_mask.sum().detach()
                                  if plan_mediation_mask is not None
                                  else torch.zeros((), device=device)),
        "optimizer_step": is_optimizer_step,
        "denoiser_rows": (1.0 - decoder_mask_B1).sum().detach(),
        "decoder_rows": decoder_mask_B1.sum().detach(),
        "group_mode": group.mode,
        "plan_present": bool(plan.x_plan_input is not None),
        # The clock the model actually saw, not the sampled one: the diagonal
        # invariant is a property of the inputs, and reporting the pre-mix draw is
        # what let a decoder-row clock mismatch sit undetected in the logs.
        "plan_clock_on_diagonal": bool(
            plan.t_plan_input is not None and torch.equal(plan.t_plan_input, t_mixed)),
        "plan_time_all_zero": bool(
            plan.t_plan_input is not None and torch.count_nonzero(plan.t_plan_input).item() == 0),
        "register_no_thinking_target": bool(group.register_only and plan.x0_plan is None),
    }
    # Collator truncation flags (conditional Stage-B). Surfaced per step so a
    # silently capped plan target shows up in the log instead of in the results.
    for key in ("prompt_truncated", "response_truncated", "thinking_truncated"):
        if key in batch:
            metrics[key] = batch[key].float().sum().detach()
    if bool(getattr(config,"engineering_smoke_report",False)):
        metrics.update({
            "response_noise_sha256": _diagnostic_tensor_hash(noise),
            "token_time_sha256": _diagnostic_tensor_hash(t),
            "branch_sha256": _diagnostic_tensor_hash(decoder_step_active),
            "plan_mask_sha256": _diagnostic_tensor_hash(plan.plan_mask),
            "plan_input_sha256": _diagnostic_tensor_hash(plan.x_plan_input),
        })
    return state, metrics
