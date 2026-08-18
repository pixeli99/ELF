"""Shared plan-target utilities for Ordered ELF."""

import torch
import torch.nn.functional as F

from utils.sampling_utils import frozen_pool_plan_target


def build_plan_target(model, x0: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Build the Ordered ELF plan target using the model's frozen whitener.

    This is the single shared implementation used by training via
    ``ELF.build_plan_target`` and by eval-time oracle probes.
    """
    pooled = frozen_pool_plan_target(x0, valid_mask, model.num_plan_slots)
    if model.plan_whiten == "none":
        return pooled
    if model.plan_whiten == "pca":
        if int(model.plan_whiten_ready.item()) == 0:
            raise RuntimeError(
                "plan_whiten='pca' used before fitting the whitener "
                "(run the pre-training stats pass)"
            )
        return (pooled - model.plan_target_mean) @ model.plan_target_proj
    return (pooled - model.plan_target_mean) / model.plan_target_std


def apply_plan_whitening(
    model, plan: torch.Tensor, plan_mask: torch.Tensor,
) -> torch.Tensor:
    """Whiten valid dynamic plan slots and force padding rows to zero."""
    if model.plan_whiten == "none":
        whitened = plan
    elif model.plan_whiten == "pca":
        if int(model.plan_whiten_ready.item()) == 0:
            raise RuntimeError("plan_whiten='pca' used before fitting the whitener")
        whitened = (plan - model.plan_target_mean) @ model.plan_target_proj
    else:
        whitened = (plan - model.plan_target_mean) / model.plan_target_std
    return whitened * plan_mask.to(whitened.dtype).unsqueeze(-1)


def masked_plan_mse(
    prediction: torch.Tensor, target: torch.Tensor, plan_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target plan shapes must match")
    values = (prediction - target).square().mean(dim=-1)
    weights = plan_mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def build_thinking_plan_target(
    plan_token_latents: torch.Tensor,
    plan_token_mask: torch.Tensor,
    resampler_encoder,
    max_plan_slots: int,
):
    """Compress valid T5 thinking rows into a dynamically padded slot batch."""
    batch_slots = []
    lengths = []
    embedding_dim = plan_token_latents.shape[-1]
    for latents, token_mask in zip(plan_token_latents, plan_token_mask):
        valid = latents[token_mask.bool()]
        groups = (valid.shape[0] + 3) // 4
        if groups > max_plan_slots:
            raise ValueError(
                f"compressed thinking requires {groups} slots, exceeds max_plan_slots={max_plan_slots}"
            )
        padded = valid.new_zeros((groups * 4, embedding_dim))
        padded[:valid.shape[0]] = valid
        if hasattr(resampler_encoder, "config") and getattr(resampler_encoder.config, "group_size", None) == 4:
            slots = resampler_encoder(padded.view(groups, 4, embedding_dim))
        else:
            slots = resampler_encoder(padded.view(groups, 4 * embedding_dim))
        batch_slots.append(slots)
        lengths.append(groups)
    k_batch = max(lengths)
    output = plan_token_latents.new_zeros((len(batch_slots), k_batch, embedding_dim))
    mask = torch.zeros((len(batch_slots), k_batch), dtype=torch.bool, device=output.device)
    for index, slots in enumerate(batch_slots):
        output[index, :slots.shape[0]] = slots
        mask[index, :slots.shape[0]] = True
    return output, mask
