"""Shared plan-target utilities for Ordered ELF."""

import torch

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
