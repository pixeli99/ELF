"""Shared masked loss primitives used by training and diagnostic probes."""

import torch
import torch.nn.functional as F


def token_feature_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-token MSE over the latent feature dimension."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    return (prediction - target).square().mean(dim=-1)


def token_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-token decoder CE, identical to the original train_step formula."""
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logits leading dimensions must match targets")
    log_probs = F.log_softmax(logits.to(torch.float32), dim=-1)
    return -log_probs.gather(-1, targets.long().unsqueeze(-1)).squeeze(-1)


def masked_token_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over valid token positions with the training denominator convention."""
    if values.shape != mask.shape:
        raise ValueError("values and mask shapes must match")
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def masked_token_feature_mse(
    prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    return masked_token_mean(token_feature_mse(prediction, target), mask)


def masked_decoder_cross_entropy(
    logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor,
) -> torch.Tensor:
    return masked_token_mean(token_cross_entropy(logits, targets), mask)
