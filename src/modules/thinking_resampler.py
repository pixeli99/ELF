#!/usr/bin/env python
"""Single-stage adjacent-token compression components for offline probes."""

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    module.requires_grad_(False)
    return module

@dataclass(frozen=True)
class ThinkingMLPConfig:
    input_dim: int = 512
    group_size: int = 4
    hidden_dim: int = 6144
    slot_dim: int = 512
    activation: str = "gelu"
    dropout: float = 0.0
    use_bias: bool = True

    @property
    def grouped_dim(self) -> int:
        return self.input_dim * self.group_size

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def group_batched_thinking(
    thinking_x0: torch.Tensor, thinking_mask: torch.Tensor, group_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Group `[B,L,D]` without letting padded rows enter a group or metric."""
    if thinking_x0.ndim != 3 or thinking_mask.shape != thinking_x0.shape[:2]:
        raise ValueError("thinking_x0/mask must have shapes [B,L,D] and [B,L]")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    mask = thinking_mask.bool()
    lengths = mask.sum(1)
    if (lengths <= 0).any():
        raise ValueError("every sample must contain at least one valid thinking token")
    k = int(torch.div(lengths.max() + group_size - 1, group_size, rounding_mode="floor"))
    padded_length = k * group_size
    clean = thinking_x0 * mask.unsqueeze(-1).to(thinking_x0.dtype)
    if padded_length > thinking_x0.shape[1]:
        clean = F.pad(clean, (0, 0, 0, padded_length - thinking_x0.shape[1]))
        mask = F.pad(mask, (0, padded_length - mask.shape[1]), value=False)
    else:
        clean, mask = clean[:, :padded_length], mask[:, :padded_length]
    groups = clean.view(clean.shape[0], k, group_size, clean.shape[-1])
    reconstruction_mask = mask.view(mask.shape[0], k, group_size)
    return groups, reconstruction_mask, reconstruction_mask.any(-1)


class ThinkingMLPEncoder(nn.Module):
    def __init__(self, config: ThinkingMLPConfig = ThinkingMLPConfig()):
        super().__init__()
        if config.activation.lower() != "gelu":
            raise ValueError("formal encoder supports activation=gelu only")
        self.config = config
        layers = [nn.Linear(config.grouped_dim, config.hidden_dim, bias=config.use_bias), nn.GELU()]
        if config.dropout:
            layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(config.hidden_dim, config.slot_dim, bias=config.use_bias))
        self.network = nn.Sequential(*layers)

    def forward(self, grouped_x0: torch.Tensor) -> torch.Tensor:
        if grouped_x0.shape[-2:] != (self.config.group_size, self.config.input_dim):
            raise ValueError("grouped_x0 has incompatible group/input dimensions")
        parameter = next(self.parameters())
        if grouped_x0.device != parameter.device:
            raise RuntimeError(
                f"grouped_x0 device {grouped_x0.device} does not match encoder "
                f"device {parameter.device}"
            )
        grouped_x0 = grouped_x0.to(dtype=parameter.dtype)
        return self.network(grouped_x0.flatten(start_dim=-2))


class ThinkingMLPDecoder(nn.Module):
    """Decoder content input is exclusively the compressed slot tensor."""
    def __init__(self, config: ThinkingMLPConfig = ThinkingMLPConfig()):
        super().__init__()
        self.config = config
        layers = [nn.Linear(config.slot_dim, config.hidden_dim, bias=config.use_bias), nn.GELU()]
        if config.dropout:
            layers.append(nn.Dropout(config.dropout))
        layers.append(nn.Linear(config.hidden_dim, config.grouped_dim, bias=config.use_bias))
        self.network = nn.Sequential(*layers)

    def forward(self, plan_slots: torch.Tensor) -> torch.Tensor:
        parameter = next(self.parameters())
        if plan_slots.device != parameter.device:
            raise RuntimeError(
                f"plan_slots device {plan_slots.device} does not match decoder "
                f"device {parameter.device}"
            )
        plan_slots = plan_slots.to(dtype=parameter.dtype)
        output = self.network(plan_slots)
        return output.view(*plan_slots.shape[:-1], self.config.group_size, self.config.input_dim)


class ThinkingMLPAutoencoder(nn.Module):
    def __init__(self, config: ThinkingMLPConfig = ThinkingMLPConfig()):
        super().__init__()
        self.config = config
        self.encoder = ThinkingMLPEncoder(config)
        self.decoder = ThinkingMLPDecoder(config)

    def forward(self, grouped_x0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        slots = self.encoder(grouped_x0)
        return slots, self.decoder(slots)

    def encode_thinking(
        self, thinking_x0: torch.Tensor, thinking_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        groups, _, plan_mask = group_batched_thinking(
            thinking_x0, thinking_mask, self.config.group_size,
        )
        return self.encoder(groups), plan_mask


class FrozenThinkingPlanEncoder(nn.Module):
    """Downstream ELF interface: token latents/mask to dynamic compressed slots/mask."""
    def __init__(self, config: ThinkingMLPConfig = ThinkingMLPConfig()):
        super().__init__()
        self.config = config
        self.encoder = ThinkingMLPEncoder(config)

    def forward(self, thinking_x0: torch.Tensor, thinking_mask: torch.Tensor):
        groups, _, plan_mask = group_batched_thinking(
            thinking_x0, thinking_mask, self.config.group_size,
        )
        return self.encoder(groups), plan_mask
