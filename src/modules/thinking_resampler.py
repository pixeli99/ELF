#!/usr/bin/env python
"""Single-stage adjacent-token compression components for offline probes."""

import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_adjacent_mlp_encoder(
    embedding_dim: int = 512, group_size: int = 4, hidden_dim: int = 1024,
) -> nn.Sequential:
    flat_dim = embedding_dim * group_size
    return nn.Sequential(
        nn.Linear(flat_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, embedding_dim),
    )


def group_adjacent_tokens(
    token_embeddings: torch.Tensor, group_size: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad `[L,D]` embeddings into `[ceil(L/K),K,D]` and a valid mask."""
    if token_embeddings.ndim != 2:
        raise ValueError(f"Expected [L,D], got {tuple(token_embeddings.shape)}")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    length, width = token_embeddings.shape
    groups = math.ceil(length / group_size)
    padded_length = groups * group_size
    padded = token_embeddings.new_zeros((padded_length, width))
    padded[:length] = token_embeddings
    mask = torch.arange(padded_length, device=token_embeddings.device) < length
    return padded.view(groups, group_size, width), mask.view(groups, group_size)


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    module.requires_grad_(False)
    return module


def split_document_indices(num_documents: int, seed: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """Deterministically split whole documents 80/10/10."""
    if num_documents < 10:
        raise ValueError("At least 10 documents are required for an 80/10/10 split")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_documents, generator=generator).tolist()
    train_end = int(num_documents * 0.8)
    val_end = train_end + int(num_documents * 0.1)
    return order[:train_end], order[train_end:val_end], order[val_end:]


def split_document_indices_90_10(
    num_documents: int, seed: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """Deterministically split whole documents into 90% train and 10% validation."""
    if num_documents < 10:
        raise ValueError("At least 10 documents are required for a 90/10 split")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(num_documents, generator=generator).tolist()
    train_end = int(num_documents * 0.9)
    return order[:train_end], order[train_end:], []


def masked_reconstruction_metrics(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return elementwise MSE and mean token cosine over valid token rows only."""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if valid_mask.shape != target.shape[:-1]:
        raise ValueError("valid_mask must match all token dimensions")
    valid = valid_mask.bool()
    selected_prediction = prediction[valid]
    selected_target = target[valid]
    if selected_target.numel() == 0:
        raise ValueError("valid_mask selects no tokens")
    mse = F.mse_loss(selected_prediction, selected_target)
    cosine = F.cosine_similarity(selected_prediction, selected_target, dim=-1).mean()
    return mse, cosine


class AdjacentMLPResampler(nn.Module):
    """One composable 4-to-1 encoder and temporary reconstruction decoder."""

    def __init__(self, embedding_dim: int = 512, group_size: int = 4, hidden_dim: int = 1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.group_size = group_size
        flat_dim = embedding_dim * group_size
        self.encoder = build_adjacent_mlp_encoder(embedding_dim, group_size, hidden_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, flat_dim),
        )

    def compress(self, groups: torch.Tensor) -> torch.Tensor:
        return self.encoder(groups.flatten(start_dim=-2))

    def reconstruct(self, slots: torch.Tensor) -> torch.Tensor:
        shape = slots.shape[:-1] + (self.group_size, self.embedding_dim)
        return self.decoder(slots).view(shape)

    def forward(self, groups: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        slots = self.compress(groups)
        return slots, self.reconstruct(slots)


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


def masked_mean_slots(groups: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    weights = valid_mask.to(groups.dtype).unsqueeze(-1)
    return (groups * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


def mean_reconstruction(groups: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    slots = masked_mean_slots(groups, valid_mask)
    return slots.unsqueeze(-2).expand_as(groups)
