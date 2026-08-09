"""Shared, model-independent pieces of the formal thinking MLP pipeline."""

import hashlib
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.modules.thinking_resampler import group_batched_thinking
from src.utils.thinking_tokenization import thinking_token_ids

LOGGER = logging.getLogger(__name__)


def load_trusted_full_training_checkpoint(path, map_location="cpu"):
    """Load complete local training state from a trusted project checkpoint.

    This explicitly enables pickle object loading, which may execute code. Only use
    it for local checkpoints produced by this project whose provenance is trusted.
    """
    LOGGER.info("Loading trusted full formal-thinking training checkpoint: %s", path)
    return torch.load(path, map_location=map_location, weights_only=False)


def load_frozen_encoder_artifact(path, map_location="cpu"):
    """Load the tensor/basic-field frozen encoder artifact without pickle objects."""
    LOGGER.info("Loading weights-only formal-thinking frozen encoder: %s", path)
    return torch.load(path, map_location=map_location, weights_only=True)


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def refuse_existing(path: str) -> None:
    if Path(path).exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")


def configure_formal_t5_tokenizer(tokenizer, model_max_length: int = 1024):
    if model_max_length != 1024:
        raise ValueError("formal thinking protocol requires model_max_length=1024")
    tokenizer.model_max_length = model_max_length
    return tokenizer


class CanonicalThinkingDataset(Dataset):
    """Tokenized dataset that deliberately exposes no response field to a model batch."""

    def __init__(self, jsonl_path: str, tokenizer, max_thinking_tokens: int = 1024):
        self.path = str(Path(jsonl_path).resolve())
        self.rows: List[Dict[str, object]] = []
        with open(jsonl_path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                row = json.loads(line)
                thinking = row.get("thinking")
                if not isinstance(thinking, str) or not thinking.strip():
                    raise ValueError(f"{jsonl_path}:{line_number}: missing or empty thinking")
                ids = thinking_token_ids(
                    tokenizer, thinking, add_special_tokens=True, truncation=False,
                )
                sample_id = str(row.get("sample_id", line_number - 1))
                if len(ids) > max_thinking_tokens:
                    raise ValueError(
                        f"{jsonl_path}:{line_number}: sample_id={sample_id} thinking length "
                        f"{len(ids)} exceeds "
                        f"formal limit {max_thinking_tokens}"
                    )
                self.rows.append({"sample_id": sample_id,
                                  "thinking_input_ids": ids})
        self.max_thinking_length = max(
            (len(row["thinking_input_ids"]) for row in self.rows), default=0,
        )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


class ThinkingCollator:
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = int(pad_token_id)

    def __call__(self, rows: Sequence[Mapping[str, object]]) -> Dict[str, object]:
        length = max(len(row["thinking_input_ids"]) for row in rows)
        ids = torch.full((len(rows), length), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), length), dtype=torch.bool)
        for index, row in enumerate(rows):
            values = torch.as_tensor(row["thinking_input_ids"], dtype=torch.long)
            ids[index, : values.numel()] = values
            mask[index, : values.numel()] = True
        return {"sample_id": [row["sample_id"] for row in rows],
                "thinking_input_ids": ids, "thinking_attention_mask": mask}


def masked_reconstruction_mse(prediction, target, reconstruction_mask):
    if prediction.shape != target.shape or reconstruction_mask.shape != target.shape[:-1]:
        raise ValueError("reconstruction shapes do not agree")
    per_token = (prediction.float() - target.float()).square().mean(-1)
    valid = reconstruction_mask.bool()
    if not valid.any():
        raise ValueError("reconstruction mask contains no valid token")
    return per_token[valid].mean()


def masked_token_cosine(prediction, target, reconstruction_mask):
    valid = reconstruction_mask.bool()
    return F.cosine_similarity(prediction.float()[valid], target.float()[valid], dim=-1).mean()


def mean_pool_reconstruction(groups, reconstruction_mask):
    weights = reconstruction_mask.to(groups.dtype).unsqueeze(-1)
    mean = (groups * weights).sum(-2) / weights.sum(-2).clamp_min(1)
    return mean.unsqueeze(-2).expand_as(groups)


class StreamingChannelMoments:
    """Mergeable FP64 channel moments over valid plan slots."""
    def __init__(self, dim: int):
        self.dim, self.count = int(dim), 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.m2 = torch.zeros(dim, dtype=torch.float64)

    def update(self, values: torch.Tensor, mask: torch.Tensor) -> None:
        selected = values.detach().double()[mask.bool()].cpu()
        if selected.numel() == 0:
            return
        n = selected.shape[0]
        mean = selected.mean(0)
        m2 = ((selected - mean).square()).sum(0)
        if self.count == 0:
            self.count, self.mean, self.m2 = n, mean, m2
            return
        total = self.count + n
        delta = mean - self.mean
        self.m2 += m2 + delta.square() * self.count * n / total
        self.mean += delta * n / total
        self.count = total

    def finalize(self):
        if self.count == 0:
            raise ValueError("no valid plan slots were observed")
        return self.mean, torch.sqrt(self.m2 / self.count), self.count


def whiten(values, mean, std, eps=1e-6):
    return (values - mean.to(values)) / (std.to(values) + eps)


def inverse_whiten(values, mean, std, eps=1e-6):
    return values * (std.to(values) + eps) + mean.to(values)


def load_data_manifest(path: str, require_formal: bool = False) -> Dict[str, object]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if require_formal and manifest.get("formal_ready") is not True:
        raise ValueError("formal data manifest must declare formal_ready=true")
    for split in ("train", "validation"):
        entry = manifest.get("splits", {}).get(split)
        if not entry or not Path(entry["path"]).is_file():
            raise ValueError(f"manifest has no readable {split} split")
        if entry.get("sha256") and sha256_file(entry["path"]) != entry["sha256"]:
            raise ValueError(f"{split} split SHA256 mismatch")
    return manifest


def canonical_t5_x0(encoder, input_ids, attention_mask, latent_mean, latent_std,
                    max_valid_length: int = 1024):
    """The one canonical runtime path: frozen T5 output then ELF normalization."""
    from src.utils.encoder_utils import encode_text
    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError("input_ids and attention_mask must both have shape [B,L]")
    valid_lengths = attention_mask.bool().sum(dim=1)
    if bool((valid_lengths > max_valid_length).any()):
        raise ValueError(
            f"thinking valid length exceeds formal limit {max_valid_length}: "
            f"maximum={int(valid_lengths.max())}"
        )
    encoder.eval().requires_grad_(False)
    with torch.no_grad():
        normalized = encode_text(
            input_ids=input_ids,
            attention_mask=attention_mask.bool(),
            encoder=encoder,
            latent_mean=latent_mean,
            latent_std=latent_std,
        )
        return normalized.float()
def resolve_decoder_response_width(model, config) -> int:
    """Resolve and cross-check the fixed response width used by an ELF checkpoint."""
    model_width = getattr(model, "max_length", None)
    config_width = getattr(config, "max_length", None)
    if model_width is None or config_width is None:
        raise ValueError("model and config must both declare max_length")
    if int(model_width) != int(config_width):
        raise ValueError(
            f"decoder response width mismatch: model={model_width}, config={config_width}"
        )
    return int(model_width)


def restore_grouped_token_layout(grouped, reconstruction_mask, valid_mask):
    """Remove only structural tail-group padding and restore original token width."""
    if grouped.ndim != 4 or reconstruction_mask.shape != grouped.shape[:-1]:
        raise ValueError("grouped latent and reconstruction_mask shapes disagree")
    if valid_mask.ndim != 2 or valid_mask.shape[0] != grouped.shape[0]:
        raise ValueError("valid_mask must have shape [B,L]")
    flat = grouped.flatten(1, 2)
    flat_group_mask = reconstruction_mask.flatten(1, 2).bool()
    token_width = valid_mask.shape[1]
    if flat.shape[1] < token_width:
        raise ValueError("grouped latent is shorter than the original token layout")
    if not torch.equal(flat_group_mask[:, :token_width], valid_mask.bool()):
        raise ValueError("group reconstruction mask does not match original token mask")
    if bool(flat_group_mask[:, token_width:].any()):
        raise ValueError("tail group padding is incorrectly marked valid")
    restored = flat[:, :token_width]
    return restored * valid_mask.bool().unsqueeze(-1).to(restored.dtype)


def adapt_to_fixed_decoder_width(latent, input_ids, valid_mask, decoder_response_width,
                                 pad_token_id):
    """Right-pad response-only tensors to the checkpoint's fixed decoder width."""
    if latent.ndim != 3 or input_ids.ndim != 2 or valid_mask.ndim != 2:
        raise ValueError("expected latent [B,L,C], input_ids/mask [B,L]")
    if latent.shape[:2] != input_ids.shape or input_ids.shape != valid_mask.shape:
        raise ValueError("latent, input_ids, and valid_mask batch/sequence shapes disagree")
    mask = valid_mask.bool()
    lengths = mask.sum(1)
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    if not torch.equal(mask, positions < lengths.unsqueeze(1)):
        raise ValueError("valid_mask must describe right-padded token sequences")
    if bool((lengths > decoder_response_width).any()) or latent.shape[1] > decoder_response_width:
        raise ValueError(
            f"response length exceeds decoder width {decoder_response_width}: "
            f"maximum_valid={int(lengths.max())}, tensor_width={latent.shape[1]}"
        )
    pad = decoder_response_width - latent.shape[1]
    padded_latent = F.pad(latent * mask.unsqueeze(-1).to(latent.dtype), (0, 0, 0, pad))
    padded_ids = F.pad(input_ids, (0, pad), value=int(pad_token_id))
    padded_mask = F.pad(mask, (0, pad), value=False)
    if not torch.equal(padded_mask.sum(1), lengths):
        raise AssertionError("fixed-width adaptation changed valid token counts")
    return padded_latent, padded_ids, padded_mask
