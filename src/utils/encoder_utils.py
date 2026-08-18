import torch
import numpy as np


@torch.no_grad()
def encode_text_components(
    input_ids,
    attention_mask,
    encoder,
    latent_mean,
    latent_std,
    use_bf16=True,
):
    """Encoder pass from text to latent with normalization."""
    autocast_enabled = bool(use_bf16) and input_ids.is_cuda
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
        latents = encoder(input_ids=input_ids, attention_mask=attention_mask, deterministic=True)
    normalized = (latents - latent_mean) / latent_std
    return latents, normalized


@torch.no_grad()
def encode_text(input_ids, attention_mask, encoder, latent_mean, latent_std, use_bf16=True):
    """Backward-compatible normalized encoder output used by training."""
    _, normalized = encode_text_components(
        input_ids, attention_mask, encoder, latent_mean, latent_std, use_bf16
    )
    return normalized


def canonical_response_inputs(input_ids, sequence_length):
    """Canonical no-prefix response labels and T5 padding mask."""
    if input_ids.ndim != 2 or sequence_length.ndim != 1:
        raise ValueError("expected input_ids [B,L] and sequence_length [B]")
    if input_ids.shape[0] != sequence_length.shape[0]:
        raise ValueError("batch dimensions disagree")
    positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    mask = positions < sequence_length.to(input_ids.device).unsqueeze(1)
    return input_ids, mask


@torch.no_grad()
def encode_response_x0(input_ids, sequence_length, encoder, latent_mean, latent_std,
                       use_bf16=True, return_components=False):
    """Canonical runtime input_ids -> frozen-T5 response x0 path for probes."""
    labels, mask = canonical_response_inputs(input_ids, sequence_length)
    raw, normalized = encode_text_components(
        labels, mask, encoder, latent_mean, latent_std, use_bf16
    )
    return (raw, normalized, mask, labels) if return_components else normalized


def build_self_attn_cond_masks(is_cond, is_valid, xp=np):
    """Build self-attention conditioning masks from cond/valid token flags."""
    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :]) |
        (~is_cond[:, :, None] & is_valid[:, None, :])
    ).astype(xp.float32)
    attention_mask = is_valid.astype(xp.float32)
    cond_seq_mask = is_cond.astype(xp.float32)
    return encoder_attention_mask, attention_mask, cond_seq_mask
