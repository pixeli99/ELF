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
def encode_conditional_x0(input_ids, valid_mask, cond_mask, encoder,
                          latent_mean, latent_std, use_bf16=True,
                          drop_condition=None):
    """Frozen-T5 x0 for a [prompt | response] window, using 2-D masks only.

    The semantics the conditional path needs are asymmetric: prompt latents must
    depend on the prompt alone, because that is all generation ever has, while
    response latents must see the prompt. Upstream expressed that as one forward
    with a 3-D attention mask, which only ever worked on transformers < 4.45
    (4.51 broadcasts it to five dimensions and raises). Two 2-D forwards give
    exactly the same latents on every version: T5 relative positions are
    unchanged because the prompt occupies the same positions in both passes.

    `drop_condition` is a per-row bool: those rows encode their response with the
    prompt hidden, which is the classifier-free-guidance null condition.
    """
    valid_mask = valid_mask.bool()
    cond_mask = cond_mask.bool()
    response_mask = valid_mask
    if drop_condition is not None and bool(drop_condition.any()):
        drop = drop_condition.reshape(-1, 1).to(valid_mask.device)
        response_mask = valid_mask & ~(drop & cond_mask)
    x0 = encode_text(input_ids, response_mask, encoder, latent_mean, latent_std,
                     use_bf16=use_bf16)
    if bool(cond_mask.any()):
        prompt_only = encode_text(input_ids, cond_mask, encoder, latent_mean, latent_std,
                                  use_bf16=use_bf16)
        x0 = torch.where(cond_mask.unsqueeze(-1), prompt_only, x0)
    return x0


@torch.no_grad()
def encode_x0(input_ids, attention_mask, encoder, latent_mean, latent_std,
              use_bf16=True, cond_mask=None, drop_condition=None):
    """Dispatch to the encoding a batch actually asks for.

    A 3-D `attention_mask` is the inherited upstream form and is passed through
    untouched. A 2-D mask with a non-empty `cond_mask` is the conditional
    Stage-B form and goes through the two-pass path above. Everything else is
    plain unconditional encoding.
    """
    if attention_mask.dim() == 3:
        return encode_text(input_ids, attention_mask, encoder, latent_mean, latent_std,
                           use_bf16=use_bf16)
    if cond_mask is not None and bool(cond_mask.any()):
        return encode_conditional_x0(input_ids, attention_mask, cond_mask, encoder,
                                     latent_mean, latent_std, use_bf16=use_bf16,
                                     drop_condition=drop_condition)
    return encode_text(input_ids, attention_mask, encoder, latent_mean, latent_std,
                       use_bf16=use_bf16)


@torch.no_grad()
def encode_thinking_x0(input_ids, attention_mask, encoder, latent_mean, latent_std,
                       use_bf16=True):
    """Frozen-T5 thinking latents in the space the Stage-A stack was fitted on.

    The 4-to-1 MLP (`tools/train_formal_thinking_mlp.py`) and its whitener
    (`tools/compute_formal_thinking_whitener.py`) both consume ELF-normalized
    latents, so every Stage-B caller must normalize before touching the frozen
    encoder. Feeding raw T5 output shrinks the MLP input by 1/latent_std and
    pushes the compressed slots off the manifold the decoder was trained on.
    """
    return encode_text(
        input_ids=input_ids, attention_mask=attention_mask, encoder=encoder,
        latent_mean=latent_mean, latent_std=latent_std, use_bf16=use_bf16,
    ).float()


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
