import torch
import numpy as np


@torch.no_grad()
def encode_text(
    input_ids,
    attention_mask,
    encoder,
    latent_mean,
    latent_std,
    use_bf16=True,
    cond_mask=None,
):
    """Encoder pass from text to latent with normalization.

    A conditional batch carries a (B, S, S) mask: condition positions attend only to the
    condition, so the condition encoding does not depend on the answer, while the rest
    attend to everything valid. transformers stopped accepting a 3D encoder mask in 4.45.
    T5Stack now writes `attention_mask[:, None, None, :]` with no rank check, which turns a
    3D mask into a 5D one and fails inside the attention with a broadcast error. The repo
    pins `transformers<4.45` for exactly this reason; this path keeps it working on newer
    ones instead, because the cluster image is not ours to pin.

    Every row of that mask is one of two vectors, the one condition positions use and the
    one the remaining positions use, so running the encoder once per row type with an
    ordinary 2D mask and selecting by position reproduces it exactly. `cond_mask` (B, S)
    says which positions are condition positions; without it, a 3D mask is accepted only
    when all its rows agree, which is the unconditional case.
    """
    autocast_enabled = bool(use_bf16) and input_ids.is_cuda
    passes = _encoder_mask_passes(attention_mask, cond_mask)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=autocast_enabled):
        if len(passes) == 1:
            latents = encoder(input_ids=input_ids, attention_mask=passes[0], deterministic=True)
        else:
            cond_rows, other_rows = passes
            latents = torch.where(
                cond_mask.unsqueeze(-1) > 0,
                encoder(input_ids=input_ids, attention_mask=cond_rows, deterministic=True),
                encoder(input_ids=input_ids, attention_mask=other_rows, deterministic=True),
            )
    return (latents - latent_mean) / latent_std


def _encoder_mask_passes(attention_mask, cond_mask):
    """Split a (B, S, S) encoder mask into the one or two 2D masks it is made of."""
    if attention_mask is None or attention_mask.dim() == 2:
        return (attention_mask,)
    if attention_mask.dim() != 3:
        raise ValueError(f"encoder attention mask must be 2D or 3D, got {tuple(attention_mask.shape)}")

    batch = attention_mask.shape[0]
    index = torch.arange(batch, device=attention_mask.device)
    if cond_mask is None or float(cond_mask.sum()) == 0.0:
        first = attention_mask[:, 0, :]
        if not torch.equal(attention_mask, first.unsqueeze(1).expand_as(attention_mask)):
            raise ValueError(
                "a 3D encoder attention mask whose rows differ needs cond_mask to say "
                "which positions are condition positions"
            )
        return (first,)

    # argmax on a 0/1 row returns the first 1, and the first 0 on its complement.
    cond_rows = attention_mask[index, cond_mask.argmax(dim=1)]
    other_rows = attention_mask[index, (1 - cond_mask).argmax(dim=1)]
    return (cond_rows, other_rows)


def build_self_attn_cond_masks(is_cond, is_valid, xp=np):
    """Build self-attention conditioning masks from cond/valid token flags."""
    encoder_attention_mask = (
        (is_cond[:, :, None] & is_cond[:, None, :]) |
        (~is_cond[:, :, None] & is_valid[:, None, :])
    ).astype(xp.float32)
    attention_mask = is_valid.astype(xp.float32)
    cond_seq_mask = is_cond.astype(xp.float32)
    return encoder_attention_mask, attention_mask, cond_seq_mask
