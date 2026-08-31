"""The conditional encoder mask must survive the transformers version in the image.

The contract the (B, S, S) mask encodes is that a condition position's latent depends on
the condition and nothing else. If that breaks, the condition silently starts leaking the
answer and every conditional number is worth nothing, so it is asserted rather than assumed.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from utils.encoder_utils import _encoder_mask_passes, build_self_attn_cond_masks, encode_text


def masks_for(cond_len, total_len, width, batch=1):
    positions = np.arange(width)[None, :].repeat(batch, axis=0)
    is_cond = positions < cond_len
    is_valid = positions < total_len
    encoder_mask, attention_mask, cond_seq_mask = build_self_attn_cond_masks(is_cond, is_valid)
    return (torch.from_numpy(encoder_mask), torch.from_numpy(attention_mask),
            torch.from_numpy(cond_seq_mask))


def test_passes_split_a_conditional_mask_into_its_two_rows():
    encoder_mask, valid, cond = masks_for(cond_len=3, total_len=7, width=10)
    cond_rows, other_rows = _encoder_mask_passes(encoder_mask, cond)
    assert torch.equal(cond_rows, cond), "condition positions attend to the condition"
    assert torch.equal(other_rows, valid), "the rest attend to everything valid"


def test_passes_collapse_when_there_is_no_condition():
    encoder_mask, valid, cond = masks_for(cond_len=0, total_len=7, width=10)
    passes = _encoder_mask_passes(encoder_mask, cond)
    assert len(passes) == 1 and torch.equal(passes[0], valid)


def test_a_two_dimensional_mask_is_passed_straight_through():
    valid = torch.ones(2, 5)
    assert _encoder_mask_passes(valid, None) == (valid,)


def test_label_dropped_rows_keep_their_own_mask():
    """Label drop cuts response->condition attention, giving a third row vector."""
    encoder_mask, valid, cond = masks_for(cond_len=3, total_len=7, width=10)
    block = (1 - cond).unsqueeze(-1) * cond.unsqueeze(1)
    dropped = encoder_mask * (1 - block)
    cond_rows, other_rows = _encoder_mask_passes(dropped, cond)
    assert torch.equal(cond_rows, cond)
    assert torch.equal(other_rows, valid * (1 - cond)), "a dropped row cannot see the condition"


def test_rows_that_disagree_without_a_cond_mask_are_rejected():
    encoder_mask, _, _ = masks_for(cond_len=3, total_len=7, width=10)
    with pytest.raises(ValueError, match="cond_mask"):
        _encoder_mask_passes(encoder_mask, None)


@pytest.mark.slow
def test_condition_latents_do_not_depend_on_the_response():
    """The whole point of the 3D mask, checked against the real t5-small."""
    from modules.t5_encoder import get_encoder

    _, encoder = get_encoder("t5-small", torch.float32)
    encoder = encoder.eval()

    width, cond_len = 16, 5
    prompt = [100, 200, 300, 400, 500]
    latents = []
    for answer in ([11, 12, 13], [900, 901, 902, 903, 904, 905]):
        ids = np.full((1, width), 1, dtype=np.int64)
        ids[0, :cond_len] = prompt
        ids[0, cond_len:cond_len + len(answer)] = answer
        encoder_mask, _, cond = masks_for(cond_len, cond_len + len(answer), width)
        latents.append(encode_text(
            input_ids=torch.from_numpy(ids), attention_mask=encoder_mask, encoder=encoder,
            latent_mean=0.0, latent_std=0.2, use_bf16=False, cond_mask=cond,
        ))

    condition_a, condition_b = latents[0][:, :cond_len], latents[1][:, :cond_len]
    assert torch.allclose(condition_a, condition_b, atol=1e-5), (
        "condition latents changed when only the answer changed: the mask is leaking"
    )
    response_a, response_b = latents[0][:, cond_len:], latents[1][:, cond_len:]
    assert not torch.allclose(response_a[:, :3], response_b[:, :3], atol=1e-3), (
        "response latents did not change when the answer changed: nothing is being encoded"
    )
