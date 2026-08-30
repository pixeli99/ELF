#!/usr/bin/env python
"""Plan probes for Ordered ELF: grafting, shuffle, and plan-token consistency.

Three cheap causal checks on a trained plan-enabled checkpoint (unconditional):

1. GRAFT  — roll the recorded plan trajectory by one sample within the batch and replay
   the token rollout with the foreign plan. If the model uses the plan, tokens should
   FOLLOW the grafted plan (large change vs. the normal rollout, but still coherent).
2. SHUFFLE-AT-DECODE — decode the normal final latent conditioned on a rolled plan.
   Tests whether the decoder read-out uses the plan at all.
3. CONSISTENCY — ||whiten(pool(final_latent)) - z_plan|| for matched vs. mismatched
   pairs. If matched ≈ mismatched, the two streams are "each talking to themselves"
   (alarm: the measured gains are likely a register effect, not planning).

Run:
  python src/plan_probes.py --config <train_yml> --checkpoint <ckpt_file_or_dir> \
      --trajectory planning_first --alpha 2.0 --steps 32 --num_samples 8
"""

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import torch
from transformers import AutoTokenizer

from configs.config import load_config_from_yaml, SamplingConfig
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.checkpoint_utils import find_latest_checkpoint
from utils.data_utils import get_pad_token_id
from utils.sampling_utils import get_sampling_steps
from utils.generation_utils import (
    _generate_samples_single_batch, _dlm_decode_batch, mask_after_eos,
)


def parse_args():
    p = argparse.ArgumentParser(description="Ordered ELF plan probes (graft / shuffle / consistency).")
    p.add_argument("--config", type=str, required=True, help="Training YAML the checkpoint was trained with.")
    p.add_argument("--checkpoint", type=str, required=True, help="Checkpoint file or directory.")
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--steps", type=int, default=32)
    p.add_argument("--trajectory", type=str, default="planning_first",
                   choices=["diagonal", "planning_first", "null"])
    p.add_argument("--alpha", type=float, default=2.0)
    p.add_argument("--plan_cfg", type=float, default=1.0)
    p.add_argument("--time_schedule", type=str, default="uniform")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--use_cpu", action="store_true")
    p.add_argument("--out", type=str, default=None, help="Where to write the JSON report.")
    return p.parse_args()


def _load_model(config, device):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, _ = get_encoder(config.encoder_model_name, torch.float32)
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model, max_length=config.max_length,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
        num_plan_slots=config.num_plan_slots,
        num_plan_time_tokens=config.num_plan_time_tokens,
        plan_whiten=config.plan_whiten,
        plan_target_dim=config.plan_target_dim,
        plan_response_attention=getattr(config, "plan_response_attention", "bidirectional"),
    ).to(device).eval()
    return model, tokenizer


def _load_weights(model, ckpt_path, device):
    if os.path.isdir(ckpt_path) or "checkpoint_" not in os.path.basename(ckpt_path):
        found = find_latest_checkpoint(ckpt_path)
        if found:
            ckpt_path = found
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["params"])  # full state incl. whitening buffers
    ema = ckpt.get("ema_params1") or {}
    if ema:
        # EMA holds parameters only; buffers stay from ckpt["params"].
        model.load_state_dict(ema, strict=False)
        print("Loaded EMA parameters.")
    return model


def _decode(z, model, config, x_plan, eos_token_id, pad_token_id, tokenizer):
    ids = _dlm_decode_batch(z=z, model=model, t_final_val=1.0, config=config,
                            self_cond_cfg_scale=1.0, x_plan=x_plan,
                            condition_token_mask=torch.zeros(
                                z.shape[:2], dtype=torch.bool, device=z.device,
                            ))
    ids = mask_after_eos(ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
    texts = [tokenizer.decode(ids[i].cpu().numpy(), skip_special_tokens=True)
             for i in range(ids.shape[0])]
    return ids, texts


def _consistency(model, latent, z_plan):
    """Mean relative distance between whiten(pool(latent)) and the plan the tokens saw."""
    valid = torch.ones(latent.shape[:2], dtype=latent.dtype, device=latent.device)
    pooled = model.build_plan_target(latent.float(), valid)
    num = (pooled - z_plan.float()).norm(dim=-1)
    den = z_plan.float().norm(dim=-1).clamp(min=1e-6)
    return float((num / den).mean())


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if config.num_plan_slots <= 0:
        raise SystemExit("This probe needs a plan-enabled config (num_plan_slots > 0).")
    device = torch.device("cpu") if (args.use_cpu or not torch.cuda.is_available()) else torch.device("cuda")

    model, tokenizer = _load_model(config, device)
    model = _load_weights(model, args.checkpoint, device)
    pad_token_id = get_pad_token_id(tokenizer)
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1

    sc = SamplingConfig(
        sampling_method="ode", num_sampling_steps=[args.steps], cfgs=[1],
        self_cond_cfg_scales=[1.0], time_schedule=args.time_schedule,
        plan_trajectory=args.trajectory, plan_lead_alpha=args.alpha,
        plan_cfg_scale=args.plan_cfg,
    )
    B = args.num_samples
    d_model = model.text_encoder_dim
    gen = torch.Generator().manual_seed(args.seed)
    t_steps = get_sampling_steps(args.steps, time_schedule=args.time_schedule,
                                 P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                                 device=device, dtype=torch.float32)
    z0 = (torch.randn((B, config.max_length, d_model), generator=gen, dtype=torch.float32)
          * config.denoiser_noise_scale).to(device)

    common = dict(model=model, t_steps=t_steps, cond_seq=None, cond_seq_mask=None,
                  config=config, sampling_config=sc, cfg_scale=1.0, self_cond_cfg_scale=1.0)

    # --- A: normal rollout, recording the plan trajectory the tokens saw ---
    gen_a = torch.Generator().manual_seed(args.seed + 1)
    lat_a, plan_a, traj_a = _generate_samples_single_batch(
        generator=gen_a, z=z0.clone(), record_plan=True, **common)
    ids_a, texts_a = _decode(lat_a, model, config, plan_a, eos_token_id, pad_token_id, tokenizer)

    # --- B: GRAFT — same token noise, plan trajectory rolled by one sample ---
    graft = [p.roll(shifts=1, dims=0) for p in traj_a]
    gen_b = torch.Generator().manual_seed(args.seed + 1)  # same plan-init draw; overridden anyway
    lat_b, plan_b = _generate_samples_single_batch(
        generator=gen_b, z=z0.clone(), plan_override=graft, **common)
    ids_b, texts_b = _decode(lat_b, model, config, plan_b, eos_token_id, pad_token_id, tokenizer)

    # --- C: SHUFFLE-AT-DECODE — normal latent, rolled plan only at the decoder read-out ---
    ids_c, texts_c = _decode(lat_a, model, config, plan_a.roll(shifts=1, dims=0),
                             eos_token_id, pad_token_id, tokenizer)

    report = {
        "trajectory": args.trajectory, "alpha": args.alpha, "steps": args.steps,
        "plan_cfg": args.plan_cfg, "num_samples": B,
        # Fraction of token positions changed by the intervention (higher = plan matters more).
        "graft_token_change_frac": float((ids_b != ids_a).float().mean()),
        "decode_shuffle_token_change_frac": float((ids_c != ids_a).float().mean()),
        # Latent-level effect of the graft on the rollout itself.
        "graft_latent_rel_change": float(
            (lat_b - lat_a).norm() / lat_a.norm().clamp(min=1e-6)),
        # Consistency: matched should be well below mismatched; matched ≈ mismatched is the
        # register-effect alarm (streams not talking to each other).
        "consistency_matched": _consistency(model, lat_a, plan_a),
        "consistency_mismatched": _consistency(model, lat_a, plan_a.roll(shifts=1, dims=0)),
        # Grafted rollout vs. the plan it was forced to follow: low = tokens followed the graft.
        "consistency_graft_followed": _consistency(model, lat_b, plan_b),
    }

    print("\n=== plan probe report ===")
    print(json.dumps(report, indent=2))
    for name, texts in (("A/normal", texts_a), ("B/graft", texts_b), ("C/decode-shuffle", texts_c)):
        print(f"\n--- {name} sample 0 ---\n{texts[0][:400]}")

    out = args.out or os.path.join(config.output_dir, "plan_probe_report.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({**report,
                   "texts_normal": texts_a, "texts_graft": texts_b,
                   "texts_decode_shuffle": texts_c}, f, ensure_ascii=False, indent=2)
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    main()
