#!/usr/bin/env python3
"""Conditional dev evaluation: prompt -> generated response -> BLEU/ROUGE.

One process, one checkpoint, one group. The prompt is a clean prefix (exactly
the training geometry); the response region and, for plan groups, all 64 plan
slots start from noise. Plan trajectory follows the group's own protocol:

    ordered   planning_first, alpha 2.0 (plan resolves ahead of the tokens)
    diagonal  t_plan == t
    register  null (slots stay noise at t_plan = 0, decode at 0)
    vanilla   no plan stream

For plan groups the GENERATED BUDGET is recorded per sample: the number of
denoised slots whose L2 norm sits above the null threshold -- plan length is
model output here, there is no mask input anywhere.
"""
import argparse, json, math, sys, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from configs.config import SamplingConfig, load_config_from_yaml
from utils.generation_utils import (_dlm_decode_batch, _generate_samples_single_batch,
                                    mask_after_eos, shift_left)
from utils.metrics_utils import compute_bleu, compute_rouge
from utils.sampling_utils import get_sampling_steps
from utils.stage_b_eval_runtime import load_model_and_encoder
from utils.thinking_tokenization import thinking_token_ids

TRAJECTORY = {"ordered": ("planning_first", 2.0), "diagonal": ("diagonal", None),
              "register": ("null", None), "vanilla": (None, None)}
NULL_SLOT_NORM = 2.0   # trained active slots sit at ||mu|| ~ 8.5; null is exactly 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(TRAJECTORY))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda"
    config = load_config_from_yaml(
        str(ROOT / f"src/configs/training_configs/train_conditional_common48w_{args.group}_10k_v1.yml"))
    model, encoder, tokenizer, resolved = load_model_and_encoder(
        config, args.checkpoint, device, load_encoder=True)
    dtype = next(model.parameters()).dtype
    print(f"group={args.group} checkpoint={resolved}", flush=True)

    dev = pq.read_table(ROOT / "data/conditional_v1/data/validation-00000.parquet").to_pylist()
    refs = {r["eval_id"]: r["reference_response"] for r in pq.read_table(
        ROOT / "data/conditional_v1/data/validation_references-00000.parquet").to_pylist()}
    dev = dev[:args.num_samples]

    trajectory, alpha = TRAJECTORY[args.group]
    sampling = SamplingConfig(
        sampling_method="sde", sde_gamma=1.5, time_schedule="logit_normal",
        num_sampling_steps=[args.nfe], cfgs=[1.0], self_cond_cfg_scales=[3.0],
        plan_trajectory=trajectory, plan_lead_alpha=alpha, plan_cfg_scale=1.0)
    generator = torch.Generator().manual_seed(args.seed)
    width, d_model = config.max_length, model.text_encoder_dim
    eos, pad = tokenizer.eos_token_id, 0

    rows, hyps, ref_list = [], [], []
    t0 = time.time()
    for start in range(0, len(dev), args.batch_size):
        chunk = dev[start:start + args.batch_size]
        bsz = len(chunk)
        prompts = [thinking_token_ids(tokenizer, r["prompt"], add_special_tokens=False,
                                      truncation=False)[:config.condition_max_tokens]
                   for r in chunk]
        input_ids = torch.full((bsz, width), pad, dtype=torch.long)
        cond_mask = torch.zeros((bsz, width), dtype=torch.float32)
        for i, p in enumerate(prompts):
            input_ids[i, :len(p)] = torch.as_tensor(p)
            cond_mask[i, :len(p)] = 1.0
        input_ids, cond_mask = input_ids.to(device), cond_mask.to(device)

        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=bool(config.use_bf16)):
            prompt_latent = encoder(input_ids=input_ids,
                                    attention_mask=cond_mask.bool(),
                                    deterministic=True).float()
        cond_seq = ((prompt_latent - config.latent_mean) / config.latent_std
                    ).to(dtype) * cond_mask.unsqueeze(-1).to(dtype)

        t_steps = get_sampling_steps(n_steps=args.nfe, time_schedule="logit_normal",
                                     P_mean=config.denoiser_p_mean,
                                     P_std=config.denoiser_p_std,
                                     device=device, dtype=dtype)
        z = (torch.randn((bsz, width, d_model), generator=generator, dtype=dtype)
             * config.denoiser_noise_scale).to(device)
        with torch.no_grad():
            latent, latent_plan = _generate_samples_single_batch(
                model=model, generator=generator, z=z, t_steps=t_steps,
                cond_seq=cond_seq, cond_seq_mask=cond_mask, config=config,
                sampling_config=sampling, cfg_scale=1.0, self_cond_cfg_scale=3.0)
            predicted = _dlm_decode_batch(
                z=latent, model=model, t_final_val=t_steps[-1].item(), config=config,
                self_cond_cfg_scale=3.0, x_plan=latent_plan,
                plan_trajectory=trajectory,
                t_plan_decode_val=(0.0 if trajectory == "null" else 1.0))
        cond_len = cond_mask.to(torch.int32).sum(dim=1)
        predicted = shift_left(predicted, cond_len, pad)
        predicted = mask_after_eos(predicted, eos_token_id=eos, pad_token_id=pad)

        budgets = (latent_plan.float().norm(dim=-1) > NULL_SLOT_NORM
                   ).sum(-1).tolist() if latent_plan is not None else [None] * bsz
        for i, row in enumerate(chunk):
            text = tokenizer.decode(predicted[i].detach().cpu().tolist(),
                                    skip_special_tokens=True)
            rows.append({"eval_id": row["eval_id"], "generated": text,
                         "generated_budget_slots": budgets[i],
                         "prompt_tokens": int(cond_len[i])})
            hyps.append(text)
            ref_list.append(refs[row["eval_id"]])
        if (start // args.batch_size) % 25 == 0:
            done = start + bsz
            rate = done / (time.time() - t0)
            print(f"  {done}/{len(dev)}  {rate:.1f} samples/s  "
                  f"eta {(len(dev)-done)/max(rate,1e-9)/60:.0f} min", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    budgets = [r["generated_budget_slots"] for r in rows if r["generated_budget_slots"] is not None]
    plens = [r["prompt_tokens"] for r in rows]
    summary = {
        "group": args.group, "checkpoint": str(resolved), "nfe": args.nfe,
        "samples": len(rows), "seed": args.seed,
        "bleu": round(compute_bleu(hyps, ref_list), 3),
        **{k: round(v, 3) for k, v in compute_rouge(hyps, ref_list).items()},
        "empty_generation_rate": round(sum(not h.strip() for h in hyps) / len(hyps), 4),
        "mean_generated_chars": round(float(np.mean([len(h) for h in hyps])), 1),
    }
    if budgets:
        summary.update({
            "budget_mean": round(float(np.mean(budgets)), 2),
            "budget_std": round(float(np.std(budgets)), 2),
            "budget_p10": float(np.percentile(budgets, 10)),
            "budget_p90": float(np.percentile(budgets, 90)),
            "corr_budget_vs_prompt_len": round(float(np.corrcoef(budgets, plens)[0, 1]), 3)
            if np.std(budgets) > 0 else 0.0,
        })
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
