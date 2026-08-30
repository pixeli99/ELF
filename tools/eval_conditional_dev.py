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
denoised slots whose L2 norm sits above the null threshold.  The fixed-width
pilot uses all configured plan slots; the budget remains a model-output probe.

``--length-mode oracle`` is a diagnostic fixed-budget protocol.  It uses only
the tokenized reference *length* to build the response validity mask; reference
content never enters the model.  Results from this mode must be labelled
length-oracle and must not be presented as free-length generation.
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


def budget_probe_is_applicable(trajectory, alpha):
    """Whether final slot norms represent a fully denoised plan endpoint."""
    return (
        trajectory not in (None, "null")
        and not (trajectory == "planning_first" and alpha < 1.0)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=list(TRAJECTORY))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--num_samples", type=int, default=2000)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split", choices=("validation", "test"), default="validation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="Resolved config YAML; defaults to the group overlay")
    ap.add_argument("--length-mode", choices=("full", "oracle"), default="full")
    ap.add_argument(
        "--free-response-tokens", type=int,
        help="Shared free-generation decode cap; defaults to the training-supported maximum",
    )
    ap.add_argument("--weights", choices=("ema", "raw"), default="ema")
    ap.add_argument(
        "--plan-trajectory",
        choices=("diagonal", "planning_first", "endpoint_correct_lagging", "null"),
        help="Override the group-default plan trajectory for validation sweeps.",
    )
    ap.add_argument(
        "--plan-lead-alpha", type=float,
        help="Planning-first clock multiplier; valid only for planning_first.",
    )
    ap.add_argument(
        "--plan-cfg-scale", type=float, default=1.0,
        help="Grid-CFG scale against the fixed initial-noise plan.",
    )
    args = ap.parse_args()

    device = "cuda"
    config_path = (Path(args.config) if args.config else
                   ROOT / f"src/configs/training_configs/train_conditional_common48w_{args.group}_10k_v1.yml")
    config = load_config_from_yaml(str(config_path))
    model, encoder, tokenizer, resolved = load_model_and_encoder(
        config, args.checkpoint, device, load_encoder=True)
    # T5 uses relative positions and the dataset contains valid sequences above
    # its legacy tokenizer warning threshold of 512; training applies the same
    # override before tokenization.
    tokenizer.model_max_length = sys.maxsize
    if args.weights == "raw":
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["params"], strict=True)
        model.eval()
    dtype = next(model.parameters()).dtype
    print(f"group={args.group} checkpoint={resolved} weights={args.weights}", flush=True)

    data_dir = ROOT / "data/conditional_v1/data"
    dev = pq.read_table(data_dir / f"{args.split}-00000.parquet").to_pylist()
    refs = {r["eval_id"]: r["reference_response"] for r in pq.read_table(
        data_dir / f"{args.split}_references-00000.parquet").to_pylist()}
    dev = dev[:args.num_samples]

    trajectory, alpha = TRAJECTORY[args.group]
    if args.plan_trajectory is not None:
        trajectory = args.plan_trajectory
        if trajectory != "planning_first":
            alpha = None
    if args.plan_lead_alpha is not None:
        if trajectory != "planning_first":
            ap.error("--plan-lead-alpha requires --plan-trajectory planning_first")
        alpha = args.plan_lead_alpha
    if trajectory == "planning_first":
        alpha = 2.0 if alpha is None else float(alpha)
        if alpha <= 0:
            ap.error("--plan-lead-alpha must be positive")
    if args.plan_cfg_scale < 0:
        ap.error("--plan-cfg-scale must be non-negative")
    if config.num_plan_slots == 0 and (
        args.plan_trajectory is not None
        or args.plan_lead_alpha is not None
        or args.plan_cfg_scale != 1.0
    ):
        ap.error("plan sampling overrides require a plan-enabled config")
    budget_probe_applicable = budget_probe_is_applicable(trajectory, alpha)
    sampling = SamplingConfig(
        sampling_method="sde", sde_gamma=1.5, time_schedule="logit_normal",
        num_sampling_steps=[args.nfe], cfgs=[1.0], self_cond_cfg_scales=[3.0],
        plan_trajectory=trajectory, plan_lead_alpha=alpha,
        plan_cfg_scale=args.plan_cfg_scale)
    width, d_model = config.max_length, model.text_encoder_dim
    # The conditional package reserves at least ``condition_max_tokens`` for
    # the prompt, so no training response can exceed the remaining width.  The
    # production generator applies the same cap after shifting off the prompt.
    # Decoding the entire 2048-position canvas would score unsupervised tail
    # positions as if they were part of the answer.
    training_response_width = width - config.condition_max_tokens
    free_response_width = (training_response_width if args.free_response_tokens is None
                           else int(args.free_response_tokens))
    if not 1 <= free_response_width <= training_response_width:
        raise ValueError(
            "free_response_tokens must be between 1 and the training-supported "
            f"maximum {training_response_width}"
        )
    eos, pad = tokenizer.eos_token_id, 0

    rows, hyps, ref_list = [], [], []
    t0 = time.time()
    for start in range(0, len(dev), args.batch_size):
        chunk = dev[start:start + args.batch_size]
        bsz = len(chunk)
        prompts = [thinking_token_ids(tokenizer, r["prompt"], add_special_tokens=False,
                                      truncation=False)[:config.condition_max_tokens]
                   for r in chunk]
        response_lengths = [
            min(len(thinking_token_ids(tokenizer, refs[r["eval_id"]],
                                       add_special_tokens=True, truncation=False)),
                width - len(prompt))
            for r, prompt in zip(chunk, prompts)
        ]
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

        # Re-seed every batch so all groups share token noise, time points and
        # SDE churn.  Plan noise comes from a separate CPU generator and cannot
        # perturb the response RNG stream.
        batch_seed = args.seed + start // args.batch_size
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        token_generator = torch.Generator().manual_seed(batch_seed)
        plan_generator = torch.Generator().manual_seed(batch_seed + 1_000_003)
        t_steps = get_sampling_steps(n_steps=args.nfe, time_schedule="logit_normal",
                                     P_mean=config.denoiser_p_mean,
                                     P_std=config.denoiser_p_std,
                                     device=device, dtype=dtype)
        z = (torch.randn((bsz, width, d_model), generator=token_generator, dtype=dtype)
             * config.denoiser_noise_scale).to(device)
        valid_mask = None
        if args.length_mode == "oracle":
            valid_mask = torch.zeros((bsz, width), dtype=torch.bool, device=device)
            for i, (prompt, response_length) in enumerate(zip(prompts, response_lengths)):
                valid_mask[i, :len(prompt) + response_length] = True
        plan_mask = initial_plan_noise = None
        if config.num_plan_slots > 0:
            plan_mask = torch.ones((bsz, config.num_plan_slots), dtype=torch.bool, device=device)
            initial_plan_noise = (
                torch.randn((bsz, config.num_plan_slots, model.plan_latent_dim),
                            generator=plan_generator, dtype=dtype)
                * config.denoiser_noise_scale
            ).to(device)
        # Reset after constructing the shared grid so the SDE response churn is
        # paired even though plan-enabled groups carry an additional stream.
        torch.manual_seed(batch_seed + 2_000_003)
        torch.cuda.manual_seed_all(batch_seed + 2_000_003)
        with torch.no_grad():
            latent, latent_plan = _generate_samples_single_batch(
                model=model, generator=token_generator, z=z, t_steps=t_steps,
                cond_seq=cond_seq, cond_seq_mask=cond_mask, config=config,
                sampling_config=sampling, cfg_scale=1.0, self_cond_cfg_scale=3.0,
                plan_mask=plan_mask, initial_plan_noise=initial_plan_noise,
                response_attention_mask=valid_mask)
            predicted = _dlm_decode_batch(
                z=latent, model=model, t_final_val=t_steps[-1].item(), config=config,
                self_cond_cfg_scale=3.0, x_plan=latent_plan,
                plan_trajectory=trajectory,
                t_plan_decode_val=(0.0 if trajectory == "null" else 1.0),
                plan_mask=plan_mask, attention_mask=valid_mask,
                condition_token_mask=cond_mask)
        cond_len = cond_mask.to(torch.int32).sum(dim=1)
        predicted = shift_left(predicted, cond_len, pad)
        if args.length_mode == "full":
            predicted = predicted[:, :free_response_width]
        raw_predicted = predicted.clone()
        predicted = mask_after_eos(predicted, eos_token_id=eos, pad_token_id=pad)
        if args.length_mode == "oracle":
            response_mask = torch.arange(width, device=device).unsqueeze(0) < torch.tensor(
                response_lengths, device=device).unsqueeze(1)
            predicted = torch.where(response_mask, predicted,
                                    torch.full_like(predicted, pad))

        # A generated-budget probe is meaningful only when the plan stream is
        # actually denoised.  Register keeps its slots as pure t_plan=0 noise;
        # thresholding those norms would mechanically report 64 "active" slots
        # even though no plan was generated.
        plan_norms = (
            latent_plan.float().norm(dim=-1)
            if latent_plan is not None and budget_probe_applicable else None
        )
        budgets = ((plan_norms > NULL_SLOT_NORM).sum(-1).tolist()
                   if plan_norms is not None else [None] * bsz)
        for i, row in enumerate(chunk):
            valid_ids = predicted[i][(predicted[i] != pad) & (predicted[i] != eos)]
            if valid_ids.numel():
                _, counts = valid_ids.unique(return_counts=True)
                unique_token_ratio = float(counts.numel() / valid_ids.numel())
                dominant_token_fraction = float(counts.max() / valid_ids.numel())
            else:
                unique_token_ratio = dominant_token_fraction = 0.0
            text = tokenizer.decode(predicted[i].detach().cpu().tolist(),
                                    skip_special_tokens=True)
            rows.append({"eval_id": row["eval_id"], "generated": text,
                         "generated_budget_slots": budgets[i],
                         "generated_plan_slot_norm_mean": (
                             float(plan_norms[i].mean()) if plan_norms is not None else None
                         ),
                         "prompt_tokens": int(cond_len[i]),
                         "response_token_budget": response_lengths[i]
                         if args.length_mode == "oracle" else free_response_width,
                         "eos_emitted": bool((raw_predicted[
                             i, :response_lengths[i] if args.length_mode == "oracle"
                             else free_response_width
                         ] == eos).any()),
                         "unique_token_ratio": unique_token_ratio,
                         "dominant_token_fraction": dominant_token_fraction})
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
    response_budgets = [r["response_token_budget"] for r in rows]
    summary = {
        "group": args.group, "checkpoint": str(resolved), "nfe": args.nfe,
        "config": str(config_path), "weights": args.weights,
        "plan_trajectory": trajectory,
        "plan_lead_alpha": alpha if trajectory == "planning_first" else None,
        "plan_cfg_scale": args.plan_cfg_scale if config.num_plan_slots > 0 else None,
        "samples": len(rows), "seed": args.seed, "split": args.split,
        "length_mode": args.length_mode,
        "free_response_token_cap": free_response_width,
        "training_response_token_cap": training_response_width,
        "gold_content_used": False,
        "gold_derived_response_length_used": args.length_mode == "oracle",
        "generated_budget_probe_applicable": budget_probe_applicable,
        "bleu": round(compute_bleu(hyps, ref_list), 3),
        **{k: round(v, 3) for k, v in compute_rouge(hyps, ref_list).items()},
        "empty_generation_rate": round(sum(not h.strip() for h in hyps) / len(hyps), 4),
        "mean_generated_chars": round(float(np.mean([len(h) for h in hyps])), 1),
        "eos_rate": round(float(np.mean([r["eos_emitted"] for r in rows])), 4),
        "mean_unique_token_ratio": round(float(np.mean(
            [r["unique_token_ratio"] for r in rows])), 4),
        "mean_dominant_token_fraction": round(float(np.mean(
            [r["dominant_token_fraction"] for r in rows])), 4),
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
        if args.length_mode == "oracle":
            summary["corr_budget_vs_reference_response_len"] = (
                round(float(np.corrcoef(budgets, response_budgets)[0, 1]), 3)
                if np.std(budgets) > 0 else 0.0
            )
        slot_norms = [r["generated_plan_slot_norm_mean"] for r in rows]
        summary["generated_plan_slot_norm_mean"] = round(float(np.mean(slot_norms)), 4)
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
