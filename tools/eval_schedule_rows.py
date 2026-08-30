#!/usr/bin/env python3
"""Free generation on unseen rows of the fixed training schedule, for any conditional config.

Unlike tools/eval_conditional_dev.py (validation / test prompts only), the rows here carry
the gold thinking and response, so configs whose condition prefix includes the thinking
(the ceiling experiment) can be evaluated, and the reference is right there for ROUGE and
the verifiable-subset accuracy. Rows are taken from schedule positions >= --start-index,
which must be past the checkpoint's training range (80000 for the 10k-step protocol).
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]
from configs.config import SamplingConfig, load_config_from_yaml
from utils.conditional_data import ConditionalPairedDataset, ConditionalSchedule, get_conditional_dataloader
from utils.encoder_utils import encode_x0
from utils.generation_utils import _dlm_decode_batch, _generate_samples_single_batch, mask_after_eos, shift_left
from utils.metrics_utils import compute_bleu, compute_rouge
from utils.sampling_utils import get_sampling_steps
from utils.stage_b_eval_runtime import load_model_and_encoder
from eval_boxed_accuracy import score as boxed_score

TRAJ = {"ordered": ("planning_first", 2.0), "diagonal": ("diagonal", None), "register": ("null", None), "vanilla": (None, None)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--checkpoint", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--start-index", type=int, default=80000); ap.add_argument("--num-samples", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8); ap.add_argument("--nfe", type=int, default=32); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--free-response-tokens", type=int, default=512); ap.add_argument("--plan-cfg-scale", type=float, default=1.0)
    ap.add_argument("--plan-trajectory", choices=("planning_first", "diagonal", "null"), default=None)
    ap.add_argument("--weights", choices=("raw", "ema"), default="raw")
    args = ap.parse_args()
    config = load_config_from_yaml(args.config); device = torch.device("cuda")
    model, encoder, tokenizer, resolved = load_model_and_encoder(config, args.checkpoint, device)
    if args.weights == "raw":
        ck = torch.load(resolved, map_location="cpu", weights_only=False); model.load_state_dict(ck["params"], strict=True); model.eval()
    step = int(torch.load(resolved, map_location="cpu", weights_only=False).get("step", 0))
    if args.start_index < step:
        raise ValueError(f"--start-index {args.start_index} overlaps rows the checkpoint saw (global step {step})")
    base = ConditionalPairedDataset(config.conditional_train_manifest, config.conditional_train_manifest_sha256, False)
    full = ConditionalSchedule(base, args.start_index + args.num_samples, config.conditional_master_seed)
    subset = torch.utils.data.Subset(full, list(range(args.start_index, args.start_index + args.num_samples)))
    loader, _ = get_conditional_dataloader(subset, tokenizer, config, batch_size=args.batch_size)
    refmap = {}
    for i in range(args.start_index, args.start_index + args.num_samples):
        row = full[i]; refmap[row["example_id"]] = row["response"]
    group = getattr(config, "group_mode", "vanilla") if config.num_plan_slots > 0 else "vanilla"
    traj, alpha = TRAJ[group]
    if args.plan_trajectory: traj = args.plan_trajectory
    sampling = SamplingConfig(sampling_method="sde", sde_gamma=1.5, time_schedule="logit_normal", num_sampling_steps=[args.nfe],
                              cfgs=[1.0], self_cond_cfg_scales=[3.0], plan_trajectory=traj or "diagonal",
                              plan_lead_alpha=alpha or 1.0, plan_cfg_scale=args.plan_cfg_scale)
    pad = int(tokenizer.pad_token_id); eos = int(tokenizer.eos_token_id or 1)
    dtype = next(model.parameters()).dtype
    rows, hyps, refs = [], [], []
    for bi, batch in enumerate(loader):
        ids = batch["input_ids"].to(device).long(); att = batch["attention_mask"].to(device).float(); cond = batch["cond_seq_mask"].to(device).float()
        with torch.inference_mode():
            x0 = encode_x0(ids, att, encoder, config.latent_mean, config.latent_std, use_bf16=bool(config.use_bf16), cond_mask=cond).to(dtype)
        bs = x0.shape[0]; seed = args.seed + bi
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        gen = torch.Generator().manual_seed(seed)
        t_steps = get_sampling_steps(args.nfe, "logit_normal", config.denoiser_p_mean, config.denoiser_p_std, device=device, dtype=dtype)
        z = (torch.randn(x0.shape, generator=gen, dtype=dtype) * config.denoiser_noise_scale).to(device)
        cond_seq = x0 * cond.unsqueeze(-1)
        plan_mask = torch.ones((bs, config.num_plan_slots), dtype=torch.bool, device=device) if config.num_plan_slots > 0 else None
        with torch.inference_mode():
            latent, final_plan = _generate_samples_single_batch(model=model, generator=gen, z=z, t_steps=t_steps, cond_seq=cond_seq, cond_seq_mask=cond,
                                                                config=config, sampling_config=sampling, cfg_scale=1.0, self_cond_cfg_scale=3.0, plan_mask=plan_mask)
            pred = _dlm_decode_batch(latent, model, t_steps[-1].item(), config, self_cond_cfg_scale=3.0, x_plan=final_plan,
                                     t_plan_decode_val=(0.0 if traj == "null" else 1.0) if final_plan is not None else None,
                                     plan_trajectory=traj, plan_mask=plan_mask, condition_token_mask=cond)
        clen = cond.to(torch.int32).sum(1)
        pred = shift_left(pred, clen, pad)[:, :args.free_response_tokens]
        eos_hit = (pred == eos).any(1)
        pred = mask_after_eos(pred, eos, pad)
        for i in range(bs):
            text = tokenizer.decode(pred[i].cpu().tolist(), skip_special_tokens=True)
            ref = refmap[batch["example_id"][i]]
            rows.append({"eval_id": batch["example_id"][i], "generated": text, "reference": ref, "eos_emitted": bool(eos_hit[i]),
                         "cond_tokens": int(clen[i]),
                         "generated_budget_slots": (int((final_plan[i].float().norm(dim=-1) > 2.0).sum()) if final_plan is not None else None)})
            hyps.append(text); refs.append(ref or "")
        print(f"  {len(rows)}/{args.num_samples}", flush=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as h:
        for r in rows: h.write(json.dumps(r, ensure_ascii=False) + "\n")
    summ = {"config": args.config, "checkpoint": str(resolved), "weights": args.weights, "rows": len(rows), "start_index": args.start_index,
            "nfe": args.nfe, "cap": args.free_response_tokens, "plan_trajectory": traj, "plan_cfg_scale": args.plan_cfg_scale,
            "bleu": compute_bleu(hyps, refs), **compute_rouge(hyps, refs),
            "eos_rate": float(np.mean([r["eos_emitted"] for r in rows])), "mean_generated_chars": float(np.mean([len(t) for t in hyps])),
            "mean_reference_chars": float(np.mean([len(r) for r in refs])),
            "verifiable": boxed_score(rows, {r["eval_id"]: r["reference"] for r in rows})}
    budgets = [r["generated_budget_slots"] for r in rows if r["generated_budget_slots"] is not None]
    if budgets: summ["budget_mean"] = float(np.mean(budgets)); summ["budget_std"] = float(np.std(budgets))
    json.dump(summ, open(str(out) + ".summary.json", "w"), indent=2)
    print(json.dumps(summ, indent=2))

if __name__ == "__main__":
    main()
