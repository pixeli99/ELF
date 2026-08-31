#!/usr/bin/env python3
"""GSM8K test accuracy: the standard axis for this line of work.

1319 held-out grade-school problems whose gold answer is the integer after `####`.
Scoring is exact match on that integer, so this measures solving rather than overlap.
Everything else we report is either in-distribution or a similarity proxy.

The arms differ in what the model emits: the no-reasoning and plan arms emit the answer
alone, the explicit-reasoning arm emits the reasoning followed by `The answer is X`. The
extractor accepts both, preferring the phrase and falling back to the last number.

    python tools/eval_gsm8k.py --config <cfg> --checkpoint <ckpt> --out <jsonl>
"""
import argparse, json, re, sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from configs.config import SamplingConfig, load_config_from_yaml
from utils.encoder_utils import encode_x0
from utils.generation_utils import (_dlm_decode_batch, _generate_samples_single_batch,
                                    mask_after_eos, shift_left)
from utils.sampling_utils import get_sampling_steps
from utils.stage_b_eval_runtime import load_model_and_encoder
from utils.thinking_tokenization import thinking_token_ids

GSM8K = Path("/cpfs01/shared/public/users/pengxiang.li/ouro_eval_outputs/hf_datasets/"
             "openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866")
ANSWER_PHRASE = re.compile(r"answer\s+is\s*:?\s*(.{0,40})", re.I)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
TRAJECTORY = {"ordered": ("planning_first", 2.0), "diagonal": ("diagonal", None),
              "register": ("null", None), "vanilla": (None, None)}


def normalize(text):
    """A gold GSM8K answer is always an integer, so compare numerically."""
    if text is None:
        return None
    cleaned = text.replace(",", "").replace("$", "").strip()
    match = NUMBER.search(cleaned)
    if not match:
        return None
    try:
        value = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return f"{value:.6g}"


def extract_prediction(text):
    phrase = ANSWER_PHRASE.search(text)
    if phrase:
        found = normalize(phrase.group(1))
        if found is not None:
            return found
    numbers = NUMBER.findall(text)
    return normalize(numbers[-1]) if numbers else None


def load_test(limit=None):
    import datasets
    table = datasets.Dataset.from_file(str(GSM8K / "gsm8k-test.arrow"))
    rows = []
    for index, row in enumerate(table):
        if limit and index >= limit:
            break
        gold = row["answer"].rsplit("####", 1)[1].strip()
        rows.append({"eval_id": f"gsm8k-test-{index:05d}", "question": row["question"],
                     "gold": gold, "gold_norm": normalize(gold)})
    return rows


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--num-samples", type=int, default=1319)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--nfe", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", choices=("raw", "ema"), default="raw")
    ap.add_argument("--plan-cfg-scale", type=float, default=1.0)
    ap.add_argument("--plan-trajectory", choices=("planning_first", "diagonal", "null"), default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    config = load_config_from_yaml(args.config)
    device = torch.device("cuda")
    model, encoder, tokenizer, resolved = load_model_and_encoder(config, args.checkpoint, device)
    if args.weights == "raw":
        checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["params"], strict=True)
        model.eval()

    group = getattr(config, "group_mode", "vanilla") if config.num_plan_slots > 0 else "vanilla"
    trajectory, alpha = TRAJECTORY[group]
    if args.plan_trajectory:
        trajectory = args.plan_trajectory
    sampling = SamplingConfig(
        sampling_method="sde", sde_gamma=1.5, time_schedule="logit_normal",
        num_sampling_steps=[args.nfe], cfgs=[1.0], self_cond_cfg_scales=[3.0],
        plan_trajectory=trajectory or "diagonal", plan_lead_alpha=alpha or 1.0,
        plan_cfg_scale=args.plan_cfg_scale)

    rows = load_test(args.num_samples)
    pad = int(tokenizer.pad_token_id)
    eos = int(tokenizer.eos_token_id or 1)
    dtype = next(model.parameters()).dtype
    width = config.max_length
    condition_cap = int(config.condition_max_tokens)
    print(f"group={group} trajectory={trajectory} rows={len(rows)} width={width} "
          f"weights={args.weights} checkpoint={resolved}", flush=True)

    records, correct, answered = [], 0, 0
    for start in range(0, len(rows), args.batch_size):
        chunk = rows[start:start + args.batch_size]
        prompts = [thinking_token_ids(tokenizer, r["question"], add_special_tokens=False)[:condition_cap]
                   for r in chunk]
        ids = np.full((len(chunk), width), pad, dtype=np.int64)
        cond_lengths = np.zeros(len(chunk), dtype=np.int64)
        for index, prompt in enumerate(prompts):
            ids[index, :len(prompt)] = prompt
            cond_lengths[index] = len(prompt)
        positions = np.arange(width)[None, :]
        is_cond = torch.from_numpy((positions < cond_lengths[:, None]).astype(np.float32)).to(device)
        valid = torch.ones((len(chunk), width), dtype=torch.float32, device=device)
        input_ids = torch.from_numpy(ids).to(device)

        with torch.inference_mode():
            x0 = encode_x0(input_ids, valid, encoder, config.latent_mean, config.latent_std,
                           use_bf16=bool(config.use_bf16), cond_mask=is_cond).to(dtype)
        seed = args.seed + start // args.batch_size
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        generator = torch.Generator().manual_seed(seed)
        t_steps = get_sampling_steps(args.nfe, "logit_normal", config.denoiser_p_mean,
                                     config.denoiser_p_std, device=device, dtype=dtype)
        z = (torch.randn(x0.shape, generator=generator, dtype=dtype)
             * config.denoiser_noise_scale).to(device)
        plan_mask = (torch.ones((len(chunk), config.num_plan_slots), dtype=torch.bool, device=device)
                     if config.num_plan_slots > 0 else None)
        with torch.inference_mode():
            latent, final_plan = _generate_samples_single_batch(
                model=model, generator=generator, z=z, t_steps=t_steps,
                cond_seq=x0 * is_cond.unsqueeze(-1), cond_seq_mask=is_cond,
                config=config, sampling_config=sampling, cfg_scale=1.0,
                self_cond_cfg_scale=3.0, plan_mask=plan_mask)
            predicted = _dlm_decode_batch(
                latent, model, t_steps[-1].item(), config, self_cond_cfg_scale=3.0,
                x_plan=final_plan,
                t_plan_decode_val=(0.0 if trajectory == "null" else 1.0) if final_plan is not None else None,
                plan_trajectory=trajectory, plan_mask=plan_mask,
                condition_token_mask=is_cond)
        predicted = shift_left(predicted, is_cond.to(torch.int32).sum(1), pad)
        predicted = mask_after_eos(predicted, eos, pad)
        for index, row in enumerate(chunk):
            text = tokenizer.decode(predicted[index].cpu().tolist(), skip_special_tokens=True)
            guess = extract_prediction(text)
            hit = guess is not None and guess == row["gold_norm"]
            correct += int(hit)
            answered += int(guess is not None)
            records.append({**{k: row[k] for k in ("eval_id", "gold")},
                            "generated": text, "predicted": guess, "correct": bool(hit)})
        print(f"  {len(records)}/{len(rows)}  accuracy so far {correct / len(records):.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = {"config": args.config, "checkpoint": str(resolved), "group": group,
               "trajectory": trajectory, "plan_cfg_scale": args.plan_cfg_scale,
               "weights": args.weights, "nfe": args.nfe, "rows": len(records),
               "accuracy": round(correct / max(len(records), 1), 4),
               "answer_rate": round(answered / max(len(records), 1), 4),
               "correct": correct,
               "mean_generated_chars": round(float(np.mean([len(r["generated"]) for r in records])), 1)}
    json.dump(summary, open(str(out) + ".summary.json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
