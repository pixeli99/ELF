#!/usr/bin/env python3
"""Compare generated plans with a frozen gold-plan upper bound on unseen rows.

This is a diagnostic only: oracle_matched uses the row's gold thinking content.
It answers whether response quality is currently limited by plan generation or
whether even an exact clean plan fails to help the response stream.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from rouge_score import rouge_scorer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from configs.config import SamplingConfig, load_config_from_yaml
from modules.plan_vae import load_frozen_plan_vae
from tools.compare_conditional_evals import paired_bootstrap
from utils.conditional_data import (
    ConditionalPairedDataset,
    ConditionalSchedule,
    get_conditional_dataloader,
)
from utils.encoder_utils import encode_x0
from utils.generation_utils import (
    _dlm_decode_batch,
    _generate_samples_single_batch,
    mask_after_eos,
    shift_left,
)
from utils.metrics_utils import compute_bleu, compute_rouge
from utils.plan_stream import build_vae_plan_target
from utils.sampling_utils import get_sampling_steps
from utils.stage_b_eval_runtime import load_model_and_encoder
from utils.thinking_tokenization import thinking_token_ids


MODES = ("generated", "oracle_matched", "oracle_shuffled")


def resolve_schedule_window(
    dataset_rows, configured_schedule_rows, checkpoint_step,
    global_batch_size, start_index, candidate_count,
):
    """Validate that an audit window starts after every row seen by the checkpoint.

    ``ConditionalSchedule`` draws one deterministic permutation of the full
    paired pool and then slices it.  Extending its row count therefore preserves
    the exact 80k training prefix while exposing a deterministic unseen tail.
    """
    values = {
        "dataset_rows": dataset_rows,
        "configured_schedule_rows": configured_schedule_rows,
        "checkpoint_step": checkpoint_step,
        "global_batch_size": global_batch_size,
        "start_index": start_index,
        "candidate_count": candidate_count,
    }
    if any(int(value) != value for value in values.values()):
        raise ValueError("schedule-window inputs must be integers")
    values = {key: int(value) for key, value in values.items()}
    if values["dataset_rows"] <= 0 or values["configured_schedule_rows"] <= 0:
        raise ValueError("dataset and configured schedule must be non-empty")
    if values["configured_schedule_rows"] > values["dataset_rows"]:
        raise ValueError("configured schedule exceeds the paired dataset")
    if values["checkpoint_step"] < 0 or values["global_batch_size"] <= 0:
        raise ValueError("checkpoint step must be non-negative and batch size positive")
    if values["start_index"] < 0 or values["candidate_count"] <= 0:
        raise ValueError("audit start must be non-negative and candidate count positive")

    seen_rows = values["checkpoint_step"] * values["global_batch_size"]
    if seen_rows > values["configured_schedule_rows"]:
        raise ValueError("checkpoint claims more rows than the configured training schedule")
    candidate_stop = values["start_index"] + values["candidate_count"]
    if candidate_stop > values["dataset_rows"]:
        raise ValueError("requested range is outside the paired dataset")
    if values["start_index"] < seen_rows:
        raise ValueError(
            "requested audit range overlaps rows already seen by the checkpoint"
        )
    return {
        "checkpoint_seen_presentation_rows": seen_rows,
        "candidate_stop": candidate_stop,
        "audit_schedule_rows": max(
            values["configured_schedule_rows"], candidate_stop,
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-index", type=int, default=40000)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--nfe", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plan-cfg-scale", type=float, default=1.0)
    parser.add_argument("--free-response-tokens", type=int, default=512)
    parser.add_argument("--exact-k-shuffle", action="store_true")
    parser.add_argument("--candidate-samples", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def dominant_fraction(ids, pad, eos):
    valid = ids[(ids != pad) & (ids != eos)]
    if not valid.numel():
        return 0.0
    _, counts = valid.unique(return_counts=True)
    return float(counts.max().item() / valid.numel())


def main():
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0 or args.nfe <= 0:
        raise ValueError("sample, batch, and NFE counts must be positive")
    if args.plan_cfg_scale < 0:
        raise ValueError("plan CFG scale must be non-negative")
    if args.exact_k_shuffle and (
        args.batch_size % 2 or args.num_samples % args.batch_size
    ):
        raise ValueError("exact-K shuffle requires even, complete batches")
    device = torch.device(args.device)
    config = load_config_from_yaml(args.config)
    if not 1 <= args.free_response_tokens <= (
        config.max_length - config.condition_max_tokens
    ):
        raise ValueError("free response cap is outside the training-supported range")

    model, encoder, tokenizer, resolved = load_model_and_encoder(
        config, args.checkpoint, device,
    )
    tokenizer.model_max_length = sys.maxsize
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["params"], strict=True)
    checkpoint_step = int(checkpoint["step"])
    del checkpoint
    model.eval()
    plan_encoder, _ = load_frozen_plan_vae(
        config.plan_vae_artifact, device=device,
        expected_sha256=config.plan_vae_artifact_sha256,
    )
    base = ConditionalPairedDataset(
        config.conditional_train_manifest,
        expected_sha256=config.conditional_train_manifest_sha256,
        verify_shards=False,
    )
    candidate_count = args.candidate_samples if args.exact_k_shuffle else args.num_samples
    window = resolve_schedule_window(
        dataset_rows=len(base),
        configured_schedule_rows=int(config.conditional_rows),
        checkpoint_step=checkpoint_step,
        global_batch_size=int(config.global_batch_size),
        start_index=args.start_index,
        candidate_count=candidate_count,
    )
    candidate_stop = window["candidate_stop"]
    schedule = ConditionalSchedule(
        base, rows=window["audit_schedule_rows"],
        master_seed=int(config.conditional_master_seed),
    )
    candidates = [schedule[index] for index in range(args.start_index, candidate_stop)]
    if args.exact_k_shuffle:
        by_k = defaultdict(list)
        for row in candidates:
            thinking_length = len(thinking_token_ids(
                tokenizer, row["thinking"], add_special_tokens=True, truncation=False,
            ))
            active_k = min(int(config.num_plan_slots), (thinking_length + 15) // 16)
            row["diagnostic_active_k"] = active_k
            by_k[active_k].append(row)
        pairs = []
        for values in by_k.values():
            pairs.extend((values[index], values[index + 1])
                         for index in range(0, len(values) - 1, 2))
        pairs.sort(key=lambda pair: min(
            pair[0]["presentation_index"], pair[1]["presentation_index"]
        ))
        if len(pairs) * 2 < args.num_samples:
            raise ValueError("candidate pool has too few exact-K donor pairs")
        selected = [row for pair in pairs[:args.num_samples // 2] for row in pair]
    else:
        selected = candidates
    loader, _ = get_conditional_dataloader(
        selected, tokenizer, config, batch_size=args.batch_size,
        num_workers=0, distributed=False,
    )

    parameter_dtype = next(model.parameters()).dtype
    pad = int(tokenizer.pad_token_id or 0)
    eos = int(tokenizer.eos_token_id or 1)
    sampling = SamplingConfig(
        sampling_method="sde", sde_gamma=1.5, time_schedule="logit_normal",
        num_sampling_steps=[args.nfe], cfgs=[1.0], self_cond_cfg_scales=[3.0],
        plan_trajectory="planning_first", plan_lead_alpha=2.0,
        plan_cfg_scale=args.plan_cfg_scale,
    )
    rows = {mode: [] for mode in MODES}
    hypotheses = {mode: [] for mode in MODES}
    references = []

    for batch_index, batch in enumerate(loader):
        batch_start = batch_index * args.batch_size
        raw_rows = selected[batch_start:batch_start + len(batch["example_id"])]
        batch_size = len(raw_rows)
        ids = batch["input_ids"].to(device).long()
        attention = batch["attention_mask"].to(device).float()
        condition = batch["cond_seq_mask"].to(device).float()
        with torch.inference_mode():
            x0 = encode_x0(
                ids, attention, encoder,
                config.latent_mean, config.latent_std,
                use_bf16=bool(config.use_bf16), cond_mask=condition,
            ).to(parameter_dtype)
            plan_target, plan_mask, _ = build_vae_plan_target(
                batch["plan_input_ids"].to(device).long(),
                batch["plan_attention_mask"].to(device).bool(),
                encoder, plan_encoder, config,
            )
            plan_target = plan_target.to(parameter_dtype)

        batch_seed = args.seed + batch_index
        torch.manual_seed(batch_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(batch_seed)
        token_generator = torch.Generator().manual_seed(batch_seed)
        plan_generator = torch.Generator().manual_seed(batch_seed + 1_000_003)
        t_steps = get_sampling_steps(
            args.nfe, "logit_normal", config.denoiser_p_mean,
            config.denoiser_p_std, device=device, dtype=parameter_dtype,
        )
        response_noise = (
            torch.randn(x0.shape, generator=token_generator, dtype=parameter_dtype)
            * config.denoiser_noise_scale
        ).to(device)
        initial_plan_noise = (
            torch.randn(plan_target.shape, generator=plan_generator,
                        dtype=parameter_dtype)
            * config.denoiser_noise_scale
        ).to(device)
        cond_seq = x0 * condition.unsqueeze(-1)

        for mode in MODES:
            torch.manual_seed(batch_seed + 2_000_003)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(batch_seed + 2_000_003)
            selected_plan = (
                plan_target if mode == "oracle_matched"
                else plan_target[torch.arange(batch_size, device=device) ^ 1]
                if mode == "oracle_shuffled" and args.exact_k_shuffle
                else plan_target.roll(shifts=1, dims=0)
                if mode == "oracle_shuffled" else None
            )
            if mode == "oracle_shuffled" and args.exact_k_shuffle:
                paired_k = [int(row["diagnostic_active_k"]) for row in raw_rows]
                if any(paired_k[index] != paired_k[index ^ 1]
                       for index in range(batch_size)):
                    raise AssertionError("exact-K donor pairing drifted inside a batch")
            override = ([selected_plan] * len(t_steps)
                        if selected_plan is not None else None)
            with torch.inference_mode():
                latent, final_plan = _generate_samples_single_batch(
                    model=model, generator=token_generator,
                    z=response_noise.clone(), t_steps=t_steps,
                    cond_seq=cond_seq, cond_seq_mask=condition,
                    config=config, sampling_config=sampling,
                    cfg_scale=1.0, self_cond_cfg_scale=3.0,
                    plan_override=override,
                    plan_override_t=1.0 if override is not None else None,
                    freeze_plan_override=override is not None,
                    plan_mask=plan_mask,
                    initial_plan_noise=(selected_plan if override is not None
                                        else initial_plan_noise),
                )
                predicted = _dlm_decode_batch(
                    latent, model, t_steps[-1].item(), config,
                    self_cond_cfg_scale=3.0, x_plan=final_plan,
                    t_plan_decode_val=1.0,
                    plan_trajectory="planning_first", plan_mask=plan_mask,
                    condition_token_mask=condition,
                )
            condition_lengths = condition.to(torch.int32).sum(dim=1)
            predicted = shift_left(predicted, condition_lengths, pad)
            raw_predicted = predicted[:, :args.free_response_tokens].clone()
            predicted = mask_after_eos(raw_predicted, eos, pad)
            for row_index, raw_row in enumerate(raw_rows):
                text = tokenizer.decode(
                    predicted[row_index].detach().cpu().tolist(),
                    skip_special_tokens=True,
                )
                record = {
                    "eval_id": raw_row["example_id"],
                    "presentation_index": int(raw_row["presentation_index"]),
                    "mode": mode,
                    "generated": text,
                    "prompt_tokens": int(condition_lengths[row_index]),
                    "response_token_budget": args.free_response_tokens,
                    "eos_emitted": bool((raw_predicted[row_index] == eos).any()),
                    "dominant_token_fraction": dominant_fraction(
                        predicted[row_index], pad, eos,
                    ),
                }
                rows[mode].append(record)
                hypotheses[mode].append(text)
        references.extend(raw_row["response"] for raw_row in raw_rows)

    metrics = {
        mode: {
            "bleu": compute_bleu(hypotheses[mode], references),
            **compute_rouge(hypotheses[mode], references),
            "mean_generated_chars": float(np.mean([len(text) for text in hypotheses[mode]])),
            "eos_rate": float(np.mean([row["eos_emitted"] for row in rows[mode]])),
            "mean_dominant_token_fraction": float(np.mean(
                [row["dominant_token_fraction"] for row in rows[mode]]
            )),
        }
        for mode in MODES
    }
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True,
    )
    def compare(left, right):
        comparison = {}
        for metric in ("rouge1", "rouge2", "rougeL"):
            delta = []
            for reference, left_text, right_text in zip(
                references, hypotheses[left], hypotheses[right],
            ):
                left_score = scorer.score(reference, left_text)[metric].fmeasure * 100
                right_score = scorer.score(reference, right_text)[metric].fmeasure * 100
                delta.append(left_score - right_score)
            comparison[metric] = paired_bootstrap(
                delta, seed=args.seed, draws=10_000,
            )
        return comparison

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(output) + ".jsonl").open("w") as handle:
        for mode in MODES:
            for row in rows[mode]:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    result = {
        "complete": True,
        "diagnostic_only": True,
        "gold_content_used_by_oracle": True,
        "checkpoint": str(Path(resolved).resolve()),
        "weights": "raw",
        "config": str(Path(args.config).resolve()),
        "candidate_schedule_range": [args.start_index, candidate_stop],
        "configured_training_schedule_rows": int(config.conditional_rows),
        "audit_schedule_rows": window["audit_schedule_rows"],
        "checkpoint_step": checkpoint_step,
        "checkpoint_seen_presentation_rows": (
            window["checkpoint_seen_presentation_rows"]
        ),
        "selected_presentation_indices": [
            int(row["presentation_index"]) for row in selected
        ],
        "exact_k_shuffle": args.exact_k_shuffle,
        "shuffle_preserves_each_sample_active_budget": args.exact_k_shuffle,
        "checkpoint_has_not_seen_schedule_range": (
            args.start_index >= window["checkpoint_seen_presentation_rows"]
        ),
        "samples": args.num_samples,
        "seed": args.seed,
        "nfe": args.nfe,
        "free_response_token_cap": args.free_response_tokens,
        "sampling": {
            "method": "sde", "sde_gamma": 1.5,
            "trajectory": "planning_first", "plan_lead_alpha": 2.0,
            "self_cond_cfg_scale": 3.0,
            "plan_cfg_scale": args.plan_cfg_scale,
        },
        "metrics": metrics,
        "oracle_matched_minus_generated": compare("oracle_matched", "generated"),
        "oracle_matched_minus_shuffled": compare("oracle_matched", "oracle_shuffled"),
        "interpretation": (
            "oracle_matched freezes the exact clean Plan-VAE target from gold thinking "
            "at t_plan=1 throughout response sampling; oracle_shuffled swaps paired plans "
            + ("with exactly equal active-slot counts" if args.exact_k_shuffle else
               "within each batch, preserving only the batch-level plan distribution")
            + "; both are diagnostics, not deployable protocols"
        ),
    }
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
