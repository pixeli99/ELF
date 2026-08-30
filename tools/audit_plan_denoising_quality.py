#!/usr/bin/env python3
"""Split raw checkpoint plan reconstruction quality into active and NULL slots."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from configs.config import SamplingConfig, load_config_from_yaml
from modules.plan_vae import load_frozen_plan_vae
from utils.generation_utils import _generate_samples_single_batch
from utils.conditional_data import (
    ConditionalPairedDataset,
    ConditionalSchedule,
    get_conditional_dataloader,
)
from utils.encoder_utils import encode_x0
from utils.plan_stream import build_vae_plan_target
from utils.sampling_utils import add_noise, get_sampling_steps
from utils.stage_b_eval_runtime import load_model_and_encoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-index", type=int, default=40000)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--token-time", type=float, default=0.5)
    parser.add_argument("--plan-times", type=float, nargs="+",
                        default=[0.0, 0.25, 0.5, 0.75, 0.9, 1.0])
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--rollout-nfe", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def accumulator():
    return {
        "active_squared_error": 0.0,
        "active_elements": 0,
        "null_squared_error": 0.0,
        "null_elements": 0,
        "active_cosine_sum": 0.0,
        "active_slots": 0,
        "active_pred_norm_sum": 0.0,
        "null_pred_norm_sum": 0.0,
        "predicted_active_slots_sum": 0.0,
        "true_active_slots_sum": 0.0,
        "samples": 0,
    }


def paired_bootstrap_interval(values, seed, draws=10000):
    tensor = torch.tensor(values, dtype=torch.float64)
    generator = torch.Generator().manual_seed(seed)
    estimates = []
    for _ in range(0, draws, 1000):
        count = min(1000, draws - len(estimates) * 1000)
        indices = torch.randint(
            0, tensor.numel(), (count, tensor.numel()), generator=generator,
        )
        estimates.append(tensor[indices].mean(dim=1))
    estimates = torch.cat(estimates)
    return [float(torch.quantile(estimates, q)) for q in (0.025, 0.975)]


def main():
    args = parse_args()
    if args.num_samples <= 0 or args.batch_size <= 0 or args.rollout_nfe <= 0:
        raise ValueError("num_samples and batch_size must be positive")
    if not 0.0 <= args.token_time <= 1.0:
        raise ValueError("token_time must be in [0,1]")
    if any(not 0.0 <= value <= 1.0 for value in args.plan_times):
        raise ValueError("plan_times must be in [0,1]")

    device = torch.device(args.device)
    config = load_config_from_yaml(args.config)
    model, encoder, tokenizer, resolved = load_model_and_encoder(
        config, args.checkpoint, device,
    )
    tokenizer.model_max_length = sys.maxsize
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["params"], strict=True)
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
    schedule = ConditionalSchedule(
        base, rows=int(config.conditional_rows),
        master_seed=int(config.conditional_master_seed),
    )
    stop = args.start_index + args.num_samples
    if args.start_index < 0 or stop > len(schedule):
        raise ValueError("requested range is outside the fixed schedule")
    subset = torch.utils.data.Subset(schedule, range(args.start_index, stop))
    loader, _ = get_conditional_dataloader(
        subset, tokenizer, config, batch_size=args.batch_size,
        num_workers=0, distributed=False,
    )

    stats = {str(value): accumulator() for value in args.plan_times}
    rollout_stats = accumulator()
    rollout_paired_minus_shuffled_cosine = []
    rollout_paired_minus_shuffled_mse = []
    zero_active_squared_error = 0.0
    zero_active_elements = 0
    true_active_slots_total = 0
    generator = torch.Generator(device=device).manual_seed(args.seed)
    parameter_dtype = next(model.parameters()).dtype

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            ids = batch["input_ids"].to(device).long()
            attention = batch["attention_mask"].to(device).float()
            condition = batch["cond_seq_mask"].to(device).float()
            x0 = encode_x0(
                ids, attention, encoder,
                config.latent_mean, config.latent_std,
                use_bf16=bool(config.use_bf16), cond_mask=condition,
            ).to(parameter_dtype)
            plan_target, plan_mask, active = build_vae_plan_target(
                batch["plan_input_ids"].to(device).long(),
                batch["plan_attention_mask"].to(device).bool(),
                encoder, plan_encoder, config,
            )
            plan_target = plan_target.to(parameter_dtype)
            active = active.bool()
            null = ~active
            batch_size = ids.shape[0]
            true_active_slots_total += int(active.sum())
            zero_active_squared_error += float(
                plan_target.float().square().masked_select(
                    active.unsqueeze(-1).expand_as(plan_target)
                ).sum().item()
            )
            zero_active_elements += int(active.sum()) * plan_target.shape[-1]

            token_t = torch.full(
                (batch_size,), args.token_time,
                dtype=parameter_dtype, device=device,
            )
            token_noise = torch.randn(
                x0.shape, dtype=parameter_dtype, device=device, generator=generator,
            )
            token_z = add_noise(
                x0, token_noise, token_t, config,
                cond_seq_mask=condition.unsqueeze(-1),
            )
            # Sampling starts with zero response self-conditioning, but clean prompt
            # latents are restored in both halves. The prompt/plan subsystem can read
            # the prompt half, so plain all-zero self-conditioning would probe an
            # out-of-distribution prompt representation.
            prompt_self_cond = condition.unsqueeze(-1) * x0
            model_input = torch.cat([token_z, prompt_self_cond], dim=-1)
            self_cond_scale = torch.ones(
                (batch_size,), dtype=parameter_dtype, device=device,
            )

            for plan_time in args.plan_times:
                plan_t = torch.full(
                    (batch_size,), plan_time,
                    dtype=parameter_dtype, device=device,
                )
                plan_noise = torch.randn(
                    plan_target.shape, dtype=parameter_dtype,
                    device=device, generator=generator,
                )
                plan_z = add_noise(plan_target, plan_noise, plan_t, config)
                with torch.amp.autocast(
                    "cuda", dtype=torch.bfloat16,
                    enabled=bool(config.use_bf16) and device.type == "cuda",
                ):
                    _, _, prediction = model(
                        model_input, token_t, deterministic=True,
                        self_cond_cfg_scale=self_cond_scale,
                        decoder_step_active=torch.zeros(
                            (batch_size,), dtype=parameter_dtype, device=device,
                        ),
                        x_plan=plan_z, t_plan=plan_t, plan_mask=plan_mask,
                        condition_token_mask=condition.bool(),
                    )
                prediction = prediction.float()
                target = plan_target.float()
                values = stats[str(plan_time)]
                active_elements = active.unsqueeze(-1).expand_as(target)
                null_elements = null.unsqueeze(-1).expand_as(target)
                values["active_squared_error"] += float(
                    (prediction - target).square().masked_select(active_elements).sum().item()
                )
                values["active_elements"] += int(active.sum()) * target.shape[-1]
                values["null_squared_error"] += float(
                    prediction.square().masked_select(null_elements).sum().item()
                )
                values["null_elements"] += int(null.sum()) * target.shape[-1]
                values["active_cosine_sum"] += float(
                    F.cosine_similarity(prediction[active], target[active], dim=-1).sum().item()
                )
                values["active_slots"] += int(active.sum())
                values["active_pred_norm_sum"] += float(prediction[active].norm(dim=-1).sum().item())
                values["null_pred_norm_sum"] += float(prediction[null].norm(dim=-1).sum().item())
                values["predicted_active_slots_sum"] += float(
                    (prediction.norm(dim=-1) > 2.0).sum(dim=-1).float().sum().item()
                )
                values["true_active_slots_sum"] += float(active.sum().item())
                values["samples"] += batch_size

            # Full inference-path plan rollout from pure noise. Plan-CFG is one
            # because it only extrapolates the response stream; the generated plan
            # itself is identical for plan-CFG 1 and 3.
            rollout_seed = args.seed + 10_000_003 + batch_index
            torch.manual_seed(rollout_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(rollout_seed)
            token_generator = torch.Generator().manual_seed(rollout_seed)
            plan_generator = torch.Generator().manual_seed(rollout_seed + 1_000_003)
            t_steps = get_sampling_steps(
                n_steps=args.rollout_nfe, time_schedule="logit_normal",
                P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
                device=device, dtype=parameter_dtype,
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
            sampling = SamplingConfig(
                sampling_method="sde", sde_gamma=1.5,
                time_schedule="logit_normal", num_sampling_steps=[args.rollout_nfe],
                cfgs=[1.0], self_cond_cfg_scales=[3.0],
                plan_trajectory="planning_first", plan_lead_alpha=2.0,
                plan_cfg_scale=1.0,
            )
            torch.manual_seed(rollout_seed + 2_000_003)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(rollout_seed + 2_000_003)
            _, generated_plan = _generate_samples_single_batch(
                model=model, generator=token_generator,
                z=response_noise, t_steps=t_steps,
                cond_seq=x0 * condition.unsqueeze(-1),
                cond_seq_mask=condition,
                config=config, sampling_config=sampling,
                cfg_scale=1.0, self_cond_cfg_scale=3.0,
                plan_mask=plan_mask,
                initial_plan_noise=initial_plan_noise,
            )
            prediction = generated_plan.float()
            target = plan_target.float()
            active_elements = active.unsqueeze(-1).expand_as(target)
            null_elements = null.unsqueeze(-1).expand_as(target)
            rollout_stats["active_squared_error"] += float(
                (prediction - target).square().masked_select(active_elements).sum().item()
            )
            rollout_stats["active_elements"] += int(active.sum()) * target.shape[-1]
            rollout_stats["null_squared_error"] += float(
                prediction.square().masked_select(null_elements).sum().item()
            )
            rollout_stats["null_elements"] += int(null.sum()) * target.shape[-1]
            rollout_stats["active_cosine_sum"] += float(
                F.cosine_similarity(prediction[active], target[active], dim=-1).sum().item()
            )
            rollout_stats["active_slots"] += int(active.sum())
            rollout_stats["active_pred_norm_sum"] += float(
                prediction[active].norm(dim=-1).sum().item()
            )
            rollout_stats["null_pred_norm_sum"] += float(
                prediction[null].norm(dim=-1).sum().item()
            )
            rollout_stats["predicted_active_slots_sum"] += float(
                (prediction.norm(dim=-1) > 2.0).sum(dim=-1).float().sum().item()
            )
            rollout_stats["true_active_slots_sum"] += float(active.sum().item())
            rollout_stats["samples"] += batch_size
            shuffled_target = target.roll(shifts=1, dims=0)
            shuffled_active = active.roll(shifts=1, dims=0)
            for row in range(batch_size):
                paired_cosine = float(F.cosine_similarity(
                    prediction[row, active[row]], target[row, active[row]], dim=-1,
                ).mean().item())
                shuffled_cosine = float(F.cosine_similarity(
                    prediction[row, shuffled_active[row]],
                    shuffled_target[row, shuffled_active[row]], dim=-1,
                ).mean().item())
                paired_mse = float(F.mse_loss(
                    prediction[row, active[row]], target[row, active[row]],
                ).item())
                shuffled_mse = float(F.mse_loss(
                    prediction[row, shuffled_active[row]],
                    shuffled_target[row, shuffled_active[row]],
                ).item())
                rollout_paired_minus_shuffled_cosine.append(
                    paired_cosine - shuffled_cosine
                )
                rollout_paired_minus_shuffled_mse.append(
                    paired_mse - shuffled_mse
                )

    finalized = {}
    for plan_time, values in stats.items():
        null_slots = values["null_elements"] // int(config.plan_target_dim)
        finalized[plan_time] = {
            "active_mse": values["active_squared_error"] / values["active_elements"],
            "null_mse": values["null_squared_error"] / values["null_elements"],
            "active_cosine": values["active_cosine_sum"] / values["active_slots"],
            "active_prediction_norm": values["active_pred_norm_sum"] / values["active_slots"],
            "null_prediction_norm": values["null_pred_norm_sum"] / null_slots,
            "predicted_budget_norm_gt_2": values["predicted_active_slots_sum"] / values["samples"],
            "true_active_budget": values["true_active_slots_sum"] / values["samples"],
            "samples": values["samples"],
        }
    result = {
        "checkpoint": str(Path(resolved).resolve()),
        "weights": "raw",
        "config": str(Path(args.config).resolve()),
        "requested_range": [args.start_index, stop],
        "samples": args.num_samples,
        "token_time": args.token_time,
        "plan_times": args.plan_times,
        "noise_seed": args.seed,
        "true_active_budget": true_active_slots_total / args.num_samples,
        "zero_prediction_active_mse": zero_active_squared_error / zero_active_elements,
        "by_plan_time": finalized,
        "full_rollout": {
            "nfe": args.rollout_nfe,
            "trajectory": "planning_first",
            "plan_lead_alpha": 2.0,
            "sampling_method": "sde",
            "sde_gamma": 1.5,
            "active_mse": rollout_stats["active_squared_error"]
            / rollout_stats["active_elements"],
            "null_mse": rollout_stats["null_squared_error"]
            / rollout_stats["null_elements"],
            "active_cosine": rollout_stats["active_cosine_sum"]
            / rollout_stats["active_slots"],
            "active_prediction_norm": rollout_stats["active_pred_norm_sum"]
            / rollout_stats["active_slots"],
            "null_prediction_norm": rollout_stats["null_pred_norm_sum"]
            / (rollout_stats["null_elements"] // int(config.plan_target_dim)),
            "predicted_budget_norm_gt_2": rollout_stats["predicted_active_slots_sum"]
            / rollout_stats["samples"],
            "true_active_budget": rollout_stats["true_active_slots_sum"]
            / rollout_stats["samples"],
            "paired_minus_shuffled_active_cosine": {
                "mean": sum(rollout_paired_minus_shuffled_cosine)
                / len(rollout_paired_minus_shuffled_cosine),
                "paired_bootstrap_95ci": paired_bootstrap_interval(
                    rollout_paired_minus_shuffled_cosine, args.seed + 31,
                ),
            },
            "paired_minus_shuffled_active_mse": {
                "mean": sum(rollout_paired_minus_shuffled_mse)
                / len(rollout_paired_minus_shuffled_mse),
                "paired_bootstrap_95ci": paired_bootstrap_interval(
                    rollout_paired_minus_shuffled_mse, args.seed + 37,
                ),
            },
        },
        "notes": {
            "active": "Plan-VAE content slots",
            "null": "Plan-VAE trailing exact-zero slots",
            "prediction": "raw plan x0 head under the prompt-causal mask",
            "budget_threshold": "predicted slot L2 norm > 2.0",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
