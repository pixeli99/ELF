#!/usr/bin/env python3
"""Measure response/plan gradient alignment on a raw Stage-B checkpoint.

The two gradients are recovered from deterministic paired backward passes:
response-only uses plan_loss_weight=0, while total uses the configured weight.
Their difference is therefore the plan-loss gradient on exactly the same row,
noise, clocks, branch, self-conditioning choice, and dropout masks.
"""

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from configs.config import load_config_from_yaml
from modules.plan_vae import load_frozen_plan_vae
from train_step import train_step
from utils.conditional_data import (
    ConditionalPairedDataset,
    ConditionalSchedule,
    get_conditional_dataloader,
)
from utils.stage_b_eval_runtime import load_model_and_encoder
from utils.train_utils import TrainState


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start-index", type=int, default=40000)
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def category(name):
    if name.startswith("blocks."):
        return "shared_blocks"
    if name.startswith(("plan_in_proj.", "plan_slot_embed", "plan_t_embedder.",
                        "plan_t_emb_tokens")):
        return "plan_input_and_clock"
    if name.startswith(("plan_norm.", "plan_head.")):
        return "plan_output_head"
    return "response_and_prefix"


def blank_stats():
    return {"dot": 0.0, "response_sq": 0.0, "plan_sq": 0.0,
            "samples": 0, "batch_cosines": []}


def add_gradient_stats(target, response_grads, named_parameters):
    batch = {"all_trainable": blank_stats()}
    for name, parameter in named_parameters:
        cat = category(name)
        batch.setdefault(cat, blank_stats())
        response = response_grads.get(name)
        total = parameter.grad
        if response is None and total is None:
            continue
        if response is None:
            response = torch.zeros_like(total)
        if total is None:
            total = torch.zeros_like(response)
        plan = total.detach() - response
        r = response.float()
        p = plan.float()
        dot = float(torch.sum(r * p).item())
        r2 = float(torch.sum(r * r).item())
        p2 = float(torch.sum(p * p).item())
        for key in ("all_trainable", cat):
            batch[key]["dot"] += dot
            batch[key]["response_sq"] += r2
            batch[key]["plan_sq"] += p2

    for key, values in batch.items():
        denominator = (values["response_sq"] * values["plan_sq"]) ** 0.5
        cosine = values["dot"] / denominator if denominator else None
        aggregate = target.setdefault(key, blank_stats())
        aggregate["dot"] += values["dot"]
        aggregate["response_sq"] += values["response_sq"]
        aggregate["plan_sq"] += values["plan_sq"]
        aggregate["samples"] += 1
        if cosine is not None:
            aggregate["batch_cosines"].append(cosine)


def finalize(values):
    denominator = (values["response_sq"] * values["plan_sq"]) ** 0.5
    response_norm = values["response_sq"] ** 0.5
    plan_norm = values["plan_sq"] ** 0.5
    cosines = values["batch_cosines"]
    return {
        "samples": values["samples"],
        "aggregate_cosine": values["dot"] / denominator if denominator else None,
        "response_gradient_norm": response_norm,
        "plan_gradient_norm": plan_norm,
        "plan_to_response_norm_ratio": plan_norm / response_norm if response_norm else None,
        "plan_projection_on_response": values["dot"] / values["response_sq"]
        if values["response_sq"] else None,
        "batch_cosine_mean": statistics.fmean(cosines) if cosines else None,
        "batch_cosine_median": statistics.median(cosines) if cosines else None,
        "batch_cosine_min": min(cosines) if cosines else None,
        "batch_cosine_max": max(cosines) if cosines else None,
        "batch_negative_fraction": sum(value < 0 for value in cosines) / len(cosines)
        if cosines else None,
    }


def main():
    args = parse_args()
    if args.num_samples <= 0 or args.start_index < 0:
        raise ValueError("sample count must be positive and start index non-negative")
    device = torch.device(args.device)
    config = load_config_from_yaml(args.config)
    if config.group_mode != "ordered" or config.plan_loss_weight <= 0:
        raise ValueError("gradient conflict audit requires a supervised ordered config")

    model, encoder, tokenizer, resolved = load_model_and_encoder(
        config, args.checkpoint, device,
    )
    tokenizer.model_max_length = sys.maxsize
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["params"], strict=True)
    del checkpoint

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
    if stop > len(schedule):
        raise ValueError(f"requested rows [{args.start_index}, {stop}) exceed schedule")
    subset = torch.utils.data.Subset(schedule, range(args.start_index, stop))
    loader, _ = get_conditional_dataloader(
        subset, tokenizer, config, batch_size=1, num_workers=0, distributed=False,
    )

    response_config = copy.deepcopy(config)
    response_config.plan_loss_weight = 0.0
    response_config.grad_accum_steps = 2
    total_config = copy.deepcopy(config)
    total_config.grad_accum_steps = 2

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    state = TrainState(
        model=model, optimizer=optimizer, step=0,
        dropout_generator=torch.Generator(device="cpu"),
    )
    named_parameters = [(name, parameter) for name, parameter in model.named_parameters()
                        if parameter.requires_grad]
    aggregate = {}
    audited_indices = []
    decoder_indices = []
    response_losses = []
    plan_losses = []
    max_metric_mismatch = 0.0

    for offset, batch in enumerate(loader):
        presentation_index = args.start_index + offset
        optimizer.zero_grad(set_to_none=True)
        state.step = 0
        _, response_metrics = train_step(
            state, encoder, batch, response_config,
            plan_encoder=plan_encoder, update_model=True,
        )
        response_grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in named_parameters if parameter.grad is not None
        }

        optimizer.zero_grad(set_to_none=True)
        state.step = 0
        _, total_metrics = train_step(
            state, encoder, batch, total_config,
            plan_encoder=plan_encoder, update_model=True,
        )
        for key in ("l2_loss", "ce_loss", "response_valid_tokens"):
            mismatch = abs(float(response_metrics[key]) - float(total_metrics[key]))
            max_metric_mismatch = max(max_metric_mismatch, mismatch)

        plan_value = float(total_metrics["plan_l2_loss"])
        if plan_value == 0.0:
            decoder_indices.append(presentation_index)
            continue
        add_gradient_stats(aggregate, response_grads, named_parameters)
        audited_indices.append(presentation_index)
        response_losses.append(float(response_metrics["loss"]))
        plan_losses.append(plan_value)

    if not audited_indices:
        raise RuntimeError("no denoiser rows were sampled; plan gradients are all zero")
    result = {
        "checkpoint": str(Path(resolved).resolve()),
        "weights": "raw",
        "config": str(Path(args.config).resolve()),
        "schedule_master_seed": int(config.conditional_master_seed),
        "requested_range": [args.start_index, stop],
        "requested_samples": args.num_samples,
        "audited_denoiser_samples": len(audited_indices),
        "decoder_samples_skipped": len(decoder_indices),
        "audited_presentation_indices": audited_indices,
        "decoder_presentation_indices": decoder_indices,
        "mean_response_loss": statistics.fmean(response_losses),
        "mean_plan_loss": statistics.fmean(plan_losses),
        "max_paired_response_metric_mismatch": max_metric_mismatch,
        "gradient_alignment": {key: finalize(values) for key, values in aggregate.items()},
        "notes": {
            "plan_gradient_definition": "grad(response + plan) - grad(response)",
            "paired_randomness": "same schedule row, noise, clocks, branch, self-conditioning, and dropout",
            "shared_blocks_scope": "parameters named blocks.*",
            "interpretation": "negative cosine means local first-order conflict under an SGD-style update; the training optimizer is Muon, so this is a diagnostic rather than an exact update simulation",
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
