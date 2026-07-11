#!/usr/bin/env python
"""Eval-time oracle-plan intervention for Ordered ELF."""

import argparse
import copy
import csv
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_ROOT = os.path.join(REPO_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from configs.config import SamplingConfig, load_config_from_yaml, load_sampling_configs
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.checkpoint_utils import find_latest_checkpoint
from utils.data_utils import get_pad_token_id, load_dataset_split
from utils.encoder_utils import encode_text
from utils.generation_utils import _dlm_decode_batch, _generate_samples_single_batch, mask_after_eos
from utils.metrics_utils import Metrics as PPLMetrics
from utils.plan_utils import build_plan_target
from utils.sampling_utils import get_sampling_steps

logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
    force=True,
)
logger = logging.getLogger("oracle_plan_eval")


@dataclass
class RunArgs:
    config_path: str
    checkpoint_path: str
    dataset: str
    num_samples: int
    num_sampling_steps: int
    modes: str
    seed: int
    paired_eval_base_seed: int
    global_batch_size: int
    eval_ppl_batch_size: int
    out_dir: str
    sampling_config_path: str
    eval_ppl_model: str
    eval_ppl_max_length: int
    dataset_cache_dir: str
    use_cpu: bool


def parse_args():
    p = argparse.ArgumentParser(description="Ordered ELF oracle-plan eval-time intervention.")
    p.add_argument("--config_path", required=True)
    p.add_argument("--checkpoint_path", required=True)
    p.add_argument("--dataset", default="embedded-language-flows/openwebtext-t5")
    p.add_argument("--num_samples", type=int, default=256)
    p.add_argument("--num_sampling_steps", type=int, default=32)
    p.add_argument("--modes", default="null,self_planning_first,oracle_matched,oracle_shuffled")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--paired_eval_base_seed", type=int, default=42)
    p.add_argument("--global_batch_size", type=int, default=4)
    p.add_argument("--eval_ppl_batch_size", type=int, default=4)
    p.add_argument("--out_dir", default="results/oracle_plan_eval/debug")
    p.add_argument(
        "--sampling_config_path",
        default="src/configs/sampling_configs/ordered_sampling_configs.yml",
        help="Sampling YAML loaded with the same loader as src/eval.py.",
    )
    p.add_argument("--eval_ppl_model", default=None)
    p.add_argument("--eval_ppl_max_length", type=int, default=None)
    p.add_argument("--dataset_cache_dir", default=None)
    p.add_argument("--use_cpu", action="store_true")
    ns = p.parse_args()
    return ns


def _resolve_checkpoint(path: str) -> str:
    if os.path.isdir(path):
        latest = find_latest_checkpoint(path)
        if latest:
            return latest
    return path


def _load_model_and_encoder(config, checkpoint_path: str, device: torch.device):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model,
        max_length=config.max_length,
        attn_drop=config.attn_dropout,
        proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
        gradient_checkpointing=False,
        num_plan_slots=config.num_plan_slots,
        num_plan_time_tokens=config.num_plan_time_tokens,
        plan_whiten=config.plan_whiten,
        plan_target_dim=config.plan_target_dim,
    ).to(device).eval()

    ckpt_path = _resolve_checkpoint(checkpoint_path)
    logger.info("Loading checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["params"])
    if ckpt.get("ema_params1"):
        model.load_state_dict(ckpt["ema_params1"], strict=False)
        logger.info("Loaded EMA parameters; plan whitener buffers kept from checkpoint params.")
    model = model.to(device).eval()
    return model, encoder, tokenizer, ckpt_path


def _take_examples(dataset_path: str, num_samples: int, cache_dir: str):
    ds = load_dataset_split(dataset_path, dataset_cache_dir=cache_dir)
    if len(ds) < num_samples:
        raise ValueError(f"Dataset has only {len(ds)} rows, requested {num_samples}.")
    cols = set(ds.column_names)
    if "input_ids" not in cols:
        raise ValueError(f"Dataset must contain input_ids; columns={ds.column_names}")
    if "sequence_length" not in cols:
        logger.warning("Dataset has no sequence_length column; falling back to len(input_ids).")
    return [ds[int(i)] for i in range(num_samples)]


def _collate_examples(examples: List[Dict], max_length: int, pad_token_id: int):
    ids_rows, lengths = [], []
    for item in examples:
        ids = np.asarray(item["input_ids"], dtype=np.int64)
        seq_len = int(item["sequence_length"]) if "sequence_length" in item else int(len(ids))
        seq_len = max(0, min(seq_len, len(ids), max_length))
        ids = ids[:max_length]
        if len(ids) < max_length:
            ids = np.concatenate(
                [ids, np.full(max_length - len(ids), pad_token_id, dtype=np.int64)]
            )
        ids_rows.append(ids)
        lengths.append(seq_len)
    input_ids = np.stack(ids_rows)
    total_lens = np.asarray(lengths, dtype=np.int32)
    pos = np.arange(max_length)[None, :]
    is_valid = pos < total_lens[:, None]
    return {
        "input_ids": torch.from_numpy(input_ids).long(),
        "attention_mask": torch.from_numpy(is_valid).float(),
        "cond_seq_mask": torch.zeros((len(examples), max_length), dtype=torch.float32),
        "sequence_length": torch.from_numpy(total_lens).long(),
    }


@torch.no_grad()
def _build_oracle_plans(model, encoder, config, examples, pad_token_id, batch_size, device):
    plans, lengths = [], []
    use_bf16 = bool(getattr(config, "use_bf16", True)) and device.type == "cuda"
    dtype = next(model.parameters()).dtype
    for start in range(0, len(examples), batch_size):
        batch = _collate_examples(examples[start:start + batch_size], config.max_length, pad_token_id)
        input_ids = batch["input_ids"].to(device)
        seq_lengths = batch["sequence_length"].to(device)
        positions = torch.arange(input_ids.shape[1], device=device).unsqueeze(0)
        enc_mask = positions < seq_lengths.unsqueeze(1)
        attn = batch["attention_mask"].to(device)
        cond = batch["cond_seq_mask"].to(device)
        x0 = encode_text(
            input_ids=input_ids,
            attention_mask=enc_mask,
            encoder=encoder,
            latent_mean=config.latent_mean,
            latent_std=config.latent_std,
            use_bf16=use_bf16,
        ).to(dtype)
        valid = attn if config.pad_token == "pad" else torch.ones_like(attn)
        valid = valid * (1.0 - cond)
        plans.append(build_plan_target(model, x0, valid).detach().cpu())
        lengths.extend(batch["sequence_length"].tolist())
    return torch.cat(plans, dim=0), lengths


def _sampling_config_for_mode(args, mode: str, sampling_configs: List[SamplingConfig]) -> SamplingConfig:
    traj = {
        "null": "null",
        "self_planning_first": "planning_first",
        "oracle_matched": "diagonal",
        "oracle_shuffled": "diagonal",
    }[mode]
    candidates = [sc for sc in sampling_configs if getattr(sc, "plan_trajectory", None) == traj]
    if mode.startswith("oracle_"):
        candidates = [sc for sc in sampling_configs if getattr(sc, "plan_trajectory", None) == "null"]
    if not candidates:
        raise ValueError(f"Sampling YAML has no trajectory template for mode={mode!r}.")
    sc = copy.deepcopy(candidates[0])
    sc.num_sampling_steps = [args.num_sampling_steps]
    sc.plan_trajectory = traj
    return sc


def _consistency(model, latent: torch.Tensor, plan: torch.Tensor):
    valid = torch.ones(latent.shape[:2], dtype=torch.float32, device=latent.device)
    target = build_plan_target(model, latent.float(), valid)
    plan_f = plan.to(device=latent.device, dtype=target.dtype)
    num = (target - plan_f).norm(dim=-1)
    den = plan_f.norm(dim=-1).clamp(min=1e-6)
    return float((num / den).mean().detach().cpu())


@torch.no_grad()
def _generate_mode(model, tokenizer, config, args, mode: str, oracle_plan: Optional[torch.Tensor],
                   shuffled_plan: Optional[torch.Tensor], pad_token_id: int, device: torch.device,
                   sampling_configs: List[SamplingConfig]):
    sc = _sampling_config_for_mode(args, mode, sampling_configs)
    cfg_scale = float(sc.cfgs[0])
    self_cond_cfg_scale = float(sc.self_cond_cfg_scales[0])
    d_model = model.text_encoder_dim
    param_dtype = next(model.parameters()).dtype
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    generated, records = [], []
    consistencies = []

    for batch_idx, start in enumerate(range(0, args.num_samples, args.global_batch_size)):
        end = min(start + args.global_batch_size, args.num_samples)
        bsz = end - start
        # Match src/generation.py paired_trajectory_eval on rank 0.
        batch_seed = (
            int(args.paired_eval_base_seed)
            + int(args.seed) * 10_000_000
            + int(args.num_sampling_steps) * 10_007
            + int(batch_idx) * 101
        )
        fork_devices = []
        if device.type == "cuda":
            fork_devices = [device.index if device.index is not None else torch.cuda.current_device()]
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(batch_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(batch_seed)
            gen = torch.Generator(device="cpu").manual_seed(batch_seed)
            t_steps = get_sampling_steps(
                n_steps=args.num_sampling_steps,
                time_schedule=sc.time_schedule,
                P_mean=config.denoiser_p_mean,
                P_std=config.denoiser_p_std,
                device=device,
                dtype=param_dtype,
            )
            if device.type == "cuda":
                z0 = torch.randn((bsz, config.max_length, d_model), dtype=param_dtype, device=device)
            else:
                z0 = torch.randn((bsz, config.max_length, d_model), generator=gen, dtype=param_dtype)
                z0 = z0.to(device)
            z0 = z0 * config.denoiser_noise_scale

        plan_source = None
        if mode == "oracle_matched":
            plan_source = oracle_plan[start:end]
        elif mode == "oracle_shuffled":
            plan_source = shuffled_plan[start:end]
        plan_override = None
        if plan_source is not None:
            plan_batch = plan_source.to(device=device, dtype=param_dtype)
            plan_override = [plan_batch for _ in range(t_steps.shape[0])]
            logger.info("%s oracle plan tensor shape: %s", mode, tuple(plan_batch.shape))

        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(batch_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(batch_seed)
            gen = torch.Generator(device="cpu").manual_seed(batch_seed)
            latent, latent_plan = _generate_samples_single_batch(
                model=model,
                generator=gen,
                z=z0.clone(),
                t_steps=t_steps,
                cond_seq=None,
                cond_seq_mask=None,
                config=config,
                sampling_config=sc,
                cfg_scale=cfg_scale,
                self_cond_cfg_scale=self_cond_cfg_scale,
                plan_override=plan_override,
                plan_override_t=1.0 if plan_override is not None else None,
                freeze_plan_override=plan_override is not None,
            )
        is_null = mode == "null"
        decode_plan = latent_plan
        t_plan_decode = 0.0 if is_null else 1.0
        ids = _dlm_decode_batch(
            z=latent,
            model=model,
            t_final_val=t_steps[-1].item(),
            config=config,
            self_cond_cfg_scale=self_cond_cfg_scale,
            x_plan=decode_plan,
            t_plan_decode_val=t_plan_decode,
            plan_trajectory=sc.plan_trajectory,
        )
        ids = mask_after_eos(ids, eos_token_id=eos_token_id, pad_token_id=pad_token_id)
        texts = [tokenizer.decode(ids[i].detach().cpu().numpy(), skip_special_tokens=True)
                 for i in range(ids.shape[0])]
        generated.extend(texts)
        if plan_source is not None:
            consistencies.append(_consistency(model, latent, plan_source.to(device)))
        for j, text in enumerate(texts):
            records.append({"id": start + j, "mode": mode, "generated": text})

    consistency = float(np.mean(consistencies)) if consistencies else None
    return generated, records, consistency, sc


def _write_outputs(out_dir: str, run_config: Dict, rows: List[Dict], sample_records: List[Dict]):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="", encoding="utf-8") as f:
        fieldnames = sorted({k for row in rows for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(out_dir, "samples.jsonl"), "w", encoding="utf-8") as f:
        for rec in sample_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    summary = {"modes": rows, "best_by_gen_ppl": min(rows, key=lambda r: r["gen_ppl"]) if rows else None}
    with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def main():
    ns = parse_args()
    args = RunArgs(**vars(ns))
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    valid_modes = {"null", "self_planning_first", "oracle_matched", "oracle_shuffled"}
    bad = [m for m in modes if m not in valid_modes]
    if bad:
        raise ValueError(f"Unknown modes: {bad}")

    device = torch.device("cpu") if args.use_cpu or not torch.cuda.is_available() else torch.device("cuda")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    config = load_config_from_yaml(args.config_path)
    sampling_configs = load_sampling_configs(args.sampling_config_path)
    config.global_batch_size = args.global_batch_size
    config.batch_size = args.global_batch_size
    config.eval_ppl_batch_size = args.eval_ppl_batch_size
    if args.eval_ppl_model is not None:
        config.eval_ppl_model = args.eval_ppl_model
    if args.eval_ppl_max_length is not None:
        config.eval_ppl_max_length = args.eval_ppl_max_length
    if config.num_plan_slots <= 0:
        raise ValueError("Oracle plan eval requires an ordered checkpoint/config with num_plan_slots > 0.")

    logger.info("mode list: %s", modes)
    logger.info("num_samples=%d num_sampling_steps=%d", args.num_samples, args.num_sampling_steps)
    logger.info("checkpoint path: %s", args.checkpoint_path)
    logger.info("dataset path: %s", args.dataset)
    logger.info("sampling config path: %s", args.sampling_config_path)
    logger.info("max_length=%d denoiser_noise_scale=%s", config.max_length,
                config.denoiser_noise_scale)
    logger.info("denoiser_p_mean=%s denoiser_p_std=%s use_bf16=%s", config.denoiser_p_mean,
                config.denoiser_p_std, bool(getattr(config, "use_bf16", True)))
    logger.info("eval_ppl_model=%s eval_ppl_max_length=%s eval_ppl_batch_size=%d retokenize=True",
                config.eval_ppl_model, config.eval_ppl_max_length, args.eval_ppl_batch_size)
    logger.info("tokenizer=%s pad_token_policy=%s skip_special_tokens=True empty_filter=True",
                config.tokenizer_name or config.encoder_model_name, config.pad_token)
    logger.info(
        "paired seed formula (rank 0): paired_base_seed + eval_seed*10000000 + "
        "steps*10007 + batch_idx*101; paired_base_seed=%d eval_seed=%d",
        args.paired_eval_base_seed, args.seed,
    )
    for mode in modes:
        effective = _sampling_config_for_mode(args, mode, sampling_configs)
        logger.info(
            "effective mode=%s sampling_method=%s num_sampling_steps=%d sde_gamma=%s "
            "time_schedule=%s cfg_scale=%s self_cond_cfg_scale=%s plan_trajectory=%s "
            "plan_lead_alpha=%s plan_cfg_scale=%s",
            mode, effective.sampling_method, args.num_sampling_steps,
            getattr(effective, "sde_gamma", 0.0), effective.time_schedule,
            effective.cfgs[0], effective.self_cond_cfg_scales[0], effective.plan_trajectory,
            getattr(effective, "plan_lead_alpha", 2.0),
            getattr(effective, "plan_cfg_scale", 1.0),
        )

    model, encoder, tokenizer, resolved_ckpt = _load_model_and_encoder(config, args.checkpoint_path, device)
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    oracle_plan = shuffled_plan = None
    lengths, perm = [], torch.empty(0, dtype=torch.long)
    if any(mode.startswith("oracle_") for mode in modes):
        examples = _take_examples(args.dataset, args.num_samples, args.dataset_cache_dir)
        oracle_plan, lengths = _build_oracle_plans(
            model, encoder, config, examples, pad_token_id, args.global_batch_size, device,
        )
        logger.info("oracle matched=True shuffled=False tensor shape: %s", tuple(oracle_plan.shape))
        shuffle_gen = torch.Generator(device="cpu").manual_seed(args.seed + 9_172_631)
        perm = torch.randperm(args.num_samples, generator=shuffle_gen)
        shuffled_plan = oracle_plan[perm]
        logger.info("oracle matched=False shuffled=True tensor shape: %s", tuple(shuffled_plan.shape))

    ppl = PPLMetrics(
        gen_ppl_eval_model_name_or_path=config.eval_ppl_model,
        eval_ppl_batch_size=args.eval_ppl_batch_size,
        eval_context_size=config.eval_ppl_max_length,
    )
    metric_rows, sample_records = [], []
    for mode in modes:
        logger.info("Running mode=%s", mode)
        logger.info("matched=%s shuffled=%s", mode == "oracle_matched", mode == "oracle_shuffled")
        texts, records, consistency, effective = _generate_mode(
            model, tokenizer, config, args, mode, oracle_plan, shuffled_plan, pad_token_id, device,
            sampling_configs,
        )
        nonempty = [t for t in texts if isinstance(t, str) and t.strip()]
        ppl.reset()
        if nonempty:
            ppl_result = ppl.record_generative_perplexity(
                text_samples=nonempty,
                max_length=config.eval_ppl_max_length,
                retokenize=True,
            )
        else:
            logger.warning("mode=%s has no nonempty samples; skipping PPL exactly as src/generation.py", mode)
            ppl_result = {"ppl": float("nan"), "mean_entropy": float("nan")}
        row = {
            "mode": mode,
            "num_samples": len(texts),
            "num_nonempty_samples": len(nonempty),
            "num_sampling_steps": args.num_sampling_steps,
            "gen_ppl": ppl_result["ppl"],
            "entropy": ppl_result["mean_entropy"],
            "consistency": consistency,
            "sampling_method": effective.sampling_method,
            "sde_gamma": getattr(effective, "sde_gamma", 0.0),
            "time_schedule": effective.time_schedule,
            "cfg_scale": effective.cfgs[0],
            "self_cond_cfg_scale": effective.self_cond_cfg_scales[0],
            "plan_trajectory": effective.plan_trajectory,
            "plan_lead_alpha": getattr(effective, "plan_lead_alpha", 2.0),
            "denoiser_noise_scale": config.denoiser_noise_scale,
            "max_length": config.max_length,
            "eval_ppl_model": config.eval_ppl_model,
            "eval_ppl_max_length": config.eval_ppl_max_length,
            "eval_ppl_batch_size": args.eval_ppl_batch_size,
        }
        metric_rows.append(row)
        sample_records.extend(records)
        logger.info("mode=%s gen_ppl=%.4f entropy=%.4f consistency=%s",
                    mode, row["gen_ppl"], row["entropy"], row["consistency"])

    run_config = asdict(args)
    run_config.update({
        "resolved_checkpoint_path": resolved_ckpt,
        "config": args.config_path,
        "modes": modes,
        "plan_shape": list(oracle_plan.shape) if oracle_plan is not None else None,
        "shuffle_permutation_first16": perm[:16].tolist(),
        "sequence_length_first16": lengths[:16],
    })
    _write_outputs(args.out_dir, run_config, metric_rows, sample_records)
    logger.info("Wrote oracle plan eval outputs to %s", args.out_dir)


if __name__ == "__main__":
    main()
