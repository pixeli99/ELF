#!/usr/bin/env python3
"""One-arm common80k generation wrapper around the original ELF sampler/scorer."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from configs.config import load_config_from_yaml
from utils.stage_b_eval_runtime import load_model_and_encoder
from utils.generation_utils import _dlm_decode_batch, _generate_samples_single_batch, mask_after_eos
from utils.metrics_utils import Metrics
from utils.sampling_utils import get_sampling_steps
from utils.stage_b_common80k_generation import (
    condition, plan_protocol, sha256_file, tensor_sha256, validate_shape_inputs,
    assert_fixed_plan, assert_truth_inputs,
)

SPLIT = Path(os.environ.get("STAGE_B_SPLIT_MANIFEST", ROOT / "data/stage_b_heldout_v2/split_manifest.json"))
SHAPE_DIR = Path(os.environ.get("STAGE_B_SHAPE_DIR", ROOT / "data/stage_b_heldout_v2/stage_b_generation_shape_test_n1000_seed45_v2"))
GPT2_SNAPSHOT = Path(os.environ.get("GPT2_LARGE_SNAPSHOT", ROOT / "artifacts/gpt2-large"))
RUN_ROOT = Path(os.environ.get("STAGE_B_RUN_ROOT", ROOT / "outputs"))
RUNS = {
    group: RUN_ROOT / f"elf_b_common80k_{group}_10k_v1"
    for group in ("ordered", "diagonal", "register", "vanilla")
}


def log_event(condition_id, stage, message):
    stamp = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    print(f"[{stamp}] [{condition_id}] [{stage}] {message}", flush=True)


def artifact_lock(snapshot=GPT2_SNAPSHOT):
    required = ("config.json", "tokenizer_config.json", "vocab.json", "merges.txt", "model.safetensors")
    if not snapshot.is_dir() or any(not (snapshot / name).is_file() for name in required):
        raise ValueError("unique local gpt2-large snapshot is incomplete")
    siblings = [path for path in snapshot.parent.iterdir() if path.is_dir()]
    if siblings != [snapshot]:
        raise ValueError(f"gpt2-large snapshot identity is not unique: {siblings}")
    return {"model_id": "gpt2-large", "snapshot": str(snapshot), "revision": snapshot.name,
            "files": {name: {"sha256": sha256_file(snapshot / name),
                              "bytes": (snapshot / name).stat().st_size} for name in required},
            "metrics_utils_sha256": sha256_file(ROOT / "src/utils/metrics_utils.py"),
            "function": "utils.metrics_utils.Metrics.record_generative_perplexity",
            "entropy_name": "token_frequency_entropy", "dtype": "bfloat16", "device": "cuda",
            "torch_version": torch.__version__}


def checkpoint_gate(path, group):
    before = sha256_file(path)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    required = ("params", "ema_params1", "opt_state", "lr_scheduler", "dropout_rng", "step")
    missing = [key for key in required if key not in ckpt]
    if missing or int(ckpt.get("step", -1)) != 80_000 or not ckpt.get("ema_params1"):
        raise ValueError(f"{group} checkpoint identity failure: missing={missing}, step={ckpt.get('step')}")
    nonfinite = 0
    for state_name in ("params", "ema_params1"):
        for value in ckpt[state_name].values():
            if torch.is_tensor(value) and value.is_floating_point() and not bool(torch.isfinite(value).all()):
                nonfinite += 1
    del ckpt
    if nonfinite or sha256_file(path) != before:
        raise ValueError(f"{group} checkpoint nonfinite/mutated")
    return {"path": str(path), "sha256": before, "global_presentations": 80_000,
            "optimizer_steps": 10_000, "ema_present": True, "finite": True}


def sampling_object(protocol):
    return type("Sampling", (), {"sampling_method": "sde", "sde_gamma": 1.5,
        "plan_trajectory": protocol["trajectory"], "plan_lead_alpha": protocol["alpha"],
        "plan_cfg_scale": 1.0})()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, choices=(4, 1000), required=True)
    parser.add_argument("--smoke-ids")
    parser.add_argument("--identity-audit-only", action="store_true")
    parser.add_argument("--work-dir")
    args = parser.parse_args()
    spec = condition(args.condition_id); group = spec["model_group"]; protocol = plan_protocol(spec)
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    shape_manifest = SHAPE_DIR / "manifest.json"; shape_data = SHAPE_DIR / "shapes.jsonl"
    rows = validate_shape_inputs(SPLIT, shape_manifest, shape_data)
    if args.samples == 4:
        wanted = json.loads(Path(args.smoke_ids).read_text())["eval_ids"]
        by_id = {row["eval_id"]: row for row in rows}; rows = [by_id[item] for item in wanted]
    else:
        rows = rows[:1000]
    run = RUNS[group]; checkpoint_path = run / "checkpoint_10000"
    cfg = load_config_from_yaml(str(run / "config.yml"))
    if getattr(cfg, "group_mode", None) != group:
        raise ValueError(f"resolved group_mode mismatch: {getattr(cfg, 'group_mode', None)} != {group}")
    if group == "vanilla" and int(cfg.num_plan_slots) != 0:
        raise ValueError("Vanilla config retains plan slots")
    if group == "register" and not bool(getattr(cfg, "plan_register_only", False)):
        raise ValueError("Register config is not register-only")
    log_event(args.condition_id, "PRECHECK", f"config={run/'config.yml'} group={group}")
    log_event(args.condition_id, "CHECKPOINT", f"path={checkpoint_path} begin CPU identity gate")
    checkpoint_lock = checkpoint_gate(checkpoint_path, group)
    log_event(args.condition_id, "PASS", f"checkpoint_sha256={checkpoint_lock['sha256']} EMA=true")
    log_event(args.condition_id, "GPT2_LOCK", f"snapshot={GPT2_SNAPSHOT}")
    gpt_lock = artifact_lock()
    log_event(args.condition_id, "PASS", f"gpt2_revision={gpt_lock['revision']}")
    if args.identity_audit_only:
        print(json.dumps({"condition": spec, "checkpoint": checkpoint_lock, "gpt2": gpt_lock}, indent=2)); return

    stage = Path(args.work_dir) if args.work_dir else out.parent / f"{out.name}.work.{os.getpid()}"
    stage.mkdir(parents=True, exist_ok=bool(args.work_dir)); start = time.time(); device = torch.device("cuda")
    log_event(args.condition_id, "LOAD", f"config/model/T5 tokenizer; device={device}")
    model, encoder, tokenizer, resolved = load_model_and_encoder(cfg, str(checkpoint_path), device, load_encoder=False)
    if encoder is not None:
        raise AssertionError("generation unexpectedly loaded T5 thinking encoder")
    # The Oracle loader has already applied ema_params1; absence was rejected by checkpoint_gate.
    log_event(args.condition_id, "LOAD", "EMA loaded; unused thinking encoder disabled")
    sampling = sampling_object(protocol); records = []; noise_hashes = {}; truth = []
    texts = []
    with torch.inference_mode():
        for index, row in enumerate(rows):
            k, length = int(row["K"]), int(row["response_length"]); response_width = int(cfg.max_length)
            if length > response_width:
                raise ValueError(f"response_length={length} exceeds checkpoint width={response_width}")
            token_gen = torch.Generator().manual_seed(int(row["token_noise_seed"]))
            plan_gen = torch.Generator().manual_seed(int(row["plan_noise_seed"]))
            token = torch.zeros((1, response_width, model.text_encoder_dim))
            token[:, :length] = (torch.randn((1, length, model.text_encoder_dim), generator=token_gen)
                                * cfg.denoiser_noise_scale)
            token_mask = (torch.arange(response_width, device=device).unsqueeze(0) < length)
            plan = plan_mask = None
            if protocol["plan_enabled"]:
                plan = torch.randn((1, k, model.plan_latent_dim), generator=plan_gen) * cfg.denoiser_noise_scale
                plan_mask = torch.ones((1, k), dtype=torch.bool, device=device)
            t_plan_probe = None if plan is None else torch.zeros((1,), device=device)
            assert_truth_inputs(group, token.to(device), token_mask, None if plan is None else plan.to(device), plan_mask, t_plan_probe)
            noise_hashes[row["eval_id"]] = {"response_initial_noise_sha256": tensor_sha256(token),
                "plan_initial_noise_sha256": None if plan is None else tensor_sha256(plan)}
            torch.manual_seed(int(row["sampling_seed"])); torch.cuda.manual_seed_all(int(row["sampling_seed"]))
            steps = get_sampling_steps(spec["steps"], "logit_normal", cfg.denoiser_p_mean,
                                       cfg.denoiser_p_std, device=device, dtype=next(model.parameters()).dtype)
            counter = {}; plan_trace = []; response_state_trace = []
            forward_inputs = []
            hook = None
            if args.samples == 4:
                def capture_forward(module, positional, kwargs):
                    x_plan = kwargs.get("x_plan"); pmask = kwargs.get("plan_mask")
                    item = {"token_sha256": tensor_sha256(positional[0]),
                            "token_shape": list(positional[0].shape),
                            "t": [float(x) for x in positional[1].detach().float().cpu()],
                            "plan_present": x_plan is not None,
                            "plan_sha256": None if x_plan is None else tensor_sha256(x_plan),
                            "plan_shape": None if x_plan is None else list(x_plan.shape),
                            "plan_mask_sha256": None if pmask is None else tensor_sha256(pmask),
                            "t_plan": None if kwargs.get("t_plan") is None else
                                [float(x) for x in kwargs["t_plan"].detach().float().cpu()]}
                    if group == "vanilla" and any(kwargs.get(name) is not None for name in ("x_plan","plan_mask","t_plan")):
                        raise AssertionError("Vanilla forward received plan inputs")
                    forward_inputs.append(item)
                hook = model.register_forward_pre_hook(capture_forward, with_kwargs=True)
            latent, final_plan, recorded = _generate_samples_single_batch(
                model, token_gen, token.to(device), steps, None, None, cfg, sampling, 1.0, 3.0,
                record_plan=True, plan_mask=plan_mask,
                initial_plan_noise=None if plan is None else plan.to(device), nfe_counter=counter,
                response_attention_mask=token_mask, response_state_trace=response_state_trace)
            if hook is not None: hook.remove()
            if group == "vanilla" and (final_plan is not None or recorded):
                raise AssertionError("Vanilla created a plan region")
            if group == "register" or (group == "ordered" and spec["alpha"] == 0.0):
                assert_fixed_plan(plan.to(device), recorded, args.condition_id)
                if any(item["plan_sha256"] != tensor_sha256(plan) for item in forward_inputs):
                    raise AssertionError("fixed plan did not reach every real model forward")
            actual = int(counter.get("model_forwards", 0))
            if actual != spec["steps"]:
                raise AssertionError(f"actual NFE {actual} != {spec['steps']}")
            ids = _dlm_decode_batch(latent, model, steps[-1].item(), cfg, 3.0,
                x_plan=final_plan, t_plan_decode_val=protocol["endpoint"],
                plan_trajectory=protocol["trajectory"], plan_mask=plan_mask,
                attention_mask=token_mask)
            eos = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
            pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            raw_ids = ids.clone(); ids = mask_after_eos(ids, eos, pad)
            ids = torch.where(token_mask, ids, torch.full_like(ids, pad))
            text = tokenizer.decode(ids[0].cpu().numpy(), skip_special_tokens=True); texts.append(text)
            generated_length = int((ids != pad).sum())
            if generated_length > length:
                raise AssertionError(f"generated_length={generated_length} exceeds response_length={length}")
            records.append({"eval_id": row["eval_id"], "condition_id": args.condition_id,
                "model_group": group, "alpha": spec["alpha"], "native_mode": group if spec["alpha"] is None else None,
                "configured_nfe": spec["steps"], "actual_forward_count": actual,
                "nonempty": bool(text.strip()), "eos": bool(((raw_ids == eos) & token_mask).any()),
                "generated_length": generated_length, "K": 0 if group == "vanilla" else k,
                **noise_hashes[row["eval_id"]]})
            if args.samples == 4:
                truth.append({"eval_id": row["eval_id"], "token_shape": list(token.shape),
                    "token_hash": tensor_sha256(token), "token_mask_shape": list(token_mask.shape),
                    "plan_present": plan is not None, "plan_shape": None if plan is None else list(plan.shape),
                    "plan_hash": None if plan is None else tensor_sha256(plan),
                    "plan_mask_shape": None if plan_mask is None else list(plan_mask.shape),
                    "token_plan_boundary": [response_width, None if plan is None else k],
                    "plan_endpoint": protocol["endpoint"], "actual_forward_count": actual,
                    "response_padding_max_abs_by_step": response_state_trace,
                    "forward_inputs": forward_inputs})
            elapsed=time.time()-start; done=index+1
            if args.samples == 4 or done % 10 == 0 or done == len(rows):
                log_event(args.condition_id, "GENERATE", f"{done}/{len(rows)} elapsed={elapsed:.1f}s samples_per_sec={done/max(elapsed,1e-9):.4f} eta={(len(rows)-done)*elapsed/done:.1f}s gpu={torch.cuda.current_device()}")

    valid_indices = [i for i, row in enumerate(records) if row["nonempty"]]
    log_event(args.condition_id, "SCORE", f"loading GPT-2 Large; valid={len(valid_indices)}/{len(records)}")
    metrics = Metrics(str(GPT2_SNAPSHOT), 2, 1024)
    score = metrics.record_generative_perplexity([texts[i] for i in valid_indices], 1024, True) if valid_indices else None
    if score:
        for pos, sample_index in enumerate(valid_indices):
            row = records[sample_index]; count = int(score["per_sample_token_count"][pos]); nll = float(score["per_sample_nll_sum"][pos])
            row.update({"gpt2_token_count": count, "gpt2_nll_sum": nll,
                "token_normalized_nll": nll / count if count else math.nan,
                "token_frequency_entropy": float(score["per_sample_token_frequency_entropy"][pos])})
    for row in records:
        for key in ("gpt2_token_count", "gpt2_nll_sum", "token_normalized_nll", "token_frequency_entropy"):
            row.setdefault(key, None)
    per_sample = stage / "per_sample.jsonl"
    with per_sample.open("w") as f:
        for row in records: f.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    (stage / "noise_hashes.json").write_text(json.dumps(noise_hashes, indent=2, sort_keys=True) + "\n")
    (stage / "progress.jsonl").write_text("".join(json.dumps({"eval_id": r["eval_id"], "complete": True}) + "\n" for r in records))
    if args.samples == 4: (stage / "runtime_truth_gate.json").write_text(json.dumps(truth, indent=2, sort_keys=True) + "\n")
    scored = [r for r in records if r["gpt2_token_count"]]
    total_nll = sum(r["gpt2_nll_sum"] for r in scored); total_tokens = sum(r["gpt2_token_count"] for r in scored)
    summary = {"complete": True, "condition_id": args.condition_id, "num_samples": len(records),
        "valid_samples": len(scored), "nonempty_rate": sum(r["nonempty"] for r in records)/len(records),
        "eos_rate": sum(r["eos"] for r in records)/len(records),
        "mean_length": statistics.mean(r["generated_length"] for r in records),
        "median_length": statistics.median(r["generated_length"] for r in records),
        "total_gpt2_nll": total_nll, "total_gpt2_scored_tokens": total_tokens,
        "corpus_gen_ppl": math.exp(total_nll/total_tokens) if total_tokens else None,
        "mean_token_frequency_entropy": statistics.mean(r["token_frequency_entropy"] for r in scored) if scored else None,
        "wall_seconds": time.time()-start, "actual_nfe_set": sorted({r["actual_forward_count"] for r in records}),
        "ema_loaded": True, "checkpoint_sha256": checkpoint_lock["sha256"],
        "gold_text_or_latent_used": False, "fixed_gold_derived_shapes_used": True}
    (stage / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    config_lock = {"condition": spec, "protocol": protocol, "checkpoint": checkpoint_lock,
        "split_manifest_sha256": sha256_file(SPLIT), "shape_manifest_sha256": sha256_file(shape_manifest),
        "shape_data_sha256": sha256_file(shape_data), "gpt2": gpt_lock,
        "generation_function": "utils.generation_utils._generate_samples_single_batch",
        "solver_function": "utils.sampling_utils._ode_step/_sde_step",
        "decode_function": "utils.generation_utils._dlm_decode_batch"}
    (stage / "config_lock.json").write_text(json.dumps(config_lock, indent=2, sort_keys=True) + "\n")
    (stage / "exit_code").write_text("0\n")
    log_event(args.condition_id, "WRITE", f"staging={stage}")
    outputs = {}
    for path in sorted(stage.iterdir()):
        if path.name not in ("manifest.json", "sha256sums.txt", "worker.log"):
            outputs[path.name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    manifest = {**summary, "model_group": group, "shape_manifest_sha256": sha256_file(shape_manifest),
        "shape_data_sha256": sha256_file(shape_data), "gpt2_input_lock": hashlib.sha256(json.dumps(gpt_lock, sort_keys=True).encode()).hexdigest(),
        "outputs": outputs}
    (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sums = [f"{sha256_file(path)}  {path.name}" for path in sorted(stage.iterdir()) if path.name not in ("sha256sums.txt", "worker.log")]
    (stage / "sha256sums.txt").write_text("\n".join(sums) + "\n")
    os.rename(stage, out)
    log_event(args.condition_id, "DONE", f"exit=0 output={out}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        condition = "unknown"
        if "--condition-id" in sys.argv:
            condition = sys.argv[sys.argv.index("--condition-id") + 1]
        log_event(condition, "FAIL", f"{type(exc).__name__}: {exc}")
        raise
