#!/usr/bin/env python
"""Audit Ordered-ELF trajectory evaluation outputs.

This script is intentionally read-only with respect to model/checkpoint/training
files. It scans generated JSONL files and metrics JSONL files, optionally checks
the eval log, and writes a compact CSV summary.
"""

import argparse
import csv
import json
import re
from pathlib import Path


RUN_RE = re.compile(
    r"^sde-steps(?P<steps>\d+)-cfg(?P<cfg>[^-]+)-"
    r"(?:(?P<sccfg>sccfg[^-]+)-)?ts_(?P<time_schedule>[^-]+)-"
    r"gamma(?P<gamma>[^-]+)-plan_(?P<plan>.+)-uncond$"
)


def parse_run_name(name):
    m = RUN_RE.match(name)
    if not m:
        return None
    plan = m.group("plan")
    if plan == "diagonal":
        trajectory, alpha = "diagonal", 1.0
    elif plan == "null":
        trajectory, alpha = "null", 0.0
    elif plan.startswith("planning_first_a"):
        alpha = float(plan.split("planning_first_a", 1)[1])
        trajectory = "planning_first" if alpha >= 1.0 else "lagging"
    else:
        trajectory, alpha = plan, ""
    return {
        "trajectory": trajectory,
        "alpha": alpha,
        "steps": int(m.group("steps")),
        "cfg": m.group("cfg"),
        "self_cond_cfg": (m.group("sccfg") or "").replace("sccfg", ""),
        "time_schedule": m.group("time_schedule"),
        "sde_gamma": m.group("gamma"),
    }


def count_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def parse_log(log_path):
    if not log_path or not log_path.exists():
        return {}, {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    info = {
        "checkpoint_152592_in_log": "checkpoint_152592" in text,
        "sampling_configs_4_in_log": "Sampling configs: 4 config(s)" in text,
        "num_samples_1000_in_log": "Num samples: 1000" in text,
        "ppl_batch_size_4_in_log": "PPL: batch_size=4" in text,
    }
    run_metrics = {}
    current_dir = None
    for line in text.splitlines():
        saved = re.search(r"Saved\s+\d+\s+generated texts to\s+(.+/all_generated_[^ ]+\.jsonl)", line)
        if saved:
            current_dir = str(Path(saved.group(1)).parent)
            run_metrics.setdefault(current_dir, {})
            continue
        ppl = re.search(r"Perplexity:\s+([0-9.]+)", line)
        if ppl and current_dir:
            run_metrics.setdefault(current_dir, {})["log_gen_ppl"] = float(ppl.group(1))
            continue
        ent = re.search(r"Mean Entropy:\s+([0-9.]+)", line)
        if ent and current_dir:
            run_metrics.setdefault(current_dir, {})["log_entropy"] = float(ent.group(1))
            current_dir = None
    return info, run_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/eval_ordered_5b_4g_full_seed42_b32")
    parser.add_argument("--log", default="logs/eval_ordered_5b_4g/eval_full_seed42_b32.log")
    parser.add_argument("--csv", default="results/ordered_5b_4g/eval_audit_summary.csv")
    parser.add_argument("--expected_samples", type=int, default=1000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    log_path = Path(args.log)
    log_info, log_metrics = parse_log(log_path)

    rows = []
    for run_dir in sorted(p for p in out_dir.iterdir() if p.is_dir()):
        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            continue
        generated_files = sorted(run_dir.glob("all_generated_*.jsonl"))
        metrics_file = run_dir / "metrics.jsonl"
        generated_file = generated_files[-1] if generated_files else None
        metrics_rows = read_jsonl(metrics_file) if metrics_file.exists() else []
        metrics = metrics_rows[-1] if metrics_rows else {}
        run_key_rel = str(run_dir)
        run_key_abs = str(run_dir.resolve())
        lm = log_metrics.get(run_key_rel) or log_metrics.get(run_key_abs) or {}
        gen_ppl = metrics.get("ppl", "")
        entropy = metrics.get("mean_entropy", "")
        log_gen_ppl = lm.get("log_gen_ppl", "")
        log_entropy = lm.get("log_entropy", "")
        row = {
            **parsed,
            "num_samples": count_jsonl(generated_file) if generated_file else 0,
            "gen_ppl": gen_ppl,
            "entropy": entropy,
            "log_gen_ppl": log_gen_ppl,
            "log_entropy": log_entropy,
            "metrics_lines": len(metrics_rows),
            "metrics_complete": bool(metrics_rows and "ppl" in metrics and "mean_entropy" in metrics),
            "log_matches_metrics": (
                log_gen_ppl == "" or (
                    abs(float(gen_ppl) - float(log_gen_ppl)) < 5e-4
                    and abs(float(entropy) - float(log_entropy)) < 5e-4
                )
            ),
            "generated_file": str(generated_file) if generated_file else "",
            "metrics_file": str(metrics_file) if metrics_file.exists() else "",
            "output_dir": str(run_dir),
        }
        rows.append(row)

    rows.sort(key=lambda r: (int(r["steps"]), str(r["trajectory"]), float(r["alpha"] or 0.0)))
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trajectory", "alpha", "steps", "num_samples", "gen_ppl", "entropy",
        "log_gen_ppl", "log_entropy", "metrics_lines", "metrics_complete",
        "log_matches_metrics", "generated_file", "metrics_file", "output_dir",
        "cfg", "self_cond_cfg", "time_schedule", "sde_gamma",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    all_1000 = all(r["num_samples"] == args.expected_samples for r in rows)
    all_metrics = all(r["metrics_complete"] for r in rows)
    all_log_match = all(r["log_matches_metrics"] for r in rows)
    print(f"rows={len(rows)}")
    print(f"all_expected_samples={all_1000}")
    print(f"all_metrics_complete={all_metrics}")
    print(f"all_log_matches_metrics={all_log_match}")
    for key, value in log_info.items():
        print(f"{key}={value}")
    print(f"wrote={csv_path}")


if __name__ == "__main__":
    main()
