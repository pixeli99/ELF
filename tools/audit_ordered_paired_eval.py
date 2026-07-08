#!/usr/bin/env python
"""Audit paired Ordered-ELF trajectory evaluation outputs."""

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
    }


def count_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_last_jsonl(path):
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last or {}


def audit(output_dir, log_path, expected_samples):
    rows = []
    out = Path(output_dir)
    for run_dir in sorted(p for p in out.iterdir() if p.is_dir()) if out.exists() else []:
        parsed = parse_run_name(run_dir.name)
        if parsed is None:
            continue
        generated_files = sorted(run_dir.glob("all_generated_*.jsonl"))
        generated_file = generated_files[-1] if generated_files else None
        metrics_file = run_dir / "metrics.jsonl"
        metrics = read_last_jsonl(metrics_file) if metrics_file.exists() else {}
        row = {
            **parsed,
            "num_samples": count_jsonl(generated_file) if generated_file else 0,
            "gen_ppl": metrics.get("ppl", ""),
            "entropy": metrics.get("mean_entropy", ""),
            "generated_file": str(generated_file) if generated_file else "",
            "metrics_file": str(metrics_file) if metrics_file.exists() else "",
            "metrics_complete": bool(
                metrics_file.exists()
                and "ppl" in metrics
                and "mean_entropy" in metrics
                and int(metrics.get("step", -1)) == 152592
            ),
        }
        rows.append(row)
    rows.sort(key=lambda r: (int(r["steps"]), str(r["trajectory"]), float(r["alpha"] or 0.0)))

    log_text = Path(log_path).read_text(encoding="utf-8", errors="replace") if Path(log_path).exists() else ""
    status = {
        "configs_found": len(rows),
        "all_1000_samples": len(rows) == 8 and all(r["num_samples"] == expected_samples for r in rows),
        "metrics_complete": len(rows) == 8 and all(r["metrics_complete"] for r in rows),
        "paired_trajectory_eval_enabled": "paired_trajectory_eval=True" in log_text,
        "strict_null_decode_enabled": "Strict null decode enabled" in log_text,
        "checkpoint_152592_in_log": "checkpoint_152592" in log_text,
    }
    return rows, status


def write_report(path, status):
    verdict = "PASS" if (
        status["configs_found"] == 8
        and status["all_1000_samples"]
        and status["metrics_complete"]
        and status["paired_trajectory_eval_enabled"]
        and status["strict_null_decode_enabled"]
        and status["checkpoint_152592_in_log"]
    ) else "CHECK_REQUIRED"
    lines = [
        "# Ordered Paired Eval Audit",
        "",
        f"Eval verdict: {verdict}",
        "",
        f"- 8/8 configs found: {status['configs_found'] == 8} ({status['configs_found']}/8)",
        f"- all expected samples: {status['all_1000_samples']}",
        f"- metrics complete: {status['metrics_complete']}",
        f"- paired_trajectory_eval enabled: {status['paired_trajectory_eval_enabled']}",
        f"- strict null decode enabled: {status['strict_null_decode_enabled']}",
        f"- checkpoint_152592 in log: {status['checkpoint_152592_in_log']}",
        "",
        "Caveats:",
        "",
        "- Lagging alpha=0.5 follows t_plan=0.5*t_tok during denoising, but current intended eval decodes normal trajectories at t_plan=1.",
        "- Null is not a normal endpoint trajectory; it remains pure-noise plan with t_plan=0 through decode.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", default="outputs/eval_ordered_5b_4g_paired_full_seed42_b32")
    parser.add_argument("--log", default="logs/eval_ordered_5b_4g/eval_paired_full_seed42_b32.log")
    parser.add_argument("--csv", default="results/ordered_5b_4g/eval_paired_trajectories_summary.csv")
    parser.add_argument("--report", default="results/ordered_5b_4g/eval_paired_audit_report.md")
    parser.add_argument("--expected_samples", type=int, default=1000)
    args = parser.parse_args()

    rows, status = audit(args.output_dir, args.log, args.expected_samples)
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trajectory", "alpha", "steps", "num_samples", "gen_ppl", "entropy",
        "generated_file", "metrics_file",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    write_report(Path(args.report), status)

    print(f"configs_found={status['configs_found']}")
    print(f"all_1000_samples={status['all_1000_samples']}")
    print(f"metrics_complete={status['metrics_complete']}")
    print(f"paired_trajectory_eval_enabled={status['paired_trajectory_eval_enabled']}")
    print(f"strict_null_decode_enabled={status['strict_null_decode_enabled']}")
    print(f"wrote_csv={csv_path}")
    print(f"wrote_report={args.report}")


if __name__ == "__main__":
    main()
