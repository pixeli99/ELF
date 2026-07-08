#!/usr/bin/env python
"""Collect Ordered-ELF trajectory eval metrics into one CSV."""

import argparse
import csv
import json
import re
from pathlib import Path


RUN_RE = re.compile(r"(?:^|-)steps(?P<steps>\d+)-.*-plan_(?P<plan>[^-]+)-uncond$")


def parse_run_dir(path: Path):
    name = path.name
    steps_match = re.search(r"(?:^|-)steps(\d+)-", name)
    steps = int(steps_match.group(1)) if steps_match else None

    if "-plan_null-" in name:
        return "null", 0.0, steps
    if "-plan_diagonal-" in name:
        return "diagonal", 1.0, steps

    lead_match = re.search(r"-plan_planning_first_a([0-9.]+)-", name)
    if lead_match:
        alpha = float(lead_match.group(1))
        trajectory = "planning_first" if alpha >= 1.0 else "lagging"
        return trajectory, alpha, steps

    return "unknown", None, steps


def read_last_jsonl(path: Path):
    last = None
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last = json.loads(line)
    return last or {}


def count_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def find_generated_file(run_dir: Path):
    files = sorted(run_dir.glob("all_generated_*.jsonl"))
    return files[-1] if files else None


def collect(input_dir: Path):
    rows = []
    for run_dir in sorted(p for p in input_dir.iterdir() if p.is_dir()):
        trajectory, alpha, steps = parse_run_dir(run_dir)
        if steps is None or trajectory == "unknown":
            continue

        metrics_path = run_dir / "metrics.jsonl"
        generated_file = find_generated_file(run_dir)
        metrics = read_last_jsonl(metrics_path) if metrics_path.exists() else {}

        row = {
            "trajectory": trajectory,
            "alpha": alpha,
            "steps": steps,
            "num_samples": count_jsonl(generated_file) if generated_file else "",
            "gen_ppl": metrics.get("ppl", ""),
            "entropy": metrics.get("mean_entropy", ""),
            "output_dir": str(run_dir),
            "generated_file": str(generated_file) if generated_file else "",
        }
        for key, value in metrics.items():
            if key == "ppl":
                key = "gen_ppl"
            elif key == "mean_entropy":
                key = "entropy"
            row.setdefault(key, value)
        rows.append(row)
    rows.sort(key=lambda r: (int(r["steps"]), str(r["trajectory"]), float(r["alpha"] or 0.0)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", default="outputs/eval_ordered_5b_4g_full")
    parser.add_argument("--output_csv", default="results/ordered_5b_4g/eval_trajectories_summary.csv")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    rows = collect(input_dir)
    if not rows:
        raise SystemExit(f"No ordered eval runs found under {input_dir}")

    fieldnames = []
    required = ["trajectory", "alpha", "steps", "num_samples", "gen_ppl", "entropy", "output_dir", "generated_file"]
    for key in required:
        fieldnames.append(key)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    main()
