#!/usr/bin/env python
"""Paired comparison of two schedule-matched ELF training logs.

The formal conditional runs log one aggregate every fixed number of microsteps.
When two runs consume the same schedule and RNG streams, matching by microstep
removes batch-difficulty variation and exposes the training cost of an
intervention. This is descriptive trajectory evidence, not an IID significance
test: adjacent intervals share an evolving model state.
"""

import argparse
import ast
import hashlib
import json
import math
import re
from pathlib import Path


RECORD_RE = re.compile(r"INFO - __main__ - (\{[^\r\n]+\})")
FIELDS = ("loss", "l2", "ce", "plan", "grad", "response_tokens", "plan_slots")


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_log(path):
    records = {}
    text = path.read_text(errors="replace")
    for match in RECORD_RE.finditer(text):
        raw = ast.literal_eval(match.group(1))
        step = int(raw["step"])
        record = {key: float(raw[key]) for key in FIELDS}
        if "mediated_rows" in raw:
            record["mediated_rows"] = int(raw["mediated_rows"])
        if step in records and records[step] != record:
            raise ValueError(f"conflicting duplicate log record at step {step}: {path}")
        records[step] = record
    if not records:
        raise ValueError(f"no training records found in {path}")
    return records


def _mean(values):
    return sum(values) / len(values)


def _pearson(xs, ys):
    x_mean, y_mean = _mean(xs), _mean(ys)
    dx = [value - x_mean for value in xs]
    dy = [value - y_mean for value in ys]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    return None if denom == 0 else sum(a * b for a, b in zip(dx, dy)) / denom


def _summary(rows):
    return {
        "intervals": len(rows),
        "mean_branch_minus_baseline": {
            key: _mean([row[f"delta_{key}"] for row in rows])
            for key in ("total_loss", "response_loss", "plan_loss", "gradient_norm")
        },
    }


def compare_logs(baseline_path, branch_path, start_step, end_step, rows_per_interval):
    baseline = parse_log(baseline_path)
    branch = parse_log(branch_path)
    steps = sorted(
        step for step in baseline.keys() & branch.keys()
        if step >= start_step and (end_step is None or step <= end_step)
    )
    if not steps:
        raise ValueError("the logs have no overlapping records in the requested range")
    expected = list(range(steps[0], steps[-1] + 1, rows_per_interval))
    if steps != expected:
        missing = sorted(set(expected) - set(steps))
        raise ValueError(f"non-contiguous paired intervals; first missing steps: {missing[:8]}")
    if end_step is not None and steps[-1] != end_step:
        raise ValueError(f"requested end_step={end_step}, latest paired step is {steps[-1]}")

    rows = []
    for step in steps:
        left, right = baseline[step], branch[step]
        if int(left["response_tokens"]) != int(right["response_tokens"]):
            raise ValueError(f"response-token mismatch at step {step}")
        if int(left["plan_slots"]) != int(right["plan_slots"]):
            raise ValueError(f"plan-slot mismatch at step {step}")
        mediated = right.get("mediated_rows")
        if mediated is None:
            raise ValueError("branch log does not contain mediated_rows")
        if not 0 <= mediated <= rows_per_interval:
            raise ValueError(f"invalid mediated_rows={mediated} at step {step}")
        left_response = left["l2"] + left["ce"]
        right_response = right["l2"] + right["ce"]
        rows.append({
            "step": step,
            "mediated_rows": mediated,
            "delta_total_loss": right["loss"] - left["loss"],
            "delta_response_loss": right_response - left_response,
            "delta_plan_loss": right["plan"] - left["plan"],
            "delta_gradient_norm": right["grad"] - left["grad"],
        })

    split = max(1, len(rows) // 2)
    low = [row for row in rows if row["mediated_rows"] < rows_per_interval / 2]
    high = [row for row in rows if row["mediated_rows"] >= rows_per_interval / 2]
    response_deltas = [row["delta_response_loss"] for row in rows]
    mediation_counts = [row["mediated_rows"] for row in rows]
    return {
        "scope": "paired_schedule_training_log_diagnostic",
        "interpretation_guardrail": (
            "Descriptive matched-trajectory evidence only; adjacent intervals are not IID."
        ),
        "baseline_log": str(baseline_path),
        "baseline_sha256": _sha256(baseline_path),
        "branch_log": str(branch_path),
        "branch_sha256": _sha256(branch_path),
        "first_microstep": steps[0],
        "last_microstep": steps[-1],
        "rows_per_interval": rows_per_interval,
        "paired_intervals": len(rows),
        "paired_microsteps": len(rows) * rows_per_interval,
        "all_response_token_counts_match": True,
        "all_plan_slot_counts_match": True,
        "mediated_rows": sum(mediation_counts),
        "mediation_rate": sum(mediation_counts) / (len(rows) * rows_per_interval),
        "overall": _summary(rows),
        "first_half": _summary(rows[:split]),
        "second_half": _summary(rows[split:]),
        "low_mediation_intervals": _summary(low) if low else None,
        "high_mediation_intervals": _summary(high) if high else None,
        "mediated_count_response_delta_pearson": _pearson(mediation_counts, response_deltas),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--branch", required=True, type=Path)
    parser.add_argument("--start-step", required=True, type=int)
    parser.add_argument("--end-step", type=int)
    parser.add_argument("--rows-per-interval", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.rows_per_interval <= 0:
        parser.error("--rows-per-interval must be positive")
    payload = compare_logs(
        args.baseline, args.branch, args.start_step, args.end_step,
        args.rows_per_interval,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
