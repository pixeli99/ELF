#!/usr/bin/env python3
"""Paired ROUGE comparison for two conditional generation JSONL artifacts."""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from rouge_score import rouge_scorer


ROOT = Path(__file__).resolve().parents[1]
METRICS = ("rouge1", "rouge2", "rougeL")


def load_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    keyed = {row["eval_id"]: row for row in rows}
    if len(keyed) != len(rows):
        raise ValueError(f"duplicate eval_id in {path}")
    return rows, keyed


def load_summary(path):
    return json.loads(Path(str(path) + ".summary.json").read_text())


def paired_bootstrap(delta, *, seed=42, draws=10_000):
    values = np.asarray(delta, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("paired deltas must be a finite nonempty vector")
    rng = np.random.default_rng(seed)
    means = np.empty(draws, dtype=np.float64)
    for start in range(0, draws, 250):
        stop = min(draws, start + 250)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return {
        "mean_delta": float(values.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "bootstrap_probability_gt_zero": float((means > 0).mean()),
        "paired_sample_win_rate": float((values > 0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-name")
    parser.add_argument("--right-name")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=10_000)
    parser.add_argument("--out")
    args = parser.parse_args()

    left_rows, left = load_rows(args.left)
    right_rows, right = load_rows(args.right)
    left_ids = [row["eval_id"] for row in left_rows]
    right_ids = [row["eval_id"] for row in right_rows]
    if left_ids != right_ids:
        raise ValueError("artifacts must have identical eval_id order")
    for eval_id in left_ids:
        for key in ("prompt_tokens", "response_token_budget"):
            if left[eval_id].get(key) != right[eval_id].get(key):
                raise ValueError(f"paired shape mismatch for {eval_id}: {key}")

    left_summary, right_summary = load_summary(args.left), load_summary(args.right)
    protocol_keys = (
        "samples", "seed", "nfe", "weights", "split", "length_mode",
        "free_response_token_cap", "training_response_token_cap",
        "gold_content_used", "gold_derived_response_length_used",
    )
    def protocol_value(summary, key):
        # Artifacts produced before split routing were validation-only.
        if key == "split":
            return summary.get(key, "validation")
        return summary.get(key)

    mismatches = {
        key: [protocol_value(left_summary, key), protocol_value(right_summary, key)]
        for key in protocol_keys
        if protocol_value(left_summary, key) != protocol_value(right_summary, key)
    }
    if mismatches:
        raise ValueError(f"evaluation protocol mismatch: {mismatches}")

    split = protocol_value(left_summary, "split")
    refs = {
        row["eval_id"]: row["reference_response"]
        for row in pq.read_table(
            ROOT / f"data/conditional_v1/data/{split}_references-00000.parquet"
        ).to_pylist()
    }
    missing = [eval_id for eval_id in left_ids if eval_id not in refs]
    if missing:
        raise ValueError(f"missing references for {len(missing)} eval IDs")

    scorer = rouge_scorer.RougeScorer(list(METRICS), use_stemmer=True)
    per_metric = {metric: {"left": [], "right": []} for metric in METRICS}
    for eval_id in left_ids:
        reference = refs[eval_id]
        left_scores = scorer.score(reference, left[eval_id]["generated"])
        right_scores = scorer.score(reference, right[eval_id]["generated"])
        for metric in METRICS:
            per_metric[metric]["left"].append(left_scores[metric].fmeasure * 100)
            per_metric[metric]["right"].append(right_scores[metric].fmeasure * 100)

    result = {
        "left": args.left_name or left_summary.get("group") or str(args.left),
        "right": args.right_name or right_summary.get("group") or str(args.right),
        "sample_count": len(left_ids),
        "protocol": {key: protocol_value(left_summary, key) for key in protocol_keys},
        "metrics": {},
        "auxiliary_metrics": {},
    }
    for metric, values in per_metric.items():
        left_values = np.asarray(values["left"])
        right_values = np.asarray(values["right"])
        result["metrics"][metric] = {
            "left_mean": float(left_values.mean()),
            "right_mean": float(right_values.mean()),
            **paired_bootstrap(
                left_values - right_values, seed=args.seed, draws=args.draws
            ),
        }

    auxiliary = {
        "generated_chars": {
            "direction": "lower_is_shorter_not_necessarily_better",
            "left": [len(left[eval_id]["generated"]) for eval_id in left_ids],
            "right": [len(right[eval_id]["generated"]) for eval_id in left_ids],
        },
        "absolute_reference_char_length_error": {
            "direction": "lower_is_better",
            "left": [abs(len(left[eval_id]["generated"]) - len(refs[eval_id]))
                     for eval_id in left_ids],
            "right": [abs(len(right[eval_id]["generated"]) - len(refs[eval_id]))
                      for eval_id in left_ids],
        },
        "eos_emitted": {
            "direction": "higher_is_better_for_free_length",
            "left": [float(left[eval_id]["eos_emitted"]) for eval_id in left_ids],
            "right": [float(right[eval_id]["eos_emitted"]) for eval_id in left_ids],
        },
        "dominant_token_fraction": {
            "direction": "lower_is_less_repetitive",
            "left": [float(left[eval_id]["dominant_token_fraction"]) for eval_id in left_ids],
            "right": [float(right[eval_id]["dominant_token_fraction"]) for eval_id in left_ids],
        },
    }
    for metric, values in auxiliary.items():
        left_values = np.asarray(values["left"])
        right_values = np.asarray(values["right"])
        result["auxiliary_metrics"][metric] = {
            "direction": values["direction"],
            "left_mean": float(left_values.mean()),
            "right_mean": float(right_values.mean()),
            **paired_bootstrap(
                left_values - right_values, seed=args.seed, draws=args.draws
            ),
        }

    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
