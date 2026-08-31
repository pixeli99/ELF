#!/usr/bin/env python3
"""Survey the HRM-Text io-cleaned corpus: which sources give usable (question, reasoning, answer) triples.

Every row is {condition: "cot"|"direct", instruction, response}. The same instruction
appears once with each condition, so pairing them on the instruction yields the triple
this project needs: the question, the reasoning a chain-of-thought model would emit, and
the short answer it ends on.

A source is usable here only if its direct response really is a short verifiable answer
rather than a second paragraph of prose, so the survey reports the answer-length
distribution and how often the answer is a bare number.

    python tools/survey_hrm.py --out data/hrm_survey.json
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

HRM = Path("/jfs/auto.prod.sz/data/ann/vlf/data/pengxiang.li/HRM-Text-data-io-cleaned-20260515/data")
NUMBER = re.compile(r"^-?[\d,]+(?:\.\d+)?$")
BOXED = re.compile(r"\\boxed")


def percentiles(values, points=(50, 90, 99)):
    if not values:
        return {}
    values = sorted(values)
    return {f"p{p}": values[min(len(values) - 1, int(len(values) * p / 100))] for p in points}


def survey_file(path):
    """One pass, keeping only what pairing and the length summaries need."""
    cot_len, direct_text = {}, {}
    counts = Counter()
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["bad_json"] += 1
                continue
            condition = row.get("condition") or ""
            instruction = row.get("instruction")
            response = row.get("response")
            if not instruction or response is None:
                counts["missing_field"] += 1
                continue
            counts["rows"] += 1
            if "cot" in condition:
                counts["cot"] += 1
                cot_len.setdefault(instruction, len(response))
            else:
                counts["direct"] += 1
                direct_text.setdefault(instruction, response)

    answer_chars, reasoning_chars = [], []
    numeric = boxed = 0
    for instruction, answer in direct_text.items():
        reasoning = cot_len.get(instruction)
        if reasoning is None:
            continue
        counts["paired"] += 1
        answer_chars.append(len(answer))
        reasoning_chars.append(reasoning)
        stripped = answer.strip()
        numeric += bool(NUMBER.match(stripped))
        boxed += bool(BOXED.search(stripped))

    paired = max(counts["paired"], 1)
    return {
        "rows": counts["rows"], "cot": counts["cot"], "direct": counts["direct"],
        "unique_cot": len(cot_len), "unique_direct": len(direct_text),
        "paired": counts["paired"],
        "pair_rate_of_cot": round(counts["paired"] / max(len(cot_len), 1), 4),
        "answer_is_bare_number": round(numeric / paired, 4),
        "answer_has_boxed": round(boxed / paired, 4),
        "answer_chars": percentiles(answer_chars),
        "reasoning_chars": percentiles(reasoning_chars),
        "bad_json": counts["bad_json"], "missing_field": counts["missing_field"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="*", default=None, help="basenames under the HRM data dir")
    parser.add_argument("--out", default="data/hrm_survey.json")
    args = parser.parse_args()

    paths = ([HRM / name for name in args.files] if args.files
             else sorted(p for p in HRM.glob("*.jsonl")))
    report = {}
    for path in paths:
        print(f"surveying {path.name} ({path.stat().st_size / 1e6:.0f} MB)...", flush=True)
        report[path.stem] = survey_file(path)
        print(f"  {json.dumps(report[path.stem])}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")
    header = f"{'source':24s} {'paired':>9s} {'pair%':>7s} {'num%':>6s} {'boxed%':>7s} {'ans p50':>8s} {'reason p50':>11s}"
    print("\n" + header)
    for name, r in sorted(report.items(), key=lambda kv: -kv[1]["paired"]):
        print(f"{name:24s} {r['paired']:9d} {100 * r['pair_rate_of_cot']:6.1f}% "
              f"{100 * r['answer_is_bare_number']:5.1f}% {100 * r['answer_has_boxed']:6.1f}% "
              f"{r['answer_chars'].get('p50', 0):8d} {r['reasoning_chars'].get('p50', 0):11d}")


if __name__ == "__main__":
    main()
