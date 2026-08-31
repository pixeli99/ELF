#!/usr/bin/env python3
"""Build the paired (prompt, thinking, answer) math package.

The HRM-Text io-cleaned corpus already carries the split this project needs: the same
question appears twice, once with `condition` containing `cot` and once with `direct`.
The cot row is the reasoning an explicit-chain-of-thought model would have to emit, the
direct row is the final answer alone. Pairing them gives:

    prompt   the question
    thinking the reasoning        -> the plan stream compresses this, it is never emitted
    answer   the final answer     -> the short target the model actually generates

which is what makes the three arms comparable on cost: no-reasoning emits `answer`,
explicit reasoning emits `thinking + answer`, ours emits `answer` with the reasoning in
the plan slots. The previous package put the reasoning inside the response, so every arm
emitted the same long text and there was no cost axis at all.

GSM8K train is direct-only in that corpus, so its pairs are rebuilt from the original
dataset, whose `answer` field is the reasoning followed by `#### N`.

    python tools/build_paired_math.py --out data/paired_math_v1
"""
import argparse, hashlib, json, os, random, re, sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

HRM = Path("/jfs/auto.prod.sz/data/ann/vlf/data/pengxiang.li/HRM-Text-data-io-cleaned-20260515")
GSM8K = Path("/cpfs01/shared/public/users/pengxiang.li/ouro_eval_outputs/hf_datasets/"
             "openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866")
SCHEMA = ["example_id", "source", "prompt", "thinking", "answer"]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/paired_math_v1")
    ap.add_argument("--max-prompt-chars", type=int, default=2000)
    ap.add_argument("--max-thinking-chars", type=int, default=4000,
                    help="~1024 t5 tokens; the plan stream cannot hold more")
    ap.add_argument("--max-answer-chars", type=int, default=120,
                    help="the answer is a final value, not a paragraph")
    ap.add_argument("--numinamath-cap", type=int, default=40000,
                    help="competition math transfers but must not swamp the grade-school mix")
    ap.add_argument("--heldout", type=int, default=1000,
                    help="in-distribution held-out rows: separates a scale floor from a domain shift")
    ap.add_argument("--rows-per-shard", type=int, default=25000)
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def row_id(source, prompt):
    return hashlib.sha256(f"{source}\x00{prompt}".encode()).hexdigest()


def pairs_from_hrm(name, path, stats):
    """Questions present as both a cot row and a direct row."""
    cot, direct = {}, {}
    with open(path) as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats[f"{name}/bad_json"] += 1
                continue
            cond = row.get("condition") or ""
            target = cot if "cot" in cond else direct
            target.setdefault(row["instruction"], row["response"])
    stats[f"{name}/cot_rows"] = len(cot)
    stats[f"{name}/direct_rows"] = len(direct)
    out = []
    for question, reasoning in cot.items():
        answer = direct.get(question)
        if answer is None:
            stats[f"{name}/unpaired"] += 1
            continue
        out.append({"example_id": row_id(name, question), "source": name,
                    "prompt": question, "thinking": reasoning, "answer": answer})
    return out


def pairs_from_gsm8k(stats):
    """GSM8K's answer field is `reasoning ... #### N`: split it at the marker."""
    import datasets
    table = datasets.Dataset.from_file(str(GSM8K / "gsm8k-train.arrow"))
    out = []
    for row in table:
        body = row["answer"]
        if "####" not in body:
            stats["gsm8k_train/no_marker"] += 1
            continue
        reasoning, answer = body.rsplit("####", 1)
        # the <<12*2=24>> calculator markers are annotation, not text the model writes
        reasoning = re.sub(r"<<[^>]*>>", "", reasoning).strip()
        out.append({"example_id": row_id("gsm8k_train", row["question"]),
                    "source": "gsm8k_train", "prompt": row["question"],
                    "thinking": reasoning, "answer": answer.strip()})
    stats["gsm8k_train/cot_rows"] = len(out)
    stats["gsm8k_train/direct_rows"] = len(out)
    return out


def keep(row, args, stats):
    if not row["prompt"].strip() or not row["thinking"].strip() or not row["answer"].strip():
        stats["drop/empty"] += 1
        return False
    if len(row["prompt"]) > args.max_prompt_chars:
        stats["drop/prompt_too_long"] += 1
        return False
    if len(row["thinking"]) > args.max_thinking_chars:
        stats["drop/thinking_too_long"] += 1
        return False
    if len(row["answer"]) > args.max_answer_chars:
        stats["drop/answer_too_long"] += 1
        return False
    if len(row["thinking"]) <= len(row["answer"]):
        stats["drop/thinking_not_longer"] += 1
        return False
    return True


def main():
    import collections
    args = parse_args()
    stats = collections.Counter()
    rng = random.Random(args.seed)

    rows = pairs_from_gsm8k(stats)
    for name in ["math_train", "omnimath", "numinamath"]:
        path = HRM / "data" / f"{name}.jsonl"
        if not path.exists():
            print(f"missing {path}, skipped", file=sys.stderr)
            continue
        found = pairs_from_hrm(name, path, stats)
        if name == "numinamath" and len(found) > args.numinamath_cap:
            rng.shuffle(found)
            stats["numinamath/capped_away"] = len(found) - args.numinamath_cap
            found = found[:args.numinamath_cap]
        rows += found
        print(f"{name}: {len(found)} pairs", flush=True)

    kept = [r for r in rows if keep(r, args, stats)]
    seen, deduped = set(), []
    for r in kept:
        if r["example_id"] in seen:
            stats["drop/duplicate"] += 1
            continue
        seen.add(r["example_id"])
        deduped.append(r)
    rng.shuffle(deduped)

    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    # Held out from the same mix, stratified by source. GSM8K test measures transfer to
    # grade-school problems; this measures the task in-distribution, so a floor on GSM8K
    # can be told apart from a floor everywhere.
    per_source = collections.defaultdict(list)
    for r in deduped:
        per_source[r["source"]].append(r)
    quota = {k: max(1, round(args.heldout * len(v) / len(deduped))) for k, v in per_source.items()}
    heldout_ids = set()
    for source, take in quota.items():
        for r in per_source[source][:take]:
            heldout_ids.add(r["example_id"])
    heldout = [r for r in deduped if r["example_id"] in heldout_ids]
    deduped = [r for r in deduped if r["example_id"] not in heldout_ids]
    pq.write_table(pa.table({k: [r[k] for r in heldout] for k in SCHEMA}), out / "data" / "heldout-00000.parquet")

    shards = []
    for index in range(0, len(deduped), args.rows_per_shard):
        chunk = deduped[index:index + args.rows_per_shard]
        name = f"train-{index // args.rows_per_shard:05d}.parquet"
        pq.write_table(pa.table({k: [r[k] for r in chunk] for k in SCHEMA}), out / "data" / name)
        shards.append(name)

    by_source = collections.Counter(r["source"] for r in deduped)
    manifest = {"rows": len(deduped), "heldout_rows": len(heldout),
                "heldout_shard": "heldout-00000.parquet", "schema": SCHEMA, "shards": shards,
                "source_counts": dict(by_source), "seed": args.seed,
                "filters": {"max_prompt_chars": args.max_prompt_chars,
                            "max_thinking_chars": args.max_thinking_chars,
                            "max_answer_chars": args.max_answer_chars},
                "build_stats": dict(stats)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(deduped)} rows to {out} in {len(shards)} shards")
    for k, v in by_source.most_common():
        print(f"  {k:16s} {v:7d}  ({100 * v / len(deduped):.1f}%)")
    print("build stats:", json.dumps(dict(stats), indent=2))


if __name__ == "__main__":
    main()
