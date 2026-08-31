#!/usr/bin/env python3
"""Build the matched no-reasoning / explicit-reasoning math datasets.

Both arms are built from the same rows, so the only thing that differs between them is
whether the target the model has to emit contains the reasoning:

    noreason   condition = question,  target = "The answer is 42"
    cot        condition = question,  target = "<reasoning>\\nThe answer is 42"

Same question, same phrasing around the answer, same scorer. The generation window is
therefore the cost axis and nothing else is.

Rows come from the HRM-Text io-cleaned corpus, which stores each question twice, once with
`condition: cot` and once with `condition: direct`; pairing them on the question gives the
reasoning and the short answer separately. Only three of its nine files actually carry both
halves (math_train, omnimath, numinamath) -- see tools/survey_hrm.py. GSM8K train is
direct-only there, so its pairs are rebuilt from the original dataset, whose `answer` field
is the reasoning followed by `#### N`.

Rows are kept only when the answer is a bare number. That is not a sample of convenience:
the whole measurement is exact match on a number, so an answer of `\\frac{1}{2}` could not
be scored, and t5-small's vocabulary has no backslash or braces to represent it with either.

    python tools/build_math_data.py --out data/math_v1
"""
import argparse
import collections
import hashlib
import json
import re
import sys
from pathlib import Path

HRM = Path("/jfs/auto.prod.sz/data/ann/vlf/data/pengxiang.li/HRM-Text-data-io-cleaned-20260515/data")
GSM8K = Path("/cpfs01/shared/public/users/pengxiang.li/ouro_eval_outputs/hf_datasets/"
             "openai___gsm8k/main/0.0.0/740312add88f781978c0658806c59bc2815b9866")
HRM_PAIRED_SOURCES = ("math_train", "omnimath", "numinamath")

ANSWER_PHRASE = "The answer is "
BARE_NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
BOXED = re.compile(r"\\boxed\s*\{([^{}]*)\}")
CALCULATOR = re.compile(r"<<[^>]*>>")
LATEX_DELIMS = re.compile(r"\\[()\[\]]|\$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/math_v1")
    parser.add_argument("--tokenizer", default="t5-small")
    parser.add_argument("--max-question-chars", type=int, default=2000)
    parser.add_argument("--max-reasoning-chars", type=int, default=6000)
    parser.add_argument("--max-answer-chars", type=int, default=30)
    parser.add_argument("--num-proc", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="debug: stop after this many rows")
    parser.add_argument("--sources", nargs="*", default=list(HRM_PAIRED_SOURCES),
                        help="which HRM files to pair; GSM8K train is always included")
    return parser.parse_args()


def normalize_answer(text):
    """A scorable answer, or None. Accepts `42`, `$42$`, `\\boxed{1,024}`, `42.`."""
    if text is None:
        return None
    candidate = text.strip()
    boxed = BOXED.search(candidate)
    if boxed:
        candidate = boxed.group(1)
    candidate = LATEX_DELIMS.sub("", candidate).strip().rstrip(".").replace(",", "").replace(" ", "")
    if not BARE_NUMBER.match(candidate):
        return None
    # String arithmetic, not float: numinamath answers run to hundreds of digits, where
    # float() overflows to infinity and every large answer would collapse to one value.
    sign = "-" if candidate.startswith("-") else ""
    whole, _, fraction = candidate.lstrip("-").partition(".")
    fraction = fraction.rstrip("0")
    whole = whole.lstrip("0") or "0"
    normalized = f"{whole}.{fraction}" if fraction else whole
    return normalized if normalized == "0" else sign + normalized


def clean_reasoning(text):
    """Drop annotation t5-small cannot represent, keep the words and the arithmetic."""
    text = BOXED.sub(r"\1", text)
    text = CALCULATOR.sub("", text)
    text = LATEX_DELIMS.sub("", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def row_id(source, question):
    return hashlib.sha256(f"{source}\x00{question}".encode()).hexdigest()[:16]


def pairs_from_hrm(name, stats):
    """Questions the corpus stores under both `cot` and `direct`."""
    path = HRM / f"{name}.jsonl"
    if not path.exists():
        print(f"missing {path}, skipped", file=sys.stderr)
        return []
    cot, direct = {}, {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                stats[f"{name}/bad_json"] += 1
                continue
            question, response = row.get("instruction"), row.get("response")
            if not question or response is None:
                stats[f"{name}/missing_field"] += 1
                continue
            target = cot if "cot" in (row.get("condition") or "") else direct
            target.setdefault(question, response)
    out = []
    for question, reasoning in cot.items():
        answer = direct.get(question)
        if answer is None:
            stats[f"{name}/unpaired"] += 1
            continue
        out.append({"source": name, "question": question, "reasoning": reasoning, "answer": answer})
    return out


def pairs_from_gsm8k(stats):
    """GSM8K's `answer` field is `reasoning ... #### N`: split it at the marker."""
    import datasets
    table = datasets.Dataset.from_file(str(GSM8K / "gsm8k-train.arrow"))
    out = []
    for row in table:
        body = row["answer"]
        if "####" not in body:
            stats["gsm8k_train/no_marker"] += 1
            continue
        reasoning, answer = body.rsplit("####", 1)
        out.append({"source": "gsm8k_train", "question": row["question"],
                    "reasoning": reasoning, "answer": answer})
    return out


def keep(row, args, stats):
    answer = normalize_answer(row["answer"])
    if answer is None:
        stats["drop/answer_not_a_number"] += 1
        return None
    question = row["question"].strip()
    reasoning = clean_reasoning(row["reasoning"])
    if not question or not reasoning:
        stats["drop/empty"] += 1
        return None
    if len(question) > args.max_question_chars:
        stats["drop/question_too_long"] += 1
        return None
    if len(reasoning) > args.max_reasoning_chars:
        stats["drop/reasoning_too_long"] += 1
        return None
    if len(reasoning) <= len(answer):
        stats["drop/reasoning_not_longer"] += 1
        return None
    if len(answer) > args.max_answer_chars:
        # A 200-digit answer is not a target a model is asked to emit, and it would
        # dominate the generation window it lands in.
        stats["drop/answer_too_long"] += 1
        return None
    return {"example_id": row_id(row["source"], question), "source": row["source"],
            "question": question, "reasoning": reasoning, "answer": answer}


def percentiles(values, points=(50, 90, 95, 99, 100)):
    values = sorted(values)
    return {f"p{p}": int(values[min(len(values) - 1, int(len(values) * p / 100 - 1e-9))])
            for p in points}


def main():
    args = parse_args()
    stats = collections.Counter()

    rows = pairs_from_gsm8k(stats)
    print(f"gsm8k_train: {len(rows)} pairs", flush=True)
    for name in args.sources:
        found = pairs_from_hrm(name, stats)
        print(f"{name}: {len(found)} pairs", flush=True)
        rows += found
    if args.limit:
        rows = rows[:args.limit]

    kept, seen = [], set()
    for row in rows:
        cleaned = keep(row, args, stats)
        if cleaned is None:
            continue
        if cleaned["example_id"] in seen:
            stats["drop/duplicate"] += 1
            continue
        seen.add(cleaned["example_id"])
        kept.append(cleaned)
    print(f"\n{len(kept)} rows survive the answer-is-a-number filter", flush=True)

    import datasets
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    base = datasets.Dataset.from_list(kept)

    def encode(batch):
        # add_special_tokens=False matches the official conditional path: the tail after
        # the target is filled with EOS by the collator and that whole tail is supervised.
        questions = tokenizer(batch["question"], add_special_tokens=False)["input_ids"]
        noreason_text = [f"{ANSWER_PHRASE}{a}" for a in batch["answer"]]
        cot_text = [f"{r}\n{ANSWER_PHRASE}{a}" for r, a in zip(batch["reasoning"], batch["answer"])]
        return {
            "condition_input_ids": questions,
            "noreason_input_ids": tokenizer(noreason_text, add_special_tokens=False)["input_ids"],
            "cot_input_ids": tokenizer(cot_text, add_special_tokens=False)["input_ids"],
            "noreason_target": noreason_text,
            "cot_target": cot_text,
        }

    encoded = base.map(encode, batched=True, batch_size=1000, num_proc=args.num_proc,
                       desc="tokenizing")

    lengths = {
        "question": percentiles([len(x) for x in encoded["condition_input_ids"]]),
        "noreason_target": percentiles([len(x) for x in encoded["noreason_input_ids"]]),
        "cot_target": percentiles([len(x) for x in encoded["cot_input_ids"]]),
    }

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for arm in ("noreason", "cot"):
        arm_ds = encoded.select_columns(
            ["example_id", "source", "question", "condition_input_ids",
             f"{arm}_input_ids", f"{arm}_target"]
        ).rename_columns({f"{arm}_input_ids": "input_ids", f"{arm}_target": "target",
                          "question": "input"})
        arm_ds.save_to_disk(str(out / arm))
        print(f"wrote {out / arm}: {len(arm_ds)} rows, columns {arm_ds.column_names}")

    manifest = {
        "rows": len(kept),
        "source_counts": dict(collections.Counter(r["source"] for r in kept)),
        "token_lengths": lengths,
        "tokenizer": args.tokenizer,
        "answer_phrase": ANSWER_PHRASE,
        "filters": {"max_question_chars": args.max_question_chars,
                    "max_reasoning_chars": args.max_reasoning_chars,
                    "max_answer_chars": args.max_answer_chars,
                    "answer_must_be_a_bare_number": True},
        "build_stats": dict(stats),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("\n" + json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
