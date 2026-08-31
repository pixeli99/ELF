#!/usr/bin/env python3
"""Score what src/eval.py generated against the GSM8K gold answers.

src/eval.py writes `{"id": i, "generated": text}` in dataset order, so `id` indexes the
eval jsonl. Scoring is exact match on the number, which is the only thing that separates
solving from producing text that looks like a solution.

Every run also reports a permutation floor: the same scorer with the predictions shuffled
against the golds. A floor is not decoration. The previous line of work reported a
retention rate of 44% from a substring-containment scorer whose floor turned out to be
28%, and built a plan on the difference.

    python tools/score_gsm8k.py --generated <dir-or-jsonl> --gold data/gsm8k_test.jsonl
"""
import argparse
import json
import math
import random
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_math_data import ANSWER_PHRASE, normalize_answer

PHRASE = re.compile(re.escape(ANSWER_PHRASE.strip()) + r"\s*:?\s*(.{0,40})", re.I)
NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract(text):
    """The number the model committed to: after the answer phrase, else the last one."""
    phrase = PHRASE.search(text)
    if phrase:
        found = normalize_answer(phrase.group(1).split()[0] if phrase.group(1).split() else "")
        if found is not None:
            return found
    numbers = NUMBER.findall(text)
    for candidate in reversed(numbers):
        found = normalize_answer(candidate)
        if found is not None:
            return found
    return None


def wilson(hits, total, z=1.96):
    """Wilson interval: usable when the count is near zero, unlike the normal one."""
    if total == 0:
        return (0.0, 0.0)
    p = hits / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (round(max(0.0, center - spread), 4), round(min(1.0, center + spread), 4))


def permutation_floor(predictions, golds, trials=200, seed=0):
    """What the scorer awards for predictions paired with the wrong questions."""
    rng = random.Random(seed)
    order = list(range(len(golds)))
    rates = []
    for _ in range(trials):
        rng.shuffle(order)
        rates.append(sum(1 for i, j in enumerate(order)
                         if i != j and predictions[i] is not None and predictions[i] == golds[j])
                     / max(len(golds), 1))
    return round(statistics.mean(rates), 4), round(statistics.stdev(rates), 4) if trials > 1 else 0.0


def find_generated(path):
    path = Path(path)
    if path.is_file():
        return path
    candidates = sorted(path.rglob("all_generated_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"no all_generated_*.jsonl under {path}")
    if len(candidates) > 1:
        print(f"note: {len(candidates)} generation files under {path}, scoring the last one",
              file=sys.stderr)
    return candidates[-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", required=True, help="all_generated_*.jsonl, or a dir holding one")
    parser.add_argument("--gold", default="data/gsm8k_test.jsonl")
    parser.add_argument("--out", default=None, help="write the per-row records here")
    parser.add_argument("--label", default=None, help="name for this arm in the summary")
    args = parser.parse_args()

    gold_rows = [json.loads(line) for line in open(args.gold, encoding="utf-8")]
    golds = [extract(row["output"]) for row in gold_rows]
    if any(g is None for g in golds):
        raise ValueError("a gold row has no extractable answer")

    generated_path = find_generated(args.generated)
    by_id = {}
    for line in open(generated_path, encoding="utf-8"):
        record = json.loads(line)
        by_id[int(record["id"])] = record["generated"]

    records, predictions, kept_golds = [], [], []
    for index, generated in sorted(by_id.items()):
        if index >= len(golds):
            raise ValueError(f"generated id {index} is past the end of {args.gold}")
        prediction = extract(generated)
        predictions.append(prediction)
        kept_golds.append(golds[index])
        records.append({"id": index, "question": gold_rows[index]["input"],
                        "gold": golds[index], "predicted": prediction,
                        "correct": prediction is not None and prediction == golds[index],
                        "generated": generated})

    total = len(records)
    correct = sum(r["correct"] for r in records)
    answered = sum(r["predicted"] is not None for r in records)
    floor, floor_sd = permutation_floor(predictions, kept_golds)
    summary = {
        "label": args.label or generated_path.parent.name,
        "generated": str(generated_path), "gold": args.gold, "rows": total,
        "accuracy": round(correct / max(total, 1), 4),
        "accuracy_ci95": wilson(correct, total),
        "permutation_floor": floor, "permutation_floor_sd": floor_sd,
        "answer_rate": round(answered / max(total, 1), 4),
        "correct": correct,
        "mean_generated_chars": round(statistics.mean([len(r["generated"]) for r in records]), 1)
        if records else 0.0,
    }
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
