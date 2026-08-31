#!/usr/bin/env python3
"""Write GSM8K test as the {input, output} jsonl that src/eval.py reads.

1319 grade-school problems whose gold answer is the integer after `####`. The target is
phrased exactly like the training targets, `The answer is 42`, so one scorer reads every
arm and no arm is favoured by the extraction.

    python tools/build_gsm8k_eval.py --out data/gsm8k_test.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_math_data import ANSWER_PHRASE, GSM8K, normalize_answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/gsm8k_test.jsonl")
    parser.add_argument("--split", default="test", choices=("test", "train"))
    args = parser.parse_args()

    import datasets
    table = datasets.Dataset.from_file(str(GSM8K / f"gsm8k-{args.split}.arrow"))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(out, "w", encoding="utf-8") as handle:
        for row in table:
            gold = normalize_answer(row["answer"].rsplit("####", 1)[1])
            if gold is None:
                raise ValueError(f"unscorable gold answer: {row['answer'][-60:]!r}")
            handle.write(json.dumps({"input": row["question"],
                                     "output": f"{ANSWER_PHRASE}{gold}"},
                                    ensure_ascii=False) + "\n")
            written += 1
    print(f"wrote {written} rows to {out}")


if __name__ == "__main__":
    main()
