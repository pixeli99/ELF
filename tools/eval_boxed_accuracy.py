"""Answer accuracy on the verifiable subset: rows whose reference ends with \\boxed{...} or '#### N'.

Reads generation jsonl files from tools/eval_conditional_dev.py (fields eval_id, generated) and
the matching *_references parquet. The model's answer is the FIRST \\boxed{} / #### in the
generation (the model is supposed to stop after answering); the reference answer is the LAST.
"""
import argparse, json, re, sys
from pathlib import Path
import pyarrow.parquet as pq

BOXED = re.compile(r"\\boxed\s*\{")
HASH = re.compile(r"####\s*([^\n]+)")

def _boxed_content(text, start):
    """Return the brace-balanced content of a \\boxed{...} beginning at `start` (index of '{')."""
    depth, i = 0, start
    while i < len(text):
        if text[i] == "{": depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0: return text[start + 1:i]
        i += 1
    return text[start + 1:]

# T5's vocabulary has no backslash or braces, so a generated "\\boxed{42}" decodes as
# "boxed42" (and "\\frac{3}{4}" as "frac34"). Generations are therefore parsed in that
# round-tripped form; references are parsed from the raw text.
ROUNDTRIP_BOXED = re.compile(r"boxed\s*(\(?[A-Za-z]\)?|-?\d[\d,]*(?:\.\d+)?(?:/\d+)?|frac\s*(-?\d+)\s*(\d+))")

def extract_answers(text):
    out = []
    for m in BOXED.finditer(text):
        out.append((m.start(), _boxed_content(text, m.end() - 1)))
    for m in HASH.finditer(text):
        out.append((m.start(), m.group(1)))
    out.sort()
    return [a for _, a in out]

def extract_generated_answers(text):
    """Answers in a decoded generation: round-tripped boxed forms, then '#### N'."""
    out = []
    for m in ROUNDTRIP_BOXED.finditer(text):
        if m.group(2) is not None:
            out.append((m.start(), f"{m.group(2)}/{m.group(3)}"))
        else:
            out.append((m.start(), m.group(1)))
    for m in HASH.finditer(text):
        out.append((m.start(), m.group(1)))
    out.sort()
    return [a for _, a in out]

def normalize(ans):
    a = ans.strip().strip("$").strip()
    a = a.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "").replace(" ", "")
    a = a.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac").replace("^\\circ", "").replace("^{\\circ}", "")
    a = a.replace("\\text{", "{").replace("\\mathrm{", "{").replace("\\%", "").replace("%", "")
    a = a.rstrip(".").strip()
    m = re.fullmatch(r"\\frac\{(-?\d+)\}\{(\d+)\}", a)
    if m: return f"{int(m.group(1))/int(m.group(2)):.6g}"
    m = re.fullmatch(r"(-?\d+)/(\d+)", a)
    if m: return f"{int(m.group(1))/int(m.group(2)):.6g}"
    m = re.fullmatch(r"-?\d[\d,]*(\.\d+)?", a)
    if m:
        try: return f"{float(a.replace(',', '')):.6g}"
        except ValueError: pass
    return a.lower()

def has_reference_answer(ref):
    return bool(extract_answers(ref[-600:]))

def score(gen_rows, refs):
    n = correct = has_answer = 0
    for r in gen_rows:
        ref = refs.get(r["eval_id"])
        if ref is None: continue
        ra = extract_answers(ref[-600:])
        if not ra: continue
        n += 1
        ga = extract_generated_answers(r["generated"])
        if ga:
            has_answer += 1
            if normalize(ga[0]) == normalize(ra[-1]): correct += 1
    return {"verifiable_rows": n, "answer_rate": round(has_answer / max(n, 1), 4),
            "accuracy": round(correct / max(n, 1), 4), "correct": correct}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--references", default=None, help="parquet with eval_id + reference_response; inferred from split if omitted")
    ap.add_argument("--split", choices=("validation", "test"), default=None)
    args = ap.parse_args()
    for f in args.files:
        split = args.split or ("test" if "test" in f else "validation")
        ref_path = args.references or f"data/conditional_v1/data/{split}_references-00000.parquet"
        t = pq.read_table(ref_path).to_pydict()
        refs = dict(zip(t["eval_id"], t["reference_response"]))
        rows = [json.loads(l) for l in open(f)]
        s = score(rows, refs); s["file"] = f; s["rows"] = len(rows); s["split"] = split
        print(json.dumps(s))

if __name__ == "__main__":
    main()
