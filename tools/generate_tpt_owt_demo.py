import argparse
import json
import re
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


PROMPT_TEMPLATE = """{context}
## End of the context

Simulate an expert's in-depth thought process as they analyze the above context, focusing on complex and informative aspects. Skip trivial details. Use Feynman technique whenever possible to ensure a deep understanding.
"""


def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()


def pick_text(example):
    for k in ["text", "content", "document", "raw_text"]:
        if k in example and isinstance(example[k], str) and len(example[k].strip()) > 100:
            return example[k].strip()
    for _, v in example.items():
        if isinstance(v, str) and len(v.strip()) > 100:
            return v.strip()
    return None


def truncate_by_tokens(text, tokenizer, max_tokens):
    ids = tokenizer(text, add_special_tokens=False).input_ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)


def apply_chat_template(tokenizer, prompt):
    messages = [{"role": "user", "content": prompt}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )


def extract_thinking(text):
    m = re.search(r"<think>(.*?)</think>", text, flags=re.S | re.I)
    if m:
        return m.group(1).strip()
    text = text.replace("<|im_end|>", "").strip()
    text = re.sub(r"^<think>", "", text, flags=re.I).strip()
    text = re.sub(r"</think>$", "", text, flags=re.I).strip()
    return text


def stop_at_think_end(text):
    idx = text.lower().find("</think>")
    if idx >= 0:
        return text[: idx + len("</think>")]
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="Skylion007/openwebtext")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--context_max_tokens", type=int, default=2048)
    parser.add_argument("--max_new_tokens", type=int, default=8192)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--min_chars", type=int, default=500)
    parser.add_argument("--out_dir", default="results/tpt_owt_demo/qwen3_0p6b")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"Loading dataset: {args.dataset}")
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    rows = []

    for ex in ds:
        raw = pick_text(ex)
        if raw is None:
            continue

        raw = clean_text(raw)
        if len(raw) < args.min_chars:
            continue

        context = truncate_by_tokens(raw, tokenizer, args.context_max_tokens)
        prompt = PROMPT_TEMPLATE.format(context=context)
        chat_text = apply_chat_template(tokenizer, prompt)

        inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        generated = tokenizer.decode(gen_ids, skip_special_tokens=False)
        generated = stop_at_think_end(generated)
        thinking = extract_thinking(generated)

        context_tokens = len(tokenizer(context, add_special_tokens=False).input_ids)
        thinking_tokens = len(tokenizer(thinking, add_special_tokens=False).input_ids)

        augmented_text = context + "\n\n<think>\n" + thinking + "\n</think>"

        row = {
            "id": len(rows),
            "source_dataset": args.dataset,
            "generator_model": args.model,
            "context_tokens": context_tokens,
            "thinking_tokens": thinking_tokens,
            "source_text": context,
            "thinking": thinking,
            "augmented_text": augmented_text,
        }
        rows.append(row)

        print(f"[{len(rows)}/{args.num_samples}] context_tokens={context_tokens}, thinking_tokens={thinking_tokens}")

        if len(rows) >= args.num_samples:
            break

    df = pd.DataFrame(rows)

    csv_path = out_dir / "tpt_owt_100_demo.csv"
    xlsx_path = out_dir / "tpt_owt_100_demo.xlsx"
    jsonl_path = out_dir / "tpt_owt_100_demo.jsonl"
    summary_path = out_dir / "tpt_owt_100_summary.json"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "num_samples": len(rows),
        "dataset": args.dataset,
        "generator_model": args.model,
        "context_max_tokens": args.context_max_tokens,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "context_tokens_mean": float(df["context_tokens"].mean()),
        "context_tokens_median": float(df["context_tokens"].median()),
        "thinking_tokens_mean": float(df["thinking_tokens"].mean()),
        "thinking_tokens_median": float(df["thinking_tokens"].median()),
        "thinking_tokens_min": int(df["thinking_tokens"].min()),
        "thinking_tokens_max": int(df["thinking_tokens"].max()),
    }

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nDone.")
    print(f"Wrote {csv_path}")
    print(f"Wrote {xlsx_path}")
    print(f"Wrote {jsonl_path}")
    print(f"Wrote {summary_path}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
