import re
import csv
from pathlib import Path

import matplotlib.pyplot as plt


OUT = Path("results/summary/loss_curves")
OUT.mkdir(parents=True, exist_ok=True)

MODEL_LOG_HINTS = {
    "ordered": [
        "logs/ordered_5b_4g/train_ordered_4g_5b.log",
        "logs/ordered_5b_4g/train_ordered_5b_4g.log",
    ],
    "register": [
        "logs/register_5b_4g/train_register_4g_5b.log",
    ],
    "vanilla": [
        "logs/vanilla_5b_4g/train_vanilla_4g_5b.log",
    ],
}

GLOBAL_BATCH_SIZE = 32
SEQ_LEN = 1024


def find_log(model):
    candidates = []
    for p in MODEL_LOG_HINTS[model]:
        p = Path(p)
        if p.exists():
            candidates.append(p)

    if not candidates:
        for p in Path("logs").rglob("*.log"):
            s = str(p).lower()
            if model in s and "train" in s and "eval" not in s:
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError(f"No training log found for {model}")

    return max(candidates, key=lambda x: x.stat().st_size)


def parse_log(model, path):
    text = path.read_text(errors="ignore")

    rows = []
    seen = set()

    for m in re.finditer(r"step=(\d+)", text):
        step = int(m.group(1))
        start = max(0, m.start() - 500)
        window = text[start:m.end() + 50]

        kvs = dict(re.findall(
            r"(loss|l2|ce|plan|plan_l2|lr|sps)=([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)",
            window,
            flags=re.IGNORECASE,
        ))

        if not any(k in kvs for k in ["loss", "l2", "ce", "plan", "plan_l2"]):
            continue

        key = (model, step)
        if key in seen:
            continue
        seen.add(key)

        row = {
            "model": model,
            "step": step,
            "tokens_b": step * GLOBAL_BATCH_SIZE * SEQ_LEN / 1e9,
            "loss": float(kvs["loss"]) if "loss" in kvs else "",
            "l2": float(kvs["l2"]) if "l2" in kvs else "",
            "ce": float(kvs["ce"]) if "ce" in kvs else "",
            "plan": float(kvs.get("plan", kvs.get("plan_l2"))) if ("plan" in kvs or "plan_l2" in kvs) else "",
            "lr": float(kvs["lr"]) if "lr" in kvs else "",
            "sps": float(kvs["sps"]) if "sps" in kvs else "",
            "log_path": str(path),
        }
        rows.append(row)

    rows.sort(key=lambda x: x["step"])
    return rows


def write_csv(rows):
    out_csv = OUT / "loss_curves_all_models.csv"
    fields = ["model", "step", "tokens_b", "loss", "l2", "ce", "plan", "lr", "sps", "log_path"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv}")
    return out_csv


def plot_metric(rows, metric, filename, ylabel=None):
    plt.figure(figsize=(8, 5))

    for model in ["vanilla", "register", "ordered"]:
        xs, ys = [], []
        for r in rows:
            if r["model"] == model and r.get(metric) != "":
                xs.append(r["tokens_b"])
                ys.append(float(r[metric]))
        if xs:
            plt.plot(xs, ys, label=model)

    plt.xlabel("Training tokens (B)")
    plt.ylabel(ylabel or metric)
    plt.title(filename.replace("_", " ").replace(".png", ""))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    out = OUT / filename
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Wrote {out}")


def main():
    all_rows = []

    for model in ["ordered", "register", "vanilla"]:
        path = find_log(model)
        print(f"{model}: {path}")
        rows = parse_log(model, path)
        print(f"  parsed {len(rows)} points")
        all_rows.extend(rows)

        if rows:
            last = rows[-1]
            print(f"  last: step={last['step']}, loss={last['loss']}, l2={last['l2']}, ce={last['ce']}, plan={last['plan']}")

    write_csv(all_rows)

    plot_metric(all_rows, "loss", "loss_total_three_groups.png", "Total loss")
    plot_metric(all_rows, "l2", "loss_l2_three_groups.png", "L2 loss")
    plot_metric(all_rows, "ce", "loss_ce_three_groups.png", "CE loss")
    plot_metric(all_rows, "plan", "loss_plan_three_groups.png", "Plan loss")

    print("\nDone. Figures are in results/summary/loss_curves/")


if __name__ == "__main__":
    main()
