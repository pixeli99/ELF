import argparse
import ast
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_float(x):
    try:
        return float(str(x).strip())
    except Exception:
        return None


def parse_log_file(path):
    rows = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = {}

            # 格式 1:
            # INFO - __main__ - {'step': '301500', 'loss': '0.7690', ...}
            if "{" in line and "}" in line:
                obj = line[line.find("{"): line.rfind("}") + 1]
                try:
                    d = ast.literal_eval(obj)
                    for k in ["step", "loss", "l2", "ce", "plan", "lr", "sps"]:
                        if k in d:
                            rec[k] = d[k]
                except Exception:
                    pass

            # 格式 2:
            # tqdm 里的 ce=0.1045, l2=0.6351, loss=0.7643, lr=..., plan=..., step=...
            for k, v in re.findall(
                r"\b(step|loss|l2|ce|plan|lr|sps)\s*=\s*([0-9.+\-eE]+)",
                line,
            ):
                rec[k] = v

            if "step" in rec and "loss" in rec:
                row = {}
                row["step"] = int(float(rec["step"]))
                for k in ["loss", "l2", "ce", "plan", "lr", "sps"]:
                    if k in rec:
                        val = parse_float(rec[k])
                        if val is not None:
                            row[k] = val
                rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No valid loss records found in {path}. "
            "Please check whether the file contains step/loss logs."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values("step")
    df = df.drop_duplicates(subset=["step"], keep="last")
    return df


def plot_loss(df, out_png):
    plt.figure(figsize=(10, 5.5), dpi=180)

    plt.plot(df["step"], df["loss"], linewidth=1.2, label="loss")

    if len(df) >= 10:
        win = max(5, min(100, len(df) // 20))
        smooth = df["loss"].rolling(win, min_periods=1).mean()
        plt.plot(df["step"], smooth, linewidth=2.2, label=f"loss rolling mean, window={win}")

    plt.title("ELF Ordered Training Loss Curve")
    plt.xlabel("Training step")
    plt.ylabel("loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ticklabel_format(style="plain", axis="x")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def plot_metrics(df, out_png):
    metrics = [m for m in ["loss", "l2", "ce", "plan"] if m in df.columns]

    plt.figure(figsize=(11, 6), dpi=180)
    for m in metrics:
        plt.plot(df["step"], df[m], linewidth=1.5, label=m)

    plt.title("ELF Ordered Training Metrics Curve")
    plt.xlabel("Training step")
    plt.ylabel("metric value")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ticklabel_format(style="plain", axis="x")
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="ordered_loss_points.txt")
    parser.add_argument("--outdir", default="figures")
    args = parser.parse_args()

    input_path = Path(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = parse_log_file(input_path)

    csv_path = outdir / "ordered_loss_points_parsed.csv"
    loss_png = outdir / "ordered_loss_curve_full.png"
    metrics_png = outdir / "ordered_metrics_curve_full.png"

    df.to_csv(csv_path, index=False)
    plot_loss(df, loss_png)
    plot_metrics(df, metrics_png)

    print("=" * 80)
    print(f"Parsed records: {len(df)}")
    print(f"Step range: {df['step'].min()} -> {df['step'].max()}")
    print(f"First loss: {df['loss'].iloc[0]:.6f}")
    print(f"Last loss:  {df['loss'].iloc[-1]:.6f}")
    print(f"Best loss:  {df['loss'].min():.6f} at step {df.loc[df['loss'].idxmin(), 'step']}")
    print("=" * 80)
    print(f"Saved CSV:     {csv_path}")
    print(f"Saved loss:    {loss_png}")
    print(f"Saved metrics: {metrics_png}")


if __name__ == "__main__":
    main()
