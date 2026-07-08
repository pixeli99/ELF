#!/usr/bin/env python
"""Plot Ordered-ELF trajectory ablation metrics."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


ORDER = ["planning_first", "diagonal", "lagging", "null"]
LABELS = {
    "planning_first": "planning_first a=2",
    "diagonal": "diagonal a=1",
    "lagging": "lagging a=0.5",
    "null": "null a=0",
}
COLORS = {
    "planning_first": "#2a6fbb",
    "diagonal": "#222222",
    "lagging": "#d17a00",
    "null": "#8a3ffc",
}


def load_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for key in ("alpha", "steps", "num_samples", "gen_ppl", "entropy"):
                if row.get(key) not in ("", None):
                    row[key] = float(row[key])
            rows.append(row)
    return rows


def metric_value(row, metric):
    value = row.get(metric)
    return None if value in ("", None) else float(value)


def plot_vs_steps(rows, metric, ylabel, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=160)
    for traj in ORDER:
        series = sorted([r for r in rows if r["trajectory"] == traj and metric_value(r, metric) is not None],
                        key=lambda r: r["steps"])
        if not series:
            continue
        ax.plot([r["steps"] for r in series], [metric_value(r, metric) for r in series],
                marker="o", linewidth=1.8, label=LABELS[traj], color=COLORS[traj])
    ax.set_xlabel("sampling steps")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} vs sampling steps")
    ax.set_xticks(sorted({int(r["steps"]) for r in rows}))
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_by_step(rows, metric, ylabel, out_path):
    steps = sorted({int(r["steps"]) for r in rows})
    fig, axes = plt.subplots(1, len(steps), figsize=(5.2 * len(steps), 4.2), dpi=160, sharey=True)
    if len(steps) == 1:
        axes = [axes]
    for ax, step in zip(axes, steps):
        vals = []
        labels = []
        colors = []
        for traj in ORDER:
            match = [r for r in rows if int(r["steps"]) == step and r["trajectory"] == traj]
            if not match or metric_value(match[0], metric) is None:
                continue
            vals.append(metric_value(match[0], metric))
            labels.append(LABELS[traj])
            colors.append(COLORS[traj])
        ax.bar(range(len(vals)), vals, color=colors)
        ax.set_title(f"{step} steps")
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
    axes[0].set_ylabel(ylabel)
    fig.suptitle(f"{ylabel} trajectory comparison")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", default="results/ordered_5b_4g/eval_trajectories_summary.csv")
    parser.add_argument("--output_dir", default="results/ordered_5b_4g/plots")
    args = parser.parse_args()

    rows = load_rows(Path(args.input_csv))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_vs_steps(rows, "gen_ppl", "gen-PPL", out / "gen_ppl_vs_steps.png")
    plot_vs_steps(rows, "entropy", "entropy", out / "entropy_vs_steps.png")
    plot_by_step(rows, "gen_ppl", "gen-PPL", out / "gen_ppl_trajectory_comparison.png")
    plot_by_step(rows, "entropy", "entropy", out / "entropy_trajectory_comparison.png")
    print(f"Wrote plots to {out}")


if __name__ == "__main__":
    main()
