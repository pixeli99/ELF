import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = Path("results/ordered_5b_4g/eval_trajectories_summary.csv")
out_dir = Path("figures")
out_dir.mkdir(exist_ok=True)

rows = []
with csv_path.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        r["steps"] = int(r["steps"])
        r["gen_ppl"] = float(r["gen_ppl"])
        r["entropy"] = float(r["entropy"])
        rows.append(r)

order = ["null", "lagging", "diagonal", "planning_first"]
labels = {
    "null": "null α=0",
    "lagging": "lagging α=0.5",
    "diagonal": "diagonal α=1",
    "planning_first": "planning-first α=2",
}

for metric, ylabel, filename in [
    ("gen_ppl", "gen-PPL ↓", "ordered_5b_trajectory_gen_ppl.png"),
    ("entropy", "Mean entropy", "ordered_5b_trajectory_entropy.png"),
]:
    plt.figure(figsize=(8, 5))
    for traj in order:
        sub = sorted([r for r in rows if r["trajectory"] == traj], key=lambda x: x["steps"])
        x = [r["steps"] for r in sub]
        y = [r[metric] for r in sub]
        plt.plot(x, y, marker="o", label=labels[traj])
    plt.xlabel("Sampling steps")
    plt.ylabel(ylabel)
    plt.title(f"Ordered 5B trajectory ablation: {ylabel}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=200)
    plt.close()

print("Saved:")
print(out_dir / "ordered_5b_trajectory_gen_ppl.png")
print(out_dir / "ordered_5b_trajectory_entropy.png")
