import ast
import csv
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_PATH = Path("logs/ordered_5b_4g/train_ordered_4g_5b.log")
OUT_DIR = Path("figures")
CSV_PATH = Path("results/ordered_5b_4g/loss_curve_points.csv")

GLOBAL_BSZ = 32
GRAD_ACCUM = 16
EFFECTIVE_BSZ = GLOBAL_BSZ * GRAD_ACCUM
SEQ_LEN = 1024
TARGET_OPT_STEPS = 9537
TARGET_TOKENS = TARGET_OPT_STEPS * EFFECTIVE_BSZ * SEQ_LEN

OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

text = LOG_PATH.read_text(errors="ignore")

records = {}

# Format 1:
# INFO - __main__ - {'step': '126900', 'loss': '0.8014', ...}
for m in re.finditer(r"INFO - __main__ - (\{.*?\})", text):
    try:
        d = ast.literal_eval(m.group(1))
    except Exception:
        continue

    if "step" not in d:
        continue

    step = int(float(d["step"]))
    rec = records.setdefault(step, {"step": step})

    for key in ["loss", "l2", "ce", "plan", "lr", "sps"]:
        if key in d:
            try:
                rec[key] = float(d[key])
            except Exception:
                pass

# Format 2:
# INFO - engine - Step 126900: loss=0.8014, l2=0.6501, ce=0.1517, plan=0.2479, lr=1.67e-03
engine_re = re.compile(
    r"INFO - engine - Step\s+(\d+):\s+"
    r"loss=([0-9.eE+-]+),\s+"
    r"l2=([0-9.eE+-]+),\s+"
    r"ce=([0-9.eE+-]+),\s+"
    r"plan=([0-9.eE+-]+),\s+"
    r"lr=([0-9.eE+-]+)"
)
for m in engine_re.finditer(text):
    step = int(m.group(1))
    rec = records.setdefault(step, {"step": step})
    rec["loss"] = float(m.group(2))
    rec["l2"] = float(m.group(3))
    rec["ce"] = float(m.group(4))
    rec["plan"] = float(m.group(5))
    rec["lr"] = float(m.group(6))

rows = []
for step in sorted(records):
    rec = records[step]
    if "loss" not in rec:
        continue

    tokens = step * GLOBAL_BSZ * SEQ_LEN
    rec["tokens_b"] = tokens / 1e9
    rec["optimizer_step_est"] = step / GRAD_ACCUM
    rec["progress"] = tokens / TARGET_TOKENS * 100
    rows.append(rec)

if not rows:
    raise RuntimeError(f"No loss records parsed from {LOG_PATH}")

# Save CSV
fields = ["step", "optimizer_step_est", "tokens_b", "progress", "loss", "l2", "ce", "plan", "lr", "sps"]
with CSV_PATH.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in fields})

x = [r["tokens_b"] for r in rows]

def series(name):
    return [r.get(name, None) for r in rows]

# Plot 1: main losses
plt.figure(figsize=(10, 6))
for name, label in [
    ("loss", "total loss"),
    ("l2", "l2 denoising loss"),
    ("ce", "CE loss"),
    ("plan", "plan loss"),
]:
    y = series(name)
    if any(v is not None for v in y):
        plt.plot(x, y, label=label)

plt.xlabel("Training tokens (B)")
plt.ylabel("Loss")
plt.title("Ordered ELF 5B Training Loss Curves")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "ordered_5b_4g_loss_curves.png", dpi=200)
plt.close()

# Plot 2: plan loss only
plt.figure(figsize=(10, 6))
plt.plot(x, series("plan"), label="plan loss")
plt.axhline(1.0, linestyle="--", label="mean-prediction baseline = 1.0")
plt.xlabel("Training tokens (B)")
plt.ylabel("Plan loss")
plt.title("Ordered ELF 5B Plan Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "ordered_5b_4g_plan_loss.png", dpi=200)
plt.close()

# Plot 3: LR curve
plt.figure(figsize=(10, 6))
plt.plot(x, series("lr"), label="learning rate")
plt.xlabel("Training tokens (B)")
plt.ylabel("Learning rate")
plt.title("Ordered ELF 5B Learning Rate Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "ordered_5b_4g_lr_curve.png", dpi=200)
plt.close()

last = rows[-1]
print("Parsed records:", len(rows))
print("Last step:", last["step"])
print("Last tokens(B):", f'{last["tokens_b"]:.4f}')
print("Progress(%):", f'{last["progress"]:.2f}')
print("Last loss:", last.get("loss"))
print("Last l2:", last.get("l2"))
print("Last ce:", last.get("ce"))
print("Last plan:", last.get("plan"))
print()
print("Saved:")
print(" -", OUT_DIR / "ordered_5b_4g_loss_curves.png")
print(" -", OUT_DIR / "ordered_5b_4g_plan_loss.png")
print(" -", OUT_DIR / "ordered_5b_4g_lr_curve.png")
print(" -", CSV_PATH)
