"""Protocol helpers for the single-process common80k Ordered oracle probe."""
import hashlib
import json
import math
import random

MODES = ("null", "self_planning_first", "oracle_matched", "oracle_shuffled")


def validate_serial_protocol(modes, load_counts, cross_gpu_gate=False,
                             matched_reference=False, consistency=False):
    if tuple(modes) != MODES:
        raise ValueError(f"modes must run serially as {MODES}")
    for name in ("model", "t5", "mlp", "whitener", "gpt2"):
        if int(load_counts.get(name, 0)) != 1:
            raise ValueError(f"{name} must load exactly once")
    if cross_gpu_gate or matched_reference or consistency:
        raise ValueError("obsolete cross-worker/reference/consistency gate enabled")
    return True


def clone_base_response(base):
    """Return independent mode inputs without mutating the registered base state."""
    return {mode: base.clone() for mode in MODES}


def validate_smoke_mapping(rows):
    if len(rows) != 4 or len({r["eval_id"] for r in rows}) != 4:
        raise ValueError("smoke mapping must contain four unique recipients")
    ks = {int(r["recipient_K"]) for r in rows}
    if not {8, 20}.issubset(ks):
        raise ValueError("smoke mapping must include K=8 and K=20")
    return True


def validate_donor_mapping(rows, expected=1000):
    if len(rows) != expected:
        raise ValueError(f"donor mapping rows={len(rows)} != {expected}")
    recipients = [r["recipient_sample_id"] for r in rows]
    donors = [r["donor_sample_id"] for r in rows]
    if len(set(recipients)) != expected or len(set(donors)) != expected:
        raise ValueError("recipient/donor IDs must be unique")
    for r in rows:
        if r["recipient_sample_id"] == r["donor_sample_id"]:
            raise ValueError("fixed point in donor mapping")
        if int(r["recipient_K"]) != int(r["donor_K"]):
            raise ValueError("nearest-K donor is forbidden")
        if r["recipient_pair_hash"] == r["donor_pair_hash"]:
            raise ValueError("donor pair identity collision")
    payload = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def paired_bootstrap(left, right, seed=42, draws=10000):
    if len(left) != len(right) or not left:
        raise ValueError("paired arrays must be nonempty and equal length")
    delta = [float(a) - float(b) for a, b in zip(left, right)]
    if not all(math.isfinite(x) for x in delta):
        raise ValueError("nonfinite paired NLL")
    rng = random.Random(seed)
    means = []
    n = len(delta)
    for _ in range(draws):
        means.append(sum(delta[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int(0.025 * draws)]
    hi = means[min(draws - 1, int(0.975 * draws))]
    return {"mean_delta_nll": sum(delta) / n, "ci95": [lo, hi],
            "win_rate": sum(x < 0 for x in delta) / n, "sample_count": n}
