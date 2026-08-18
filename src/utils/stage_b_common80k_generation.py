"""Identity and orchestration helpers for common80k Generation Evaluation v1."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch

SHAPE_MANIFEST_SHA = "10b3282441a608a8d30409cd1072095d0e37831b57c65c34552a203e8787f58b"
SHAPE_DATA_SHA = "00ea35649134374c1f93ab5b26ff948e3df5950786e269adfea370c5eeb5b23d"
SPLIT_MANIFEST_SHA = "738e43edd417124c9145efbfb98f68488997da793df4a0e280e30887d53b745f"
ALLOWED_SHAPE_FIELDS = frozenset({"eval_id", "source_sample_hash", "K", "plan_mask_length",
    "response_length", "response_mask_length", "token_noise_seed", "plan_noise_seed",
    "sampling_seed", "audit_source_hash"})

CONDITIONS = (
    ("ordered_alpha2p0_steps8", "ordered", 2.0, 8),
    ("ordered_alpha2p0_steps32", "ordered", 2.0, 32),
    ("ordered_alpha1p0_steps8", "ordered", 1.0, 8),
    ("ordered_alpha1p0_steps32", "ordered", 1.0, 32),
    ("ordered_alpha0p5_steps8", "ordered", 0.5, 8),
    ("ordered_alpha0p5_steps32", "ordered", 0.5, 32),
    ("ordered_alpha0p0_steps8", "ordered", 0.0, 8),
    ("ordered_alpha0p0_steps32", "ordered", 0.0, 32),
    ("diagonal_alpha1p0_steps8", "diagonal", 1.0, 8),
    ("diagonal_alpha1p0_steps32", "diagonal", 1.0, 32),
    ("register_native_steps8", "register", None, 8),
    ("register_native_steps32", "register", None, 32),
    ("vanilla_native_steps8", "vanilla", None, 8),
    ("vanilla_native_steps32", "vanilla", None, 32),
)
CONDITION_BY_ID = {row[0]: row for row in CONDITIONS}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_sha256(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(str((tuple(value.shape), value.dtype)).encode() + value.numpy().tobytes()).hexdigest()


def condition(condition_id):
    if condition_id not in CONDITION_BY_ID:
        raise ValueError(f"unknown condition_id: {condition_id}")
    cid, group, alpha, steps = CONDITION_BY_ID[condition_id]
    return {"condition_id": cid, "model_group": group, "alpha": alpha, "steps": steps}


def plan_protocol(spec):
    group, alpha = spec["model_group"], spec["alpha"]
    if group == "vanilla":
        return {"plan_enabled": False, "trajectory": None, "alpha": None, "endpoint": None}
    if group == "register":
        return {"plan_enabled": True, "trajectory": "null", "alpha": 0.0, "endpoint": 0.0}
    value = float(alpha)
    trajectory = "null" if value == 0 else ("diagonal" if value == 1 else "planning_first")
    return {"plan_enabled": True, "trajectory": trajectory, "alpha": value,
            "endpoint": min(1.0, value)}


def validate_shape_inputs(split_manifest, shape_manifest, shape_data, rows_expected=1000):
    if sha256_file(split_manifest) != SPLIT_MANIFEST_SHA:
        raise ValueError("split manifest SHA256 mismatch")
    if sha256_file(shape_manifest) != SHAPE_MANIFEST_SHA:
        raise ValueError("Generation Shape manifest SHA256 mismatch")
    if sha256_file(shape_data) != SHAPE_DATA_SHA:
        raise ValueError("Generation Shape data SHA256 mismatch")
    split = json.loads(Path(split_manifest).read_text())
    manifest = json.loads(Path(shape_manifest).read_text())
    required = {"complete": True, "generation_gold_content_used": False,
                "stage_b_heldout": True, "end_to_end_heldout": False,
                "stage_a_mlp_seen": True, "whitener_seen": True}
    for key, expected in required.items():
        if manifest.get(key) != expected:
            raise ValueError(f"shape manifest gate failed: {key}")
    zero = split.get("zero_intersections", {})
    if not (split.get("complete") and zero and all(int(value) == 0 for value in zero.values())):
        raise ValueError("split completeness/content hash gate failed")
    rows = []
    for line in Path(shape_data).read_text().splitlines():
        row = json.loads(line)
        if set(row) - ALLOWED_SHAPE_FIELDS:
            raise ValueError(f"gold/unknown shape fields: {sorted(set(row)-ALLOWED_SHAPE_FIELDS)}")
        k, n = int(row["K"]), int(row["response_length"])
        if not 1 <= k <= 255 or int(row["plan_mask_length"]) != k:
            raise ValueError("invalid runtime K/mask")
        if not 1 <= n <= 1024 or int(row["response_mask_length"]) != n:
            raise ValueError("invalid response length/mask")
        rows.append(row)
    if len(rows) != rows_expected or len({r["eval_id"] for r in rows}) != rows_expected:
        raise ValueError("shape row/identity count mismatch")
    return rows


def select_smoke(rows, seed=42, count=4):
    score = lambda r: hashlib.sha256(f"{seed}\0{r['eval_id']}".encode()).hexdigest()
    bands = ((3, 64), (65, 128), (129, 192), (193, 255))
    chosen = []
    for low, high in bands:
        candidates = sorted((r for r in rows if low <= int(r["K"]) <= high), key=score)
        if candidates: chosen.append(candidates[0])
    used_k = {int(row["K"]) for row in chosen}
    for row in sorted(rows, key=score):
        if len(chosen) == count: break
        if int(row["K"]) not in used_k:
            chosen.append(row); used_k.add(int(row["K"]))
    if len(chosen) != count or max(int(r["K"]) for r in chosen) <= 16:
        raise ValueError("cannot construct four distinct-K smoke rows including K>16")
    return chosen


def assert_truth_inputs(group, token, token_mask, plan, plan_mask, t_plan):
    if token.shape[:2] != token_mask.shape or token_mask.dtype != torch.bool:
        raise ValueError("invalid token input/mask")
    if group == "vanilla":
        if any(value is not None for value in (plan, plan_mask, t_plan)):
            raise ValueError("Vanilla must not contain a plan path")
        return
    if plan is None or plan_mask is None or t_plan is None:
        raise ValueError(f"{group} requires plan tensor/mask/time")
    if plan.shape[:2] != plan_mask.shape or plan_mask.dtype != torch.bool:
        raise ValueError("invalid plan input/mask")


def assert_fixed_plan(initial, trace, label):
    if any(not torch.equal(initial, value) for value in trace):
        raise ValueError(f"{label} plan/register changed")


def validate_resume_arm(path, expected):
    path = Path(path); manifest_path = path / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = json.loads(manifest_path.read_text())
    for key, value in expected.items():
        if manifest.get(key) != value:
            return False
    if not manifest.get("complete") or not (path / "per_sample.jsonl").is_file():
        return False
    for name, meta in manifest.get("outputs", {}).items():
        target = path / name
        if not target.is_file() or sha256_file(target) != meta["sha256"]:
            return False
    return True


def all_workers_finished(pid_rows):
    for row in pid_rows:
        try:
            os.kill(int(row["pid"]), 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            return False
        return False
    return True
