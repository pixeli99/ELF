#!/usr/bin/env python3
"""Build provenance-only Stage-B dev/test manifests from the paired train pool."""

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.tpt_million_data import CONTROL_RE, stable_hash
from tools.build_tpt_global_candidates import exact_record_hash, sha256_file

EXPECTED_POOL_ROWS = 488_692
EXPECTED_SCHEDULE_ROWS = 80_000
EXPECTED_SCHEDULE_SHA256 = "eb840ba3dc78e90aa91f91c1543f47e30f4fe10260ac16c249320a56694dd766"
T5_ID = "t5-small"
T5_REVISION = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
TOKENIZER_HASHES = {
    "spiece.model": "d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86",
    "tokenizer.json": "d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066",
    "tokenizer_config.json": "d1e7146101aa96282057f374aa9c3b260fba2109e5990edb1775d6efc70ffa3c",
}
K_BUCKETS = ((3, 32), (33, 64), (65, 128), (129, 192), (193, 255))
R_BUCKETS = ((1, 64), (65, 128), (129, 256), (257, 512), (513, 1024))
DEV_FIELDS = (
    "eval_id", "split", "sample_id", "source_id", "source", "pair_hash",
    "canonical_prompt_hash", "source_shard", "byte_offset", "thinking_tokens",
    "response_tokens", "pair_tokens", "K", "response_length", "stratum",
    "token_noise_seed", "plan_noise_seed", "token_time_seed", "plan_time_seed",
    "branch_seed", "thinking_response_hash", "exact_record_identity",
)
SHAPE_FIELDS = (
    "eval_id", "source_sample_hash", "K", "plan_mask_length", "response_length",
    "response_mask_length", "token_noise_seed", "plan_noise_seed", "sampling_seed",
    "audit_source_hash",
)


def stable_int(seed, identity, stream):
    digest = hashlib.sha256(f"{seed}\0{identity}\0{stream}".encode()).digest()
    return int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)


def bucket_index(value, buckets):
    for index, (low, high) in enumerate(buckets):
        if low <= value <= high:
            return index
    raise ValueError(f"value outside bucket definitions: {value}")


def stratum_key(row):
    ki = bucket_index(int(row["K"]), K_BUCKETS)
    ri = bucket_index(int(row["response_length"]), R_BUCKETS)
    return (str(row["source"]), ki, ri)


def stratum_name(key):
    source, ki, ri = key
    return f"{source}|K={K_BUCKETS[ki][0]}-{K_BUCKETS[ki][1]}|R={R_BUCKETS[ri][0]}-{R_BUCKETS[ri][1]}"


def largest_remainder(counts, total):
    available = sum(counts.values())
    if total > available:
        raise ValueError(f"requested {total} rows from capacity {available}")
    targets = {key: total * count // available for key, count in counts.items()}
    remaining = total - sum(targets.values())
    order = sorted(counts, key=lambda key: (-(total * counts[key] % available), key))
    for key in order[:remaining]:
        targets[key] += 1
    return targets


def redistribute_to_capacity(targets, capacities):
    quotas = {key: min(value, capacities.get(key, 0)) for key, value in targets.items()}
    deficits = [(key, targets[key] - quotas[key]) for key in sorted(targets) if targets[key] > quotas[key]]
    for origin, deficit in deficits:
        for _ in range(deficit):
            choices = [key for key in capacities if quotas.get(key, 0) < capacities[key]]
            if not choices:
                raise ValueError("insufficient capacity during deterministic redistribution")
            source, ki, ri = origin
            chosen = min(
                choices,
                key=lambda key: (
                    key[0] != source,
                    abs(key[1] - ki) + abs(key[2] - ri),
                    abs(key[1] - ki), abs(key[2] - ri), key,
                ),
            )
            quotas[chosen] = quotas.get(chosen, 0) + 1
    for key in capacities:
        quotas.setdefault(key, 0)
    return quotas


def select_stratified(rows, total, seed, excluded_prompts=None, excluded_contents=None):
    excluded_prompts = excluded_prompts or set()
    excluded_contents = excluded_contents or set()
    eligible = [row for row in rows
                if row["canonical_prompt_hash"] not in excluded_prompts
                and row["thinking_response_hash"] not in excluded_contents]
    raw_counts = Counter(stratum_key(row) for row in eligible)
    content_representatives = {}
    for row in eligible:
        content = row["thinking_response_hash"]
        rank = stable_int(seed, f"{content}\0{row['sample_id']}", "content_representative")
        current = content_representatives.get(content)
        if current is None or (rank, row["sample_id"]) < current[0]:
            content_representatives[content] = ((rank, row["sample_id"]), row)
    representatives = {}
    for _, row in content_representatives.values():
        prompt = row["canonical_prompt_hash"]
        rank = stable_int(seed, f"{prompt}\0{row['sample_id']}", "representative")
        current = representatives.get(prompt)
        if current is None or (rank, row["sample_id"]) < current[0]:
            representatives[prompt] = ((rank, row["sample_id"]), row)
    by_stratum = defaultdict(list)
    for _, row in representatives.values():
        by_stratum[stratum_key(row)].append(row)
    capacities = {key: len(value) for key, value in by_stratum.items()}
    targets = largest_remainder(raw_counts, total)
    quotas = redistribute_to_capacity(targets, capacities)
    selected = []
    for key in sorted(by_stratum):
        ranked = sorted(
            by_stratum[key],
            key=lambda row: (stable_int(seed, row["sample_id"], "selection"), row["sample_id"]),
        )
        selected.extend(ranked[:quotas[key]])
    selected.sort(key=lambda row: (stable_int(seed, row["sample_id"], "output"), row["sample_id"]))
    if len(selected) != total or len({row["canonical_prompt_hash"] for row in selected}) != total \
            or len({row["thinking_response_hash"] for row in selected}) != total:
        raise AssertionError("stratified selection count/prompt/content uniqueness failure")
    audit = []
    for key in sorted(set(raw_counts) | set(capacities) | set(quotas)):
        audit.append({"stratum": stratum_name(key), "raw_rows": raw_counts.get(key, 0),
                      "unique_prompt_content_capacity": capacities.get(key, 0),
                      "largest_remainder_target": targets.get(key, 0), "actual": quotas.get(key, 0),
                      "redistributed": quotas.get(key, 0) - targets.get(key, 0)})
    return selected, audit


def read_pointer(path, offset):
    with open(path, "rb") as handle:
        handle.seek(int(offset))
        line = handle.readline()
    if not line:
        raise ValueError(f"empty source pointer: {path}@{offset}")
    return json.loads(line)


def assert_pointer_identity(record, expected_sample_id, expected_source_id, expected_pair_hash,
                            expected_thinking_tokens, expected_response_tokens):
    actual_id = str(record.get("sample_id") or record.get("source_id") or "")
    actual_pair = str(record.get("exact_record_hash") or record.get("pair_hash") or "")
    derived_source = f"{record['source']}|{record['source_config']}|{actual_id}"
    if actual_id != expected_sample_id or actual_pair != expected_pair_hash \
            or derived_source != expected_source_id \
            or int(record["thinking_tokens"]) != int(expected_thinking_tokens) \
            or int(record["response_tokens"]) != int(expected_response_tokens):
        raise ValueError(f"source pointer identity mismatch: {expected_sample_id}")


def load_selection(paired):
    metadata = {}
    actual_rows = 0
    for entry in paired["splits"]["train"]["selection_files"]:
        path = Path(entry["path"])
        if path.stat().st_size != int(entry["bytes"]) or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"selection shard identity mismatch: {path}")
        rows = 0
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                sid = str(row.get("sample_id") or "")
                source_id = str(row.get("source_id") or "")
                if not sid or not source_id or sid in metadata:
                    raise ValueError(f"invalid/duplicate paired selection identity: {sid}")
                metadata[sid] = (source_id, str(row["pair_hash"]), int(row["thinking_tokens"]))
                rows += 1
        if rows != int(entry["rows"]):
            raise ValueError(f"selection shard row mismatch: {path}")
        actual_rows += rows
    if actual_rows != EXPECTED_POOL_ROWS or len(metadata) != EXPECTED_POOL_ROWS:
        raise ValueError(f"paired selection count mismatch: {actual_rows}/{len(metadata)}")
    return metadata


def load_schedule(schedule_path, selection, expected_sha=EXPECTED_SCHEDULE_SHA256):
    if sha256_file(schedule_path) != expected_sha:
        raise ValueError("common schedule SHA256 mismatch")
    rows, train = 0, {"sample_id": set(), "source_id": set(), "pair_hash": set(),
                      "prompt": set(), "content": set(), "exact_record_identity": set()}
    pointers = []
    with open(schedule_path, encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            sid = str(item["sample_id"])
            if int(item["presentation_index"]) != rows or int(item["optimizer_step"]) != rows // 8 + 1 \
                    or int(item["within_step"]) != rows % 8:
                raise ValueError(f"schedule index/step mismatch at row {rows}")
            if sid not in selection or sid in train["sample_id"] \
                    or item["source_id"] in train["source_id"] or item["pair_hash"] in train["pair_hash"]:
                raise ValueError(f"schedule missing/duplicate identity: {sid}")
            source_id, pair_hash, thinking_tokens = selection[sid]
            if source_id != item["source_id"] or pair_hash != item["pair_hash"] \
                    or thinking_tokens != int(item["thinking_tokens"]):
                raise ValueError(f"schedule/selection identity mismatch: {sid}")
            record = read_pointer(item["source_shard_path"], item["source_byte_offset"])
            assert_pointer_identity(record, sid, source_id, pair_hash, thinking_tokens,
                                    int(item["response_tokens"]))
            prompt_hash = stable_hash(record["prompt"])
            if prompt_hash != record["canonical_prompt_hash"]:
                raise ValueError(f"schedule prompt normalization/hash mismatch: {sid}")
            content_hash = str(record.get("thinking_response_hash") or
                               stable_hash(record["thinking"], record["response"]))
            if content_hash != stable_hash(record["thinking"], record["response"]):
                raise ValueError(f"schedule thinking/response hash mismatch: {sid}")
            exact_identity = str(record.get("exact_record_hash") or exact_record_hash(record))
            if exact_identity != pair_hash:
                raise ValueError(f"schedule exact-record identity mismatch: {sid}")
            train["sample_id"].add(sid); train["source_id"].add(source_id)
            train["pair_hash"].add(pair_hash); train["prompt"].add(prompt_hash)
            train["content"].add(content_hash); train["exact_record_identity"].add(exact_identity)
            pointers.append((sid, str(Path(item["source_shard_path"]).resolve()), int(item["source_byte_offset"])))
            rows += 1
    if rows != EXPECTED_SCHEDULE_ROWS or len(train["sample_id"]) != EXPECTED_SCHEDULE_ROWS:
        raise ValueError(f"common schedule row count mismatch: {rows}")
    return train, pointers


def load_tokenizer():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(T5_ID, revision=T5_REVISION, local_files_only=True)
    tokenizer.model_max_length = 1024
    snapshot = Path(tokenizer.init_kwargs.get("name_or_path", ""))
    if not snapshot.is_dir():
        snapshot = Path(os.environ.get("HF_HUB_CACHE", os.environ.get("HF_HOME", ""))) / \
            f"hub/models--{T5_ID.replace('/', '--')}/snapshots/{T5_REVISION}"
    files = {}
    for name, expected in TOKENIZER_HASHES.items():
        path = snapshot / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"tokenizer artifact mismatch: {path}")
        files[name] = {"path": str(path.resolve()), "sha256": expected, "bytes": path.stat().st_size}
    if int(tokenizer.eos_token_id) != 1 or int(tokenizer.pad_token_id) != 0:
        raise ValueError("T5 EOS/PAD semantics mismatch")
    return tokenizer, files


def validate_tokenization(tokenizer, rows):
    thinking = tokenizer([row["thinking"] for row in rows], add_special_tokens=True,
                         truncation=False)["input_ids"]
    response = tokenizer([row["response"] for row in rows], add_special_tokens=True,
                         truncation=False)["input_ids"]
    eos = int(tokenizer.eos_token_id)
    for row, thinking_ids, response_ids in zip(rows, thinking, response):
        sid = row["sample_id"]
        for name, ids, expected in (("thinking", thinking_ids, row["thinking_tokens"]),
                                    ("response", response_ids, row["response_tokens"])):
            if not ids or ids[-1] != eos or ids.count(eos) != 1 or len(ids) != int(expected):
                raise ValueError(f"{name} tokenizer/EOS/count mismatch: {sid}")


def scan_candidates(paired, selection, train, tokenizer, token_batch_size=128):
    pool_manifest_path = Path(paired["paired_source_pool_manifest"])
    if sha256_file(pool_manifest_path) != paired["paired_source_pool_manifest_sha256"]:
        raise ValueError("paired canonical pool manifest SHA256 mismatch")
    pool = json.loads(pool_manifest_path.read_text())
    if not pool.get("complete") or int(pool["rows"]) != 670_595:
        raise ValueError("canonical exact-dedup pool manifest mismatch")
    found, candidates, pending = set(), [], []
    rejected = Counter()

    def consume_batch():
        if not pending:
            return
        validate_tokenization(tokenizer, pending)
        for row in pending:
            sid = row["sample_id"]
            source_id, pair_hash, expected_thinking = selection[sid]
            prompt_hash = stable_hash(row["prompt"])
            actual_exact = exact_record_hash(row)
            actual_tr = stable_hash(row["thinking"], row["response"])
            stored_tr = str(row.get("thinking_response_hash") or actual_tr)
            derived_source = f"{row['source']}|{row['source_config']}|{sid}"
            if prompt_hash != row["canonical_prompt_hash"] or actual_exact != row["exact_record_hash"] \
                    or stored_tr != actual_tr:
                raise ValueError(f"canonical hash mismatch: {sid}")
            if source_id != derived_source or pair_hash != actual_exact or expected_thinking != int(row["thinking_tokens"]):
                raise ValueError(f"paired/canonical identity mismatch: {sid}")
            thinking_tokens, response_tokens = int(row["thinking_tokens"]), int(row["response_tokens"])
            pair_tokens = int(row["pair_tokens"])
            if pair_tokens != thinking_tokens + response_tokens or pair_tokens > 1024:
                raise ValueError(f"pair token gate mismatch: {sid}")
            if sid in train["sample_id"] or source_id in train["source_id"] or pair_hash in train["pair_hash"]:
                rejected["train_row_identity"] += 1
            elif prompt_hash in train["prompt"]:
                rejected["train_prompt_cluster"] += 1
            elif stored_tr in train["content"]:
                rejected["train_content_cluster"] += 1
            else:
                candidates.append({
                    "sample_id": sid, "source_id": source_id, "source": row["source"],
                    "pair_hash": pair_hash, "canonical_prompt_hash": prompt_hash,
                    "thinking_response_hash": stored_tr,
                    "exact_record_identity": actual_exact,
                    "source_shard": row["_source_shard"], "byte_offset": row["_byte_offset"],
                    "thinking_tokens": thinking_tokens, "response_tokens": response_tokens,
                    "pair_tokens": pair_tokens, "K": (thinking_tokens + 3) // 4,
                    "response_length": response_tokens,
                })
        pending.clear()

    total_pool_rows = 0
    for entry in pool["shards"]:
        path = pool_manifest_path.parent / entry["name"]
        if path.stat().st_size != int(entry["bytes"]) or sha256_file(path) != entry["sha256"]:
            raise ValueError(f"canonical shard identity mismatch: {path}")
        rows = 0
        with path.open("rb") as handle:
            while True:
                offset = handle.tell(); line = handle.readline()
                if not line:
                    break
                rows += 1; total_pool_rows += 1
                row = json.loads(line)
                sid = str(row.get("sample_id") or row.get("source_id") or "")
                if sid not in selection:
                    continue
                if sid in found:
                    raise ValueError(f"duplicate selected canonical identity: {sid}")
                found.add(sid)
                if not isinstance(row.get("thinking"), str) or not row["thinking"].strip():
                    raise ValueError(f"empty thinking in paired pool: {sid}")
                if not isinstance(row.get("response"), str) or not row["response"].strip():
                    raise ValueError(f"empty response in paired pool: {sid}")
                if CONTROL_RE.search(row["thinking"]) or CONTROL_RE.search(row["response"]):
                    raise ValueError(f"residual control token in paired pool: {sid}")
                if not row.get("eligible") or row.get("reject_reason") is not None:
                    raise ValueError(f"ineligible record in paired pool: {sid}")
                row["_source_shard"] = str(path.resolve()); row["_byte_offset"] = offset
                pending.append(row)
                if len(pending) >= token_batch_size:
                    consume_batch()
        if rows != int(entry["rows"]):
            raise ValueError(f"canonical shard row count mismatch: {path}")
    consume_batch()
    if total_pool_rows != int(pool["rows"]) or found != set(selection):
        raise ValueError(f"paired resolution accounting mismatch: pool={total_pool_rows}, found={len(found)}")
    if rejected["train_row_identity"] != EXPECTED_SCHEDULE_ROWS:
        raise ValueError(f"row-level train exclusion mismatch: {dict(rejected)}")
    return candidates, rejected, pool_manifest_path


def make_eval_rows(selected, split, seed):
    rows = []
    for row in selected:
        eval_id = stable_hash("stage_b_heldout_v2", split, seed, row["sample_id"])
        item = {
            "eval_id": eval_id, "split": split,
            **{field: row[field] for field in ("sample_id", "source_id", "source", "pair_hash",
                                                "canonical_prompt_hash", "source_shard", "byte_offset",
                                                "thinking_tokens", "response_tokens", "pair_tokens", "K",
                                                "response_length")},
            "stratum": stratum_name(stratum_key(row)),
            "token_noise_seed": stable_int(seed, eval_id, "token_noise"),
            "plan_noise_seed": stable_int(seed, eval_id, "plan_noise"),
            "token_time_seed": stable_int(seed, eval_id, "token_time"),
            "plan_time_seed": stable_int(seed, eval_id, "plan_time"),
            "branch_seed": stable_int(seed, eval_id, "branch"),
            "thinking_response_hash": row["thinking_response_hash"],
            "exact_record_identity": row["exact_record_identity"],
        }
        if tuple(item) != DEV_FIELDS:
            raise AssertionError("Dev/Test output schema drift")
        rows.append(item)
    return rows


def make_shape_rows(test_rows, seed=45, n=1000):
    selected, audit = select_stratified(test_rows, n, seed)
    output = []
    for row in selected:
        eval_id = stable_hash("stage_b_generation_shape_test_v2", seed, row["eval_id"])
        item = {
            "eval_id": eval_id,
            "source_sample_hash": stable_hash(row["sample_id"]),
            "K": int(row["K"]), "plan_mask_length": int(row["K"]),
            "response_length": int(row["response_length"]),
            "response_mask_length": int(row["response_length"]),
            "token_noise_seed": stable_int(seed, eval_id, "token_noise"),
            "plan_noise_seed": stable_int(seed, eval_id, "plan_noise"),
            "sampling_seed": stable_int(seed, eval_id, "sampling"),
            "audit_source_hash": stable_hash(row["source_shard"], row["byte_offset"],
                                               row["sample_id"], row["pair_hash"]),
        }
        if tuple(item) != SHAPE_FIELDS:
            raise AssertionError("generation shape output contains forbidden fields")
        output.append(item)
    return output, audit, {row["source_sample_hash"] for row in output}


def assert_disjoint(train, dev, test, shape_source_hashes=None):
    sets = {}
    for name, rows in (("dev", dev), ("test", test)):
        sets[name] = {field: {row[field] for row in rows}
                      for field in ("sample_id", "source_id", "pair_hash", "canonical_prompt_hash",
                                    "thinking_response_hash", "exact_record_identity")}
        if len(sets[name]["canonical_prompt_hash"]) != len(rows):
            raise AssertionError(f"{name} prompt hash is not unique")
        if len(sets[name]["thinking_response_hash"]) != len(rows):
            raise AssertionError(f"{name} content hash is not unique")
    checks = {}
    train_names = {"sample_id": train["sample_id"], "source_id": train["source_id"],
                   "pair_hash": train["pair_hash"], "canonical_prompt_hash": train["prompt"],
                   "thinking_response_hash": train["content"],
                   "exact_record_identity": train["exact_record_identity"]}
    for field in train_names:
        for split in ("dev", "test"):
            count = len(train_names[field] & sets[split][field])
            checks[f"train_{split}_{field}_intersection"] = count
            if count:
                raise AssertionError(f"train/{split} {field} overlap: {count}")
        count = len(sets["dev"][field] & sets["test"][field])
        checks[f"dev_test_{field}_intersection"] = count
        if count:
            raise AssertionError(f"dev/test {field} overlap: {count}")
    if shape_source_hashes is not None:
        test_hashes = {stable_hash(row["sample_id"]) for row in test}
        if not shape_source_hashes <= test_hashes:
            raise AssertionError("generation shape subset contains a non-test sample")
        checks["generation_non_test_count"] = len(shape_source_hashes - test_hashes)
    return checks


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
    return {"path": path.name, "rows": len(rows), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def write_split(stage, dirname, rows, purpose, seed, distribution, semantics, extra=None):
    directory = stage / dirname; directory.mkdir()
    data_name = "manifest_rows.jsonl" if "generation" not in dirname else "shapes.jsonl"
    file_info = write_jsonl(directory / data_name, rows)
    audit_path = directory / "audit.json"
    audit_path.write_text(json.dumps({"rows": len(rows), "distribution": distribution},
                                     indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "complete": True, "formal_ready": False, "stage_b_heldout": True,
        "end_to_end_heldout": False, "stage_a_mlp_seen": True, "whitener_seen": True,
        "eligible_for_stage_b_model_selection": False, "purpose": purpose,
        "seed": seed, "rows": len(rows), "file": file_info, "distribution": distribution,
        "audit": {"path": audit_path.name, "sha256": sha256_file(audit_path),
                  "bytes": audit_path.stat().st_size},
        **semantics,
    }
    if extra:
        manifest.update(extra)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sums = []
    for path in sorted((directory / name for name in (data_name, "audit.json", "manifest.json")),
                       key=lambda value: value.name):
        sums.append(f"{sha256_file(path)}  {path.name}")
    (directory / "sha256sums.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
    return {"path": str((Path(dirname) / "manifest.json")), "sha256": sha256_file(manifest_path),
            "data_sha256": file_info["sha256"], "rows": len(rows)}


def build(args):
    out = Path(args.output_dir)
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    stage = out.parent / f".{out.name}.staging.{os.getpid()}"
    if stage.exists():
        raise FileExistsError(stage)
    stage.mkdir(parents=True)
    try:
        paired_path, schedule_path = Path(args.paired_manifest), Path(args.schedule)
        if sha256_file(paired_path) != args.paired_manifest_sha256:
            raise ValueError("paired manifest SHA256 mismatch")
        paired = json.loads(paired_path.read_text())
        if not paired.get("complete") or not paired.get("training_ready") \
                or int(paired.get("rows", -1)) != EXPECTED_POOL_ROWS:
            raise ValueError("paired manifest complete/rows gate failed")
        selection = load_selection(paired)
        train, _ = load_schedule(schedule_path, selection, args.schedule_sha256)
        tokenizer, tokenizer_files = load_tokenizer()
        candidates, rejected, pool_manifest_path = scan_candidates(
            paired, selection, train, tokenizer, args.token_batch_size,
        )
        row_level_remaining = EXPECTED_POOL_ROWS - EXPECTED_SCHEDULE_ROWS
        if row_level_remaining != 408_692:
            raise AssertionError("row-level arithmetic changed")
        prompt_remaining = row_level_remaining - rejected["train_prompt_cluster"]
        content_filtered = rejected["train_content_cluster"]
        eligible_remaining = len(candidates)
        test_selected, test_distribution = select_stratified(candidates, 10_000, 44)
        test_prompts = {row["canonical_prompt_hash"] for row in test_selected}
        test_contents = {row["thinking_response_hash"] for row in test_selected}
        dev_selected, dev_distribution = select_stratified(
            candidates, 2_000, 43, test_prompts, test_contents,
        )
        dev_rows = make_eval_rows(dev_selected, "dev", 43)
        test_rows = make_eval_rows(test_selected, "test", 44)
        shape_rows, shape_distribution, shape_sources = make_shape_rows(test_rows, 45, 1000)
        intersections = assert_disjoint(train, dev_rows, test_rows, shape_sources)
        code_files = [Path(__file__).resolve(), ROOT / "src/utils/tpt_million_data.py",
                      ROOT / "src/utils/thinking_tokenization.py", ROOT / "src/utils/data_utils.py"]
        semantics = {
            "paired_manifest": str(paired_path.resolve()),
            "paired_manifest_sha256": sha256_file(paired_path),
            "canonical_pool_manifest": str(pool_manifest_path.resolve()),
            "canonical_pool_manifest_sha256": sha256_file(pool_manifest_path),
            "common_schedule": str(schedule_path.resolve()),
            "common_schedule_sha256": sha256_file(schedule_path),
            "normalization": "src.utils.tpt_million_data.stable_hash(prompt)",
            "normalization_source_sha256": sha256_file(ROOT / "src/utils/tpt_million_data.py"),
            "tokenizer": {"id": T5_ID, "revision": T5_REVISION, "files": tokenizer_files,
                          "add_special_tokens": True, "terminal_eos_exactly_once": True,
                          "truncation": False, "eos_token_id": 1, "pad_token_id": 0},
            "bucket_definitions": {"K": K_BUCKETS, "response_length": R_BUCKETS},
            "selection_policy": "test-first; stable-hash representative; integer largest-remainder; same-source then adjacent K/response bucket redistribution",
            "thinking_response_hash": "existing canonical hash validated against stable_hash(thinking,response)",
            "code_sha256": {str(path.relative_to(ROOT)): sha256_file(path) for path in code_files},
            "stage_b_train_seen": False, "generation_gold_content_used": False,
            "semantic_near_dedup_performed": False,
        }
        outputs = {}
        outputs["dev"] = write_split(
            stage, "stage_b_dev_n2000_seed43_v2", dev_rows,
            "engineering_and_protocol_development", 43, dev_distribution, semantics,
        )
        outputs["test"] = write_split(
            stage, "stage_b_test_n10000_seed44_v2", test_rows,
            "final_stage_b_evaluation_only", 44, test_distribution, semantics,
        )
        outputs["generation_shape_test"] = write_split(
            stage, "stage_b_generation_shape_test_n1000_seed45_v2", shape_rows,
            "paired_unconditional_generation_comparison", 45, shape_distribution, semantics,
            {"gold_content_used": False, "generation_gold_content_used": False,
             "allowed_fields": SHAPE_FIELDS,
             "subset_of_test_manifest": outputs["test"]["path"],
             "subset_of_test_manifest_sha256": outputs["test"]["sha256"]},
        )
        build_manifest = {
            "complete": True, "formal_ready": False, "version": 2,
            "input_rows": EXPECTED_POOL_ROWS, "train_schedule_rows": EXPECTED_SCHEDULE_ROWS,
            "row_level_remaining": row_level_remaining,
            "prompt_cluster_filtered_remaining": prompt_remaining,
            "content_cluster_additional_filtered": content_filtered,
            "final_eligible_rows": eligible_remaining,
            "exclusion_counts": dict(rejected), "outputs": outputs,
            "zero_intersections": intersections,
            "generation_allowed_fields": SHAPE_FIELDS,
            "generation_gold_content_used": False,
            "stage_b_heldout": True, "end_to_end_heldout": False,
            "stage_b_train_seen": False, "stage_a_mlp_seen": True, "whitener_seen": True,
            "eligible_for_stage_b_model_selection": False,
            "semantic_near_dedup_performed": False,
            **semantics,
        }
        audit_report = {
            "complete": True, "input_rows": EXPECTED_POOL_ROWS,
            "row_level_remaining": row_level_remaining,
            "prompt_cluster_filtered_remaining": prompt_remaining,
            "content_cluster_additional_filtered": content_filtered,
            "final_eligible_rows": eligible_remaining,
            "exclusion_counts": dict(rejected), "zero_intersections": intersections,
        }
        (stage / "audit_report.json").write_text(
            json.dumps(audit_report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with (stage / "bucket_counts.csv").open("x", newline="", encoding="utf-8") as handle:
            fields = ["split", "stratum", "raw_rows", "unique_prompt_content_capacity",
                      "largest_remainder_target", "actual", "redistributed"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for split, distribution in (("test", test_distribution), ("dev", dev_distribution),
                                        ("generation_shape_test", shape_distribution)):
                for row in distribution:
                    writer.writerow({"split": split, **row})
        (stage / "README.md").write_text(
            "# Stage-B heldout v2\n\nProvenance-only Stage-B heldout manifests. "
            "They exclude train prompt and thinking-response content clusters. "
            "They are not end-to-end held out because Stage-A and whitening saw the source pool.\n",
            encoding="utf-8")
        manifest_path = stage / "split_manifest.json"
        manifest_path.write_text(json.dumps(build_manifest, indent=2, sort_keys=True) + "\n")
        sums = []
        for path in sorted(stage.rglob("*")):
            if path.is_file() and path.name != "sha256sums.txt":
                sums.append(f"{sha256_file(path)}  {path.relative_to(stage)}")
        (stage / "sha256sums.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")
        os.rename(stage, out)
        print(json.dumps({"output": str(out), "prompt_cluster_filtered_remaining": prompt_remaining,
                          "content_cluster_additional_filtered": content_filtered,
                          "final_eligible_rows": eligible_remaining,
                          "outputs": outputs,
                          "manifest_sha256": sha256_file(out / "split_manifest.json")}, indent=2))
    except Exception as error:
        (stage / "failure.json").write_text(json.dumps({"complete": False, "error": str(error)}, indent=2) + "\n")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paired-manifest", required=True)
    parser.add_argument("--paired-manifest-sha256", default="ed61921c9e6f464275d67ab8765987ece8d2594f80231b6b6bb3a407f0299980")
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--schedule-sha256", default=EXPECTED_SCHEDULE_SHA256)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--token-batch-size", type=int, default=128)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
