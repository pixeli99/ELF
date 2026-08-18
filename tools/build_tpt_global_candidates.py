#!/usr/bin/env python3
"""Build deterministic canonical and exact-deduplicated TPT million v1 pools."""
import argparse
import csv
import hashlib
import heapq
import json
import math
import os
import shutil
import sqlite3
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.tpt_million_data import CONTROL_RE, normalize_text

EXPECTED = {
    "gta": 97228,
    "nemotron": 472414,
    "dolphin-deepseek": 53309,
    "dolphin-flash": 47647,
}
EXPECTED_TOTAL = 670598
T5_ID = "t5-small"
T5_REVISION = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4"
TOKENIZER_FILES = ("spiece.model", "tokenizer.json", "tokenizer_config.json")
LENGTH_BUCKETS = ((0, 256), (257, 512), (513, 768), (769, 1024))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def framed_hash(*fields):
    payload = json.dumps([normalize_text(str(value)) for value in fields], ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def exact_record_hash(record):
    return framed_hash(record.get("prompt", ""), record["thinking"], record["response"])


def tokenizer_provenance(snapshot):
    snapshot = Path(snapshot)
    if snapshot.name != T5_REVISION:
        raise RuntimeError("T5 snapshot revision mismatch")
    files = {}
    for name in TOKENIZER_FILES:
        path = snapshot / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[name] = {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
    return {"model_id": T5_ID, "revision": T5_REVISION, "files": files,
            "add_special_tokens": True, "terminal_eos_exactly_once": True,
            "truncation": False, "eos_token_id": 1, "pad_token_id": 0}


def _manifest_shards(manifest_path, manifest):
    base = Path(manifest_path).parent
    accepted = [item for item in manifest.get("shards", []) if item.get("name", "").startswith("eligible-")]
    if not accepted:
        raise RuntimeError(f"no accepted shards in {manifest_path}")
    return [(base / item["name"], item) for item in accepted]


def validate_input_manifests(named_manifests, tokenizer_info):
    report = {"passed": False, "sources": {}, "expected_total": EXPECTED_TOTAL,
              "tokenizer_provenance": tokenizer_info, "gta_scope_note":
              "GTA raw_rows=97228 here is the prefiltered eligible import, not the original 430788-row corpus."}
    total = 0
    manifests = []
    for name, path in named_manifests:
        manifest = json.loads(Path(path).read_text())
        if not manifest.get("complete") or manifest.get("formal_ready") is not False:
            raise RuntimeError(f"invalid complete/formal_ready state: {path}")
        if manifest.get("accepted_rows") != EXPECTED[name]:
            raise RuntimeError(f"accepted count mismatch for {name}")
        protocol = manifest.get("tokenizer", {})
        required_protocol = {"id": T5_ID, "revision": T5_REVISION, "add_special_tokens": True,
                             "terminal_eos_exactly_once": True, "truncation": False}
        if protocol != required_protocol:
            raise RuntimeError(f"tokenizer protocol mismatch for {name}: {protocol}")
        frozen_path = Path(manifest["source_manifest"])
        if sha256_file(frozen_path) != manifest["source_manifest_sha256"]:
            raise RuntimeError(f"frozen source manifest hash mismatch for {name}")
        frozen = json.loads(frozen_path.read_text())
        source = manifest.get("source", {})
        for key in ("dataset", "revision", "config", "split"):
            if source.get(key) != frozen.get(key):
                raise RuntimeError(f"source identity mismatch for {name}: {key}")
        actual_rows = 0
        shard_report = []
        for shard_path, shard in _manifest_shards(path, manifest):
            if not shard_path.is_file() or sha256_file(shard_path) != shard["sha256"]:
                raise RuntimeError(f"accepted shard integrity failure: {shard_path}")
            rows = sum(1 for _ in shard_path.open("rb"))
            if rows != shard["rows"]:
                raise RuntimeError(f"accepted shard row mismatch: {shard_path}")
            actual_rows += rows
            shard_report.append({"path": str(shard_path), "rows": rows, "bytes": shard_path.stat().st_size,
                                 "sha256": shard["sha256"]})
        if actual_rows != EXPECTED[name]:
            raise RuntimeError(f"actual accepted rows mismatch for {name}")
        report["sources"][name] = {"manifest": str(path), "manifest_sha256": sha256_file(path),
                                   "accepted_rows": actual_rows, "source": source, "shards": shard_report,
                                   "tokenizer": protocol, "tokenizer_files": tokenizer_info["files"]}
        total += actual_rows
        manifests.append((name, manifest, shard_report))
    if total != EXPECTED_TOTAL:
        raise RuntimeError(f"four-source total {total} != {EXPECTED_TOTAL}")
    report.update({"passed": True, "actual_total": total, "all_sources_share_tokenizer": True})
    return report, manifests


def validate_record(record, source_name):
    required = ("sample_id", "source", "source_revision", "source_config", "source_file",
                "source_row_index", "raw_sha256", "prompt", "thinking", "response", "prompt_tokens",
                "thinking_tokens", "response_tokens", "pair_tokens", "full_tokens", "task_class",
                "classification_reason", "classifier_version", "provenance_type",
                "canonical_prompt_hash", "pair_hash", "eligible")
    missing = [key for key in required if key not in record]
    if missing:
        raise RuntimeError(f"{source_name}: missing canonical fields {missing}")
    if not isinstance(record["thinking"], str) or not record["thinking"].strip():
        raise RuntimeError(f"{source_name}: empty thinking")
    if not isinstance(record["response"], str) or not record["response"].strip():
        raise RuntimeError(f"{source_name}: empty response")
    if CONTROL_RE.search(record["thinking"]) or CONTROL_RE.search(record["response"]):
        raise RuntimeError(f"{source_name}: residual control token")
    thinking = int(record["thinking_tokens"])
    response = int(record["response_tokens"])
    pair = int(record["pair_tokens"])
    if thinking <= 0 or response <= 0 or pair != thinking + response or pair > 1024:
        raise RuntimeError(f"{source_name}: invalid pair token accounting")
    if not record["eligible"] or record.get("reject_reason") is not None:
        raise RuntimeError(f"{source_name}: accepted shard contains ineligible row")


class ShardWriter:
    def __init__(self, root, rows_per_shard):
        self.root, self.rows_per_shard = Path(root), rows_per_shard
        self.handle = None
        self.path = None
        self.shard_rows = self.total_rows = 0
        self.shards = []

    def write(self, record):
        if self.handle is None or self.shard_rows == self.rows_per_shard:
            self.close_shard()
            self.path = self.root / f"part-{len(self.shards):05d}.jsonl"
            self.handle = self.path.open("x", encoding="utf-8")
            self.shard_rows = 0
        self.handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        self.shard_rows += 1
        self.total_rows += 1

    def close_shard(self):
        if self.handle is None:
            return
        self.handle.close()
        self.shards.append({"name": self.path.name, "rows": self.shard_rows,
                            "bytes": self.path.stat().st_size, "sha256": sha256_file(self.path)})
        self.handle = None

    def close(self):
        self.close_shard()
        return self.shards


def percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))]


def length_summary(values):
    return {"rows": len(values), "token_sum": sum(values), "mean": statistics.fmean(values) if values else None,
            "p50": percentile(values, .50), "p90": percentile(values, .90),
            "p95": percentile(values, .95), "p99": percentile(values, .99), "max": max(values) if values else None,
            "buckets": {f"{lo}-{hi}": sum(lo <= value <= hi for value in values) for lo, hi in LENGTH_BUCKETS}}


def atomic_promote(staging, target):
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"refusing overwrite: {target}")
    os.rename(staging, target)


def build(args):
    report_target = Path(args.report_dir)
    all_target = Path(args.canonical_all_dir)
    dedup_target = Path(args.dedup_dir)
    for path in (report_target, all_target, dedup_target):
        if path.exists():
            raise FileExistsError(f"refusing overwrite: {path}")
    report_stage = report_target.parent / f".{report_target.name}.staging.{os.getpid()}"
    all_stage = all_target.parent / f".{all_target.name}.staging.{os.getpid()}"
    dedup_stage = dedup_target.parent / f".{dedup_target.name}.staging.{os.getpid()}"
    for path in (report_stage, all_stage, dedup_stage):
        path.mkdir(parents=True)
    try:
        tokenizer_info = tokenizer_provenance(args.tokenizer_snapshot)
        named = [(item.split("=", 1)[0], item.split("=", 1)[1]) for item in args.input_manifest]
        if [name for name, _ in named] != ["gta", "nemotron", "dolphin-deepseek", "dolphin-flash"]:
            raise RuntimeError("input manifest order must be gta,nemotron,dolphin-deepseek,dolphin-flash")
        gate, manifests = validate_input_manifests(named, tokenizer_info)
        (report_stage / "input_gate_report.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
        db = sqlite3.connect(report_stage / "build.sqlite3")
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("CREATE TABLE members(exact_hash TEXT, sample_id TEXT PRIMARY KEY, source TEXT, prompt_hash TEXT, tr_hash TEXT)")
        db.execute("CREATE TABLE exact_clusters(exact_hash TEXT PRIMARY KEY, keeper_id TEXT, keeper_key TEXT, n INTEGER)")
        all_writer = ShardWriter(all_stage, args.shard_rows)
        source_counts = Counter()
        task_lengths = defaultdict(list)
        classifier_versions = defaultdict(set)
        source_ids = set()
        review_heap = []
        max_pair = 0
        max_prompt = 0
        prompt_over_1024 = 0
        view_path = report_stage / "task_view_ids.jsonl"
        with view_path.open("x", encoding="utf-8") as view_out:
          for source_name, _, shards in manifests:
            for shard in shards:
              with open(shard["path"], encoding="utf-8") as handle:
                for line in handle:
                        record = json.loads(line)
                        validate_record(record, source_name)
                        identity = f"{record['source']}|{record['source_config']}|{record['sample_id']}"
                        if identity in source_ids:
                            raise RuntimeError(f"duplicate source identity: {identity}")
                        source_ids.add(identity)
                        exact = exact_record_hash(record)
                        tr_hash = framed_hash(record["thinking"], record["response"])
                        record["exact_record_hash"] = exact
                        record["thinking_response_hash"] = tr_hash
                        all_writer.write(record)
                        source_counts[source_name] += 1
                        pair = int(record["pair_tokens"])
                        max_pair = max(max_pair, pair)
                        max_prompt = max(max_prompt, int(record["prompt_tokens"]))
                        prompt_over_1024 += int(record["prompt_tokens"] > 1024)
                        key = (source_name, record["task_class"])
                        task_lengths[key].append(pair)
                        classifier_versions[key].add(record["classifier_version"])
                        if record["task_class"] == "general_text_v1": view = "general_text_v1"
                        elif record["task_class"] in {"math_primary","code_primary","mcq","tool_or_structured","multimodal_dependent"}: view = "auxiliary_specialized"
                        else: view = "other_unknown"
                        view_out.write(json.dumps({"sample_id":record["sample_id"],"source":source_name,
                                                   "task_class":record["task_class"],"view":view},sort_keys=True)+"\n")
                        keeper_key = f"{record['source']}\0{record['source_config']}\0{record['sample_id']}"
                        db.execute("INSERT INTO members VALUES(?,?,?,?,?)", (exact, record["sample_id"], source_name,
                                                                            record["canonical_prompt_hash"], tr_hash))
                        current = db.execute("SELECT keeper_key,n FROM exact_clusters WHERE exact_hash=?", (exact,)).fetchone()
                        if current is None:
                            db.execute("INSERT INTO exact_clusters VALUES(?,?,?,1)", (exact, record["sample_id"], keeper_key))
                        else:
                            if keeper_key < current[0]:
                                db.execute("UPDATE exact_clusters SET keeper_id=?,keeper_key=?,n=n+1 WHERE exact_hash=?",
                                           (record["sample_id"], keeper_key, exact))
                            else:
                                db.execute("UPDATE exact_clusters SET n=n+1 WHERE exact_hash=?", (exact,))
                        rank = int(hashlib.sha256(("42|" + identity).encode()).hexdigest(), 16)
                        item = (-rank, identity, {k: record.get(k) for k in
                                ("sample_id", "source", "source_config", "source_row_index", "raw_sha256",
                                 "task_class", "classification_reason", "classifier_version", "pair_tokens")})
                        if len(review_heap) < args.review_rows:
                            heapq.heappush(review_heap, item)
                        elif item > review_heap[0]:
                            heapq.heapreplace(review_heap, item)
            db.commit()
        all_shards = all_writer.close()
        if sum(source_counts.values()) != EXPECTED_TOTAL or dict(source_counts) != EXPECTED:
            raise RuntimeError(f"source reconciliation failed: {dict(source_counts)}")
        duplicate_clusters = db.execute("SELECT count(*) FROM exact_clusters WHERE n>1").fetchone()[0]
        unique_exact = db.execute("SELECT count(*) FROM exact_clusters").fetchone()[0]
        duplicate_rows = EXPECTED_TOTAL - unique_exact
        dedup_writer = ShardWriter(dedup_stage, args.shard_rows)
        mapping_path = report_stage / "exact_duplicate_mapping.jsonl"
        with mapping_path.open("x", encoding="utf-8") as mapping:
            for shard in all_shards:
                with (all_stage / shard["name"]).open(encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        keeper = db.execute("SELECT keeper_id FROM exact_clusters WHERE exact_hash=?",
                                            (record["exact_record_hash"],)).fetchone()[0]
                        if record["sample_id"] == keeper:
                            dedup_writer.write(record)
                        else:
                            mapping.write(json.dumps({"duplicate_id": record["sample_id"], "keeper_id": keeper,
                                                      "exact_record_hash": record["exact_record_hash"],
                                                      "source": record["source"], "source_config": record["source_config"]},
                                                     sort_keys=True) + "\n")
        dedup_shards = dedup_writer.close()
        prompt_path = report_stage / "prompt_clusters.jsonl"
        conflict_path = report_stage / "response_conflicts.jsonl"
        prompt_clusters = prompt_conflicts = 0
        with prompt_path.open("x", encoding="utf-8") as prompt_out, conflict_path.open("x", encoding="utf-8") as conflict_out:
            query = "SELECT prompt_hash,count(*),count(DISTINCT tr_hash),group_concat(DISTINCT source) FROM members GROUP BY prompt_hash ORDER BY prompt_hash"
            for prompt_hash, count, variants, sources in db.execute(query):
                item = {"canonical_prompt_hash": prompt_hash, "rows": count,
                        "thinking_response_variants": variants, "sources": sorted(sources.split(","))}
                prompt_out.write(json.dumps(item, sort_keys=True) + "\n")
                prompt_clusters += 1
                if variants > 1:
                    conflict_out.write(json.dumps(item, sort_keys=True) + "\n")
                    prompt_conflicts += 1
        matrix = Counter()
        cursor = db.execute("SELECT m.exact_hash,m.source FROM members m JOIN exact_clusters c USING(exact_hash) WHERE c.n>1 ORDER BY m.exact_hash,m.source")
        previous = None
        sources = []
        for exact, source in cursor:
            if previous is not None and exact != previous:
                for i, left in enumerate(sources):
                    for right in sources[i + 1:]: matrix[(left, right)] += 1
                sources = []
            previous = exact; sources.append(source)
        if sources:
            for i, left in enumerate(sources):
                for right in sources[i + 1:]: matrix[(left, right)] += 1
        distribution = []
        for (source, task), values in sorted(task_lengths.items()):
            summary = length_summary(values)
            distribution.append({"source": source, "task_class": task, "classifier_version":
                                 ",".join(sorted(classifier_versions[(source, task)])),
                                 "provisional": True, **summary})
        with (report_stage / "source_task_distribution.csv").open("x", newline="", encoding="utf-8") as handle:
            fields = ["source", "task_class", "rows", "share", "token_sum", "mean", "p50", "p90", "p95", "p99", "max", "classifier_version", "provisional"]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for row in distribution:
                flat = {k: row[k] for k in fields if k != "share"}; flat["share"] = row["rows"] / source_counts[row["source"]]
                writer.writerow(flat)
        (report_stage / "source_task_distribution.json").write_text(json.dumps(distribution, indent=2), encoding="utf-8")
        (report_stage / "length_distribution.json").write_text(json.dumps({
            f"{source}|{task}": length_summary(values) for (source, task), values in sorted(task_lengths.items())}, indent=2), encoding="utf-8")
        duplicate_summary = {"rows_before": EXPECTED_TOTAL, "exact_duplicate_clusters": duplicate_clusters,
                             "removed_rows": duplicate_rows, "unique_rows": unique_exact,
                             "source_source_matrix": {"|".join(key): value for key, value in sorted(matrix.items())},
                             "mapping_sha256": sha256_file(mapping_path)}
        (report_stage / "exact_duplicate_summary.json").write_text(json.dumps(duplicate_summary, indent=2), encoding="utf-8")
        general_tasks = {"general_text_v1"}
        general_unique = db.execute("SELECT count(DISTINCT c.exact_hash) FROM exact_clusters c JOIN members m USING(exact_hash) WHERE m.sample_id IN (SELECT sample_id FROM members)").fetchone()[0]
        # Exact general count is derived from the deduplicated output to preserve keeper classification.
        general_unique = 0
        for shard in dedup_shards:
            with (dedup_stage / shard["name"]).open(encoding="utf-8") as handle:
                general_unique += sum(json.loads(line)["task_class"] in general_tasks for line in handle)
        gap = {"all_eligible_pre_dedup": EXPECTED_TOTAL, "gap_to_1m_pre_dedup": 1000000-EXPECTED_TOTAL,
               "exact_unique": unique_exact, "gap_to_1m_exact": 1000000-unique_exact,
               "gap_to_1p02m_exact": 1020000-unique_exact, "provisional_general_unique": general_unique,
               "gap_general_to_1m": 1000000-general_unique,
               "note": "329402 is only the minimum pre-dedup gap; prompt policy, quotas, near-duplicate handling, decontamination and holdouts are not frozen."}
        (report_stage / "gap_report.json").write_text(json.dumps(gap, indent=2), encoding="utf-8")
        (report_stage / "candidate_pool_summary.json").write_text(json.dumps({
            "complete": True, "formal_ready": False, "source_counts": dict(source_counts),
            "canonical_all_rows": EXPECTED_TOTAL, "exact_unique_rows": unique_exact,
            "prompt_clusters": prompt_clusters, "response_conflict_clusters": prompt_conflicts,
            "max_pair_tokens": max_pair, "pair_tokens_over_1024": 0,
            "prompt_tokens_over_1024": prompt_over_1024, "max_prompt_tokens": max_prompt,
            "classification_is_provisional": True,
        }, indent=2), encoding="utf-8")
        with (report_stage / "manual_review_sample_manifest.jsonl").open("x", encoding="utf-8") as handle:
            for _, _, item in sorted(review_heap, reverse=True): handle.write(json.dumps(item, sort_keys=True) + "\n")
        input_md = f"# Input gate\n\nPASS. Four complete accepted sources total 670,598 rows. GTA's 97,228 rows are a prefiltered eligible import from the original 430,788-row corpus; this is not a 100% raw acceptance rate.\n\nAll source manifests declare the same pinned T5 tokenizer protocol. Tokenizer-file hashes are recorded in `input_gate_report.json`.\n\nThe logged `1683 > 1024` warning arose while tokenizing the auxiliary prompt field without truncation. No model forward occurred. The accepted pair maximum is {max_pair}; accepted rows with pair_tokens > 1024: 0. Prompt length does not participate in the current pair gate.\n"
        (report_stage / "input_gate_report.md").write_text(input_md, encoding="utf-8")
        gap_md = f"# Gap report\n\n| Pool | Rows | Gap to 1M | Gap to 1.02M |\n|---|---:|---:|---:|\n| All eligible, pre-dedup | {EXPECTED_TOTAL:,} | {1000000-EXPECTED_TOTAL:,} | {1020000-EXPECTED_TOTAL:,} |\n| Exact deduplicated | {unique_exact:,} | {1000000-unique_exact:,} | {1020000-unique_exact:,} |\n| Provisional general only | {general_unique:,} | {1000000-general_unique:,} | n/a |\n\nThe 329,402 figure is only the minimum pre-dedup gap. It must not be used to launch Qwen generation before prompt policy, task quotas, near-duplicate policy, decontamination, and holdouts are frozen.\n"
        (report_stage / "gap_report.md").write_text(gap_md, encoding="utf-8")
        all_manifest = {"complete": True, "formal_ready": False, "rows": EXPECTED_TOTAL,
                        "rows_per_shard": args.shard_rows, "deterministic_source_order": [n for n,_ in named],
                        "tokenizer_provenance": tokenizer_info, "source_lineage": gate["sources"], "shards": all_shards}
        dedup_manifest = {"complete": True, "formal_ready": False, "rows": unique_exact,
                          "dedup_rule": "exact_record_hash only; stable source/config/sample_id keeper",
                          "rows_per_shard": args.shard_rows, "shards": dedup_shards,
                          "mapping_sha256": sha256_file(mapping_path)}
        (all_stage / "manifest.json").write_text(json.dumps(all_manifest, indent=2), encoding="utf-8")
        (dedup_stage / "manifest.json").write_text(json.dumps(dedup_manifest, indent=2), encoding="utf-8")
        db.close(); os.unlink(report_stage / "build.sqlite3")
        for suffix in ("-wal", "-shm"):
            path = report_stage / ("build.sqlite3" + suffix)
            if path.exists(): path.unlink()
        build_manifest = {"complete": True, "formal_ready": False, "canonical_all": str(all_target),
                          "canonical_exact_dedup": str(dedup_target), "report_dir": str(report_target),
                          "input_gate_sha256": sha256_file(report_stage/"input_gate_report.json"),
                          "exact_duplicate_mapping_sha256": sha256_file(mapping_path),
                          "outputs": {p.name: sha256_file(p) for p in report_stage.iterdir() if p.is_file()}}
        (report_stage / "build_manifest.json").write_text(json.dumps(build_manifest, indent=2), encoding="utf-8")
        atomic_promote(all_stage, all_target); atomic_promote(dedup_stage, dedup_target); atomic_promote(report_stage, report_target)
        print(json.dumps({"canonical_all": EXPECTED_TOTAL, "exact_unique": unique_exact,
                          "prompt_conflicts": prompt_conflicts, "formal_ready": False}, indent=2))
    except Exception as exc:
        (report_stage / "failure.json").write_text(json.dumps({"complete": False, "error": str(exc)}, indent=2), encoding="utf-8")
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", action="append", required=True, help="name=/absolute/manifest.json")
    parser.add_argument("--tokenizer_snapshot", required=True)
    parser.add_argument("--canonical_all_dir", required=True)
    parser.add_argument("--dedup_dir", required=True)
    parser.add_argument("--report_dir", required=True)
    parser.add_argument("--shard_rows", type=int, default=50000)
    parser.add_argument("--review_rows", type=int, default=400)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
