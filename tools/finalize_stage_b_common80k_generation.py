#!/usr/bin/env python3
"""Strict 14-arm finalizer for common80k Generation Evaluation v1."""
import argparse, csv, hashlib, json, math, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from utils.stage_b_common80k_generation import CONDITIONS, sha256_file, validate_resume_arm, all_workers_finished

PID_SCHEMA_V2 = ("schema_version", "pid", "gpu", "condition_id", "worker_log_work", "worker_log_final")


def _rows(path):
    return [line for line in path.read_text().splitlines() if line.strip()]


def parse_pid_rows(path, root):
    """Read versioned PID metadata plus the historical 3/4-column formats."""
    lines = _rows(path)
    if not lines:
        return []
    first = lines[0].split("\t")
    result = []
    if "condition_id" in first:
        reader = csv.DictReader(lines, delimiter="\t")
        fields = tuple(reader.fieldnames or ())
        if fields != PID_SCHEMA_V2:
            raise ValueError(f"pids.tsv line 1: invalid header {fields}; expected {PID_SCHEMA_V2}")
        for line_no, row in enumerate(reader, 2):
            missing = [key for key in PID_SCHEMA_V2 if row.get(key) in (None, "")]
            if missing:
                raise ValueError(f"pids.tsv line {line_no}: missing fields {missing}")
            if row["schema_version"] != "2":
                raise ValueError(f"pids.tsv line {line_no}: unsupported schema_version={row['schema_version']}")
            result.append({**row, "line_number": line_no})
    else:
        for line_no, line in enumerate(lines, 1):
            values = line.split("\t")
            if len(values) == 3:
                pid, gpu, cid = values; work_log = ""
            elif len(values) == 4:
                pid, gpu, cid, work_log = values
            else:
                raise ValueError(f"pids.tsv line {line_no}: headerless schema requires 3 or 4 fields, got {len(values)}")
            result.append({"schema_version": "legacy", "pid": pid, "gpu": gpu,
                           "condition_id": cid, "worker_log_work": work_log,
                           "worker_log_final": str(root / cid / "worker.log"), "line_number": line_no})
    seen = set()
    for row in result:
        line_no = row["line_number"]
        try: int(row["pid"]); int(row["gpu"])
        except ValueError as exc: raise ValueError(f"pids.tsv line {line_no}: pid/gpu must be integers") from exc
        cid = row["condition_id"]
        if cid in seen: raise ValueError(f"pids.tsv line {line_no}: duplicate condition_id={cid}")
        seen.add(cid)
        final_log = root / cid / "worker.log"
        row["worker_log"] = str(final_log if final_log.is_file() else Path(row.get("worker_log_work", "")))
    return result


def parse_exit_rows(path):
    """Read legacy 2-column and current 4-column exit metadata without dropping fields."""
    lines = _rows(path)
    if not lines: return {}
    first = lines[0].split("\t"); result = {}
    if "condition_id" in first:
        reader = csv.DictReader(lines, delimiter="\t")
        required = {"condition_id", "exit_code"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"exit_codes.tsv line 1: missing required header fields {sorted(required)}")
        iterator = ((n, row) for n, row in enumerate(reader, 2))
    else:
        def legacy():
            for n, line in enumerate(lines, 1):
                values=line.split("\t")
                if len(values) not in (2,3,4):
                    raise ValueError(f"exit_codes.tsv line {n}: unsupported headerless field count={len(values)}")
                yield n,{"condition_id":values[0],"exit_code":values[1]}
        iterator=legacy()
    for line_no,row in iterator:
        cid=row.get("condition_id"); code=row.get("exit_code")
        if not cid or code is None: raise ValueError(f"exit_codes.tsv line {line_no}: missing condition_id/exit_code")
        if cid in result: raise ValueError(f"exit_codes.tsv line {line_no}: duplicate condition_id={cid}")
        try: result[cid]=int(code)
        except ValueError as exc: raise ValueError(f"exit_codes.tsv line {line_no}: invalid exit_code={code}") from exc
    return result


def bootstrap(values, seed=42, draws=10_000):
    import numpy as np
    values = np.asarray(values, dtype=np.float64); rng = np.random.default_rng(seed)
    means = np.empty(draws)
    for start in range(0, draws, 250):
        stop = min(draws, start + 250)
        means[start:stop] = values[rng.integers(0, len(values), (stop-start, len(values)))].mean(1)
    return {"mean_delta_nll": float(values.mean()), "ci95": np.quantile(means, [.025,.975]).tolist(),
            "paired_win_rate": float((values < 0).mean()), "n": len(values)}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--root",required=True); p.add_argument("--expected-rows",type=int,required=True)
    p.add_argument("--validate-arm"); p.add_argument("--smoke-prerequisite"); p.add_argument("--check-root-only",action="store_true")
    a=p.parse_args(); root=Path(a.root)
    if a.smoke_prerequisite:
        smoke=json.loads((Path(a.smoke_prerequisite)/"manifest.json").read_text())
        if not smoke.get("complete") or smoke.get("condition_count")!=14: raise ValueError("formal requires complete 14-arm smoke")
    expected={"shape_data_sha256":"00ea35649134374c1f93ab5b26ff948e3df5950786e269adfea370c5eeb5b23d"}
    if a.check_root_only:
        manifest=json.loads((root/"manifest.json").read_text())
        if not manifest.get("complete") or manifest.get("condition_count")!=14 or manifest.get("expected_rows_per_arm")!=a.expected_rows:
            raise RuntimeError("root manifest incomplete/mismatched")
        for cid,group,alpha,steps in CONDITIONS:
            if not validate_resume_arm(root/cid,{**expected,"condition_id":cid,"num_samples":a.expected_rows}):
                raise RuntimeError(f"{cid}: arm validation failed")
        for name,meta in manifest.get("outputs",{}).items():
            if sha256_file(root/name)!=meta["sha256"]:raise RuntimeError(f"root output hash mismatch: {name}")
        print("GENERATION_EVALUATION_GATE_PASS");return
    if a.validate_arm:
        spec=next(row for row in CONDITIONS if row[0]==a.validate_arm)
        print("PASS" if validate_resume_arm(root/spec[0], {**expected,"condition_id":spec[0],"num_samples":a.expected_rows}) else "FAIL")
        raise SystemExit(0 if validate_resume_arm(root/spec[0], {**expected,"condition_id":spec[0],"num_samples":a.expected_rows}) else 1)
    pid_rows=parse_pid_rows(root/"pids.tsv", root) if (root/"pids.tsv").is_file() else []
    if not all_workers_finished(pid_rows): raise RuntimeError("workers still running; refusing finalize")
    exits=parse_exit_rows(root/"exit_codes.tsv")
    summaries={}; samples={}; noises={}; common=None; gpt_locks=set(); checkpoint_locks={}
    for cid,group,alpha,steps in CONDITIONS:
        if exits.get(cid)!=0: raise RuntimeError(f"{cid}: missing/nonzero exit code")
        if not validate_resume_arm(root/cid,{**expected,"condition_id":cid,"num_samples":a.expected_rows}):
            raise RuntimeError(f"{cid}: incomplete/hash-invalid arm")
        manifest=json.loads((root/cid/"manifest.json").read_text()); summaries[cid]=json.loads((root/cid/"summary.json").read_text())
        gpt_locks.add(manifest["gpt2_input_lock"]); checkpoint_locks[cid]=manifest["checkpoint_sha256"]
        rows=[json.loads(x) for x in (root/cid/"per_sample.jsonl").read_text().splitlines()]; samples[cid]={r["eval_id"]:r for r in rows}
        noises[cid]=json.loads((root/cid/"noise_hashes.json").read_text())
        ids=[eid for eid,row in samples[cid].items() if row.get("token_normalized_nll") is not None]
        common=ids if common is None else [x for x in common if x in set(ids)]
    (root/"common_valid_eval_ids.json").write_text(json.dumps({"eval_ids":common},indent=2)+"\n")
    first=next(iter(noises.values()))
    for cid,group,alpha,steps in CONDITIONS:
        for eid in common:
            if noises[cid][eid]["response_initial_noise_sha256"] != first[eid]["response_initial_noise_sha256"]:
                raise RuntimeError(f"{cid}: response initial noise mismatch")
    plan_arms=[cid for cid,group,alpha,steps in CONDITIONS if group!="vanilla"]
    plan_first=noises[plan_arms[0]]
    for cid in plan_arms[1:]:
        for eid in common:
            if noises[cid][eid]["plan_initial_noise_sha256"] != plan_first[eid]["plan_initial_noise_sha256"]:
                raise RuntimeError(f"{cid}: plan initial noise mismatch")
    paired=[]; raw_deltas=[]; ids=list(summaries)
    for left in ids:
        for right in ids:
            if left>=right:continue
            values=[]
            for eid in common:
                x,y=samples[left][eid],samples[right][eid]
                if x["token_normalized_nll"] is not None and y["token_normalized_nll"] is not None:
                    values.append(x["token_normalized_nll"]-y["token_normalized_nll"])
            if values:
                paired.append({"left":left,"right":right,**bootstrap(values)})
                for eid,value in zip(common,values): raw_deltas.append({"eval_id":eid,"left":left,"right":right,"delta_nll":value})
    with (root/"paired_nll_differences.jsonl").open("w") as f:
        for row in raw_deltas:f.write(json.dumps(row,sort_keys=True)+"\n")
    (root/"paired_bootstrap_summary.json").write_text(json.dumps(paired,indent=2,sort_keys=True)+"\n")
    ordered=[]; native=[]
    for cid,group,alpha,steps in CONDITIONS:
        row={"condition_id":cid,"group":group,"alpha":alpha,"steps":steps,**summaries[cid]}
        if group=="ordered":ordered.append(row)
        if group!="ordered" or alpha in (2.0,1.0):native.append(row)
    for name,rows in (("ordered_trajectory_table.csv",ordered),("four_group_native_table.csv",native)):
        with (root/name).open("w",newline="") as f:
            fields=("condition_id","group","alpha","steps","corpus_gen_ppl","mean_token_frequency_entropy","valid_samples","eos_rate","mean_length")
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows({k:r.get(k) for k in fields} for r in rows)
    if len(gpt_locks)!=1: raise RuntimeError("GPT-2 input lock differs across arms")
    (root/"report.md").write_text("Generation Evaluation v1. Gen-PPL: lower is better. Token-frequency entropy is an auxiliary diversity/degeneration metric and has no simple direction. Absolute PPL is not compared with old OWT.\n")
    output_names=("common_valid_eval_ids.json","paired_nll_differences.jsonl","paired_bootstrap_summary.json",
                  "ordered_trajectory_table.csv","four_group_native_table.csv","report.md")
    manifest={"complete":True,"condition_count":14,"expected_rows_per_arm":a.expected_rows,
        "common_valid_eval_ids":len(common),"gold_text_or_latent_used":False,"fixed_gold_derived_shapes_used":True,
        "entropy_display_name":"Token-frequency entropy","checkpoint_selection":False,
        "shape_data_sha256":expected["shape_data_sha256"],"gpt2_input_lock":next(iter(gpt_locks)),
        "checkpoint_locks":checkpoint_locks,
        "pid_schema":"versioned_v2_or_validated_legacy","worker_processes":[
            {key:row.get(key) for key in ("pid","gpu","condition_id","worker_log")} for row in pid_rows],
        "outputs":{name:{"bytes":(root/name).stat().st_size,"sha256":sha256_file(root/name)} for name in output_names},
        "conditions":[row[0] for row in CONDITIONS]}
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
    (root/"sha256sums.txt").write_text("".join(f"{sha256_file(root/name)}  {name}\n" for name in (*output_names,"manifest.json")))

if __name__=="__main__":main()
