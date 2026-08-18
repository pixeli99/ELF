#!/usr/bin/env python3
"""CPU-only visible preflight for common80k Generation Evaluation."""
import argparse, json, os, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from utils.stage_b_common80k_generation import CONDITIONS, sha256_file, validate_shape_inputs, validate_resume_arm
from tools.eval_stage_b_common80k_generation import GPT2_SNAPSHOT, RUNS, artifact_lock

def log(stage,message,status="CHECK"):
 stamp=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
 print(f"[{stamp}] [preflight] [{status}] [{stage}] {message}",flush=True)

def gate(stage,expected,actual,ok):
 log(stage,f"expected={expected} actual={actual}","PASS" if ok else "FAIL")
 if not ok:raise ValueError(f"{stage}: expected={expected}, actual={actual}")

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('smoke','formal'),required=True);p.add_argument('--root',required=True)
 p.add_argument('--resume',type=int,choices=(0,1),required=True);p.add_argument('--gpu-ids',required=True);p.add_argument('--max-parallel',type=int,required=True)
 a=p.parse_args();root=Path(a.root);gpus=a.gpu_ids.split(',')
 gate('CONDITIONS',14,len(CONDITIONS),len(CONDITIONS)==14)
 gate('GPU_IDS','unique nonnegative integers',gpus,len(gpus)>0 and len(set(gpus))==len(gpus) and all(x.isdigit() for x in gpus))
 gate('MAX_PARALLEL',f'1..{len(gpus)}',a.max_parallel,1<=a.max_parallel<=len(gpus))
 split=ROOT/'results/tpt_million_v1/stage_b_heldout_v2/split_manifest.json';shape_dir=split.parent/'stage_b_generation_shape_test_n1000_seed45_v2'
 rows=validate_shape_inputs(split,shape_dir/'manifest.json',shape_dir/'shapes.jsonl')
 log('SHAPE',f"split_sha={sha256_file(split)} manifest_sha={sha256_file(shape_dir/'manifest.json')} data_sha={sha256_file(shape_dir/'shapes.jsonl')} rows={len(rows)}",'PASS')
 gpt=artifact_lock();log('GPT2',f"snapshot={gpt['snapshot']} revision={gpt['revision']} weights_sha={gpt['files']['model.safetensors']['sha256']}",'PASS')
 checkpoint_locks={}
 for group,run in RUNS.items():
  path=run/'checkpoint_10000';gate(f'CHECKPOINT_{group}','exists',path,path.is_file())
  checkpoint_locks[group]=sha256_file(path);log(f'CHECKPOINT_{group}',f"sha256={checkpoint_locks[group]}",'PASS')
 if a.resume==0:gate('OUTPUT','absent',str(root),not root.exists())
 else:gate('OUTPUT','existing resumable root',str(root),root.is_dir())
 statuses=[];expected_rows=4 if a.mode=='smoke' else 1000
 for cid,group,alpha,steps in CONDITIONS:
  valid=validate_resume_arm(root/cid,{'condition_id':cid,'num_samples':expected_rows,
    'shape_data_sha256':'00ea35649134374c1f93ab5b26ff948e3df5950786e269adfea370c5eeb5b23d'}) if root.is_dir() else False
  action='SKIP_HASH_VALID' if a.resume and valid else 'RUN'
  statuses.append({'condition_id':cid,'group':group,'alpha':alpha,'steps':steps,'action':action})
  log('ARM_PLAN',f"condition={cid} group={group} alpha={alpha} nfe={steps} action={action}",'PASS')
 pending=[x for x in statuses if x['action']=='RUN'];waves=[pending[i:i+a.max_parallel] for i in range(0,len(pending),a.max_parallel)]
 log('SCHEDULE',f"wave_sizes={[len(x) for x in waves]} total_pending={len(pending)}",'PASS')
 print(json.dumps({'complete':True,'mode':a.mode,'root':str(root),'resume':a.resume,'gpu_ids':gpus,
  'max_parallel':a.max_parallel,'conditions':statuses,'wave_sizes':[len(x) for x in waves],
  'checkpoint_sha256':checkpoint_locks,'gpt2_revision':gpt['revision']},sort_keys=True),flush=True)
 log('RESULT','GENERATION_PREFLIGHT_GATE_PASS','PASS')

if __name__=='__main__':main()
