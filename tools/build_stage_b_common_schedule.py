#!/usr/bin/env python3
"""Build the immutable metadata-only 80k Stage-B presentation schedule."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from utils.data_utils import FormalStageBPairedDataset

STREAMS=('response_noise','token_time','branch','plan_noise','plan_time')
def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def derive(master,index,sid,stream):
 return int(hashlib.sha256(f'common80k-v1\0{master}\0{index}\0{sid}\0{stream}'.encode()).hexdigest()[:16],16)%(2**63-1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--paired-manifest',required=True);p.add_argument('--output',required=True);p.add_argument('--rows',type=int,default=80000);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
 out=Path(a.output)
 if out.exists():raise FileExistsError(out)
 dataset=FormalStageBPairedDataset(a.paired_manifest,tokenizer=None,max_length=1024)
 if len(dataset)!=488692 or a.rows>len(dataset):raise ValueError(f'unexpected paired rows: {len(dataset)}')
 indices=torch.randperm(len(dataset),generator=torch.Generator().manual_seed(a.seed))[:a.rows].tolist()
 stage=Path(str(out)+f'.staging.{os.getpid()}');stage.mkdir(parents=True)
 schedule=stage/'common_schedule_80k_seed42_v1.jsonl';seen=set()
 with schedule.open('w') as handle:
  for presentation,dataset_index in enumerate(indices):
   path,offset,sid,source_id,thinking_tokens,response_tokens=dataset.rows[dataset_index]
   with open(path,'rb') as source:source.seek(offset);record=json.loads(source.readline())
   if str(record.get('sample_id') or record.get('source_id'))!=sid:raise ValueError(f'source pointer drift: {sid}')
   pair_hash=record.get('exact_record_hash') or record.get('pair_hash')
   if not pair_hash or sid in seen:raise ValueError(f'missing pair hash or duplicate ID: {sid}')
   seen.add(sid);row={'presentation_index':presentation,'optimizer_step':presentation//8+1,'within_step':presentation%8,'sample_id':sid,'source_id':source_id,'pair_hash':pair_hash,'source_shard_path':path,'source_byte_offset':offset,'thinking_tokens':thinking_tokens,'response_tokens':response_tokens,'K':(thinking_tokens+3)//4,'response_length':response_tokens}
   for stream in STREAMS:row[f'{stream}_seed']=derive(a.seed,presentation,sid,stream)
   handle.write(json.dumps(row,sort_keys=True)+'\n')
 if len(seen)!=a.rows:raise ValueError('unique schedule accounting failed')
 source_hashes={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'src/train.py',ROOT/'src/train_step.py',ROOT/'src/utils/data_utils.py')}
 manifest={'complete':True,'version':1,'master_seed':a.seed,'rows':a.rows,'unique_sample_ids':len(seen),'microsteps':a.rows,'effective_batch':8,'optimizer_steps':a.rows//8,'coverage':a.rows/len(dataset),'shuffle':False,'selection':'torch.randperm(dataset_rows, generator=manual_seed(42))[:80000]','paired_manifest':str(Path(a.paired_manifest).resolve()),'paired_manifest_sha256':sha(a.paired_manifest),'schedule_file':schedule.name,'schedule_sha256':sha(schedule),'torch_version':torch.__version__,'source_sha256':source_hashes,'seed_streams':list(STREAMS),'no_text_content':True}
 (stage/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.rename(stage,out)
if __name__=='__main__':main()
