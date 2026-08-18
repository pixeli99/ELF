#!/usr/bin/env python3
import argparse,json,os,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from utils.stage_b_common80k_generation import sha256_file,validate_shape_inputs
from utils.stage_b_oracle_content_probe import map_shape_recipients,exact_k_unique_mapping
from utils.tpt_million_data import stable_hash
from utils.stage_b_oracle_content_probe import read_pointer

def eligible_fallback_rows(split, recipients, tests):
 paired=json.loads(Path(split['paired_manifest']).read_text());selection=set()
 for entry in paired['splits']['train']['selection_files']:
  with open(entry['path']) as f:
   for line in f:selection.add(json.loads(line)['sample_id'])
 schedule=ROOT/'results/tpt_million_v1/common_schedule_80k_seed42_v1/common_schedule_80k_seed42_v1.jsonl'
 train={'sample_id':set(),'source_id':set(),'pair_hash':set(),'prompt':set(),'content':set()}
 with schedule.open() as f:
  for line in f:
   s=json.loads(line);r=read_pointer(s['source_shard_path'],s['source_byte_offset'],s['sample_id'])
   train['sample_id'].add(s['sample_id']);train['source_id'].add(s['source_id']);train['pair_hash'].add(s['pair_hash'])
   train['prompt'].add(r['canonical_prompt_hash']);train['content'].add(r['thinking_response_hash'])
 rec_counts=Counter(int(r['K']) for _,r in recipients);test_counts=Counter(int(r['K']) for r in tests)
 needed={k for k,n in rec_counts.items() if test_counts[k] <= n+1}
 pool_manifest=Path(paired['paired_source_pool_manifest']);pool=json.loads(pool_manifest.read_text());out=[]
 for entry in pool['shards']:
  path=pool_manifest.parent/entry['name']
  with path.open('rb') as f:
   while True:
    offset=f.tell();line=f.readline()
    if not line:break
    r=json.loads(line);sid=str(r.get('sample_id') or r.get('source_id') or '')
    k=(int(r.get('thinking_tokens',0))+3)//4
    if sid not in selection or k not in needed:continue
    source_id=f"{r['source']}|{r['source_config']}|{sid}";pair=r.get('exact_record_hash') or r.get('pair_hash')
    if sid in train['sample_id'] or source_id in train['source_id'] or pair in train['pair_hash'] or r['canonical_prompt_hash'] in train['prompt'] or r['thinking_response_hash'] in train['content']:continue
    out.append({'sample_id':sid,'source_id':source_id,'pair_hash':pair,'canonical_prompt_hash':r['canonical_prompt_hash'],
      'thinking_response_hash':r['thinking_response_hash'],'exact_record_identity':pair,'K':k,'source_shard':str(path),'byte_offset':offset})
 return out,sorted(needed)
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
 if out.exists():raise FileExistsError(out)
 b=ROOT/'results/tpt_million_v1/stage_b_heldout_v2';td=b/'stage_b_test_n10000_seed44_v2';sd=b/'stage_b_generation_shape_test_n1000_seed45_v2'
 shapes=validate_shape_inputs(b/'split_manifest.json',sd/'manifest.json',sd/'shapes.jsonl')
 tests=[json.loads(x) for x in (td/'manifest_rows.jsonl').read_text().splitlines()]
 if len(tests)!=10000:raise ValueError('Test must contain 10,000 rows')
 recipients=map_shape_recipients(shapes,tests);split=json.loads((b/'split_manifest.json').read_text())
 fallback,expanded_k=eligible_fallback_rows(split,recipients,tests);pool=tests+[r for r in fallback if r['sample_id'] not in {x['sample_id'] for x in tests}]
 mapping,audit=exact_k_unique_mapping(recipients,pool,42);audit.update({'donor_pool':'heldout_v2_test_plus_filtered_eligible_fallback','test_pool_rows':len(tests),'eligible_fallback_rows':len(fallback),'expanded_K':expanded_k})
 stage=out.parent/f'{out.name}.staging.{os.getpid()}';stage.mkdir(parents=True)
 data=stage/'donor_mapping.jsonl';data.write_text(''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in mapping))
 manifest={'complete':True,**audit,'seed':42,'shape_data_sha256':sha256_file(sd/'shapes.jsonl'),'test_data_sha256':sha256_file(td/'manifest_rows.jsonl'),
 'data':{'name':data.name,'rows':len(mapping),'bytes':data.stat().st_size,'sha256':sha256_file(data)},'privileged_gold_associated':True,'gold_response_used':False}
 (stage/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');os.rename(stage,out);print(json.dumps(manifest,sort_keys=True))
if __name__=='__main__':main()
