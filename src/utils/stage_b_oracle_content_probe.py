"""Pure preflight helpers for the common80k Ordered oracle-content probe."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
from utils.tpt_million_data import stable_hash

MODES=('null','self_planning_first','oracle_matched','oracle_shuffled')
IDENTITY_FIELDS=('sample_id','source_id','pair_hash','canonical_prompt_hash','thinking_response_hash','exact_record_identity')

def score(seed,*parts):return hashlib.sha256(('\0'.join(map(str,(seed,*parts)))).encode()).hexdigest()

def select_probe_mapping(rows,count):
    """Keep formal order unchanged; smoke deterministically includes scarce K=8 and K=20."""
    if count == len(rows):return list(rows)
    if count != 4:raise ValueError('probe mapping count must be 4 or the full mapping')
    selected=[next((r for r in rows if int(r['recipient_K'])==k),None) for k in (8,20)]
    remaining=[r for r in rows if r not in selected]
    selected.extend((min(remaining,key=lambda r:(int(r['recipient_K']),r['eval_id'])),
                     max(remaining,key=lambda r:(int(r['recipient_K']),r['eval_id']))))
    if any(r is None for r in selected) or len({r['eval_id'] for r in selected})!=4:
        raise ValueError('deterministic smoke K coverage unavailable')
    return selected

def map_shape_recipients(shape_rows,test_rows):
    by_hash={stable_hash(r['sample_id']):r for r in test_rows}
    if len(by_hash)!=len(test_rows):raise ValueError('test source_sample_hash collision')
    out=[]
    for s in shape_rows:
        r=by_hash.get(s['source_sample_hash'])
        if r is None:raise ValueError(f"shape recipient absent from Test: {s['eval_id']}")
        if int(r['K'])!=int(s['K']) or int(r['response_length'])!=int(s['response_length']):raise ValueError(f"shape/Test mismatch: {s['eval_id']}")
        out.append((s,r))
    if len(out)!=1000 or len({r['sample_id'] for _,r in out})!=1000:raise ValueError('expected 1,000 unique recipients')
    return out

def identities_disjoint(a,b):
    return all(str(a.get(k,''))!=str(b.get(k,'')) for k in IDENTITY_FIELDS)

def exact_k_unique_mapping(recipients,donor_pool,seed=42):
    by_k=defaultdict(list)
    for d in donor_pool:by_k[int(d['K'])].append(d)
    for k in by_k:by_k[k].sort(key=lambda d:score(seed,'donor',d['sample_id']))
    used=set();mapping=[];counts={str(k):len(v) for k,v in sorted(by_k.items())}
    for shape,r in recipients:
        candidates=[d for d in by_k.get(int(r['K']),[]) if d['sample_id'] not in used and identities_disjoint(r,d)]
        if not candidates:raise ValueError(f"exact-K unique donor unavailable before model load: eval_id={shape['eval_id']} sample_id={r['sample_id']} K={r['K']} pool_count={len(by_k.get(int(r['K']),[]))}")
        d=min(candidates,key=lambda x:score(seed,shape['eval_id'],x['sample_id']))
        used.add(d['sample_id']);mapping.append({'eval_id':shape['eval_id'],'recipient_sample_id':r['sample_id'],'donor_sample_id':d['sample_id'],
          'recipient_K':int(r['K']),'donor_K':int(d['K']),'recipient_pair_hash':r['pair_hash'],'donor_pair_hash':d['pair_hash'],
          'recipient_source_shard':r['source_shard'],'recipient_byte_offset':int(r['byte_offset']),
          'donor_source_shard':d['source_shard'],'donor_byte_offset':int(d['byte_offset'])})
    if len({x['donor_sample_id'] for x in mapping})!=len(mapping):raise AssertionError('donors are not globally unique')
    if any(x['recipient_sample_id']==x['donor_sample_id'] or x['recipient_K']!=x['donor_K'] for x in mapping):raise AssertionError('fixed point or non-exact K')
    payload=''.join(json.dumps(x,sort_keys=True,separators=(',',':'))+'\n' for x in mapping).encode()
    return mapping,{'recipient_count':len(mapping),'donor_unique_count':len({x['donor_sample_id'] for x in mapping}),'fixed_point_count':0,
      'nearest_k_used':False,'donor_pool':'heldout_v2_test_n10000','K_pool_counts':counts,'mapping_sha256':hashlib.sha256(payload).hexdigest()}

def read_pointer(path,offset,expected_id):
    with open(path,'rb') as f:f.seek(int(offset));row=json.loads(f.readline())
    if str(row.get('sample_id') or row.get('source_id'))!=expected_id:raise ValueError(f'pointer identity mismatch: {expected_id}')
    return row

def oracle_model_input(row):
    """Return only privileged thinking; never expose response to the generator batch."""
    thinking=row.get('thinking')
    if not isinstance(thinking,str) or not thinking.strip():raise ValueError('oracle thinking missing')
    return {'thinking':thinking,'sample_id':row['sample_id']}
