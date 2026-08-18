import hashlib,json,tempfile,unittest
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tools')]
from configs.config import load_config_from_yaml
from utils.data_utils import FormalStageBScheduleDataset
from build_stage_b_common_schedule import derive,STREAMS

class Base:
 def __init__(self,root):
  self.rows=[];self.items=[]
  for i in range(160):
   path=root/f'{i}.jsonl';path.write_text(json.dumps({'sample_id':f's{i}','thinking':'t','response':'r'})+'\n')
   self.rows.append((str(path),0,f's{i}',f'source{i}',4+i%5,8+i%7));self.items.append({'sample_id':f's{i}','source_id':f'source{i}','thinking_input_ids':[1],'response_input_ids':[1]})
 def __getitem__(self,i):return dict(self.items[i])

class TestSchedule(unittest.TestCase):
 def test_seed_derivation_is_stateless_and_stream_separated(self):
  a=[derive(42,3,'sample',x) for x in STREAMS]
  self.assertEqual(a,[derive(42,3,'sample',x) for x in STREAMS]);self.assertEqual(len(set(a)),5)
 def test_schedule_reader_preserves_order_and_metadata(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td);base=Base(root);schedule=root/'schedule.jsonl'
   with schedule.open('w') as f:
    for i,row in enumerate(base.rows):
     item={'presentation_index':i,'optimizer_step':i//8+1,'within_step':i%8,'sample_id':row[2],'source_id':row[3],'pair_hash':f'p{i}','source_shard_path':row[0],'source_byte_offset':0,'thinking_tokens':row[4],'response_tokens':row[5]}
     item.update({f'{s}_seed':derive(42,i,row[2],s) for s in STREAMS});f.write(json.dumps(item)+'\n')
   ds=FormalStageBScheduleDataset(base,schedule)
   self.assertEqual(len(ds),160);self.assertEqual([ds[i]['sample_id'] for i in range(160)],[f's{i}' for i in range(160)])
   self.assertEqual(ds[9]['presentation_index'],9);self.assertIn('branch_seed',ds[9])
 def test_overlay_protocols(self):
  expected={'ordered':(255,.15,0.,1.,False),'diagonal':(255,0.,1.,1.,False),'register':(255,0.,0.,0.,True),'vanilla':(0,0.,0.,0.,False)}
  for group,want in expected.items():
   cfg=load_config_from_yaml(str(ROOT/'src/configs/training_configs'/f'train_stage_b_common80k_{group}_10k_v1.yml'))
   self.assertEqual((cfg.num_plan_slots,cfg.plan_done_frac,cfg.plan_diag_frac,cfg.plan_loss_weight,cfg.plan_register_only),want)
   self.assertEqual(cfg.grad_accum_steps,8);self.assertEqual(cfg.max_optimizer_steps,10000);self.assertEqual(cfg.seed,42)
if __name__=='__main__':unittest.main()
