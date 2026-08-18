import sys
from pathlib import Path
import unittest, torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from utils.stage_b_oracle_serial import *

def mapping(n=4):
 return [dict(eval_id=f'e{i}',recipient_sample_id=f'r{i}',donor_sample_id=f'd{i}',recipient_K=(8,20,40,200)[i%4],donor_K=(8,20,40,200)[i%4],recipient_pair_hash=f'rp{i}',donor_pair_hash=f'dp{i}') for i in range(n)]

class SerialOracleTests(unittest.TestCase):
 def test_mode_order_and_single_load(self):
  self.assertTrue(validate_serial_protocol(MODES,{x:1 for x in ('model','t5','mlp','whitener','gpt2')}))
  with self.assertRaises(ValueError):validate_serial_protocol(MODES[::-1],{x:1 for x in ('model','t5','mlp','whitener','gpt2')})

 def test_obsolete_gates_rejected(self):
  kw={x:1 for x in ('model','t5','mlp','whitener','gpt2')}
  for flag in ('cross_gpu_gate','matched_reference','consistency'):
   with self.subTest(flag=flag),self.assertRaises(ValueError):validate_serial_protocol(MODES,kw,**{flag:True})

 def test_base_clone_is_independent_and_equal(self):
  x=torch.randn(1,1024,3);m=clone_base_response(x)
  self.assertTrue(all(torch.equal(v,x) for v in m.values()));m['null'].add_(1)
  self.assertTrue(torch.equal(m['self_planning_first'],x));self.assertFalse(torch.equal(m['null'],x))

 def test_smoke_covers_k8_k20(self):
  self.assertTrue(validate_smoke_mapping(mapping()))
  with self.assertRaises(ValueError):validate_smoke_mapping(mapping()[2:])

 def test_exact_k_unique_no_fixed_mapping(self):
  rows=mapping();self.assertEqual(validate_donor_mapping(rows,4),validate_donor_mapping(rows,4))
  bad=mapping();bad[0]['donor_K']=9
  with self.assertRaises(ValueError):validate_donor_mapping(bad,4)

 def test_bootstrap_deterministic_and_direction(self):
  a=paired_bootstrap([1,2,3],[2,3,4],draws=100);b=paired_bootstrap([1,2,3],[2,3,4],draws=100)
  self.assertEqual(a,b);self.assertEqual(a['mean_delta_nll'],-1);self.assertEqual(a['win_rate'],1)

 def test_evaluator_has_single_serial_loop_and_no_old_consistency(self):
  text=(ROOT/'tools/eval_stage_b_oracle_serial.py').read_text()
  self.assertIn('for mode in MODES',text);self.assertNotIn('_consistency(',text)
  self.assertNotIn('build_plan_target',text);self.assertIn('response SDE random stream differs across modes',text)
  self.assertIn('torch.cuda.manual_seed_all',text)

 def test_launcher_is_single_process_single_gpu(self):
  text=(ROOT/'scripts/run_stage_b_oracle_serial.sh').read_text()
  self.assertNotIn('torchrun',text);self.assertNotIn('MAX_PARALLEL',text);self.assertIn('GPU_ID',text)
  self.assertEqual(text.count('eval_stage_b_oracle_serial.py'),1)

 def test_handoff_keeps_only_stable_serial_entry(self):
  guide=(ROOT/'CODE_GUIDE.md').read_text()
  self.assertIn('Stable serial four-mode Oracle probe',guide)
  self.assertFalse((ROOT/'tools/diagnose_stage_b_planning_first.py').exists())

if __name__=='__main__':unittest.main()
