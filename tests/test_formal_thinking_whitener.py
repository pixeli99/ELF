import tempfile,unittest
from pathlib import Path
import torch
from tools.compute_formal_thinking_whitener import DualMoments,state_sha
from src.modules.thinking_resampler import group_batched_thinking
from src.utils.formal_thinking_mlp import whiten,inverse_whiten,refuse_existing
class Tests(unittest.TestCase):
 def test_welford_raw_direct_population(self):
  x=torch.randn(101,7);m=DualMoments(7);m.update(x[:33]);m.update(x[33:]);mean,var,rmean,rvar=m.finish();self.assertTrue(torch.allclose(mean,x.double().mean(0),atol=1e-14));self.assertTrue(torch.allclose(var,x.double().var(0,unbiased=False),atol=1e-14));self.assertTrue(torch.allclose(mean,rmean,atol=1e-14));self.assertTrue(torch.allclose(var,rvar,atol=1e-14))
 def test_batch_size_invariance(self):
  x=torch.randn(99,3);a=DualMoments(3);b=DualMoments(3)
  for z in x.split(7):a.update(z)
  for z in x.split(13):b.update(z)
  self.assertTrue(torch.allclose(a.finish()[0],b.finish()[0],atol=1e-14));self.assertTrue(torch.allclose(a.finish()[1],b.finish()[1],atol=1e-14))
 def test_plan_slots_and_padding(self):
  for n,k in ((1,1),(2,1),(3,1),(4,1),(5,2)):
   x=torch.randn(1,n,2);mask=torch.ones(1,n,dtype=torch.bool);g,rm,pm=group_batched_thinking(x,mask,4);self.assertEqual(int(pm.sum()),k);self.assertEqual(int(rm.sum()),n);self.assertTrue((g[~rm]==0).all())
 def test_padding_excluded(self):
  values=torch.tensor([[1.,2.],[3.,4.],[999.,999.]]);mask=torch.tensor([1,1,0],dtype=torch.bool);m=DualMoments(2);m.update(values[mask]);self.assertEqual(m.n,2);self.assertTrue(torch.equal(m.finish()[0],torch.tensor([2.,3.],dtype=torch.float64)))
 def test_whiten_inverse_and_finite(self):
  x=torch.randn(20,5);mean=x.double().mean(0);std=x.double().var(0,unbiased=False).sqrt();y=whiten(x,mean,std,1e-6);self.assertTrue(torch.isfinite(y).all());self.assertTrue(torch.allclose(inverse_whiten(y,mean,std,1e-6),x,atol=1e-6))
 def test_state_hash_and_output_refusal(self):
  model=torch.nn.Linear(3,2);a=state_sha(model.state_dict());b=state_sha(model.state_dict());self.assertEqual(a,b)
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(FileExistsError):refuse_existing(d)
if __name__=="__main__":unittest.main()
