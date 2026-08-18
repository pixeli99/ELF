import inspect,json,tempfile,unittest
from unittest import mock
import numpy as np
from types import SimpleNamespace
from pathlib import Path
import torch
from src.modules.thinking_resampler import (ThinkingMLPAutoencoder,ThinkingMLPConfig,
 FrozenThinkingPlanEncoder,ThinkingMLPDecoder,freeze_module,group_batched_thinking)
from src.utils.formal_thinking_mlp import (CanonicalThinkingDataset,StreamingChannelMoments,
 ShardedThinkingDataset,load_data_manifest,
 canonical_t5_x0,configure_formal_t5_tokenizer,inverse_whiten,masked_reconstruction_mse,
 adapt_to_fixed_decoder_width,load_frozen_encoder_artifact,
 load_trusted_full_training_checkpoint,mean_pool_reconstruction,refuse_existing,
 resolve_decoder_response_width,restore_grouped_token_layout,whiten)
from tools.train_formal_thinking_mlp import is_better_validation,restore_checkpoint
from src.utils.generation_utils import _dlm_decode_logits_batch

class FakeTokenizer:
 pad_token_id=0
 def encode(self,text,add_special_tokens=True): return [ord(c)%31+2 for c in text]+([1] if add_special_tokens else [])
 def __call__(self,text,add_special_tokens=True,truncation=False):
  self.last_truncation=truncation;return {"input_ids":self.encode(text,add_special_tokens)}

class FormalThinkingMLPTests(unittest.TestCase):
 def _fixed_width(self,length,width=1024):
  latent=torch.randn(1,length,3);ids=torch.arange(length).view(1,-1);mask=torch.ones(1,length,dtype=torch.bool)
  return adapt_to_fixed_decoder_width(latent,ids,mask,width,0)
 def test_fixed_width_639_and_1024_no_truncation(self):
  for length in (639,1024):
   latent,ids,mask=self._fixed_width(length);self.assertEqual(latent.shape,(1,1024,3));self.assertEqual(ids.shape,(1,1024));self.assertEqual(int(mask.sum()),length);self.assertTrue(torch.equal(ids[0,:length],torch.arange(length)));self.assertTrue((latent[:,length:]==0).all());self.assertFalse(mask[:,length:].any())
 def test_fixed_width_1025_rejected(self):
  with self.assertRaisesRegex(ValueError,"1025"):self._fixed_width(1025)
 def test_restore_tail_group_padding_and_condition_shapes(self):
  x=torch.randn(1,7,4);mask=torch.ones(1,7,dtype=torch.bool);groups,rmask,_=group_batched_thinking(x,mask,4);mean=mean_pool_reconstruction(groups,rmask);restored=[x,restore_grouped_token_layout(groups,rmask,mask),restore_grouped_token_layout(mean,rmask,mask)];adapted=[adapt_to_fixed_decoder_width(value,torch.arange(7).view(1,-1),mask,16,0) for value in restored];self.assertTrue(all(item[0].shape==(1,16,4) and int(item[2].sum())==7 for item in adapted));self.assertEqual(rmask.flatten().tolist(),[True]*7+[False])
 def test_decoder_helper_optional_response_mask(self):
  class FakeModel:
   def __init__(self):self.calls=[]
   def __call__(self,z,t,**kwargs):self.calls.append((z.clone(),kwargs));return z,None if False else torch.cat([z,z.new_zeros(z.shape[0],z.shape[1],1)],-1),None
  config=SimpleNamespace(num_self_cond_cfg_tokens=0,self_cond_prob=0,use_bf16=False);z=torch.randn(2,9,3);model=FakeModel();old=_dlm_decode_logits_batch(z,model,1.0,config,0.0);self.assertNotIn("attention_mask",{k:v for k,v in model.calls[0][1].items() if v is not None});mask=torch.tensor([[1]*7+[0]*2,[1]*9],dtype=torch.bool);new=_dlm_decode_logits_batch(z,model,1.0,config,0.0,attention_mask=mask);self.assertTrue(torch.equal(model.calls[1][1]["attention_mask"],mask));self.assertEqual(new.shape[:2],(2,9));self.assertTrue(torch.equal(old,new));self.assertEqual(model.calls[1][0].shape[1],9);self.assertIsNone(model.calls[1][1]["x_plan"])
 def test_decoder_width_resolution(self):
  self.assertEqual(resolve_decoder_response_width(SimpleNamespace(max_length=1024),SimpleNamespace(max_length=1024)),1024)
 def test_divisible_and_tail_masks(self):
  x=torch.randn(2,12,8);m=torch.zeros(2,12,dtype=torch.bool);m[0,:8]=1;m[1,:10]=1
  g,rm,pm=group_batched_thinking(x,m,4);self.assertEqual(g.shape,(2,3,4,8));self.assertEqual(rm[1,2].tolist(),[True,True,False,False]);self.assertEqual(pm.tolist(),[[True,True,False],[True,True,True]])
 def test_padding_does_not_change_loss_or_mean(self):
  target=torch.randn(1,2,4,3);mask=torch.tensor([[[1,1,1,1],[1,0,0,0]]],dtype=torch.bool);pred=target.clone();self.assertEqual(float(masked_reconstruction_mse(pred,target,mask)),0)
  altered=target.clone();altered[~mask]=1e6;self.assertTrue(torch.equal(mean_pool_reconstruction(target,mask)[mask],mean_pool_reconstruction(altered,mask)[mask]))
 def test_shapes_gradients_and_decoder_api(self):
  cfg=ThinkingMLPConfig(input_dim=8,group_size=4,hidden_dim=16,slot_dim=8);model=ThinkingMLPAutoencoder(cfg);groups=torch.randn(2,3,4,8);slots,recon=model(groups);self.assertEqual(slots.shape,(2,3,8));self.assertEqual(recon.shape,(2,3,4,8));recon.square().mean().backward();self.assertTrue(all(p.grad is not None for p in model.parameters()));self.assertEqual(list(inspect.signature(ThinkingMLPDecoder.forward).parameters),["self","plan_slots"])
 def test_parameter_count(self):
  model=ThinkingMLPAutoencoder();self.assertEqual(sum(p.numel() for p in model.encoder.parameters()),15735296);self.assertEqual(sum(p.numel() for p in model.decoder.parameters()),15736832);self.assertEqual(sum(p.numel() for p in model.parameters()),31472128)
 def test_missing_thinking_never_falls_back(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"x.jsonl";p.write_text(json.dumps({"sample_id":"x","response":"fallback"})+"\n")
   with self.assertRaisesRegex(ValueError,"thinking"):CanonicalThinkingDataset(str(p),FakeTokenizer())
 def test_dataset_batch_surface(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"x.jsonl";p.write_text(json.dumps({"sample_id":"x","thinking":"abc","response":"secret"})+"\n");row=CanonicalThinkingDataset(str(p),FakeTokenizer())[0];self.assertEqual(set(row),{"sample_id","thinking_input_ids"});self.assertEqual(row["thinking_input_ids"][-1],1)
 def test_train_only_sharded_manifest_and_dataset_surface(self):
  with tempfile.TemporaryDirectory() as d:
   shard=Path(d)/"train.jsonl";shard.write_text(json.dumps({"sample_id":"x","thinking":"abc","thinking_tokens":4,"response":"secret"})+"\n")
   manifest=Path(d)/"manifest.json";manifest.write_text(json.dumps({"training_ready":True,"formal_ready":False,
    "splits":{"train":{"shards":[str(shard)]},"validation":None,"test":None}}))
   loaded=load_data_manifest(str(manifest),require_formal=True,train_only=True);self.assertIsNone(loaded["splits"]["validation"])
   row=ShardedThinkingDataset([str(shard)],FakeTokenizer())[0]
   self.assertEqual(set(row),{"sample_id","thinking_input_ids"});self.assertNotIn("response",row)
   with self.assertRaises(ValueError):load_data_manifest(str(manifest),require_formal=True,train_only=False)
 def test_formal_lengths_525_and_1024_pass_1025_fails(self):
  tokenizer=configure_formal_t5_tokenizer(FakeTokenizer());self.assertEqual(tokenizer.model_max_length,1024)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"x.jsonl"
   for chars,expected in ((524,525),(1023,1024)):
    p.write_text(json.dumps({"thinking":"x"*chars})+"\n");ds=CanonicalThinkingDataset(str(p),tokenizer);self.assertEqual(ds.max_thinking_length,expected);self.assertEqual(len(ds[0]["thinking_input_ids"]),expected);self.assertFalse(tokenizer.last_truncation)
   p.write_text(json.dumps({"thinking":"x"*1024})+"\n")
   with self.assertRaisesRegex(ValueError,"1025"):CanonicalThinkingDataset(str(p),tokenizer)
 def test_canonical_x0_keyword_contract_and_normalized_return(self):
  encoder=torch.nn.Linear(2,2);ids=torch.tensor([[2,1,0]]);mask=torch.tensor([[1,1,0]],dtype=torch.bool);raw=torch.randn(1,3,2);normalized=torch.randn(1,3,2,dtype=torch.bfloat16)
  with mock.patch("src.utils.encoder_utils.encode_text_components",return_value=(raw,normalized)) as helper:
   result=canonical_t5_x0(encoder,ids,mask,0.0,0.2)
  self.assertEqual(result.dtype,torch.float32);self.assertTrue(torch.equal(result,normalized.float()));helper.assert_called_once_with(input_ids=ids,attention_mask=mask,encoder=encoder,latent_mean=0.0,latent_std=0.2)
 def test_forward_length_gate_ignores_padding(self):
  encoder=torch.nn.Linear(2,2);ids=torch.zeros(1,1025,dtype=torch.long);mask=torch.zeros_like(ids,dtype=torch.bool);mask[:,:1024]=1
  with mock.patch("src.utils.encoder_utils.encode_text_components",return_value=(torch.empty(0),torch.empty(0))) as helper:
   canonical_t5_x0(encoder,ids,mask,0.0,0.2);self.assertEqual(helper.call_count,1)
  mask[:,:]=1
  with mock.patch("src.utils.encoder_utils.encode_text_components") as helper:
   with self.assertRaisesRegex(ValueError,"1025"):canonical_t5_x0(encoder,ids,mask,0.0,0.2)
   helper.assert_not_called()
 def test_freeze_t5_and_trainable_mlp(self):
  t5=freeze_module(torch.nn.Linear(3,3));self.assertFalse(t5.training);self.assertFalse(any(p.requires_grad for p in t5.parameters()));self.assertTrue(all(p.requires_grad for p in ThinkingMLPAutoencoder(ThinkingMLPConfig(input_dim=2,group_size=2,hidden_dim=3,slot_dim=2)).parameters()))
 def test_bf16_input_fp32_autoencoder_forward_backward_step(self):
  cfg=ThinkingMLPConfig(input_dim=8,group_size=4,hidden_dim=16,slot_dim=8);model=ThinkingMLPAutoencoder(cfg);groups=torch.randn(2,3,4,8,dtype=torch.bfloat16);mask=torch.ones(2,3,4,dtype=torch.bool)
  slots,reconstruction=model(groups);manual_slots,manual_reconstruction=model(groups.float())
  self.assertEqual(slots.dtype,torch.float32);self.assertEqual(reconstruction.dtype,torch.float32);self.assertTrue(torch.equal(slots,manual_slots));self.assertTrue(torch.equal(reconstruction,manual_reconstruction))
  loss=masked_reconstruction_mse(reconstruction,groups,mask);self.assertEqual(loss.dtype,torch.float32);self.assertTrue(torch.isfinite(loss));optimizer=torch.optim.AdamW(model.parameters());optimizer.zero_grad();loss.backward();self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()));optimizer.step()
 def test_frozen_encoder_bf16_and_whitening_match_fp32(self):
  cfg=ThinkingMLPConfig(input_dim=4,group_size=4,hidden_dim=8,slot_dim=4);encoder=freeze_module(FrozenThinkingPlanEncoder(cfg));x=torch.randn(2,6,4,dtype=torch.bfloat16);mask=torch.tensor([[1,1,1,1,1,0],[1,1,1,0,0,0]],dtype=torch.bool)
  slots_bf16,plan_mask=encoder(x,mask);slots_fp32,plan_mask_fp32=encoder(x.float(),mask);self.assertEqual(slots_bf16.dtype,torch.float32);self.assertTrue(torch.equal(slots_bf16,slots_fp32));self.assertTrue(torch.equal(plan_mask,plan_mask_fp32));self.assertEqual(plan_mask.dtype,torch.bool)
  a=StreamingChannelMoments(4);b=StreamingChannelMoments(4);a.update(slots_bf16,plan_mask);b.update(slots_fp32,plan_mask_fp32);am,astd,an=a.finalize();bm,bstd,bn=b.finalize();self.assertEqual(an,bn);self.assertTrue(torch.equal(am,bm));self.assertTrue(torch.equal(astd,bstd))
 def test_checkpoint_roundtrip_and_step(self):
  cfg=ThinkingMLPConfig(input_dim=2,group_size=2,hidden_dim=3,slot_dim=2);a=ThinkingMLPAutoencoder(cfg);opt=torch.optim.AdamW(a.parameters());sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1)
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/"c.pt";torch.save({"encoder":a.encoder.state_dict(),"decoder":a.decoder.state_dict(),"optimizer":opt.state_dict(),"scheduler":sch.state_dict(),"optimizer_step":7},p);b=ThinkingMLPAutoencoder(cfg);opt2=torch.optim.AdamW(b.parameters());sch2=torch.optim.lr_scheduler.LambdaLR(opt2,lambda _:1);state=restore_checkpoint(p,b,opt2,sch2);x=torch.randn(1,1,2,2);self.assertTrue(torch.equal(a(x)[1],b(x)[1]));self.assertEqual(state["optimizer_step"],7)
 def test_trusted_full_checkpoint_and_weights_only_frozen_artifact(self):
  cfg=ThinkingMLPConfig(input_dim=2,group_size=2,hidden_dim=3,slot_dim=2);model=ThinkingMLPAutoencoder(cfg);optimizer=torch.optim.AdamW(model.parameters());loss=model(torch.randn(1,1,2,2))[1].square().mean();loss.backward();optimizer.step()
  with tempfile.TemporaryDirectory() as d:
   full=Path(d)/"best.pt";torch.save({"encoder":model.encoder.state_dict(),"decoder":model.decoder.state_dict(),"optimizer":optimizer.state_dict(),"numpy_rng_state":np.random.get_state()},full)
   with self.assertRaises(Exception):torch.load(full,map_location="cpu",weights_only=True)
   state=load_trusted_full_training_checkpoint(full,map_location="cpu");self.assertIn("numpy_rng_state",state);self.assertTrue(all(torch.isfinite(value).all() for value in state["encoder"].values()));self.assertTrue(all(torch.equal(value,state["encoder"][key]) for key,value in model.encoder.state_dict().items()));self.assertTrue(all(torch.equal(value,state["decoder"][key]) for key,value in model.decoder.state_dict().items()))
   frozen=Path(d)/"frozen_encoder.pt";torch.save({"encoder":model.encoder.state_dict(),"model_config":cfg.to_dict(),"parameter_count":sum(p.numel() for p in model.encoder.parameters()),"downstream_requires_grad":False},frozen);artifact=load_frozen_encoder_artifact(frozen,map_location="cpu");restored=FrozenThinkingPlanEncoder(ThinkingMLPConfig(**artifact["model_config"]));restored.encoder.load_state_dict(artifact["encoder"],strict=True);original=FrozenThinkingPlanEncoder(cfg);original.encoder.load_state_dict(model.encoder.state_dict(),strict=True);x=torch.randn(1,3,2);mask=torch.ones(1,3,dtype=torch.bool);self.assertTrue(torch.equal(original(x,mask)[0],restored(x,mask)[0]))
 def test_best_and_output_refusal(self):
  self.assertTrue(is_better_validation(.1,.2));self.assertFalse(is_better_validation(.3,.2))
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(FileExistsError):refuse_existing(d)
 def test_whitening_masks_and_inverse(self):
  m=StreamingChannelMoments(2);v=torch.tensor([[[1.,2.],[100.,200.]],[[3.,4.],[5.,6.]]]);mask=torch.tensor([[1,0],[1,1]],dtype=torch.bool);m.update(v,mask);mean,std,n=m.finalize();self.assertEqual(n,3);self.assertTrue(torch.equal(mean,torch.tensor([3.,4.],dtype=torch.float64)));x=torch.randn(4,2);self.assertTrue(torch.allclose(inverse_whiten(whiten(x,mean,std),mean,std),x,atol=1e-6))

if __name__=="__main__":unittest.main()
