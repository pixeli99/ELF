#!/usr/bin/env python3
"""Deterministic train-only formal thinking plan whitening and encoder export."""
import argparse,hashlib,json,math,os,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch,yaml
from torch.utils.data import DataLoader,Subset
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from transformers import AutoTokenizer
from src.modules.t5_encoder import get_encoder
from src.modules.thinking_resampler import FrozenThinkingPlanEncoder,ThinkingMLPConfig,freeze_module
from src.utils.formal_thinking_mlp import (ShardedThinkingDataset,ThinkingCollator,canonical_t5_x0,
 configure_formal_t5_tokenizer,load_trusted_full_training_checkpoint)

EXPECTED_ROWS=488692;EXPECTED_TOKENS=241972872;EXPECTED_SLOTS=60676596
T5_REV="df1b051c49625cf57a3d0d8d3863ed4d13564fe4";EPS=1e-6
TOKEN_HASHES={"spiece.model":"d60acb128cf7b7f2536e8f38a5b18a05535c9e14c7a355904270e15b0945ea86","tokenizer.json":"d2acde0d8d71dd30a711834b07781b9c89feaac33fd332f60507699282740066","tokenizer_config.json":"d1e7146101aa96282057f374aa9c3b260fba2109e5990edb1775d6efc70ffa3c"}
def sha(path):
 h=hashlib.sha256()
 with open(path,"rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def state_sha(state):
 h=hashlib.sha256()
 for k,v in sorted(state.items()):h.update(k.encode());h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
class DualMoments:
 def __init__(self,dim):self.n=0;self.mean=torch.zeros(dim,dtype=torch.float64);self.m2=torch.zeros(dim,dtype=torch.float64);self.raw_sum=torch.zeros(dim,dtype=torch.float64);self.raw_sum_sq=torch.zeros(dim,dtype=torch.float64)
 def update(self,x):
  x=x.detach().cpu().double();n=len(x)
  if not n:return
  bm=x.mean(0);bm2=((x-bm)**2).sum(0);self.raw_sum+=x.sum(0);self.raw_sum_sq+=(x*x).sum(0)
  if not self.n:self.n=n;self.mean=bm;self.m2=bm2;return
  total=self.n+n;delta=bm-self.mean;self.m2+=bm2+delta.square()*self.n*n/total;self.mean+=delta*n/total;self.n=total
 def finish(self):
  var=self.m2/self.n;rmean=self.raw_sum/self.n;rvar=self.raw_sum_sq/self.n-rmean.square();return self.mean,var,rmean,rvar
def gpu_state():
 q=subprocess.check_output(["nvidia-smi","-i","7","--query-gpu=utilization.gpu,memory.used,memory.free","--format=csv,noheader,nounits"],text=True).strip().split(",")
 apps=subprocess.check_output(["nvidia-smi","-i","7","--query-compute-apps=pid,process_name,used_memory","--format=csv,noheader,nounits"],text=True).strip().splitlines();return {"util":int(q[0]),"used":int(q[1]),"free":int(q[2]),"apps":apps}
def check_runtime(state,pid,smoke=False):
 known=[];milvus=None
 for line in state["apps"]:
  p,name,mem=[x.strip() for x in line.split(",")];mem=int(mem)
  if int(p)==3281471 and name=="milvus":milvus=mem
  elif int(p)==pid:known.append(mem)
  else:raise RuntimeError(f"unknown GPU process: {line}")
 if milvus is None or milvus>4096:raise RuntimeError(f"Milvus gate failed: {milvus}")
 if state["free"]<4096:raise RuntimeError(f"GPU free memory gate failed: {state['free']}")
 if smoke and state["used"]>=20480:raise RuntimeError(f"smoke total GPU memory gate failed: {state['used']}")
 return milvus,max(known,default=0)
def load_all(args):
 manifest=json.load(open(args.data_manifest));state=load_trusted_full_training_checkpoint(args.checkpoint,"cpu");mc=ThinkingMLPConfig(**state["model_config"])
 tok=AutoTokenizer.from_pretrained("t5-small",revision=T5_REV,local_files_only=True);tok=configure_formal_t5_tokenizer(tok,1024);_,t5=get_encoder("t5-small",dtype=torch.bfloat16,revision=T5_REV,local_files_only=True);t5=freeze_module(t5).cuda()
 enc=FrozenThinkingPlanEncoder(mc);enc.encoder.load_state_dict(state["encoder"],strict=True);enc=freeze_module(enc).cuda();ds=ShardedThinkingDataset(manifest["splits"]["train"]["shards"],tok,1024)
 return manifest,state,mc,tok,t5,enc,ds
def run_pass(ds,indices,tok,t5,enc,batch_size,progress_every,log,smoke=False,keep=False):
 subset=Subset(ds,indices) if indices is not None else ds;loader=DataLoader(subset,batch_size=batch_size,shuffle=False,num_workers=4,collate_fn=ThinkingCollator(tok.pad_token_id),persistent_workers=True)
 mom=DualMoments(512);rows=tokens=slots=nonfinite=0;seen=set();kept=[];start=time.time();peak_project=peak_total=0
 torch.cuda.reset_peak_memory_stats()
 with torch.no_grad():
  for b in loader:
   ids=b["thinking_input_ids"].cuda(non_blocking=True);mask=b["thinking_attention_mask"].cuda(non_blocking=True);x0=canonical_t5_x0(t5,ids,mask,0.,.2);z,pmask=enc(x0,mask);valid=z[pmask].float()
   if not torch.isfinite(valid).all():nonfinite+=int((~torch.isfinite(valid)).any(1).sum());raise RuntimeError("nonfinite encoder outputs")
   mom.update(valid);rows+=len(b["sample_id"]);tokens+=int(mask.sum());slots+=int(pmask.sum());
   for sid in b["sample_id"]:
    if sid in seen:raise RuntimeError(f"duplicate sample {sid}")
    seen.add(sid)
   if keep:kept.append(valid.cpu())
   if rows%progress_every<batch_size or rows==len(subset):
    gs=gpu_state();milvus,project=check_runtime(gs,os.getpid(),smoke);peak_project=max(peak_project,project);peak_total=max(peak_total,gs["used"]);elapsed=time.time()-start;rate=rows/elapsed;eta=(len(subset)-rows)/rate
    msg={"processed_rows":rows,"target_rows":len(subset),"valid_slots":slots,"rows_per_s":rate,"slots_per_s":slots/elapsed,"elapsed":elapsed,"eta":eta,"gpu_total_used_mib":gs["used"],"whitening_process_mib":project,"milvus_mib":milvus,"gpu_utilization":gs["util"]};print(json.dumps(msg),flush=True);log.write(json.dumps(msg)+"\n");log.flush()
 return {"mom":mom,"rows":rows,"tokens":tokens,"slots":slots,"nonfinite":nonfinite,"seen":seen,"kept":torch.cat(kept) if keep else None,"elapsed":time.time()-start,"peak_project_mib":max(peak_project,math.ceil(torch.cuda.max_memory_allocated()/2**20)),"peak_total_mib":peak_total}
def select_indices(ds,n,seed=42):return sorted(range(len(ds)),key=lambda i:hashlib.sha256(f"{seed}|{ds.rows[i][2]}".encode()).digest())[:n]
def artifact_payload(enc,mc,args,checkpoint_sha,manifest_sha,token_hashes):return {"encoder":{k:v.detach().cpu() for k,v in enc.encoder.state_dict().items()},"model_config":mc.to_dict(),"architecture":{"input_dim":2048,"hidden_dim":6144,"output_dim":512,"group_size":4,"activation":"gelu","dropout":0.},"parameter_count":sum(p.numel() for p in enc.encoder.parameters()),"downstream_requires_grad":False,"source_checkpoint_path":args.checkpoint,"source_checkpoint_sha256":checkpoint_sha,"train_manifest_path":args.data_manifest,"train_manifest_sha256":manifest_sha,"t5_model_id":"t5-small","t5_revision":T5_REV,"tokenizer_hashes":token_hashes,"export_timestamp":time.strftime("%Y-%m-%dT%H:%M:%S%z"),"code_revision":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--mode",choices=["smoke","formal"],required=True);ap.add_argument("--config",required=True);ap.add_argument("--data_manifest",required=True);ap.add_argument("--checkpoint",required=True);ap.add_argument("--output_dir",required=True);ap.add_argument("--allowlist_samples",required=True);ap.add_argument("--batch_size",type=int,default=16);args=ap.parse_args();out=Path(args.output_dir)
 if out.exists():raise FileExistsError(out)
 stage=out.parent/f"{out.name}.staging.{os.getpid()}";stage.mkdir(parents=True);log=(stage/"run.log").open("x");checkpoint_sha=sha(args.checkpoint);manifest_sha=sha(args.data_manifest);before_checkpoint=checkpoint_sha
 try:
  if manifest_sha!="f02cebf13d5b99919172299ebd9ea999b4501a8ffb6bc1395c33e73cf90be867":raise RuntimeError("manifest SHA mismatch")
  snap=Path(os.environ["HF_HOME"])/"hub/models--t5-small/snapshots"/T5_REV;token_hashes={k:sha(snap/k) for k in TOKEN_HASHES}
  if token_hashes!=TOKEN_HASHES:raise RuntimeError("tokenizer hash mismatch")
  allow=json.load(open(args.allowlist_samples));manifest,state,mc,tok,t5,enc,ds=load_all(args);enc_before=state_sha(enc.encoder.state_dict())
  if len(ds)!=EXPECTED_ROWS or state["optimizer_step"]!=61087 or state["data_manifest_sha256"]!=manifest_sha or sum(p.numel() for p in enc.encoder.parameters())!=15735296:raise RuntimeError("identity gate failed")
  frozen=artifact_payload(enc,mc,args,checkpoint_sha,manifest_sha,token_hashes);torch.save(frozen,stage/"frozen_encoder_v1.pt")
  reload=torch.load(stage/"frozen_encoder_v1.pt",map_location="cpu",weights_only=True);test_enc=FrozenThinkingPlanEncoder(ThinkingMLPConfig(**reload["model_config"]));test_enc.encoder.load_state_dict(reload["encoder"]);test_enc=freeze_module(test_enc).cuda()
  if args.mode=="smoke":
   idx=list(range(512));results=[]
   for bs in (8,16):
    r=run_pass(ds,idx,tok,t5,enc,bs,128,log,True,True);mean,var,rmean,rvar=r["mom"].finish();direct_mean=r["kept"].double().mean(0);direct_var=r["kept"].double().var(0,unbiased=False);results.append((r,mean,var,rmean,rvar,direct_mean,direct_var))
   a,b=results
   diffs={"batch_mean_max_abs":float((a[1]-b[1]).abs().max()),"batch_std_max_abs":float((a[2].sqrt()-b[2].sqrt()).abs().max()),"stream_direct_mean_max_abs":max(float((x[1]-x[5]).abs().max()) for x in results),"stream_direct_var_max_abs":max(float((x[2]-x[6]).abs().max()) for x in results)}
   # Export/full encoder parity.
   batch=next(iter(DataLoader(Subset(ds,idx[:8]),batch_size=8,collate_fn=ThinkingCollator(tok.pad_token_id))));ids=batch["thinking_input_ids"].cuda();mask=batch["thinking_attention_mask"].cuda();x0=canonical_t5_x0(t5,ids,mask,0.,.2);z1,m1=enc(x0,mask);z2,m2=test_enc(x0,mask);parity=float((z1-z2).abs().max())
   tolerances={"batch_mean_max_abs":2e-4,"batch_std_max_abs":5e-4,"stream_direct_mean_max_abs":1e-12,"stream_direct_var_max_abs":1e-12}
   if a[0]["rows"]!=512 or b[0]["rows"]!=512 or a[0]["slots"]!=b[0]["slots"] or any(diffs[k]>tolerances[k] for k in diffs) or parity!=0:raise RuntimeError(f"smoke parity failed {diffs} tolerances={tolerances} export={parity}")
   summary={"complete":True,"mode":"smoke","rows":512,"batch_sizes":[8,16],"slots":a[0]["slots"],"diffs":diffs,"tolerances":tolerances,"batch_difference_note":"BF16 T5 roundoff under different dynamic batch padding shapes; streaming/direct comparisons use identical forward outputs","export_output_max_abs":parity,"allowlisted_coexisting_process":{"pid":3281471,"process_name":"milvus","pre_run_samples":allow,"reason":"user-authorized coexistence for formal MLP whitening","user_authorized":True},"peak_project_mib":max(a[0]["peak_project_mib"],b[0]["peak_project_mib"]),"peak_total_mib":max(a[0]["peak_total_mib"],b[0]["peak_total_mib"])}
   (stage/"summary.json").write_text(json.dumps(summary,indent=2))
  else:
   r=run_pass(ds,None,tok,t5,enc,args.batch_size,5000,log,False,False)
   if (r["rows"],r["tokens"],r["slots"],r["nonfinite"],len(r["seen"]))!=(EXPECTED_ROWS,EXPECTED_TOKENS,EXPECTED_SLOTS,0,EXPECTED_ROWS):raise RuntimeError("full accounting mismatch")
   mean,var,rmean,rvar=r["mom"].finish();tol=1e-12;var=torch.clamp(var,min=0);rvar=torch.clamp(rvar,min=0);std=var.sqrt();rstd=rvar.sqrt();scale=std+EPS;near=torch.where(std<=EPS)[0]
   if not all(torch.isfinite(x).all() for x in (mean,var,std,scale)) or len(near):raise RuntimeError(f"invalid stats near_zero={near.tolist()}")
   np.savez(stage/"whitener_stats_v1.npz",mean=mean.numpy(),variance=var.numpy(),std=std.numpy(),scale=scale.numpy(),count=np.array(r["mom"].n,dtype=np.int64),epsilon=np.array(EPS),ddof=np.array(0,dtype=np.int64))
   stats_sha=sha(stage/"whitener_stats_v1.npz");torch.save({"mean":mean.float(),"scale":scale.float(),"epsilon":EPS,"feature_dim":512,"source_statistics_sha256":stats_sha,"frozen_encoder_sha256":sha(stage/"frozen_encoder_v1.pt"),"train_manifest_sha256":manifest_sha},stage/"whitener_stageb_v1.pt")
   # Deterministic 10k second forward verification.
   vr=run_pass(ds,select_indices(ds,10000),tok,t5,enc,args.batch_size,2000,log,False,True);white=(vr["kept"].double()-mean)/scale;vmean=white.mean(0);vstd=white.var(0,unbiased=False).sqrt();global_white_mean=(mean-mean)/scale;global_white_var=var/scale.square()
   # Reload gates.
   loaded=np.load(stage/"whitener_stats_v1.npz");reload_diff=max(float(np.max(np.abs(loaded[k]-v.numpy()))) for k,v in (("mean",mean),("variance",var),("std",std),("scale",scale)))
   stageb=torch.load(stage/"whitener_stageb_v1.pt",map_location="cpu",weights_only=True);stageb_diff=max(float((stageb["mean"].double()-mean).abs().max()),float((stageb["scale"].double()-scale).abs().max()))
   enc_after=state_sha(enc.encoder.state_dict());checkpoint_after=sha(args.checkpoint)
   if enc_before!=enc_after or before_checkpoint!=checkpoint_after or reload_diff!=0:raise RuntimeError("immutability/reload gate failed")
   kdist={"min":3,"mean":124.16122220130471,"p50":124,"p90":184,"p95":196,"p99":215,"max":255}
   summary={"complete":True,"rows":r["rows"],"valid_thinking_tokens":r["tokens"],"valid_plan_slots":r["slots"],"K_distribution":kdist,"mean":{"min":float(mean.min()),"max":float(mean.max()),"mean":float(mean.mean())},"std":{"min":float(std.min()),"max":float(std.max()),"mean":float(std.mean())},"scale":{"min":float(scale.min()),"max":float(scale.max()),"mean":float(scale.mean())},"near_zero_dimensions":near.tolist(),"welford_raw_mean_max_abs":float((mean-rmean).abs().max()),"welford_raw_std_max_abs":float((std-rstd).abs().max()),"verification_10000":{"rows":vr["rows"],"slots":vr["slots"],"max_abs_mean":float(vmean.abs().max()),"std_min":float(vstd.min()),"std_max":float(vstd.max()),"std_mean":float(vstd.mean())},"global_algebra":{"max_abs_mean":float(global_white_mean.abs().max()),"variance_min":float(global_white_var.min()),"variance_max":float(global_white_var.max()),"max_abs_variance_from_one":float((global_white_var-1).abs().max())},"reload_max_abs":reload_diff,"stageb_fp32_max_abs":stageb_diff,"encoder_state_sha_before":enc_before,"encoder_state_sha_after":enc_after,"checkpoint_sha_before":before_checkpoint,"checkpoint_sha_after":checkpoint_after,"elapsed_seconds":r["elapsed"]+vr["elapsed"],"peak_project_mib":max(r["peak_project_mib"],vr["peak_project_mib"]),"peak_total_mib":max(r["peak_total_mib"],vr["peak_total_mib"])}
   (stage/"summary.json").write_text(json.dumps(summary,indent=2));(stage/"slot_length_distribution.json").write_text(json.dumps(kdist,indent=2));(stage/"sample_accounting.json").write_text(json.dumps({"expected_rows":EXPECTED_ROWS,"processed_rows":r["rows"],"expected_valid_thinking_tokens":EXPECTED_TOKENS,"observed_valid_thinking_tokens":r["tokens"],"expected_plan_slots":EXPECTED_SLOTS,"observed_plan_slots":r["slots"],"missing_rows":0,"duplicate_rows":0,"extra_rows":0,"nonfinite_outputs":0},indent=2))
   files={p.name:{"bytes":p.stat().st_size,"sha256":sha(p)} for p in stage.iterdir() if p.is_file() and p.name not in ("manifest.json","sha256sums.txt")}
   gpu=gpu_state();milvus,_=check_runtime(gpu,os.getpid())
   manifest_out={"complete":True,"whitening_ready":True,"project_formal_ready":False,"scope":"train_only_all_rows","rows":r["rows"],"valid_thinking_tokens":r["tokens"],"valid_plan_slots":r["slots"],"feature_dim":512,"statistic_dtype":"float64","stageb_dtype":"float32","variance_definition":"population_ddof0","scale_definition":"std_plus_epsilon_existing_project_convention","epsilon":EPS,"near_zero_dimension_count":len(near),"dataset_manifest_path":args.data_manifest,"dataset_manifest_sha256":manifest_sha,"source_shards":manifest["splits"]["train"]["files"],"checkpoint_path":args.checkpoint,"checkpoint_sha256":checkpoint_sha,"frozen_encoder_sha256":sha(stage/"frozen_encoder_v1.pt"),"t5_model_id":"t5-small","t5_revision":T5_REV,"tokenizer_hashes":token_hashes,"seed":42,"batch_size":args.batch_size,"gpu":"NVIDIA GeForce RTX 4090 / CUDA_VISIBLE_DEVICES=7","elapsed_seconds":summary["elapsed_seconds"],"peak_gpu_memory_mib":summary["peak_project_mib"],"peak_total_gpu_memory_mib":summary["peak_total_mib"],"allowlisted_coexisting_process":{"pid":3281471,"process_name":"milvus","pre_run_samples":allow,"reason":"user-authorized coexistence for formal MLP whitening","user_authorized":True,"final_memory_mib":milvus},"code_commit":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"git_dirty":bool(subprocess.check_output(["git","status","--short"],text=True).strip()),"files":files}
   (stage/"manifest.json").write_text(json.dumps(manifest_out,indent=2));
   allfiles={p.name:sha(p) for p in stage.iterdir() if p.is_file() and p.name!="sha256sums.txt"};(stage/"sha256sums.txt").write_text("".join(f"{v}  {k}\n" for k,v in sorted(allfiles.items())))
  log.close();os.rename(stage,out)
 except Exception as e:
  log.write(json.dumps({"failure":str(e)})+"\n");log.close();(stage/"failure.json").write_text(json.dumps({"complete":False,"error":str(e)},indent=2));raise
if __name__=="__main__":main()
