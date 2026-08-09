#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
import torch,yaml
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from transformers import AutoTokenizer
from src.modules.t5_encoder import get_encoder
from src.modules.thinking_resampler import FrozenThinkingPlanEncoder,ThinkingMLPConfig,freeze_module
from src.utils.formal_thinking_mlp import *
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--config",required=True);ap.add_argument("--data_manifest",required=True);ap.add_argument("--frozen_encoder",required=True);ap.add_argument("--output_dir",required=True);args=ap.parse_args()
 cfg=yaml.safe_load(Path(args.config).read_text());manifest=load_data_manifest(args.data_manifest,require_formal=True);refuse_existing(args.output_dir);out=Path(args.output_dir);out.mkdir(parents=True)
 tok=AutoTokenizer.from_pretrained(cfg["t5_model_id"],revision=cfg["t5_revision"],local_files_only=True);tok=configure_formal_t5_tokenizer(tok,cfg["max_thinking_tokens"]);_,t5=get_encoder(cfg["t5_model_id"],dtype=torch.bfloat16,revision=cfg["t5_revision"],local_files_only=True);t5=freeze_module(t5).cuda()
 state=load_frozen_encoder_artifact(args.frozen_encoder,map_location="cpu");mc=ThinkingMLPConfig(**state["model_config"]);enc=FrozenThinkingPlanEncoder(mc);enc.encoder.load_state_dict(state["encoder"],strict=True);enc=freeze_module(enc).cuda();ds=CanonicalThinkingDataset(manifest["splits"]["train"]["path"],tok,max_thinking_tokens=cfg["max_thinking_tokens"])
 moments=StreamingChannelMoments(mc.slot_dim)
 with torch.no_grad():
  for b in DataLoader(ds,batch_size=cfg["validation_batch_size"],collate_fn=ThinkingCollator(tok.pad_token_id),num_workers=cfg["num_workers"]):
   ids=b["thinking_input_ids"].cuda();mask=b["thinking_attention_mask"].cuda();x0=canonical_t5_x0(t5,ids,mask,cfg["latent_mean"],cfg["latent_std"]);slots,pmask=enc(x0,mask);moments.update(slots,pmask)
 mean,std,count=moments.finalize();eps=float(cfg.get("whitening_eps",1e-6));torch.save({"mean":mean.float(),"std":std.float(),"count":count,"eps":eps},out/"plan_whitener.pt")
 (out/"manifest.json").write_text(json.dumps({"complete":True,"count":count,"dim":mc.slot_dim,"eps":eps,"train_split_sha256":manifest["splits"]["train"].get("sha256"),"t5_revision":cfg["t5_revision"],"encoder_sha256":sha256_file(args.frozen_encoder)},indent=2))
if __name__=="__main__":main()
