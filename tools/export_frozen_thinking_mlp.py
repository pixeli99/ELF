#!/usr/bin/env python3
import argparse, json, shutil, sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.modules.thinking_resampler import (FrozenThinkingPlanEncoder, ThinkingMLPAutoencoder,
                                             ThinkingMLPConfig)
from src.utils.formal_thinking_mlp import (load_trusted_full_training_checkpoint,
                                            refuse_existing, sha256_file)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--checkpoint",required=True); ap.add_argument("--output_dir",required=True); args=ap.parse_args()
    refuse_existing(args.output_dir); out=Path(args.output_dir); out.mkdir(parents=True)
    state=load_trusted_full_training_checkpoint(args.checkpoint,map_location="cpu"); cfg=ThinkingMLPConfig(**state["model_config"])
    model=ThinkingMLPAutoencoder(cfg); model.encoder.load_state_dict(state["encoder"],strict=True); model.decoder.load_state_dict(state["decoder"],strict=True)
    model.eval().requires_grad_(False); enc_params=sum(p.numel() for p in model.encoder.parameters())
    wrapper=FrozenThinkingPlanEncoder(cfg); wrapper.encoder.load_state_dict(model.encoder.state_dict(),strict=True); wrapper.eval().requires_grad_(False)
    torch.save({"encoder":wrapper.encoder.state_dict(),"model_config":cfg.to_dict(),"parameter_count":enc_params,"downstream_requires_grad":False,"interface":{"input":["thinking_x0[B,L,512]","thinking_mask[B,L]"],"output":["plan_slots[B,K,512]","plan_mask[B,K]"]}},out/"frozen_encoder.pt")
    shutil.copy2(args.checkpoint,out/"autoencoder_audit_checkpoint.pt")
    (out/"encoder_config.json").write_text(json.dumps({**cfg.to_dict(),"parameter_count":enc_params,"eval_mode_required":True},indent=2))
    manifest={k:state.get(k) for k in ("t5_model_id","t5_revision","data_manifest_path","data_manifest_sha256","best_validation_mse","optimizer_step")}; manifest["source_checkpoint_sha256"]=sha256_file(args.checkpoint)
    (out/"training_manifest.json").write_text(json.dumps(manifest,indent=2))
    lines=[f"{sha256_file(str(p))}  {p.name}" for p in sorted(out.iterdir()) if p.name!="source_hashes.sha256"]
    (out/"source_hashes.sha256").write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()
