#!/usr/bin/env python3
"""Optional no-grad token-decodability audit; never contributes to MLP training."""
import argparse,csv,json,sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from configs.config import load_config_from_yaml
from tools.eval_oracle_plan import _load_model_and_encoder
from src.modules.thinking_resampler import ThinkingMLPAutoencoder,ThinkingMLPConfig,group_batched_thinking
from src.utils.formal_thinking_mlp import (CanonicalThinkingDataset,ThinkingCollator,
 adapt_to_fixed_decoder_width,canonical_t5_x0,configure_formal_t5_tokenizer,
 load_data_manifest,load_trusted_full_training_checkpoint,mean_pool_reconstruction,
 refuse_existing,resolve_decoder_response_width,restore_grouped_token_layout)
from src.utils.generation_utils import _dlm_decode_logits_batch

def main():
 ap=argparse.ArgumentParser(description="Evaluate frozen thinking latent token decodability (not teacher-forced training CE).")
 ap.add_argument("--elf_config",required=True);ap.add_argument("--elf_checkpoint",required=True);ap.add_argument("--autoencoder_checkpoint",required=True);ap.add_argument("--data_manifest",required=True);ap.add_argument("--split",default="validation");ap.add_argument("--output_dir",required=True);args=ap.parse_args()
 manifest=load_data_manifest(args.data_manifest);refuse_existing(args.output_dir);out=Path(args.output_dir);out.mkdir(parents=True)
 device=torch.device("cuda");config=load_config_from_yaml(args.elf_config);model,t5,tokenizer,_=_load_model_and_encoder(config,args.elf_checkpoint,device);response_width=resolve_decoder_response_width(model,config);tokenizer=configure_formal_t5_tokenizer(tokenizer,response_width)
 state=load_trusted_full_training_checkpoint(args.autoencoder_checkpoint,map_location="cpu");ae=ThinkingMLPAutoencoder(ThinkingMLPConfig(**state["model_config"]));ae.encoder.load_state_dict(state["encoder"],strict=True);ae.decoder.load_state_dict(state["decoder"],strict=True);ae.to(device).eval().requires_grad_(False)
 ds=CanonicalThinkingDataset(manifest["splits"][args.split]["path"],tokenizer,max_thinking_tokens=response_width); totals={k:{"ce_sum":0.,"correct":0,"valid":0,"samples":0,"missing":0,"nonfinite":0} for k in ("clean_t5","mlp_reconstruction","mean_pool_reconstruction")}
 with torch.no_grad():
  for b in DataLoader(ds,batch_size=4,collate_fn=ThinkingCollator(tokenizer.pad_token_id)):
   ids=b["thinking_input_ids"].to(device);mask=b["thinking_attention_mask"].to(device);lengths=mask.sum(1)
   if bool((lengths>response_width).any()):
    bad=int(torch.where(lengths>response_width)[0][0]);raise ValueError(f"sample_id={b['sample_id'][bad]} thinking length {int(lengths[bad])} exceeds decoder response width {response_width}")
   x0=canonical_t5_x0(t5,ids,mask,config.latent_mean,config.latent_std,max_valid_length=response_width);groups,rmask,_=group_batched_thinking(x0,mask,ae.config.group_size);_,recon=ae(groups);mean_recon=mean_pool_reconstruction(groups,rmask);candidates={"clean_t5":x0,"mlp_reconstruction":restore_grouped_token_layout(recon,rmask,mask),"mean_pool_reconstruction":restore_grouped_token_layout(mean_recon,rmask,mask)}
   expected_valid=int(mask.sum())
   for name,value in candidates.items():
    latent,padded_ids,padded_mask=adapt_to_fixed_decoder_width(value,ids,mask,response_width,tokenizer.pad_token_id)
    if int(padded_mask.sum())!=expected_valid:raise AssertionError("condition valid-token denominator changed")
    logits=_dlm_decode_logits_batch(latent,model,1.0,config,getattr(config,"self_cond_cfg_scale",0.0),attention_mask=padded_mask)
    if tuple(logits.shape[:2])!=(ids.shape[0],response_width):raise AssertionError("decoder logits are not response-only fixed-width logits")
    ce=torch.nn.functional.cross_entropy(logits[padded_mask].float(),padded_ids[padded_mask],reduction="sum");correct=(logits.argmax(-1)[padded_mask]==padded_ids[padded_mask]).sum();row=totals[name];row["samples"]+=ids.shape[0];row["valid"]+=expected_valid
    if not bool(torch.isfinite(ce)):row["nonfinite"]+=1
    else:row["ce_sum"]+=float(ce);row["correct"]+=int(correct)
 summary={name:{"mean_token_ce":v["ce_sum"]/v["valid"],"token_accuracy":v["correct"]/v["valid"],"sample_count":v["samples"],"valid_token_count":v["valid"],"missing_count":v["missing"],"nonfinite_count":v["nonfinite"]} for name,v in totals.items()}
 payload={"metric_note":"frozen ELF decoder token-decodability diagnostic; not MLP training loss, not teacher-forced autoregressive CE, and not evidence of reasoning semantics","decoder_response_width":response_width,"conditions":summary};(out/"summary.json").write_text(json.dumps(payload,indent=2));(out/"report.md").write_text("# Thinking MLP Token Decodability\n\nThis is a frozen ELF decoder token-decodability diagnostic. It is not the MLP training loss, not teacher-forced autoregressive CE, and does not establish reasoning semantics.\n")
if __name__=="__main__":main()
