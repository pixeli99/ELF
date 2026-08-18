#!/usr/bin/env python3
"""Offline trainer for the frozen-T5, formal 4-to-1 thinking autoencoder."""

import argparse, csv, json, os, random, sys
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from src.modules.t5_encoder import get_encoder
from src.modules.thinking_resampler import (ThinkingMLPAutoencoder, ThinkingMLPConfig,
                                             group_batched_thinking)
from src.utils.formal_thinking_mlp import (CanonicalThinkingDataset, ThinkingCollator,
    ShardedThinkingDataset,
    canonical_t5_x0, configure_formal_t5_tokenizer, load_data_manifest, masked_reconstruction_mse,
    load_trusted_full_training_checkpoint, masked_token_cosine, mean_pool_reconstruction,
    refuse_existing, sha256_file)
from transformers import AutoTokenizer


def checkpoint_payload(model, optimizer, scheduler, epoch, step, best, cfg, args):
    return {"encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "epoch": epoch, "optimizer_step": step, "best_validation_mse": best,
            "model_config": model.config.to_dict(), "t5_model_id": cfg["t5_model_id"],
            "t5_revision": cfg["t5_revision"], "data_manifest_path": str(Path(args.data_manifest).resolve()),
            "data_manifest_sha256": sha256_file(args.data_manifest),
            "source_sha256": {str(path.relative_to(ROOT)): sha256_file(str(path)) for path in
                (ROOT / "src/modules/thinking_resampler.py", ROOT / "src/utils/formal_thinking_mlp.py",
                 ROOT / "tools/train_formal_thinking_mlp.py")},
            "rng_state": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                          "python": random.getstate()}}


def restore_checkpoint(path, model, optimizer=None, scheduler=None):
    state = load_trusted_full_training_checkpoint(path, map_location="cpu")
    model.encoder.load_state_dict(state["encoder"], strict=True)
    model.decoder.load_state_dict(state["decoder"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"]); scheduler.load_state_dict(state["scheduler"])
    return state


def is_better_validation(value, best): return float(value) < float(best)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--data_manifest", required=True)
    ap.add_argument("--output_dir"); ap.add_argument("--resume"); ap.add_argument("--warm_start")
    ap.add_argument("--max_optimizer_steps", type=int)
    ap.add_argument("--formal", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    train_only = bool(cfg.get("train_only", False))
    manifest = load_data_manifest(args.data_manifest, require_formal=args.formal, train_only=train_only)
    out = Path(args.output_dir or cfg["output_dir"])
    if args.resume:
        if not out.is_dir(): raise FileNotFoundError("resume output directory does not exist")
    else:
        refuse_existing(str(out)); out.mkdir(parents=True)
    seed = int(cfg["seed"]); random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg["precision"] == "bf16" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(cfg["t5_model_id"], revision=cfg["t5_revision"],
        local_files_only=cfg.get("local_files_only", True))
    tokenizer = configure_formal_t5_tokenizer(tokenizer, cfg["max_thinking_tokens"])
    _, encoder = get_encoder(cfg["t5_model_id"], dtype=dtype, revision=cfg["t5_revision"],
        local_files_only=cfg.get("local_files_only", True))
    encoder = encoder.to(device).eval().requires_grad_(False)
    split_names = ("train",) if train_only else ("train", "validation")
    datasets = {}
    for s in split_names:
        entry = manifest["splits"][s]
        if entry.get("shards"):
            datasets[s] = ShardedThinkingDataset(entry["shards"], tokenizer,
                max_thinking_tokens=cfg["max_thinking_tokens"])
        else:
            datasets[s] = CanonicalThinkingDataset(entry["path"], tokenizer,
                max_thinking_tokens=cfg["max_thinking_tokens"])
    collate = ThinkingCollator(tokenizer.pad_token_id)
    loaders = {s: DataLoader(datasets[s], batch_size=cfg[f"{s}_batch_size"], shuffle=s == "train",
        num_workers=cfg["num_workers"], collate_fn=collate) for s in datasets}
    mc = ThinkingMLPConfig(**cfg["model"]); model = ThinkingMLPAutoencoder(mc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]),
                                  weight_decay=float(cfg["weight_decay"]))
    total_updates = max(1, len(loaders["train"]) * cfg["epochs"] // cfg["gradient_accumulation_steps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0., 1-step/total_updates))
    start_epoch, step, best = 0, 0, float("inf")
    if args.resume:
        state = restore_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch, step, best = state["epoch"] + 1, state["optimizer_step"], state["best_validation_mse"]
    elif args.warm_start:
        restore_checkpoint(args.warm_start, model)
    log_path = out / "metrics.csv"; fields = ["epoch","optimizer_step","split","reconstruction_mse",
        "token_cosine","mean_pool_mse","valid_token_count","valid_plan_slot_count","learning_rate","gradient_norm"]
    with log_path.open("a", newline="") as log:
        writer=csv.DictWriter(log,fieldnames=fields)
        if log.tell()==0: writer.writeheader()
        for epoch in range(start_epoch, cfg["epochs"]):
            model.train(); optimizer.zero_grad(set_to_none=True)
            for batch_index,batch in enumerate(loaders["train"]):
                ids=batch["thinking_input_ids"].to(device); mask=batch["thinking_attention_mask"].to(device)
                x0=canonical_t5_x0(encoder,ids,mask,cfg["latent_mean"],cfg["latent_std"])
                groups,rmask,pmask=group_batched_thinking(x0,mask,mc.group_size)
                slots,pred=model(groups); loss=masked_reconstruction_mse(pred,groups,rmask)
                (loss/cfg["gradient_accumulation_steps"]).backward()
                if (batch_index+1)%cfg["gradient_accumulation_steps"]==0 or batch_index+1==len(loaders["train"]):
                    grad=float(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip"])); optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True); step+=1
                    writer.writerow(dict(epoch=epoch,optimizer_step=step,split="train",reconstruction_mse=float(loss),token_cosine="",mean_pool_mse="",valid_token_count=int(rmask.sum()),valid_plan_slot_count=int(pmask.sum()),learning_rate=scheduler.get_last_lr()[0],gradient_norm=grad)); log.flush()
                    if cfg.get("save_every", 0) and step % int(cfg["save_every"]) == 0:
                        torch.save(checkpoint_payload(model,optimizer,scheduler,epoch,step,best,cfg,args),out/f"checkpoint_{step}.pt")
                    max_steps=args.max_optimizer_steps or cfg.get("max_optimizer_steps")
                    if max_steps and step >= int(max_steps): break
            if train_only:
                torch.save(checkpoint_payload(model,optimizer,scheduler,epoch,step,best,cfg,args),out/"final.pt")
                if max_steps and step >= int(max_steps): break
                continue
            model.eval(); sums=torch.zeros(5,dtype=torch.float64,device=device)
            with torch.no_grad():
                for batch in loaders["validation"]:
                    ids=batch["thinking_input_ids"].to(device); mask=batch["thinking_attention_mask"].to(device)
                    x0=canonical_t5_x0(encoder,ids,mask,cfg["latent_mean"],cfg["latent_std"])
                    groups,rmask,pmask=group_batched_thinking(x0,mask,mc.group_size); _,pred=model(groups)
                    n=rmask.sum(); sums[0] += masked_reconstruction_mse(pred,groups,rmask).double()*n; sums[1] += masked_token_cosine(pred,groups,rmask).double()*n; sums[2] += masked_reconstruction_mse(mean_pool_reconstruction(groups,rmask),groups,rmask).double()*n; sums[3] += n; sums[4] += pmask.sum()
            val=(sums[:3]/sums[3]).cpu().tolist()
            writer.writerow(dict(epoch=epoch,optimizer_step=step,split="validation",reconstruction_mse=val[0],token_cosine=val[1],mean_pool_mse=val[2],valid_token_count=int(sums[3]),valid_plan_slot_count=int(sums[4]),learning_rate=scheduler.get_last_lr()[0],gradient_norm="")); log.flush()
            payload=checkpoint_payload(model,optimizer,scheduler,epoch,step,min(best,val[0]),cfg,args)
            torch.save(payload,out/"last.pt")
            if is_better_validation(val[0],best): best=val[0]; torch.save(payload,out/"best.pt")
    (out/"training_manifest.json").write_text(json.dumps({"complete":True,"train_only":train_only,
        "best_validation_mse":None if train_only else best,"optimizer_step":step,
        "data_manifest":args.data_manifest,"max_thinking_tokens":cfg["max_thinking_tokens"],
        "observed_max_thinking_length":max(dataset.max_thinking_length for dataset in datasets.values())},indent=2))

if __name__ == "__main__": main()
