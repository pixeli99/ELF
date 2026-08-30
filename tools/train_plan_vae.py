#!/usr/bin/env python3
"""Train the frozen Plan-VAE that turns gold thinking into K plan slots.

Extracted from the v7 scratchpad script and parameterised, because the v7 artifact
(beta 0.1, free bits 1.0, lambda 0.5) turned out to keep only 23% of the gold numeric
answers through an encode/decode round trip, while a plain per-span PCA at the same
8192-float budget keeps 44%. The KL/aux pressure, not the budget, is what removes the
content the response would need -- so beta, free bits and the aux weight have to be
swept and gated on tools/eval_plan_vae_roundtrip.py, not on reconstruction MSE.

  python tools/train_plan_vae.py --beta 0.02 --free-bits 4 --aux-weight 0.5 --out data/plan_vae_runs/b0.02
"""
import argparse, csv, json, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from transformers import AutoTokenizer
from modules.t5_encoder import get_encoder
from modules.plan_vae import PlanVAE, K_MAX, Z_DIM, WIDTH

REV, LATENT_STD = "df1b051c49625cf57a3d0d8d3863ed4d13564fe4", 0.2


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--free-bits", type=float, default=1.0, help="nats per slot exempt from the KL penalty")
    ap.add_argument("--aux-weight", type=float, default=0.5, help="weight on predicting the pooled response latent")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--kl-warmup", type=int, default=4000)
    ap.add_argument("--eval-rows", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--span-decode", action="store_true",
                    help="add the direct slot->own-span linear decode path")
    ap.add_argument("--span-input", choices=("mean", "flat"), default="mean",
                    help="encoder input per slot: the span mean, or the whole span")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = "cuda"
    torch.manual_seed(args.seed)

    data = ROOT / "data/conditional_v1/data"
    thinking, response = [], []
    for shard in sorted(data.glob("train-*.parquet")):
        t = pq.read_table(shard, columns=["thinking", "response"])
        thinking += t.column("thinking").to_pylist()
        response += t.column("response").to_pylist()
    order = torch.randperm(len(thinking), generator=torch.Generator().manual_seed(7)).tolist()
    eval_idx, train_idx = order[:args.eval_rows], order[args.eval_rows:]
    print(f"train {len(train_idx)}  eval {len(eval_idx)}", flush=True)

    tok = AutoTokenizer.from_pretrained("t5-small", revision=REV, use_fast=True)
    _, t5 = get_encoder("t5-small", dtype=torch.float32, revision=REV)
    t5 = t5.to(dev).eval().requires_grad_(False)

    @torch.no_grad()
    def encode(texts, cap=1024):
        enc = tok(texts, add_special_tokens=True, truncation=True, max_length=cap,
                  padding=True, return_tensors="pt")
        ids, mask = enc["input_ids"].to(dev), enc["attention_mask"].to(dev).bool()
        return t5(input_ids=ids, attention_mask=mask).float() / LATENT_STD, mask

    model = PlanVAE(span_decode=args.span_decode, span_input=args.span_input).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * max(0.05, 1 - s / args.steps))

    def losses(think_texts, resp_texts, beta_eff):
        x, mask = encode(think_texts)
        with torch.no_grad():
            r, rmask = encode(resp_texts)
            r_target = (r * rmask.unsqueeze(-1)).sum(1) / rmask.sum(1, keepdim=True).clamp_min(1)
        mu, logvar, active = model.posterior(x, mask)
        z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        recon, aux_pred = model.decode(z, x.shape[1], active)
        recon_mse = ((recon - x).square().mean(-1) * mask.float()).sum() / mask.float().sum()
        kl_slot = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1)     # (B, K) nats
        kl_term = kl_slot.clamp_min(args.free_bits).mean() / Z_DIM
        aux_mse = F.mse_loss(aux_pred, r_target)
        loss = recon_mse + beta_eff * kl_term + args.aux_weight * aux_mse
        used = (kl_slot > args.free_bits).float().sum(-1)
        return loss, recon_mse, kl_slot, aux_mse, used

    @torch.no_grad()
    def evaluate():
        model.eval()
        recs, kls, actives, lens = [], [], [], []
        for i in range(0, len(eval_idx), args.batch_size):
            take = eval_idx[i:i + args.batch_size]
            x, mask = encode([thinking[j] for j in take])
            mu, logvar, active = model.posterior(x, mask)
            recon, _ = model.decode(mu, x.shape[1], active)
            kl_slot = 0.5 * (mu.square() + logvar.exp() - 1 - logvar).sum(-1)
            valid = mask.float()
            recs.append(float(((recon - x).square().mean(-1) * valid).sum() / valid.sum()))
            kls.append(float(kl_slot.mean()))
            actives += (kl_slot > args.free_bits).float().sum(-1).tolist()
            lens += mask.sum(-1).tolist()
        model.train()
        actives, lens = np.array(actives), np.array(lens, dtype=float)
        return {"recon_mse": round(float(np.mean(recs)), 4),
                "kl_per_slot": round(float(np.mean(kls)), 3),
                "active_mean": round(float(actives.mean()), 2),
                "active_std": round(float(actives.std()), 2),
                "corr_active_vs_len": round(float(np.corrcoef(actives, lens)[0, 1]), 3)
                if actives.std() > 0 else 0.0}

    log = csv.writer(open(out / "log.csv", "w", newline=""))
    log.writerow(["step", "loss", "recon", "kl_slot", "aux", "active_mean", "lr"])
    t0, pointer, epoch_order = time.time(), 0, None
    for step in range(1, args.steps + 1):
        if epoch_order is None or pointer + args.batch_size > len(epoch_order):
            epoch_order = torch.randperm(len(train_idx)).tolist(); pointer = 0
        take = [train_idx[j] for j in epoch_order[pointer:pointer + args.batch_size]]
        pointer += args.batch_size
        beta_eff = args.beta * (min(1.0, step / args.kl_warmup) if args.kl_warmup else 1.0)
        loss, recon, kl_slot, aux, used = losses(
            [thinking[i] for i in take], [response[i] for i in take], beta_eff)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        if step % 100 == 0:
            log.writerow([step, f"{float(loss):.4f}", f"{float(recon):.4f}",
                          f"{float(kl_slot.mean()):.2f}", f"{float(aux):.4f}",
                          f"{float(used.mean()):.1f}", f"{sched.get_last_lr()[0]:.2e}"])
        if step % 500 == 0:
            print(f"step {step}/{args.steps} loss {float(loss):.4f} recon {float(recon):.4f} "
                  f"kl/slot {float(kl_slot.mean()):.2f} ({(time.time()-t0)/step:.2f}s/step)", flush=True)
        if step % 4000 == 0 or step == args.steps:
            report = evaluate()
            print(f"EVAL step {step}: {json.dumps(report)}", flush=True)
            (out / f"eval_{step}.json").write_text(json.dumps(report, indent=2))
    torch.save({"state_dict": model.state_dict(), "beta": args.beta, "k_max": K_MAX,
                "z_dim": Z_DIM, "free_bits": args.free_bits, "lambda": args.aux_weight,
                "steps": args.steps, "span_decode": args.span_decode,
                "span_input": args.span_input}, out / "final.pt")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
