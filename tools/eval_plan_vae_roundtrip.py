#!/usr/bin/env python3
"""Acceptance metric for a Plan-VAE artifact: does the gold answer survive the bottleneck?

gold thinking -> T5 latents -> VAE encode/decode -> ELF decoder head -> text, then check
whether the reference's numeric answer string is still there. Reconstruction MSE does not
separate a plan that keeps the reasoning from one that keeps only its shape: the v7
artifact sits at MSE 0.479 (mean-pool baseline 0.539) yet keeps only 23% of the answers,
below a per-span PCA at the same budget (44%).
"""
import argparse, json, re, sys
from pathlib import Path
import torch, pyarrow.parquet as pq
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from configs.config import load_config_from_yaml
from modules.plan_vae import load_frozen_plan_vae
from utils.encoder_utils import encode_thinking_x0
from utils.generation_utils import _dlm_decode_logits_batch
from utils.stage_b_eval_runtime import load_model_and_encoder

ANSWER = re.compile(r"\\boxed\s*\{([^}]*)\}|####\s*([^\n]+)")
DECODER_CONFIG = "src/configs/training_configs/train_conditional_common48w_vanilla_causal_eos_v5.yml"
DECODER_CKPT = "outputs/elf_b_conditional_common48w_vanilla_causal_eos_v5/checkpoint_10000"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--rows", type=int, default=128)
    ap.add_argument("--shard", default="data/conditional_v1/data/train-00018.parquet")
    ap.add_argument("--decoder-config", default=DECODER_CONFIG)
    ap.add_argument("--decoder-checkpoint", default=DECODER_CKPT)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    cfg = load_config_from_yaml(args.decoder_config)
    model, enc, tok, resolved = load_model_and_encoder(cfg, args.decoder_checkpoint, dev)
    raw = torch.load(resolved, map_location="cpu", weights_only=False)
    model.load_state_dict(raw["params"], strict=True); model.eval()
    vae, meta = load_frozen_plan_vae(args.artifact, dev)
    t = pq.read_table(args.shard, columns=["thinking", "response"]).to_pydict()
    rows = []
    for th, rp in zip(t["thinking"], t["response"]):
        m = list(ANSWER.finditer(rp[-400:]))
        if not m:
            continue
        a = (m[-1].group(1) or m[-1].group(2) or "").strip()
        if re.fullmatch(r"-?[\d,]+(\.\d+)?", a):
            rows.append((th, a.replace(",", "")))
        if len(rows) >= args.rows:
            break
    kept_clean = kept_round = 0
    width = cfg.max_length
    with torch.no_grad():
        for i in range(0, len(rows), 8):
            chunk = rows[i:i + 8]
            e = tok([c[0] for c in chunk], max_length=1024, padding="max_length",
                    truncation=True, return_tensors="pt")
            ids, mask = e["input_ids"].to(dev), e["attention_mask"].to(dev)
            x0 = encode_thinking_x0(ids, mask, enc, cfg.latent_mean, cfg.latent_std).float()
            mu, _, active = vae.posterior(x0, mask.bool())
            recon = vae.decode(mu, x0.shape[1], active=active)[0].float()
            for name, lat in (("clean", x0), ("roundtrip", recon)):
                buf = lat.new_zeros((lat.shape[0], width, lat.shape[2])); buf[:, :lat.shape[1]] = lat
                bm = mask.new_zeros((mask.shape[0], width)); bm[:, :mask.shape[1]] = mask
                pred = _dlm_decode_logits_batch(
                    buf.to(next(model.parameters()).dtype), model, 1.0, cfg, 1.0,
                    attention_mask=bm.bool()).argmax(-1)
                for j, (_, ans) in enumerate(chunk):
                    text = tok.decode(pred[j][bm[j].bool()].cpu().tolist(),
                                      skip_special_tokens=True).replace(",", "")
                    if ans in text:
                        if name == "clean":
                            kept_clean += 1
                        else:
                            kept_round += 1
    n = len(rows)
    result = {"artifact": args.artifact, "meta": {k: v for k, v in meta.items()}, "rows": n,
              "answer_kept_clean_latents": round(kept_clean / max(n, 1), 4),
              "answer_kept_after_roundtrip": round(kept_round / max(n, 1), 4)}
    print(json.dumps(result))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
