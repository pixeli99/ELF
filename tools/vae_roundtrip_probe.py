"""What survives the Plan-VAE bottleneck? gold thinking -> T5 latents -> VAE(64x128) -> decode -> text.
argv: <cfg> <ckpt (for the decoder head)> [n_rows]"""
import sys, re, torch, pyarrow.parquet as pq
sys.path.insert(0, "src")
from configs.config import load_config_from_yaml
from modules.plan_vae import load_frozen_plan_vae
from utils.encoder_utils import encode_thinking_x0
from utils.generation_utils import _dlm_decode_logits_batch
from utils.stage_b_eval_runtime import load_model_and_encoder
cfg = load_config_from_yaml(sys.argv[1]); ckpt = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 64
dev = torch.device("cuda")
model, enc, tok, resolved = load_model_and_encoder(cfg, ckpt, dev)
raw = torch.load(resolved, map_location="cpu", weights_only=False); model.load_state_dict(raw["params"], strict=True); model.eval()
vcfg = load_config_from_yaml("src/configs/training_configs/train_conditional_common48w_ordered_causal_eos_v5.yml")
vae, _ = load_frozen_plan_vae(vcfg.plan_vae_artifact, dev, vcfg.plan_vae_artifact_sha256)
t = pq.read_table("data/conditional_v1/data/train-00018.parquet", columns=["thinking","response"]).to_pydict()
pat = re.compile(r"\\boxed\s*\{([^}]*)\}|####\s*([^\n]+)")
rows = []
for th, rp in zip(t["thinking"], t["response"]):
    m = list(pat.finditer(rp[-400:]))
    if m:
        ans = (m[-1].group(1) or m[-1].group(2) or "").strip()
        if re.fullmatch(r"-?[\d,]+(\.\d+)?", ans): rows.append((th, ans.replace(",", "")))
    if len(rows) >= N: break
print(f"rows with a numeric gold answer: {len(rows)}")
kept_orig = kept_rt = 0; shown = 0
with torch.no_grad():
    for i in range(0, len(rows), 8):
        chunk = rows[i:i+8]
        e = tok([c[0] for c in chunk], max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
        ids, m = e["input_ids"].to(dev), e["attention_mask"].to(dev)
        x0 = encode_thinking_x0(ids, m, enc, cfg.latent_mean, cfg.latent_std).float()
        mu, _, active = vae.posterior(x0, m.bool())
        recon = vae.decode(mu, x0.shape[1], active=active)[0].float()
        # the ELF decoder's RoPE is built for config.max_length; place the 1024-position
        # thinking latents at the front of a full-width buffer and mask the rest out
        W = cfg.max_length
        for name, lat in (("orig", x0), ("roundtrip", recon)):
            buf = lat.new_zeros((lat.shape[0], W, lat.shape[2])); buf[:, :lat.shape[1]] = lat
            bm = m.new_zeros((m.shape[0], W)); bm[:, :m.shape[1]] = m
            logits = _dlm_decode_logits_batch(buf.to(next(model.parameters()).dtype), model, 1.0, cfg, 1.0, attention_mask=bm.bool())
            pred = logits.argmax(-1)
            for j, (th, ans) in enumerate(chunk):
                text = tok.decode(pred[j][bm[j].bool()].cpu().tolist(), skip_special_tokens=True)
                hit = ans in text.replace(",", "")
                if name == "orig": kept_orig += hit
                else:
                    kept_rt += hit
                    if shown < 3:
                        shown += 1
                        print(f"\n  gold answer {ans!r} | in round-trip text: {hit}")
                        print(f"  thinking  : {th[:200]!r}")
                        print(f"  roundtrip : {text[:200]!r}")
n = len(rows)
print(f"\ngold answer string present after ELF-decoding the clean T5 latents: {kept_orig}/{n} = {kept_orig/n:.2f}")
print(f"gold answer string present after the Plan-VAE round trip          : {kept_rt}/{n} = {kept_rt/n:.2f}")
