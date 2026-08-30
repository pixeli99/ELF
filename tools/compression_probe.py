"""Is the plan bottleneck a capacity limit or a bad autoencoder?

Same budget, three compressions of the gold thinking latents, each decoded back to
1024x512 and then read out with the ELF decoder head; score = does the gold numeric
answer still appear in the decoded text.

  span_mean_512   64 spans x 512 dims (32768 floats), per-span linear decoder
  span_pca_128    64 spans x 128 dims  (8192 floats, the VAE's budget), per-span linear decoder
  plan_vae_128    64 slots x 128 dims  (8192 floats), the trained VAE's own cross-attention decoder
"""
import sys, re, torch, pyarrow.parquet as pq
sys.path.insert(0, "src")
from configs.config import load_config_from_yaml
from modules.plan_vae import load_frozen_plan_vae, span_pool, SPAN, K_MAX
from utils.encoder_utils import encode_thinking_x0
from utils.generation_utils import _dlm_decode_logits_batch
from utils.stage_b_eval_runtime import load_model_and_encoder
N_FIT, N_EVAL = int(sys.argv[1]) if len(sys.argv) > 1 else 1024, 64
cfg = load_config_from_yaml("src/configs/training_configs/train_conditional_common48w_vanilla_causal_eos_v5.yml")
vcfg = load_config_from_yaml("src/configs/training_configs/train_conditional_common48w_ordered_causal_eos_v5.yml")
dev = torch.device("cuda")
model, enc, tok, resolved = load_model_and_encoder(cfg, "outputs/elf_b_conditional_common48w_vanilla_causal_eos_v5/checkpoint_10000", dev)
raw = torch.load(resolved, map_location="cpu", weights_only=False); model.load_state_dict(raw["params"], strict=True); model.eval()
vae, _ = load_frozen_plan_vae(vcfg.plan_vae_artifact, dev, vcfg.plan_vae_artifact_sha256)
t = pq.read_table("data/conditional_v1/data/train-00018.parquet", columns=["thinking","response"]).to_pydict()
pat = re.compile(r"\\boxed\s*\{([^}]*)\}|####\s*([^\n]+)")
fit_rows, eval_rows = [], []
for th, rp in zip(t["thinking"], t["response"]):
    m = list(pat.finditer(rp[-400:]))
    ans = None
    if m:
        a = (m[-1].group(1) or m[-1].group(2) or "").strip()
        if re.fullmatch(r"-?[\d,]+(\.\d+)?", a): ans = a.replace(",", "")
    if ans and len(eval_rows) < N_EVAL: eval_rows.append((th, ans))
    elif len(fit_rows) < N_FIT: fit_rows.append(th)
    if len(fit_rows) >= N_FIT and len(eval_rows) >= N_EVAL: break
print(f"fit rows {len(fit_rows)}, eval rows {len(eval_rows)}")
def latents(texts):
    out_x, out_m = [], []
    for i in range(0, len(texts), 16):
        e = tok(texts[i:i+16], max_length=1024, padding="max_length", truncation=True, return_tensors="pt")
        ids, m = e["input_ids"].to(dev), e["attention_mask"].to(dev)
        with torch.no_grad(): out_x.append(encode_thinking_x0(ids, m, enc, cfg.latent_mean, cfg.latent_std).float().cpu())
        out_m.append(m.cpu())
    return torch.cat(out_x), torch.cat(out_m)
xf, mf = latents(fit_rows)
# per-span views: (rows*K, SPAN*512) targets, span-mean and span-PCA codes
def spans_of(x):                       # (B,1024,512) -> (B*K, SPAN, 512)
    B = x.shape[0]; return x.view(B, K_MAX, SPAN, 512)
Sf = spans_of(xf).reshape(-1, SPAN * 512)
mean_f = spans_of(xf).mean(2).reshape(-1, 512)
keep = Sf.abs().sum(1) > 0                     # drop all-pad spans from the fit
Sf, mean_f = Sf[keep], mean_f[keep]
mu_s = Sf.mean(0)
U, S, V = torch.pca_lowrank((Sf - mu_s).to(dev), q=128, center=False)   # PCA over full span content
Vp = V[:, :128].cpu()
def ridge(X, Y, lam=1e2):
    X = X.to(dev); Y = Y.to(dev)
    xm, ym = X.mean(0), Y.mean(0)
    Xc, Yc = X - xm, Y - ym
    W = torch.linalg.solve(Xc.T @ Xc + lam * torch.eye(X.shape[1], device=dev), Xc.T @ Yc)
    return (W, xm, ym)
Wm = ridge(mean_f, Sf)                                   # 512 -> SPAN*512
Wp = ridge(((Sf - mu_s).to(dev) @ Vp.to(dev)).cpu(), Sf)  # 128 -> SPAN*512
def apply_ridge(R, X):
    W, xm, ym = R; return (X.to(dev) - xm) @ W + ym
xe, me = latents([r[0] for r in eval_rows])
answers = [r[1] for r in eval_rows]
W_ = cfg.max_length
def read_out(lat, mask):
    hits = 0
    for i in range(0, lat.shape[0], 8):
        l, m = lat[i:i+8].to(dev), mask[i:i+8].to(dev)
        buf = l.new_zeros((l.shape[0], W_, 512)); buf[:, :l.shape[1]] = l
        bm = m.new_zeros((m.shape[0], W_)); bm[:, :m.shape[1]] = m
        with torch.no_grad():
            pred = _dlm_decode_logits_batch(buf.to(next(model.parameters()).dtype), model, 1.0, cfg, 1.0, attention_mask=bm.bool()).argmax(-1)
        for j in range(l.shape[0]):
            text = tok.decode(pred[j][bm[j].bool()].cpu().tolist(), skip_special_tokens=True).replace(",", "")
            hits += answers[i + j] in text
    return hits
recons = {}
recons["uncompressed"] = xe
sp = spans_of(xe)
recons["span_mean_512"] = apply_ridge(Wm, sp.mean(2).reshape(-1, 512)).reshape(xe.shape[0], K_MAX, SPAN, 512).reshape(xe.shape[0], 1024, 512).cpu()
code = (sp.reshape(-1, SPAN * 512).to(dev) - mu_s.to(dev)) @ Vp.to(dev)
recons["span_pca_128"] = apply_ridge(Wp, code.cpu()).reshape(xe.shape[0], K_MAX, SPAN, 512).reshape(xe.shape[0], 1024, 512).cpu()
with torch.no_grad():
    mu, _, active = vae.posterior(xe.to(dev), me.to(dev).bool())
    recons["plan_vae_128"] = vae.decode(mu, 1024, active=active)[0].float().cpu()
print(f"\ngold numeric answer recovered from the ELF-decoded reconstruction (n={len(answers)}):")
for name, lat in recons.items():
    floats = {"uncompressed": 1024 * 512, "span_mean_512": K_MAX * 512, "span_pca_128": K_MAX * 128, "plan_vae_128": K_MAX * 128}[name]
    h = read_out(lat, me)
    print(f"  {name:16s} {floats:7d} floats/row  answer kept {h}/{len(answers)} = {h/len(answers):.2f}")
