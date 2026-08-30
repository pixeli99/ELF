"""How much does the plan tell about the response beyond the prompt?  Ridge probes on frozen T5 latents.
argv: [n_rows] [gpu]"""
import sys, torch, numpy as np, pyarrow.parquet as pq, collections
sys.path.insert(0, "src")
from configs.config import load_config_from_yaml
from modules.t5_encoder import get_encoder
from modules.plan_vae import load_frozen_plan_vae, span_pool
from utils.encoder_utils import encode_thinking_x0
from transformers import AutoTokenizer
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
cfg = load_config_from_yaml("src/configs/training_configs/train_conditional_common48w_ordered_10k_v1.yml")
dev = torch.device("cuda"); torch.manual_seed(0)
tok = AutoTokenizer.from_pretrained("t5-small")
_, enc = get_encoder("t5-small", torch.float32); enc = enc.to(dev).eval()
vae, _ = load_frozen_plan_vae(cfg.plan_vae_artifact, dev, cfg.plan_vae_artifact_sha256)
t = pq.read_table("data/conditional_v1/data/train-00001.parquet", columns=["prompt","thinking","response"]).to_pydict()
rows = [(p, th, r) for p, th, r in zip(t["prompt"], t["thinking"], t["response"]) if len(r) > 20][:N]
print("rows", len(rows))
def enc_texts(texts, max_len):
    e = tok(texts, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    ids, m = e["input_ids"].to(dev), e["attention_mask"].to(dev)
    with torch.no_grad():
        x = encode_thinking_x0(ids, m, enc, cfg.latent_mean, cfg.latent_std).float()
    return x, m.bool(), ids
def mean_pool(x, m):
    mf = m.unsqueeze(-1).float(); return (x * mf).sum(1) / mf.sum(1).clamp_min(1)
def spans(x, m, k, span=16):
    pooled, active = span_pool(x, m)      # (B, 64, 512)
    return pooled[:, :k].reshape(x.shape[0], -1)
F = collections.defaultdict(list); Y = collections.defaultdict(list); resp_ids_all = []
B = 32
for i in range(0, len(rows), B):
    chunk = rows[i:i+B]
    xp, mp, _ = enc_texts([r[0] for r in chunk], 512)
    xt, mt, _ = enc_texts([r[1] for r in chunk], 1024)
    xr, mr, rid = enc_texts([r[2] for r in chunk], 512)
    with torch.no_grad():
        mu, logvar, active = vae.posterior(xt, mt)
    F["prompt_mean"].append(mean_pool(xp, mp).cpu())
    F["prompt_spans8"].append(spans(xp, mp, 8).cpu())
    F["plan_mu"].append(mu.reshape(mu.shape[0], -1).float().cpu())
    F["plan_budget"].append(active.sum(1, keepdim=True).float().cpu())
    tp, _ = span_pool(xt, mt)                                   # (B,64,512) raw span pool of thinking
    F["think_spans8"].append(tp[:, :8].reshape(tp.shape[0], -1).cpu())
    F["think_mean"].append(mean_pool(xt, mt).cpu())
    F["think_super8"].append(tp.reshape(tp.shape[0], 8, 8, 512).mean(2).reshape(tp.shape[0], -1).cpu())
    Y["resp_mean"].append(mean_pool(xr, mr).cpu())
    Y["resp_spans4"].append(spans(xr, mr, 4).cpu())
    resp_ids_all.append(rid.cpu())
    if i % 1600 == 0: print("encoded", i, flush=True)
F = {k: torch.cat(v) for k, v in F.items()}; Y = {k: torch.cat(v) for k, v in Y.items()}
rid = torch.cat(resp_ids_all)
ntr = int(len(rows) * 0.875)
# bag-of-words target: top-2000 token ids in train responses (excluding pad/eos)
cnt = torch.bincount(rid[:ntr].reshape(-1), minlength=32128); cnt[:2] = 0
vocab = cnt.topk(2000).indices
bow = torch.zeros(len(rows), 2000)
for j, v in enumerate(vocab.tolist()):
    bow[:, j] = (rid == v).any(1).float()
Y["resp_bow2000"] = bow
def ridge_r2(Xtr, Ytr, Xva, Yva):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr = ((Xtr - mu) / sd).to(dev); Xva = ((Xva - mu) / sd).to(dev)
    ym = Ytr.mean(0); Ytr = (Ytr - ym).to(dev); Yva = (Yva - ym).to(dev)
    XtX = Xtr.T @ Xtr; XtY = Xtr.T @ Ytr; eye = torch.eye(Xtr.shape[1], device=dev)
    best = -1e9
    for lam in [1e1, 1e2, 1e3, 1e4, 1e5]:
        W = torch.linalg.solve(XtX + lam * eye, XtY)
        r2 = 1 - ((Xva @ W - Yva) ** 2).sum() / (Yva ** 2).sum()
        best = max(best, r2.item())
    return best
sets = {
    "prompt_mean": ["prompt_mean"], "prompt_spans8": ["prompt_spans8"],
    "plan_mu": ["plan_mu"], "plan_budget": ["plan_budget"],
    "prompt_spans8+plan_budget": ["prompt_spans8", "plan_budget"],
    "prompt_spans8+plan_mu": ["prompt_spans8", "plan_mu"],
    "think_mean": ["think_mean"], "think_super8": ["think_super8"], "think_spans8": ["think_spans8"],
    "prompt_spans8+think_mean": ["prompt_spans8", "think_mean"],
    "prompt_spans8+think_super8": ["prompt_spans8", "think_super8"],
    "prompt_spans8+think_spans8": ["prompt_spans8", "think_spans8"],
}
print(f"{'features':32s} " + " ".join(f"{k:>14s}" for k in Y))
for name, keys in sets.items():
    X = torch.cat([F[k] for k in keys], 1)
    r2s = [ridge_r2(X[:ntr], Y[k][:ntr], X[ntr:], Y[k][ntr:]) for k in Y]
    print(f"{name:32s} " + " ".join(f"{r:14.3f}" for r in r2s), flush=True)
