"""Equal-dimension comparison: PCA-512 of each candidate plan representation -> response targets."""
import sys, torch, pyarrow.parquet as pq, collections
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
def enc_texts(texts, max_len):
    e = tok(texts, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
    ids, m = e["input_ids"].to(dev), e["attention_mask"].to(dev)
    with torch.no_grad(): x = encode_thinking_x0(ids, m, enc, cfg.latent_mean, cfg.latent_std).float()
    return x, m.bool(), ids
def mean_pool(x, m):
    mf = m.unsqueeze(-1).float(); return (x * mf).sum(1) / mf.sum(1).clamp_min(1)
F = collections.defaultdict(list); Y = collections.defaultdict(list); rids = []
for i in range(0, len(rows), 32):
    chunk = rows[i:i+32]
    xp, mp, _ = enc_texts([r[0] for r in chunk], 512); xt, mt, _ = enc_texts([r[1] for r in chunk], 1024); xr, mr, rid = enc_texts([r[2] for r in chunk], 512)
    with torch.no_grad(): mu, logvar, active = vae.posterior(xt, mt)
    tp, _ = span_pool(xt, mt)
    F["prompt_mean"].append(mean_pool(xp, mp).cpu())
    F["prompt_spans16"].append(span_pool(xp, mp)[0][:, :16].reshape(len(chunk), -1).cpu())
    F["plan_mu"].append(mu.reshape(len(chunk), -1).float().cpu())
    F["plan_mu_actmean"].append(((mu * active.unsqueeze(-1)).sum(1) / active.sum(1, keepdim=True).clamp_min(1)).float().cpu())
    F["think_mean"].append(mean_pool(xt, mt).cpu())
    F["think_spans64"].append(tp.reshape(len(chunk), -1).cpu())
    Y["resp_mean"].append(mean_pool(xr, mr).cpu()); Y["resp_spans4"].append(span_pool(xr, mr)[0][:, :4].reshape(len(chunk), -1).cpu()); rids.append(rid.cpu())
F = {k: torch.cat(v) for k, v in F.items()}; Y = {k: torch.cat(v) for k, v in Y.items()}; rid = torch.cat(rids)
ntr = int(len(rows) * 0.875)
cnt = torch.bincount(rid[:ntr].reshape(-1), minlength=32128); cnt[:2] = 0; vocab = cnt.topk(2000).indices
Y["resp_bow2000"] = torch.stack([(rid == v).any(1).float() for v in vocab.tolist()], 1)
def pca(X, k):
    mu = X[:ntr].mean(0); Xc = (X - mu).to(dev)
    U, S, V = torch.pca_lowrank(Xc[:ntr], q=min(k, Xc.shape[1]), center=False)
    return (Xc @ V[:, :k]).cpu()
def ridge_r2(Xtr, Ytr, Xva, Yva):
    mu, sd = Xtr.mean(0), Xtr.std(0).clamp_min(1e-6)
    Xtr = ((Xtr - mu) / sd).to(dev); Xva = ((Xva - mu) / sd).to(dev); ym = Ytr.mean(0); Ytr = (Ytr - ym).to(dev); Yva = (Yva - ym).to(dev)
    XtX = Xtr.T @ Xtr; XtY = Xtr.T @ Ytr; eye = torch.eye(Xtr.shape[1], device=dev); best = -1e9
    for lam in [1e0, 1e1, 1e2, 1e3, 1e4]:
        W = torch.linalg.solve(XtX + lam * eye, XtY); best = max(best, (1 - ((Xva @ W - Yva) ** 2).sum() / (Yva ** 2).sum()).item())
    return best
feats = {"prompt_mean(512)": F["prompt_mean"], "prompt_spans16_pca512": pca(F["prompt_spans16"], 512),
         "plan_mu_actmean(128)": F["plan_mu_actmean"], "plan_mu_pca128": pca(F["plan_mu"], 128), "plan_mu_pca512": pca(F["plan_mu"], 512),
         "think_mean(512)": F["think_mean"], "think_spans64_pca512": pca(F["think_spans64"], 512), "think_spans64_pca128": pca(F["think_spans64"], 128)}
feats["prompt_pca512+plan_mu_pca512"] = torch.cat([feats["prompt_spans16_pca512"], feats["plan_mu_pca512"]], 1)
feats["prompt_pca512+think_spans64_pca512"] = torch.cat([feats["prompt_spans16_pca512"], feats["think_spans64_pca512"]], 1)
feats["prompt_pca512+think_mean"] = torch.cat([feats["prompt_spans16_pca512"], F["think_mean"]], 1)
print(f"{'features':36s} " + " ".join(f"{k:>14s}" for k in Y))
for name, X in feats.items():
    print(f"{name:36s} " + " ".join(f"{ridge_r2(X[:ntr], Y[k][:ntr], X[ntr:], Y[k][ntr:]):14.3f}" for k in Y), flush=True)
