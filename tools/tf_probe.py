"""Teacher-forced response denoising with oracle / shuffled / null plan.  argv: <cfg> <ckpt> [n_batches] [seed]"""
import sys, torch, torch.nn.functional as F
sys.path.insert(0, "src")
from configs.config import load_config_from_yaml
from utils.stage_b_eval_runtime import load_model_and_encoder
from utils.encoder_utils import encode_x0
from utils.plan_stream import build_vae_plan_target
from utils.sampling_utils import add_noise
from utils.conditional_data import ConditionalPairedDataset, ConditionalSchedule, get_conditional_dataloader
from utils.generation_utils import _dlm_decode_logits_batch
from modules.plan_vae import load_frozen_plan_vae
cfg = load_config_from_yaml(sys.argv[1]); ckpt = sys.argv[2]; nb = int(sys.argv[3]) if len(sys.argv) > 3 else 6; seed = int(sys.argv[4]) if len(sys.argv) > 4 else 12345
dev = torch.device("cuda")
model, encoder, tok, _ = load_model_and_encoder(cfg, ckpt, dev)
raw = torch.load(ckpt, map_location="cpu", weights_only=False); model.load_state_dict(raw["params"], strict=True); model.eval()   # raw weights like the formal evals
vae, _ = load_frozen_plan_vae(cfg.plan_vae_artifact, dev, cfg.plan_vae_artifact_sha256)
base = ConditionalPairedDataset(cfg.conditional_train_manifest, cfg.conditional_train_manifest_sha256, False)
loader, _ = get_conditional_dataloader(ConditionalSchedule(base, 8 * nb, seed), tok, cfg, batch_size=8)
res = {}
torch.manual_seed(0)
with torch.no_grad():
    for batch in loader:
        ids = batch["input_ids"].to(dev).long(); am = batch["attention_mask"].to(dev).float(); cm = batch["cond_seq_mask"].to(dev).float()
        x0 = encode_x0(ids, am, encoder, cfg.latent_mean, cfg.latent_std, cond_mask=cm).float()
        x0_plan, pmask, active = build_vae_plan_target(batch["plan_input_ids"].to(dev).long(), batch["plan_attention_mask"].to(dev).bool(), encoder, vae, cfg)
        x0_plan = x0_plan.float(); B = x0.shape[0]
        resp = (am > 0) & (cm == 0)                      # real response tokens
        lengths = am.sum(1, keepdim=True); pos = torch.arange(x0.shape[1], device=dev)[None, :]
        band = (pos >= lengths) & (pos < lengths + 64)   # EOS band
        plans = {"oracle": (x0_plan, 1.0), "shuffled": (x0_plan.roll(1, dims=0), 1.0),
                 "null": (torch.randn_like(x0_plan) * cfg.denoiser_noise_scale, 0.0)}
        for t in [0.1, 0.3, 0.5, 0.7, 0.9]:
            tb = torch.full((B,), t, device=dev)
            eps = torch.randn_like(x0)
            for name, (xp, tp) in plans.items():
                z = add_noise(x0, eps, tb, cfg, cond_seq_mask=cm.unsqueeze(-1))
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(torch.cat([z, torch.zeros_like(z)], -1), tb, attention_mask=am.bool(), deterministic=True,
                                self_cond_cfg_scale=torch.ones(B, device=dev), x_plan=xp, t_plan=torch.full((B,), tp, device=dev),
                                plan_mask=pmask, condition_token_mask=cm.bool())
                xhat = out[0].float()
                mse_resp = F.mse_loss(xhat[resp], x0[resp]).item(); mse_band = F.mse_loss(xhat[band], x0[band]).item()
                pred = _dlm_decode_logits_batch(xhat, model, 1.0, cfg, 1.0, x_plan=xp, t_plan_decode_val=tp, plan_mask=pmask, attention_mask=am.bool()).argmax(-1) if False else None
                res.setdefault((t, name), []).append((mse_resp, mse_band))
print("teacher-forced x0 MSE on response tokens / EOS band (lower is better); rows unseen by training")
for (t, name), v in sorted(res.items()):
    print(f"t={t:.1f} plan={name:8s} resp_mse={sum(a for a,_ in v)/len(v):.4f}  band_mse={sum(b for _,b in v)/len(v):.4f}")
