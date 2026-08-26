"""Plan-VAE: thinking -> K_max span-anchored variational plan slots.

Extracted verbatim from the v7 training script (scratchpad/plan_vae.py) so the
repository can load the frozen artifact without importing training code.
Design constants (span 16, K_max 64, z 128, d_model 512) are properties of the
trained artifact and are read back from it at load time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

K_MAX, Z_DIM, D_MODEL, WIDTH = 64, 128, 512, 512

SPAN = 1024 // K_MAX      # 16 tokens per slot, absolute spans


def span_pool(x, mask):
    """Fixed slot assignment: slot k = masked mean of tokens [16k, 16k+16).

    The v1-v5 bisection showed learned query routing never escapes the
    ignore-the-slots basin (recon pinned at the positional-prior floor, 0.65),
    while a decoder fed these fixed span means beats the pool baseline within
    3k steps. So the aggregation STRUCTURE is fixed and only transforms are
    learned. Spans past a short sample pool to zero -- those slots' posteriors
    fall back to the prior, which is the soft-dynamic-K mechanism.
    """
    B, L, C = x.shape
    pad = (SPAN - L % SPAN) % SPAN
    xp = F.pad(x, (0, 0, 0, pad))
    mp = F.pad(mask, (0, pad)).float()
    k_here = xp.shape[1] // SPAN
    grouped = xp.view(B, k_here, SPAN, C)
    weights = mp.view(B, k_here, SPAN, 1)
    pooled = (grouped * weights).sum(2) / weights.sum(2).clamp_min(1)
    active = weights.sum(2).squeeze(-1) > 0            # (B, k_here)
    if k_here < K_MAX:
        pooled = F.pad(pooled, (0, 0, 0, K_MAX - k_here))
        active = F.pad(active, (0, K_MAX - k_here))
    return pooled[:, :K_MAX], active[:, :K_MAX]


class SlotBlock(nn.Module):
    """Self-attention + FFN over the 64 slots (capacity re-allocation)."""

    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(D_MODEL, 8, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, 2048), nn.GELU(),
                                 nn.Linear(2048, D_MODEL))
        self.n1, self.n2 = nn.LayerNorm(D_MODEL), nn.LayerNorm(D_MODEL)

    def forward(self, q, active):
        # Inactive (empty-span) slots are excluded as KEYS: content cannot leak
        # into them through attention. In v6 this leak is exactly what smeared
        # information across all 64 slots and erased the dynamic budget.
        h = self.n1(q)
        a, _ = self.attn(h, h, h, key_padding_mask=~active, need_weights=False)
        q = q + a
        return q + self.ffn(self.n2(q))


class Block(nn.Module):
    def __init__(self, self_attend):
        super().__init__()
        self.cross = nn.MultiheadAttention(D_MODEL, 8, batch_first=True)
        self.self_attn = (nn.MultiheadAttention(D_MODEL, 8, batch_first=True)
                          if self_attend else None)
        self.ffn = nn.Sequential(nn.Linear(D_MODEL, 2048), nn.GELU(),
                                 nn.Linear(2048, D_MODEL))
        self.n1, self.n2, self.n3 = (nn.LayerNorm(D_MODEL) for _ in range(3))
        # Keys/values must be normalized too: raw T5 latents have per-token norm
        # ~23, which saturates the attention logits at init and freezes the
        # routing (the beta=0 probe showed huge KL with recon stuck at the
        # positional-prior floor -- information present, decoder unable to route).
        self.nm = nn.LayerNorm(D_MODEL)

    def forward(self, q, memory, memory_pad):
        m = self.nm(memory)
        a, _ = self.cross(self.n1(q), m, m, key_padding_mask=memory_pad,
                          need_weights=False)
        q = q + a
        if self.self_attn is not None:
            a, _ = self.self_attn(self.n2(q), self.n2(q), self.n2(q), need_weights=False)
            q = q + a
        return q + self.ffn(self.n3(q))


SPAN = 1024 // K_MAX      # 16 tokens per slot, absolute spans


class PlanVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool_in = nn.Linear(WIDTH, D_MODEL)
        self.slot_pos = nn.Parameter(torch.randn(1, K_MAX, D_MODEL) * 0.02)
        self.enc = nn.ModuleList([SlotBlock() for _ in range(2)])
        self.mu = nn.Linear(D_MODEL, Z_DIM)
        self.logvar = nn.Linear(D_MODEL, Z_DIM)
        # Near-deterministic posterior at init: with the default sigma ~ 1 the
        # sampled z is mostly noise, the decoder learns to ignore the slots and
        # the posterior collapses before any information gets through (that is
        # exactly what round 1 measured). bias -4 -> sigma ~ 0.14 at step 0.
        nn.init.constant_(self.logvar.bias, -4.0)
        self.slot_in = nn.Linear(Z_DIM, D_MODEL)
        # Decoder-side slot identity. Without it a position query can only route
        # to a slot by CONTENT, which requires the encoder to have specialised
        # already -- a chicken-and-egg that kept v1-v3 pinned at the
        # ignore-the-slots solution (~0.64, the positional-prior floor).
        # With identity keys, "position i reads slot seg(i)" is linearly learnable.
        self.slot_id = nn.Parameter(torch.randn(1, K_MAX, D_MODEL) * 0.02)
        self.pos = nn.Parameter(torch.randn(1, 1024, D_MODEL) * 0.02)
        self.dec = nn.ModuleList([Block(False) for _ in range(2)])
        self.out = nn.Linear(D_MODEL, WIDTH)
        self.aux = nn.Sequential(nn.Linear(D_MODEL, 1024), nn.GELU(),
                                 nn.Linear(1024, WIDTH))

    def posterior(self, x, mask):
        pooled, active = span_pool(x, mask)
        q = self.pool_in(pooled) + self.slot_pos
        for block in self.enc:
            q = block(q, active)
        gate = active.unsqueeze(-1).float()
        # Inactive slots are pinned to the prior: mu = 0, logvar = 0 (sigma = 1),
        # KL exactly 0. The zero vector IS the null code -- in Stage B the tail
        # of x0_plan is exact zeros, so "how long is the plan" becomes generated
        # content instead of an input mask.
        mu = self.mu(q) * gate
        logvar = self.logvar(q).clamp(-8, 8) * gate
        return mu, logvar, active

    def decode(self, z, length, active=None):
        s = self.slot_in(z) + self.slot_id
        if active is None:
            active = torch.ones(z.shape[:2], dtype=torch.bool, device=z.device)
        p = self.pos[:, :length].expand(z.shape[0], -1, -1)
        for block in self.dec:
            p = block(p, s, None)
        gate = active.unsqueeze(-1).float()
        s_active = (s * gate).sum(1) / gate.sum(1).clamp_min(1)
        return self.out(p), self.aux(s_active)




def load_frozen_plan_vae(artifact_path, device="cpu", expected_sha256=None):
    """Load the exported Plan-VAE and freeze it. Returns (model, artifact_meta)."""
    import hashlib
    from pathlib import Path

    data = Path(artifact_path).read_bytes()
    if expected_sha256 and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"plan VAE artifact SHA256 mismatch: {artifact_path}")
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if int(artifact["k_max"]) != K_MAX or int(artifact["z_dim"]) != Z_DIM:
        raise ValueError("plan VAE artifact dimensions do not match this module")
    model = PlanVAE()
    model.load_state_dict(artifact["state_dict"], strict=True)
    model = model.to(device).eval().requires_grad_(False)
    meta = {k: v for k, v in artifact.items() if k != "state_dict"}
    return model, meta
