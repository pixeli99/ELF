"""ELF transformer model."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from modules.layers import (
    Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN,
    TextRotaryEmbeddingFast, TimestepEmbedder,
    DEFAULT_KERNEL_INIT, DEFAULT_BIAS_INIT, ZERO_INIT, NORMAL_INIT_002,
    _make_linear,
)


PLAN_RESPONSE_ATTENTION_MODES = (
    "bidirectional",
    "causal_bottleneck",
    "prompt_causal_bottleneck",
)
PLAN_WHITEN_MODES = ("none", "zscore", "pca", "external")


def build_plan_response_attention_mask(
    token_valid: torch.Tensor,
    plan_valid: torch.Tensor,
    prefix_len: int,
    mode_len: int,
    num_time_tokens: int,
    num_plan_time_tokens: int,
    mode: str,
    condition_token_mask: Optional[torch.Tensor] = None,
    plan_mediation_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build padding/key permissions for the runtime [prefix, mode, plan, response] layout."""
    if mode not in PLAN_RESPONSE_ATTENTION_MODES:
        raise ValueError(f"unknown plan_response_attention={mode!r}")
    batch, response_len = token_valid.shape
    plan_len = plan_valid.shape[1]
    device = token_valid.device
    prefix_valid = torch.ones((batch, prefix_len), dtype=torch.bool, device=device)
    mode_valid = torch.ones((batch, mode_len), dtype=torch.bool, device=device)
    valid = torch.cat((prefix_valid, mode_valid, plan_valid.bool(), token_valid.bool()), dim=1)
    if plan_mediation_mask is not None:
        if tuple(plan_mediation_mask.shape) != (batch,):
            raise ValueError("plan_mediation_mask must have shape [B]")
        if plan_mediation_mask.device != device:
            raise ValueError("plan_mediation_mask must be on the same device as token_valid")
        if plan_mediation_mask.dtype != torch.bool:
            if not bool(((plan_mediation_mask == 0) | (plan_mediation_mask == 1)).all()):
                raise ValueError("plan_mediation_mask must be bool or contain only 0/1")
            plan_mediation_mask = plan_mediation_mask.bool()
        if mode != "prompt_causal_bottleneck" and bool(plan_mediation_mask.any()):
            raise ValueError(
                "plan mediation requires plan_response_attention='prompt_causal_bottleneck'"
            )
    if mode == "bidirectional":
        return valid

    total = valid.shape[1]
    allowed = valid[:, None, :].expand(batch, total, total).clone()
    plan_start = prefix_len + mode_len
    plan_end = plan_start + plan_len
    plan_time_start = num_time_tokens
    plan_time_end = min(prefix_len, plan_time_start + num_plan_time_tokens)
    if mode == "prompt_causal_bottleneck":
        if condition_token_mask is None:
            raise ValueError(
                "prompt_causal_bottleneck requires condition_token_mask with shape [B, S]"
            )
        if tuple(condition_token_mask.shape) != (batch, response_len):
            raise ValueError("condition_token_mask must have shape [B, S]")
        if condition_token_mask.device != device:
            raise ValueError("condition_token_mask must be on the same device as token_valid")
        if condition_token_mask.dtype != torch.bool:
            if not bool(((condition_token_mask == 0) | (condition_token_mask == 1)).all()):
                raise ValueError("condition_token_mask must be bool or contain only 0/1")
            condition_token_mask = condition_token_mask.bool()
        if bool((condition_token_mask & ~token_valid.bool()).any()):
            raise ValueError("condition_token_mask must be a subset of token_valid")

        # Prompt tokens and plan tokens form a closed subsystem at every layer:
        # they may read each other and the plan clock, but never response/shared
        # prefix/mode states that could already contain response information.
        side = torch.zeros((batch, total), dtype=torch.bool, device=device)
        side[:, plan_time_start:plan_time_end] = True
        side[:, plan_start:plan_end] = plan_valid.bool()
        side[:, plan_end:] = condition_token_mask
        allowed = torch.where(side[:, :, None], side[:, None, :], allowed)
        if plan_mediation_mask is not None and bool(plan_mediation_mask.any()):
            # On selected training rows, response/shared queries may still read
            # the plan, but cannot bypass it by reading prompt-token keys. Prompt
            # and plan queries retain the closed prompt-aware subsystem above.
            prompt_keys = torch.zeros((batch, total), dtype=torch.bool, device=device)
            prompt_keys[:, plan_end:] = condition_token_mask
            bypass = (
                plan_mediation_mask[:, None, None]
                & ~side[:, :, None]
                & prompt_keys[:, None, :]
            )
            allowed &= ~bypass
        # Invalid plan queries produce no attention result and are zeroed at the output head.
        allowed[:, plan_start:plan_end, :] &= plan_valid[:, :, None].bool()
        return allowed

    # Shared time/CFG/mode tokens may absorb response information in earlier layers.
    # The entire plan side (plan-time + plan slots) therefore reads only itself.
    plan_query_parts = []
    if plan_time_end > plan_time_start:
        plan_query_parts.append(torch.arange(plan_time_start, plan_time_end, device=device))
    plan_query_parts.append(torch.arange(plan_start, plan_end, device=device))
    plan_queries = torch.cat(plan_query_parts)
    allowed[:, plan_queries, :] = False
    allowed[:, plan_queries, plan_time_start:plan_time_end] = True
    allowed[:, plan_queries, plan_start:plan_end] = plan_valid[:, None, :]
    # Invalid plan queries produce no attention result and are zeroed at the output head.
    allowed[:, plan_start:plan_end, :] &= plan_valid[:, :, None]
    return allowed


class ELFBlock(nn.Module):
    """ELF Transformer block."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

    def forward(self, x: torch.Tensor, rope_fn: Optional[nn.Module] = None,
                attention_mask: Optional[torch.Tensor] = None,
                deterministic: bool = True) -> torch.Tensor:
        x_normed = self.norm1(x)
        attn_out = self.attn(x_normed, rope_fn, attention_mask=attention_mask,
                             deterministic=deterministic)
        x = x + attn_out

        x_normed = self.norm2(x)
        mlp_out = self.mlp(x_normed, deterministic=deterministic)
        x = x + mlp_out
        return x


class ELF(nn.Module):
    """Text ELF Transformer.

    Ordered ELF (optional, gated by ``num_plan_slots > 0``): in addition to the S token
    positions, K learnable *planning slots* are carried through the network as a second
    stream with its own noise clock ``t_plan``. The plan slots sit between the mode tokens
    and the token positions (layout ``[prefix, mode, plan, tokens]``), participate in full
    bidirectional attention, are stripped before the token output head, and have their own
    zero-init velocity head. When ``num_plan_slots == 0`` (or ``x_plan is None``) the model
    is byte-for-byte the original ELF.
    """

    def __init__(
        self,
        text_encoder_dim: int,
        max_length: int,
        hidden_size: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        bottleneck_dim: int = 128,
        num_time_tokens: int = 4,
        num_self_cond_cfg_tokens: int = 4,
        num_model_mode_tokens: int = 0,
        vocab_size: int = 0,
        gradient_checkpointing: bool = False,
        num_plan_slots: int = 0,
        num_plan_time_tokens: int = 4,
        plan_whiten: str = "zscore",
        plan_target_dim: int = 0,
        plan_response_attention: str = "bidirectional",
    ):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.attn_drop = attn_drop
        self.proj_drop = proj_drop
        self.bottleneck_dim = bottleneck_dim
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size
        self.gradient_checkpointing = gradient_checkpointing
        self.num_plan_slots = num_plan_slots
        self.num_plan_time_tokens = num_plan_time_tokens
        if plan_response_attention not in PLAN_RESPONSE_ATTENTION_MODES:
            raise ValueError(f"unknown plan_response_attention={plan_response_attention!r}")
        self.plan_response_attention = plan_response_attention

        # Self-conditioning input projection (only used when input is [z, x_pred]).
        self.self_cond_proj = _make_linear(2 * text_encoder_dim, text_encoder_dim, bias=True)

        # Text bottleneck projection.
        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)

        # Time / SC-CFG embedders + learned prefix tokens.
        if num_time_tokens <= 0:
            raise ValueError("num_time_tokens must be positive for prefix time conditioning")
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, num_time_tokens, hidden_size))
        NORMAL_INIT_002(self.t_emb_tokens)

        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(torch.empty(1, num_self_cond_cfg_tokens, hidden_size))
            NORMAL_INIT_002(self.self_cond_cfg_tokens)

        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(torch.empty(1, num_model_mode_tokens, hidden_size))
            NORMAL_INIT_002(self.mode_tokens)

        # Planning stream: K learnable slots + their own clock (second set of prefix time tokens).
        # The plan stream lives in its own latent space of dimension `plan_latent_dim`: equal to
        # text_encoder_dim for plan_whiten in {"none", "zscore"}, or plan_target_dim for "pca"
        # (whitened PCA projection of the pooled target — lower-dim, more abstract slots).
        self.plan_whiten = plan_whiten
        if (num_plan_slots > 0 and plan_target_dim > 0
                and plan_whiten in ("pca", "external")):
            # "external": the target maker (e.g. the frozen Plan-VAE) already
            # produces standardized latents in its own space; the model carries
            # no whitening buffers for it.
            self.plan_latent_dim = plan_target_dim
        else:
            self.plan_latent_dim = text_encoder_dim
        if num_plan_slots > 0:
            self.plan_in_proj = _make_linear(self.plan_latent_dim, hidden_size, bias=True)
            self.plan_slot_embed = nn.Parameter(torch.empty(1, num_plan_slots, hidden_size))
            NORMAL_INIT_002(self.plan_slot_embed)
            self.plan_t_embedder = TimestepEmbedder(hidden_size)
            self.plan_t_emb_tokens = nn.Parameter(torch.empty(1, num_plan_time_tokens, hidden_size))
            NORMAL_INIT_002(self.plan_t_emb_tokens)
            self.plan_norm = RMSNorm(hidden_size)
            # Zero-init velocity head (like FinalLayer): plan_output starts at 0 for stability.
            self.plan_head = _make_linear(
                hidden_size, self.plan_latent_dim, bias=True,
                kernel_init=ZERO_INIT, bias_init=ZERO_INIT,
            )
            # Frozen target-maker whitening stats, fit once before training (identity until then).
            # Persistent buffers: saved/restored with checkpoints, so resume/eval reuse the fit.
            # plan_whiten == "external" carries none: the external encoder owns its geometry.
            if plan_whiten != "external":
                self.register_buffer("plan_target_mean", torch.zeros(text_encoder_dim))
                if plan_whiten == "pca":
                    self.register_buffer("plan_target_proj",
                                         torch.zeros(text_encoder_dim, self.plan_latent_dim))
                else:
                    self.register_buffer("plan_target_std", torch.ones(text_encoder_dim))
                self.register_buffer("plan_whiten_ready", torch.zeros((), dtype=torch.uint8))

        head_dim = hidden_size // num_heads
        # Prefix + mode + plan tokens carry no rotary position; only real token positions do.
        prefix_total = num_model_mode_tokens + num_time_tokens
        if num_self_cond_cfg_tokens > 0:
            prefix_total += num_self_cond_cfg_tokens
        if num_plan_slots > 0:
            prefix_total += num_plan_slots + num_plan_time_tokens
        self.feat_rope = TextRotaryEmbeddingFast(
            dim=head_dim, pt_seq_len=max_length, num_empty_token=prefix_total,
        )

        self.blocks = nn.ModuleList()
        q1, q3 = depth // 4, depth // 4 * 3
        for i in range(depth):
            in_drop_range = q3 > i >= q1
            self.blocks.append(ELFBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if in_drop_range else 0.0,
                proj_drop=proj_drop if in_drop_range else 0.0,
            ))

        # Final flow-matching output head.
        self.final_layer = FinalLayer(hidden_size, patch_size=1, out_channels=text_encoder_dim)

        # Factored decoder unembedding: hidden -> text_encoder_dim -> vocab.
        bn = text_encoder_dim
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, bn))
        self.proj_bias = nn.Parameter(torch.empty(bn))
        self.unembed_kernel = nn.Parameter(torch.empty(bn, vocab_size))
        self.unembed_bias = nn.Parameter(torch.empty(vocab_size))
        DEFAULT_KERNEL_INIT(self.proj_kernel)
        DEFAULT_BIAS_INIT(self.proj_bias)
        DEFAULT_KERNEL_INIT(self.unembed_kernel)
        DEFAULT_BIAS_INIT(self.unembed_bias)

    @torch.no_grad()
    def set_plan_whitener(self, mean: torch.Tensor,
                          std: Optional[torch.Tensor] = None,
                          proj: Optional[torch.Tensor] = None) -> None:
        """Install frozen whitening stats for the plan target maker (fit once, then frozen)."""
        if self.plan_whiten == "external":
            raise RuntimeError(
                "plan_whiten='external': the plan target arrives pre-standardized "
                "(frozen Plan-VAE); the model owns no whitening buffers")
        self.plan_target_mean.copy_(mean.to(self.plan_target_mean))
        if self.plan_whiten == "pca":
            if proj is None:
                raise ValueError("plan_whiten='pca' requires a projection matrix")
            self.plan_target_proj.copy_(proj.to(self.plan_target_proj))
        elif std is not None:
            self.plan_target_std.copy_(std.to(self.plan_target_std))
        self.plan_whiten_ready.fill_(1)

    def build_plan_target(self, x0: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Frozen plan target: masked mean-pool of x0 into K slots, then whiten.

        x0: (N, S, C) clean token latents; valid_mask: (N, S). Returns (N, K, plan_latent_dim).
        The whitening stats are dataset constants (buffers); the plan is never un-whitened —
        it only ever conditions the model, so its latent space is free to be standardized.
        """
        from utils.plan_utils import build_plan_target
        return build_plan_target(self, x0, valid_mask)

    def build_context(self, t: torch.Tensor,
                      t_plan: Optional[torch.Tensor] = None,
                      self_cond_cfg_scale: Optional[torch.Tensor] = None) -> list:
        B = t.shape[0]
        prefix_tokens = []

        time_emb = self.t_embedder(t)  # (B, hidden)
        prefix_tokens.append(
            self.t_emb_tokens.expand(B, -1, -1) + time_emb.unsqueeze(1)
        )

        # Planning-stream clock: a second set of in-context time tokens carrying t_plan.
        if self.num_plan_slots > 0 and self.num_plan_time_tokens > 0 and t_plan is not None:
            plan_time_emb = self.plan_t_embedder(t_plan)
            prefix_tokens.append(
                self.plan_t_emb_tokens.expand(B, -1, -1) + plan_time_emb.unsqueeze(1)
            )

        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)
            prefix_tokens.append(
                self.self_cond_cfg_tokens.expand(B, -1, -1) + sc_emb.unsqueeze(1)
            )
        return prefix_tokens

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        deterministic: bool = True,
        self_cond_cfg_scale: Optional[torch.Tensor] = None,
        decoder_step_active: Optional[bool] = None,
        x_plan: Optional[torch.Tensor] = None,
        t_plan: Optional[torch.Tensor] = None,
        plan_mask: Optional[torch.Tensor] = None,
        condition_token_mask: Optional[torch.Tensor] = None,
        plan_mediation_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """x: (N, S, C) or (N, S, 2C) with self-cond. t: (N,). attention_mask: (N, S), 1=valid.

        x_plan: (N, K, plan_latent_dim) noised plan latent (no self-cond). t_plan: (N,) plan clock.
        condition_token_mask: (N, S), 1 for clean prompt positions. Required by
        prompt_causal_bottleneck so prompt/plan cannot indirectly read the response.
        plan_mediation_mask: optional (N,) training intervention. Selected rows
        force non-plan queries to receive prompt information through plan slots.
        Returns (output, decoder_logits, plan_output). plan_output is (N, K, plan_latent_dim) or None.
        """
        B = x.shape[0]
        # Plan slots are a structural part of a plan-enabled model (like mode tokens):
        # they are always present so the precomputed RoPE empty-token count stays
        # consistent across forwards. When called without explicit plan inputs (e.g.
        # plan-unaware sampling), default to a clean placeholder plan (zeros at t_plan=1);
        # real training always passes x_plan / t_plan.
        plan_enabled = self.num_plan_slots > 0
        if plan_enabled:
            if attention_mask is None:
                attention_mask = torch.ones(
                    x.shape[:2], dtype=torch.bool, device=x.device,
                )
            if x_plan is None:
                x_plan = x.new_zeros((B, self.num_plan_slots, self.plan_latent_dim))
            runtime_plan_slots = x_plan.shape[1]
            if runtime_plan_slots > self.num_plan_slots:
                raise ValueError(
                    f"runtime plan has {runtime_plan_slots} slots, exceeds "
                    f"max_plan_slots={self.num_plan_slots}"
                )
            if plan_mask is None:
                plan_mask = torch.ones(
                    (B, runtime_plan_slots), dtype=torch.bool, device=x.device,
                )
            elif tuple(plan_mask.shape) != (B, runtime_plan_slots):
                raise ValueError("plan_mask must have shape [B, K_batch]")
            if t_plan is None:
                t_plan = t.new_ones((t.shape[0],))
            response_attention_mask = attention_mask.bool()
        else:
            runtime_plan_slots = 0

        # Self-conditioning: input is [z, x_pred] when 2x encoder dim
        with torch.amp.autocast('cuda', enabled=False):
            if x.shape[-1] == 2 * self.text_encoder_dim:
                x = self.self_cond_proj(x.float())
            x = self.text_proj(x.float())
            if plan_enabled:
                plan_hidden_in = (
                    self.plan_in_proj(x_plan.float())
                    + self.plan_slot_embed[:, :runtime_plan_slots]
                )
            context_prefix_tokens = self.build_context(t, t_plan, self_cond_cfg_scale)

        # Insert plan slots before mode tokens so the final layout is [prefix, mode, plan, tokens].
        plan_offset = 0
        if plan_enabled:
            x = torch.cat([plan_hidden_in, x], dim=1)
            plan_offset = runtime_plan_slots

        # Prepend learnable model-mode tokens (gated by decoder_step_active).
        # decoder_step_active may be None / Python bool / (B,) tensor — the last
        # form supports per-example branching at training time.
        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode_tokens = self.mode_tokens.expand(B, -1, -1)
            if decoder_step_active is None:
                active_gate = 0.0
            elif isinstance(decoder_step_active, torch.Tensor) and decoder_step_active.dim() > 0:
                active_gate = decoder_step_active.to(mode_tokens.dtype).view(-1, 1, 1)
            else:
                active_gate = float(decoder_step_active)
            mode_tokens = mode_tokens * active_gate
            x = torch.cat([mode_tokens, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None and not plan_enabled:
                mode_mask = torch.ones((B, self.num_model_mode_tokens),
                                       dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        prefix_len = 0
        if context_prefix_tokens:
            prefix_tokens = torch.cat(context_prefix_tokens, dim=1)
            prefix_len = prefix_tokens.shape[1]
            x = torch.cat([prefix_tokens, x], dim=1)
            if attention_mask is not None and not plan_enabled:
                prefix_mask = torch.ones((B, prefix_len),
                                         dtype=attention_mask.dtype, device=attention_mask.device)
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        if plan_enabled:
            # Built from scratch over the full [prefix, mode, plan, response] layout,
            # so the running concatenation above is skipped rather than discarded.
            attention_mask = build_plan_response_attention_mask(
                token_valid=response_attention_mask,
                plan_valid=plan_mask,
                prefix_len=prefix_len,
                mode_len=model_mode_offset,
                num_time_tokens=self.num_time_tokens,
                num_plan_time_tokens=self.num_plan_time_tokens,
                mode=self.plan_response_attention,
                condition_token_mask=condition_token_mask,
                plan_mediation_mask=plan_mediation_mask,
            )

        use_checkpoint = self.gradient_checkpointing and self.training and torch.is_grad_enabled()
        runtime_empty_tokens = prefix_len + model_mode_offset + plan_offset
        rope_fn = lambda value: self.feat_rope(
            value, num_empty_token=runtime_empty_tokens,
        )
        for block in self.blocks:
            if use_checkpoint:
                def _block_forward(hidden: torch.Tensor, block: ELFBlock = block) -> torch.Tensor:
                    return block(hidden, rope_fn=rope_fn, attention_mask=attention_mask,
                                 deterministic=deterministic)

                x = checkpoint(_block_forward, x, use_reentrant=False)
            else:
                x = block(x, rope_fn=rope_fn, attention_mask=attention_mask,
                          deterministic=deterministic)

        # Split out plan slots (between mode tokens and token positions) and token positions.
        plan_start = prefix_len + model_mode_offset
        token_start = plan_start + plan_offset
        plan_hidden_out = x[:, plan_start:token_start] if plan_enabled else None
        x = x[:, token_start:]

        # Factored decoder unembedding + heads (fp32).
        with torch.amp.autocast('cuda', enabled=False):
            decoder_logits = None
            if decoder_step_active is not None:
                x_f32 = x.float()
                hidden = F.gelu(x_f32 @ self.proj_kernel + self.proj_bias, approximate="tanh")
                decoder_logits = hidden @ self.unembed_kernel + self.unembed_bias
            output = self.final_layer(x.float())
            plan_output = None
            if plan_enabled:
                plan_output = self.plan_head(self.plan_norm(plan_hidden_out.float()))
                plan_output = plan_output * plan_mask.to(plan_output.dtype).unsqueeze(-1)
        return output, decoder_logits, plan_output


# Model factory functions
def ELF_B(**kwargs): return ELF(depth=12, hidden_size=768,  num_heads=12, **kwargs)
def ELF_M(**kwargs): return ELF(depth=24, hidden_size=1056, num_heads=16, **kwargs)
def ELF_L(**kwargs): return ELF(depth=32, hidden_size=1280, num_heads=16, **kwargs)

ELF_models = {
    'ELF-B': ELF_B, 'ELF-M': ELF_M, 'ELF-L': ELF_L,
}
