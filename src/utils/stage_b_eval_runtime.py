"""Stable runtime shared by formal common80k generation and serial Oracle evaluation."""
import logging
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from utils.checkpoint_utils import find_latest_checkpoint

logger = logging.getLogger(__name__)


def _resolve_checkpoint(path):
    path = str(path)
    if Path(path).is_dir():
        latest = find_latest_checkpoint(path)
        if latest:
            return latest
    return path

def load_model_and_encoder(config, checkpoint_path, device, load_encoder=True):
    """Restore the current model structure and require EMA parameters for evaluation."""
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    if load_encoder:
        encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
        encoder = encoder.to(device).eval().requires_grad_(False)
    else:
        encoder_config, encoder = AutoConfig.from_pretrained(config.encoder_model_name), None
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size, num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim, gradient_checkpointing=False,
        num_plan_slots=config.num_plan_slots, num_plan_time_tokens=config.num_plan_time_tokens,
        plan_whiten=config.plan_whiten, plan_target_dim=config.plan_target_dim,
        plan_response_attention=getattr(config, "plan_response_attention", "bidirectional"),
    ).to(device).eval()
    resolved = _resolve_checkpoint(checkpoint_path)
    checkpoint = torch.load(resolved, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["params"])
    if not checkpoint.get("ema_params1"):
        raise ValueError("formal evaluation requires EMA parameters; raw fallback is forbidden")
    model.load_state_dict(checkpoint["ema_params1"], strict=False)
    return model.to(device).eval(), encoder, tokenizer, resolved
