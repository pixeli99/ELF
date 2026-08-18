"""Stable runtime shared by formal common80k generation and serial Oracle evaluation."""
import logging
from pathlib import Path
import torch
from transformers import AutoConfig, AutoTokenizer
from modules.model import ELF_models
from modules.t5_encoder import get_encoder
from modules.thinking_resampler import ThinkingMLPConfig, ThinkingMLPEncoder, freeze_module
from utils.checkpoint_utils import find_latest_checkpoint
from utils.plan_utils import apply_plan_whitening, build_plan_response_attention_mask, build_thinking_plan_target
from utils.stage_b_oracle_content_probe import oracle_model_input, read_pointer

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

def load_thinking_plan_stack(config, device):
    """Load frozen T5 and the exported nonlinear adjacent-4 MLP encoder once."""
    _, t5 = get_encoder(config.encoder_model_name, torch.float32)
    t5 = t5.to(device).eval().requires_grad_(False)
    artifact = torch.load(config.frozen_thinking_encoder, map_location="cpu", weights_only=True)
    encoder = ThinkingMLPEncoder(ThinkingMLPConfig(hidden_dim=6144))
    encoder.load_state_dict(artifact["encoder"], strict=True)
    return t5, freeze_module(encoder.to(device))

@torch.inference_mode()
def build_clean_thinking_plan(meta, which, tokenizer, t5, encoder, model, config, device):
    """Canonical thinking -> T5 -> adjacent-4 MLP -> whitener construction."""
    prefix = which + "_"
    row = read_pointer(meta[prefix + "source_shard"], meta[prefix + "byte_offset"],
                       meta[prefix + "sample_id"])
    item = oracle_model_input(row)
    encoded = tokenizer(item["thinking"], add_special_tokens=True, truncation=False,
                        return_tensors="pt")
    ids = encoded["input_ids"].to(device)
    mask = ids.ne(tokenizer.pad_token_id)
    if ids.shape[1] > 1024:
        raise ValueError(f'oracle thinking exceeds 1024: {item["sample_id"]}')
    use_bf16 = bool(config.use_bf16) and device.type == "cuda"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
        latent = t5(input_ids=ids, attention_mask=mask, deterministic=True).float()
    raw, plan_mask = build_thinking_plan_target(
        latent, mask, encoder, max_plan_slots=config.max_plan_slots)
    plan = apply_plan_whitening(model, raw, plan_mask).float()
    expected = int(meta["recipient_K"] if which == "recipient" else meta["donor_K"])
    if int(plan_mask.sum()) != expected:
        raise ValueError("oracle plan K mismatch")
    return plan, plan_mask

def attention_truth(model, response_mask, plan_mask):
    prefix_len = model.num_time_tokens + model.num_plan_time_tokens + model.num_self_cond_cfg_tokens
    mode_len = model.num_model_mode_tokens
    mask = build_plan_response_attention_mask(
        response_mask.bool(), plan_mask.bool(), prefix_len, mode_len,
        model.num_time_tokens, model.num_plan_time_tokens, model.plan_response_attention)
    total = prefix_len + mode_len + plan_mask.shape[1] + response_mask.shape[1]
    allowed = mask[:, None, :].expand(-1, total, -1) if mask.ndim == 2 else mask
    plan_start, plan_end = prefix_len + mode_len, prefix_len + mode_len + plan_mask.shape[1]
    expected = response_mask[:, :, None] & plan_mask[:, None, :]
    invalid_keys = ~torch.cat((torch.ones_like(response_mask[:, :prefix_len + mode_len]),
                               plan_mask, response_mask), dim=1).bool()
    return {
        "response_reads_valid_plan": bool(allowed[:, plan_end:, plan_start:plan_end].masked_select(expected).all()),
        "padding_keys_invisible": not bool(allowed.masked_select(invalid_keys[:, None, :].expand_as(allowed)).any()),
        "layout": ["prefix", "mode", "plan", "response"],
        "prefix_len": prefix_len, "mode_len": mode_len,
    }
