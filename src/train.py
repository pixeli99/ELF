#!/usr/bin/env python
"""Training script for the ELF."""

import argparse
import copy
import hashlib
import logging
import os
import json
import subprocess
import sys
import time
import contextlib

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import AutoTokenizer

from modules.t5_encoder import get_encoder
from modules.plan_vae import SPAN as PLAN_VAE_SPAN, load_frozen_plan_vae
from modules.thinking_resampler import (
    build_adjacent_mlp_encoder, freeze_module, ThinkingMLPConfig, ThinkingMLPEncoder,
)
from utils.logging_utils import log_for_0
from utils.checkpoint_utils import (
    save_checkpoint, load_checkpoint, find_latest_checkpoint,
    load_model_params_from_checkpoint,
)
from utils.train_utils import (
    TrainState, prefetch_to_device, get_optimizer, create_learning_rate_fn,
    attach_lr_scheduler,
)
from generation import run_generation
from configs.config import load_config_from_yaml, apply_config_overrides, load_sampling_configs, SamplingConfig
from modules.model import ELF_models
from utils.data_utils import (
    get_dataloader, get_thinking_dataloader, prepare_batch, load_dataset,
    load_thinking_jsonl_splits, get_pad_token_id,
    FormalStageBPairedDataset, FormalStageBCollator, FormalStageBScheduleDataset,
)
from utils.encoder_utils import encode_text, encode_thinking_x0
from utils.dolma_data import DolmaStreamDataset, get_dolma_dataloader
from utils.conditional_data import (ConditionalPairedDataset, ConditionalSchedule,
                                    get_conditional_dataloader)
from utils.plan_stream import assert_group_protocol, compress_thinking_to_slots, resolve_group
from utils.sampling_utils import frozen_pool_plan_target
from utils.plan_utils import build_thinking_plan_target
from train_step import train_step

try:
    import wandb
except ImportError:
    wandb = None

# Logging: no timestamps; suppress noisy checkpoint loggers; unbuffered stdout
logging.basicConfig(
    format="%(levelname)s - %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO, force=True,
)
logger = logging.getLogger(__name__)
sys.stdout.reconfigure(line_buffering=True)


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gpu_memory_snapshot():
    try:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        physical_gpu = visible[local_rank] if local_rank < len(visible) else visible[0]
        line = subprocess.check_output([
            "nvidia-smi", "-i", physical_gpu, "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ], text=True).strip().splitlines()[0]
        processes = subprocess.check_output([
            "nvidia-smi", "-i", physical_gpu, "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ], text=True).strip().splitlines()
        return int(line), processes
    except Exception:
        return None, []


def _init_distributed():
    """Initialize torch.distributed if launched via torchrun."""
    if "WORLD_SIZE" in os.environ and not dist.is_initialized():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            dist.init_process_group(backend="nccl")
        else:
            dist.init_process_group(backend="gloo")


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def _sample_seed(base_seed: int, sample_ids, epoch: int) -> int:
    """Stable per-sample RNG seed, independent of rank/world size (formal B=1 DDP)."""
    if len(sample_ids) != 1:
        raise ValueError("deterministic formal Stage-B DDP requires micro_batch_per_gpu=1")
    payload = f"formal-stage-b-v1\0{base_seed}\0{epoch}\0{sample_ids[0]}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _is_eval_epoch(config, current_epoch: int) -> bool:
    return config.eval_freq >= 1 and current_epoch % config.eval_freq == 0


def uses_paired_thinking(config) -> bool:
    """Stage-B sources whose data is paired thinking/response documents."""
    return getattr(config, "plan_source", "frozen_pool") in ("thinking_mlp_4to1", "span_vae")


def should_run_validation(config, current_epoch: int) -> bool:
    """Stage-B uses eval cadence for no-grad validation loss, not generation."""
    return (
        _is_eval_epoch(config, current_epoch)
        and uses_paired_thinking(config)
        and not getattr(config, "formal_stage_b_manifest", None)
        and not getattr(config, "conditional_train_manifest", None)
    )


def should_run_generation(config, current_epoch: int) -> bool:
    """Generation is opt-in and remains the legacy frozen-pool evaluation path."""
    return (
        _is_eval_epoch(config, current_epoch)
        and bool(getattr(config, "online_eval", False))
        and not uses_paired_thinking(config)
    )


@torch.no_grad()
def evaluate_validation_loss(
    state, encoder, plan_encoder, dataloader, config, generator,
    replicated_across_ranks: bool = False,
):
    """Evaluate the training objective without backward, optimizer, EMA, or step changes."""
    was_training = state.model.training
    eval_config = copy.copy(config)
    eval_config.label_drop_prob = 0.0
    device = next(state.model.parameters()).device
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    count = torch.zeros((), dtype=torch.float64, device=totals.device)
    eval_state = copy.copy(state)
    eval_state.dropout_generator = generator
    fork_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=fork_devices):
            torch.manual_seed(generator.initial_seed())
            if device.type == "cuda":
                torch.cuda.manual_seed_all(generator.initial_seed())
            for batch in dataloader:
                batch = prepare_batch(batch, eval_config, generator=generator)
                batch_size = int(batch["input_ids"].shape[0])
                _, metrics = train_step(
                    eval_state, encoder=encoder, batch=batch, config=eval_config,
                    plan_encoder=plan_encoder, update_model=False,
                )
                totals += torch.stack([
                    metrics["loss"], metrics["l2_loss"], metrics["ce_loss"],
                    metrics["plan_l2_loss"],
                ]).double() * batch_size
                count += batch_size
    finally:
        state.model.train(was_training)
    if dist.is_initialized():
        dist.all_reduce(totals)
        dist.all_reduce(count)
        if replicated_across_ranks:
            totals /= dist.get_world_size()
            count /= dist.get_world_size()
    means = totals / count.clamp_min(1.0)
    return {
        "validation_loss": float(means[0]),
        "validation_l2_loss": float(means[1]),
        "validation_ce_loss": float(means[2]),
        "validation_plan_l2_loss": float(means[3]),
        "validation_documents": int(count.item()),
    }


@torch.no_grad()
def _fit_plan_whitener(model, encoder, train_dataset, config, device, pad_token_id, world):
    """Fit the frozen plan-target whitener on a few batches before training starts.

    Accumulates per-dim mean and variance (zscore) or the full covariance (pca) of the RAW
    mean-pooled plan target over valid slots, all_reduces across DDP ranks, and installs the
    stats as persistent model buffers (`ELF.set_plan_whitener`). Checkpoints carry the
    buffers, so resumed runs skip this pass (guarded by `plan_whiten_ready`).

    Motivation (measured on real t5-small latents): the raw pooled target has std ~0.47
    (info-bearing centered part ~0.28) vs ~0.84 for tokens under the shared noise scale —
    without whitening the plan resolves LATER than the tokens in SNR terms on the diagonal,
    the opposite of the design intent.
    """
    log_for_0(f"Fitting plan-target whitener ({config.plan_whiten}) "
              f"on up to {config.plan_whiten_batches} batches...")
    loader = get_dataloader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        num_workers=0, drop_last=True,
        max_seq_length=config.max_length, pad_token_id=pad_token_id,
        max_input_seq_length=config.max_input_length,
        distributed=(world > 1),
    )
    # Dedicated generator: the stats pass must not consume the training RNG stream.
    stats_gen = torch.Generator(device="cpu").manual_seed(config.seed + 20_260_702)
    C = model.text_encoder_dim
    use_pca = config.plan_whiten == "pca"
    n = torch.zeros((), dtype=torch.float64, device=device)
    s1 = torch.zeros((C,), dtype=torch.float64, device=device)
    s2 = (torch.zeros((C, C), dtype=torch.float64, device=device) if use_pca
          else torch.zeros((C,), dtype=torch.float64, device=device))
    use_bf16 = bool(getattr(config, "use_bf16", True)) and device.type == "cuda"

    it = iter(loader)
    for _ in range(config.plan_whiten_batches):
        try:
            batch = next(it)
        except StopIteration:
            break
        batch = prepare_batch(batch, config, generator=stats_gen)
        input_ids = batch["input_ids"].to(device, non_blocking=True).long()
        enc_mask = batch["encoder_attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
        cond_mask = batch["cond_seq_mask"].to(device, dtype=torch.float32, non_blocking=True)
        attn_mask = batch["attention_mask"].to(device, dtype=torch.float32, non_blocking=True)
        x0 = encode_text(
            input_ids=input_ids, attention_mask=enc_mask, encoder=encoder,
            latent_mean=config.latent_mean, latent_std=config.latent_std,
            use_bf16=use_bf16,
        ).float()
        valid = attn_mask if config.pad_token == "pad" else torch.ones_like(attn_mask)
        valid = valid * (1 - cond_mask)
        pooled, slot_mask = frozen_pool_plan_target(
            x0, valid, config.num_plan_slots, return_slot_mask=True,
        )
        flat = pooled.reshape(-1, C).double()
        m = slot_mask.reshape(-1, 1).double()
        n += m.sum()
        s1 += (flat * m).sum(dim=0)
        if use_pca:
            s2 += (flat * m).T @ flat
        else:
            s2 += (flat.square() * m).sum(dim=0)
    del loader

    if dist.is_initialized():
        dist.all_reduce(n)
        dist.all_reduce(s1)
        dist.all_reduce(s2)
    n = torch.clamp(n, min=1.0)
    mean = s1 / n
    if use_pca:
        cov = s2 / n - torch.outer(mean, mean)
        cov = 0.5 * (cov + cov.T)
        evals, evecs = torch.linalg.eigh(cov)  # ascending
        k = model.plan_latent_dim
        top = torch.argsort(evals, descending=True)[:k]
        lam = torch.clamp(evals[top], min=1e-6)
        proj = evecs[:, top] / lam.sqrt()[None, :]  # y = (x - mean) @ proj -> per-dim unit variance
        model.set_plan_whitener(mean.float(), proj=proj.float())
        total_var = torch.clamp(evals.clamp(min=0).sum(), min=1e-12)
        log_for_0(f"Plan whitener (pca): {int(n.item())} slots, {k} components, "
                  f"{float(evals[top].sum() / total_var):.1%} variance explained")
    else:
        var = torch.clamp(s2 / n - mean.square(), min=1e-6)
        model.set_plan_whitener(mean.float(), std=var.sqrt().float())
        log_for_0(f"Plan whitener (zscore): {int(n.item())} slots, "
                  f"raw pooled std ~ {float(var.sqrt().mean()):.3f}")


@torch.no_grad()
def _fit_thinking_plan_whitener(
    model, encoder, plan_encoder, train_dataset, tokenizer, config, device, world,
):
    """Fit whitening moments from valid compressed-thinking slots only."""
    loader = get_thinking_dataloader(
        train_dataset, tokenizer, batch_size=config.batch_size,
        max_length=config.max_length, shuffle=True, num_workers=0,
        drop_last=False, distributed=(world > 1),
        plan_add_special_tokens=config.thinking_plan_add_special_tokens,
    )
    width = model.text_encoder_dim
    n = torch.zeros((), dtype=torch.float64, device=device)
    s1 = torch.zeros(width, dtype=torch.float64, device=device)
    s2 = torch.zeros(width, dtype=torch.float64, device=device)
    for batch_index, batch in enumerate(loader):
        if batch_index >= config.plan_whiten_batches:
            break
        raw_plan, plan_mask = compress_thinking_to_slots(
            batch["plan_input_ids"].to(device).long(),
            batch["plan_attention_mask"].to(device).bool(),
            encoder, plan_encoder, config,
        )
        valid = raw_plan[plan_mask].double()
        n += valid.shape[0]
        s1 += valid.sum(dim=0)
        s2 += valid.square().sum(dim=0)
    if dist.is_initialized():
        dist.all_reduce(n)
        dist.all_reduce(s1)
        dist.all_reduce(s2)
    n = n.clamp_min(1.0)
    mean = s1 / n
    variance = (s2 / n - mean.square()).clamp_min(1e-6)
    model.set_plan_whitener(mean.float(), std=variance.sqrt().float())
    log_for_0(
        f"Thinking plan whitener: {int(n.item())} valid slots, "
        f"raw std ~ {float(variance.sqrt().mean()):.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train ELF Diffusion Model (PyTorch).")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file to override defaults.")
    parser.add_argument(
        "--config_override", action="append", default=[],
        help="Override config values (field_name=value). Can be specified multiple times.",
    )
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU even when CUDA is available.")
    return parser.parse_args()


def run_training(config, *, force_cpu: bool = False):
    _init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cpu") if force_cpu or not torch.cuda.is_available() else torch.device(f"cuda:{local_rank}")
    rank = _rank()
    world = _world_size()

    log_for_0("=" * 60)
    log_for_0("ELF Diffusion Model Training (PyTorch)")
    log_for_0("=" * 60)
    log_for_0(f"Model: {config.model}")
    log_for_0(f"Encoder Model: {config.encoder_model_name}")
    log_for_0(f"Encoder Checkpoint: {config.encoder_checkpoint}")
    log_for_0(f"Data: {config.data_path}")
    log_for_0(f"Max sequence length: {config.max_length}")
    log_for_0(f"Output dir: {config.output_dir}")
    log_for_0(f"HF Repo ID: {config.hf_repo_id}")
    log_for_0(f"Batch size per device: {config.batch_size}")
    log_for_0(f"Number of epochs: {config.epochs}")
    log_for_0(f"PyTorch device: {device}, world_size={world}")
    log_for_0(f"BF16 autocast: {bool(getattr(config, 'use_bf16', True)) and device.type == 'cuda'}")
    log_for_0(f"Gradient checkpointing: {bool(getattr(config, 'gradient_checkpointing', True))}")
    log_for_0("=" * 60)

    if config.resume and config.init_from:
        raise ValueError("Config cannot set both resume and init_from: resume restores training state; "
                         "init_from only warm-starts model weights.")
    if bool(getattr(config, "diagnostic_run", False)):
        group = resolve_group(config)
        log_for_0(f"DIAGNOSTIC run: formal group protocol not asserted (group_mode={group.mode})")
    else:
        group = assert_group_protocol(config)
    group_mode = group.mode

    if config.use_wandb and rank == 0 and wandb is not None:
        wandb_config = {k: getattr(config, k) for k in dir(config) if not k.startswith("_")}
        wandb_tags = config.wandb_tag.split(",") if config.wandb_tag else None
        wandb.init(
            project=config.wandb_project, entity=config.wandb_entity,
            name=config.wandb_run_name, id=config.wandb_run_name, resume=config.wandb_resume,
            tags=wandb_tags, config=wandb_config, dir="/tmp",
        )
        resume_suffix = f" (resume={config.wandb_resume}, id={config.wandb_run_name})"
        log_for_0(f"Wandb initialized: {wandb.run.url}{resume_suffix}")

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    # Per-rank seed so stochastic draws (decoder/denoiser branch coin,
    # timesteps, noise) diverge across ranks. A shared seed would make every
    # rank take the same branch in lockstep, producing spiky decoder gradients
    # instead of an evenly-mixed CE/L2 reduction.
    g = torch.Generator(device="cpu").manual_seed(config.seed + rank)

    # TF32 for fp32 matmuls on Ampere/Hopper (no hyperparameter change).
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log_for_0("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name or config.encoder_model_name)
    if uses_paired_thinking(config):
        # T5 uses relative positions and accepts these observed 1122-token thinking
        # trajectories; disable the tokenizer's legacy 512-token warning only.
        tokenizer.model_max_length = sys.maxsize
    pad_token_id = get_pad_token_id(tokenizer, config.pad_token)
    log_for_0(f"Using {'EOS' if config.pad_token == 'eos' else 'PAD'} token for padding: {pad_token_id}")

    if uses_paired_thinking(config):
        if group_mode != "vanilla":
            if config.max_plan_slots is None:
                raise ValueError("paired thinking Stage-B requires max_plan_slots")
            if config.num_plan_slots != config.max_plan_slots:
                raise ValueError("num_plan_slots must equal max_plan_slots for Stage-B")
        if config.conditional_train_manifest:
            paired = ConditionalPairedDataset(
                config.conditional_train_manifest,
                expected_sha256=config.conditional_train_manifest_sha256,
                verify_shards=bool(config.conditional_verify_shards),
            )
            rows = int(config.conditional_rows or len(paired))
            train_dataset = ConditionalSchedule(
                paired, rows=rows, master_seed=int(config.conditional_master_seed),
            )
            eval_dataset = None
            log_for_0(f"Conditional Stage-B: pool={len(paired)} schedule={len(train_dataset)} "
                      f"seed={config.conditional_master_seed}")
            log_for_0(f"Schedule fingerprint: {train_dataset.fingerprint()}")
        elif config.formal_stage_b_manifest:
            if (config.formal_stage_b_manifest_sha256
                    and _sha256_file(config.formal_stage_b_manifest)
                    != config.formal_stage_b_manifest_sha256):
                raise ValueError("formal Stage-B manifest SHA256 mismatch")
            train_dataset = FormalStageBPairedDataset(
                config.formal_stage_b_manifest, tokenizer, max_length=config.max_length,
            )
            if config.formal_stage_b_schedule:
                train_dataset = FormalStageBScheduleDataset(train_dataset,
                    config.formal_stage_b_schedule,config.formal_stage_b_schedule_sha256)
            eval_dataset = None
            log_for_0(f"Formal paired Stage-B documents: train={len(train_dataset)} val=0 test=0")
        else:
            if not config.thinking_data_path:
                raise ValueError("thinking_mlp_4to1 requires thinking_data_path")
            document_splits = load_thinking_jsonl_splits(
                config.thinking_data_path, config.seed, config.thinking_split,
            )
            train_dataset, eval_dataset = document_splits["train"], document_splits["val"]
            log_for_0(
                f"Thinking documents: train={len(train_dataset)} val={len(eval_dataset)} "
                f"test={len(document_splits['test'])}"
            )
    elif config.dolma_data_dir:
        train_dataset = DolmaStreamDataset(
            config.dolma_data_dir, tokenizer, max_length=config.max_length,
            min_tokens=config.dolma_min_tokens,
            samples_per_epoch=config.dolma_samples_per_epoch,
            seed=config.seed, rank=rank, world=world,
        )
        eval_dataset = None
    else:
        train_dataset, eval_dataset = load_dataset(config)

    log_for_0(f"Loading Encoder config: {config.encoder_model_name}...")
    encoder_config, encoder = get_encoder(config.encoder_model_name, torch.float32)
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    log_for_0(f"Encoder d_model: {encoder_config.d_model}")

    plan_encoder = None
    if (getattr(config, "plan_source", "frozen_pool") == "span_vae"
            and group_mode in ("ordered", "diagonal")):
        if encoder_config.d_model != 512:
            raise ValueError("span_vae requires t5-small width 512")
        if not config.plan_vae_artifact:
            raise ValueError("span_vae requires plan_vae_artifact")
        plan_encoder, vae_meta = load_frozen_plan_vae(
            config.plan_vae_artifact, device=device,
            expected_sha256=config.plan_vae_artifact_sha256,
        )
        if int(vae_meta["k_max"]) != config.num_plan_slots                 or int(vae_meta["z_dim"]) != config.plan_target_dim:
            raise ValueError("plan VAE artifact does not match num_plan_slots/plan_target_dim")
        log_for_0(f"Loaded frozen Plan-VAE (beta={vae_meta['beta']}, "
                  f"K={vae_meta['k_max']}, z={vae_meta['z_dim']}) from {config.plan_vae_artifact}")
    elif (getattr(config, "plan_source", "frozen_pool") == "thinking_mlp_4to1"
            and group_mode in ("ordered", "diagonal")):
        if encoder_config.d_model != 512:
            raise ValueError("thinking_mlp_4to1 Stage-B requires t5-small width 512")
        artifact_path = config.frozen_thinking_encoder or config.thinking_resampler_checkpoint
        if not artifact_path:
            raise ValueError("thinking_mlp_4to1 requires a frozen encoder artifact")
        if config.frozen_encoder_sha256 and _sha256_file(artifact_path) != config.frozen_encoder_sha256:
            raise ValueError("frozen thinking encoder SHA256 mismatch")
        probe = torch.load(artifact_path, map_location="cpu", weights_only=True)
        is_formal_encoder = "architecture" in probe
        if is_formal_encoder:
            architecture = probe["architecture"]
            if architecture != {
                "input_dim": 2048, "hidden_dim": 6144, "output_dim": 512,
                "group_size": 4, "activation": "gelu", "dropout": 0.0,
            }:
                raise ValueError(f"unexpected formal encoder architecture: {architecture}")
            encoder_state = probe["encoder"]
            hidden_dim = 6144
            if (config.frozen_stage_a_checkpoint_sha256
                    and probe.get("source_checkpoint_sha256")
                    != config.frozen_stage_a_checkpoint_sha256):
                raise ValueError("frozen encoder source checkpoint SHA256 mismatch")
        else:
            encoder_state = probe.get("encoder")
            hidden_dim = 1024
        if encoder_state is None:
            raise ValueError("Stage-A artifact is missing encoder state")
        if is_formal_encoder:
            plan_encoder = ThinkingMLPEncoder(ThinkingMLPConfig(hidden_dim=hidden_dim))
        else:
            plan_encoder = build_adjacent_mlp_encoder(512, 4, hidden_dim)
        plan_encoder.load_state_dict(encoder_state, strict=True)
        plan_encoder = freeze_module(plan_encoder.to(device))
        log_for_0(
            f"Loaded frozen Stage-A plan encoder only from "
            f"{artifact_path}"
        )

    log_for_0(f"Creating {config.model} model...")
    # Use the full tokenizer length for CE heads; tokenizer.vocab_size can exclude
    # added special tokens that still appear in tokenized Qwen targets.
    try:
        vocab_size = len(tokenizer)
    except TypeError:
        vocab_size = tokenizer.vocab_size
    log_for_0(f"Tokenizer vocab: CE head={vocab_size}")
    model = ELF_models[config.model](
        text_encoder_dim=encoder_config.d_model, max_length=config.max_length,
        attn_drop=config.attn_dropout, proj_drop=config.proj_dropout,
        num_time_tokens=config.num_time_tokens,
        num_self_cond_cfg_tokens=config.num_self_cond_cfg_tokens,
        vocab_size=vocab_size,
        num_model_mode_tokens=config.num_model_mode_tokens,
        bottleneck_dim=config.bottleneck_dim,
        gradient_checkpointing=bool(getattr(config, "gradient_checkpointing", True)),
        num_plan_slots=config.num_plan_slots,
        num_plan_time_tokens=config.num_plan_time_tokens,
        plan_whiten=config.plan_whiten,
        plan_target_dim=config.plan_target_dim,
        plan_response_attention=getattr(config, "plan_response_attention", "bidirectional"),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log_for_0(f"ELF parameters: {total_params:,}")
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log_for_0(f"Total trainable parameters: {total_trainable:,}")

    if config.init_from:
        load_model_params_from_checkpoint(
            model, config.init_from, strict=False, prefer_ema=True,
        )
        if getattr(config, "plan_source", "frozen_pool") == "thinking_mlp_4to1" and group_mode != "vanilla":
            stage_b = config.formal_stage_b_manifest or config.conditional_train_manifest
            if stage_b and config.num_plan_slots > 0 and config.num_plan_slots != 16:
                source = torch.load(config.init_from, map_location="cpu", weights_only=False)
                source_state = source.get("ema_params1") or source.get("params")
                old_slots = source_state.get("plan_slot_embed")
                if old_slots is None or old_slots.ndim != 3 or old_slots.shape[1] != 16:
                    raise ValueError("ordered warm-start lacks the expected K=16 plan_slot_embed")
                resized = F.interpolate(
                    old_slots.float().transpose(1, 2), size=config.num_plan_slots,
                    mode="linear", align_corners=True,
                ).transpose(1, 2)
                with torch.no_grad():
                    model.plan_slot_embed.copy_(resized.to(model.plan_slot_embed))
                log_for_0(
                    f"Deterministically interpolated ordered K=16 plan_slot_embed to "
                    f"capacity K={config.num_plan_slots}; no random slot expansion."
                )
            model.plan_target_mean.zero_()
            if hasattr(model, "plan_target_std"):
                model.plan_target_std.fill_(1.0)
            model.plan_whiten_ready.zero_()
            log_for_0(
                "Reinitialized plan whitening buffers for thinking_mlp_4to1; "
                "old frozen-pool statistics are intentionally not reused."
            )

    if config.thinking_whitener_artifact and group_mode in ("ordered", "diagonal"):
        if not (config.formal_stage_b_manifest or config.conditional_train_manifest):
            raise ValueError("precomputed thinking whitener requires a Stage-B manifest")
        whitening_manifest = os.path.join(
            os.path.dirname(config.thinking_whitener_artifact), "manifest.json",
        )
        if (config.whitening_manifest_sha256
                and _sha256_file(whitening_manifest) != config.whitening_manifest_sha256):
            raise ValueError("thinking whitening manifest SHA256 mismatch")
        whitener = torch.load(
            config.thinking_whitener_artifact, map_location="cpu", weights_only=True,
        )
        if int(whitener.get("feature_dim", -1)) != 512:
            raise ValueError("thinking whitener feature_dim must be 512")
        mean, scale = whitener["mean"].float(), whitener["scale"].float()
        if mean.shape != (512,) or scale.shape != (512,) or not torch.isfinite(mean).all() \
                or not torch.isfinite(scale).all() or not bool((scale > 0).all()):
            raise ValueError("invalid precomputed thinking whitener")
        model.set_plan_whitener(mean.to(device), std=scale.to(device))
        log_for_0("Loaded frozen train-only all-row thinking whitener; stats pass disabled.")

    # Keep initialization identical across ranks, then make runtime stochastic
    # ops (e.g. dropout) rank-specific.
    torch.manual_seed(config.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed + rank)

    if config.global_batch_size is not None:
        log_for_0(f"Using global batch size: {config.global_batch_size}")
        total_batch_size = config.global_batch_size
        local_batch_size = total_batch_size // world
        config.batch_size = local_batch_size
    elif config.batch_size is not None:
        log_for_0(f"Using batch size per device: {config.batch_size}")
        total_batch_size = config.batch_size * world
        local_batch_size = config.batch_size
        config.global_batch_size = total_batch_size
    else:
        raise ValueError("Either global_batch_size or batch_size must be specified")
    if (config.formal_stage_b_manifest or config.conditional_train_manifest) and local_batch_size != 1:
        raise ValueError("schedule-seeded Stage-B requires batch_size=1 per device")

    steps_per_epoch = len(train_dataset) // total_batch_size
    num_train_steps = steps_per_epoch * config.epochs
    if config.warmup_steps >= 0:
        num_warmup_steps = config.warmup_steps
    elif config.warmup_epochs is not None:
        num_warmup_steps = int(config.warmup_epochs * steps_per_epoch)
    else:
        num_warmup_steps = 0

    # Gradient accumulation: LR schedule is parameterized in optimizer steps
    grad_accum_steps = config.grad_accum_steps
    num_optimizer_steps = num_train_steps // grad_accum_steps
    num_warmup_optimizer_steps = num_warmup_steps // grad_accum_steps
    if getattr(config, "warmup_optimizer_steps", None) is not None:
        num_warmup_optimizer_steps = int(config.warmup_optimizer_steps)
        if not 0 <= num_warmup_optimizer_steps <= num_optimizer_steps:
            raise ValueError("warmup_optimizer_steps must be within the optimizer horizon")

    # Effective learning rate (scaled with effective batch size, including grad accum)
    if config.lr is None or config.lr <= 0:
        if config.lr is not None:
            log_for_0(f"Configured lr={config.lr} is non-positive; recomputing from blr={config.blr}")
        config.lr = config.blr * (total_batch_size * grad_accum_steps) / 256

    log_for_0(
        f"World={world} | batch local={local_batch_size}, total={total_batch_size} | "
        f"steps/epoch={steps_per_epoch}, total_train={num_train_steps}, "
        f"warmup_micro={num_warmup_steps}, warmup_optimizer={num_warmup_optimizer_steps}, "
        f"lr={config.lr:.2e}"
    )
    if grad_accum_steps > 1:
        log_for_0(
            f"Grad accum={grad_accum_steps}, effective batch={total_batch_size * grad_accum_steps}, "
            f"optimizer steps={num_optimizer_steps}"
        )

    lr_fn = create_learning_rate_fn(
        num_train_steps=num_optimizer_steps, num_warmup_steps=num_warmup_optimizer_steps,
        learning_rate=config.lr, schedule=config.lr_schedule, min_lr=config.min_lr,
    )
    optimizer = get_optimizer(model, config, lr=config.lr, grad_accum_steps=grad_accum_steps)
    lr_scheduler = attach_lr_scheduler(optimizer, lr_fn)

    state = TrainState(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler,
        ema_params1=TrainState.init_ema(model),
        step=0, epoch=0, dropout_generator=g,
    )

    # Auto-resume: if no explicit resume/init_from path, check output_dir for existing checkpoints.
    if not config.resume and not config.init_from:
        auto_ckpt = find_latest_checkpoint(config.output_dir)
        if auto_ckpt:
            config.resume = config.output_dir
            log_for_0(f"Auto-resuming from {auto_ckpt}")

    start_epoch, resume_step = 0, 0
    resume_epoch_fractional = 0.0  # Fractional epoch for save-point tracking
    if config.resume:
        try:
            ckpt_path = config.resume
            if "checkpoint_" not in ckpt_path:
                ckpt_path = find_latest_checkpoint(ckpt_path) or ckpt_path
            state, resume_step = load_checkpoint(ckpt_path, state)
            resume_epoch_fractional = float(state.epoch)
            start_epoch = int(state.epoch)
            log_for_0(f"Resumed from step {resume_step} (epoch {resume_epoch_fractional:.2f})")
        except Exception as e:
            log_for_0(f"Error loading checkpoint: {e}")
            log_for_0("Continuing training from scratch")

    # Fit the frozen plan-target whitener once before wrapping (science-arm scale fix).
    # Resume restores the fitted buffers from the checkpoint (plan_whiten_ready == 1);
    # in DDP every rank runs the pass on its own shard and the moments are all_reduced,
    # so all ranks install identical stats before the DDP broadcast.
    if (config.num_plan_slots > 0 and not config.plan_register_only
            and config.plan_whiten not in ("none", "external")
            and int(state.model.plan_whiten_ready.item()) == 0):
        if getattr(config, "plan_source", "frozen_pool") == "thinking_mlp_4to1":
            if config.plan_whiten == "pca":
                raise ValueError("Stage-B single-layer probe currently supports none/zscore whitening")
            _fit_thinking_plan_whitener(
                state.model, encoder, plan_encoder, train_dataset, tokenizer,
                config, device, world,
            )
        else:
            _fit_plan_whitener(state.model, encoder, train_dataset, config, device, pad_token_id, world)

    # torch.compile before DDP so only the inner module is compiled and
    # checkpoint I/O (which uses unwrap_model -> _orig_mod) still works.
    if device.type == "cuda" and bool(getattr(config, "use_compile", True)):
        log_for_0("Compiling ELF model with torch.compile (first step will be slower)...")
        state = state.replace(model=torch.compile(state.model))

    if world > 1:
        # find_unused_parameters=False is safe: 0-mult sinks in train_step
        # (`0 * net_out.sum()` for CE, `0 * decoder_logits.sum()` for L2)
        # keep every head in the autograd graph on every step.
        ddp_find_unused = bool(getattr(config, "ddp_find_unused_parameters", False))
        log_for_0(f"DDP find_unused_parameters: {ddp_find_unused}")
        state = state.replace(model=DDP(
            state.model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=ddp_find_unused,
            gradient_as_bucket_view=True,
            broadcast_buffers=False,
        ))

    os.makedirs(config.output_dir, exist_ok=True)

    if rank == 0:
        config_dict = {
            k: ([vars(sc) for sc in v] if isinstance(v, list) and v and isinstance(v[0], SamplingConfig) else v)
            for k, v in vars(config).items()
        }
        config_path = os.path.join(config.output_dir, "config.yml")
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        log_for_0(f"Config saved to {config_path}")

    if uses_paired_thinking(config):
        if config.formal_stage_b_manifest or config.conditional_train_manifest:
            if config.conditional_train_manifest:
                train_dataloader, _ = get_conditional_dataloader(
                    train_dataset, tokenizer, config, batch_size=local_batch_size,
                    num_workers=config.num_workers, distributed=False,
                )
                validation_dataloader = None
                return_conditional = True
            else:
                return_conditional = False
            collator = FormalStageBCollator(pad_token_id, config.max_length)
            if config.formal_stage_b_schedule and world != 1:
                raise ValueError("common schedule baselines require independent single-GPU processes")
            if not return_conditional:
                sampler = (torch.utils.data.SequentialSampler(train_dataset)
                           if config.formal_stage_b_schedule else
                           torch.utils.data.distributed.DistributedSampler(
                               train_dataset, num_replicas=world, rank=rank, shuffle=True,
                               seed=config.seed, drop_last=True))
                train_dataloader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=local_batch_size, sampler=sampler,
                    num_workers=config.num_workers, drop_last=True, collate_fn=collator,
                    pin_memory=True, persistent_workers=config.num_workers > 0,
                )
                validation_dataloader = None
        else:
            train_dataloader = get_thinking_dataloader(
                train_dataset, tokenizer, batch_size=local_batch_size,
                max_length=config.max_length, shuffle=True,
                num_workers=config.num_workers, drop_last=True,
                distributed=(world > 1),
                plan_add_special_tokens=config.thinking_plan_add_special_tokens,
            )
            validation_dataloader = get_thinking_dataloader(
                eval_dataset, tokenizer,
                batch_size=min(local_batch_size, max(1, len(eval_dataset))),
                max_length=config.max_length, shuffle=False,
                num_workers=config.num_workers, drop_last=False,
                distributed=False,
                plan_add_special_tokens=config.thinking_plan_add_special_tokens,
            )
    elif config.dolma_data_dir:
        validation_dataloader = None
        train_dataloader = get_dolma_dataloader(
            train_dataset, pad_token_id, batch_size=local_batch_size,
            max_length=config.max_length, num_workers=config.num_workers,
        )
    else:
        validation_dataloader = None
        train_dataloader = get_dataloader(
            train_dataset, batch_size=local_batch_size, shuffle=True,
            num_workers=config.num_workers, drop_last=True,
            max_seq_length=config.max_length, pad_token_id=pad_token_id,
            max_input_seq_length=config.max_input_length,
            distributed=(world > 1),
        )

    log_for_0("\n" + "=" * 60)
    log_for_0("Checkpoint and Evaluation Schedule")
    log_for_0("=" * 60)
    log_for_0(
        f"Steps/epoch={steps_per_epoch}, epochs={config.epochs}, total={steps_per_epoch * config.epochs} | "
        f"save every {config.save_freq} epoch(s), eval every {config.eval_freq} epoch(s)"
    )

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)
    log_for_0(f"Sampling configs: {len(config.sampling_configs)} config(s)")

    log_for_0("\n" + "=" * 60)
    log_for_0("Starting Training")
    log_for_0("=" * 60)

    if resume_step > 0:
        global_step = resume_step
        # Skip already-processed batches within the current epoch on resume
        steps_to_skip_in_epoch = resume_step - start_epoch * steps_per_epoch
    else:
        global_step = start_epoch * steps_per_epoch
        steps_to_skip_in_epoch = 0
    state.step = global_step
    optimizer_step_count = global_step // max(grad_accum_steps, 1)

    last_log_step = global_step
    train_metrics = []
    smoke_records, smoke_sample_ids = [], []
    smoke_peak_total_mib = 0
    last_log_time = time.time()

    # Track last save point for fractional save_freq; use fractional epoch from
    # checkpoint to avoid re-saving immediately after resume.
    last_save_epoch = resume_epoch_fractional if resume_step > 0 else float(start_epoch)

    _TRUNCATION_KEYS = ("prompt_truncated", "response_truncated", "thinking_truncated")

    truncation_totals = {key: 0.0 for key in _TRUNCATION_KEYS}

    for epoch in range(start_epoch, config.epochs):
        log_for_0(f"\nEpoch {epoch + 1}/{config.epochs}")

        # Free device buffers from previous epoch before allocating new ones, to avoid
        # transient OOM at epoch boundaries.
        if epoch > start_epoch:
            del train_loader, train_iterator
            train_metrics = []
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if world > 1 and hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)

        train_iterator = iter(train_dataloader)
        train_loader = prefetch_to_device(train_iterator, size=4)

        initial_pbar = (resume_step - start_epoch * steps_per_epoch) if (epoch == start_epoch and resume_step > 0) else 0
        epoch_pbar = tqdm(
            total=steps_per_epoch, desc=f"Epoch {epoch + 1}", initial=initial_pbar,
            mininterval=1.0, disable=rank != 0,
        )

        for step_in_epoch, batch in enumerate(train_loader):
            is_first_step = step_in_epoch == 0 and epoch == start_epoch
            if is_first_step:
                log_for_0("Performing initial training step, this may take longer...")
            # Skip already-processed batches when resuming mid-epoch
            if epoch == start_epoch and step_in_epoch < steps_to_skip_in_epoch:
                continue
            if config.formal_stage_b_manifest or config.conditional_train_manifest:
                identities = list(batch.get("sample_id") or batch.get("example_id") or [])
                seed = _sample_seed(config.seed, identities, epoch)
                torch.manual_seed(seed)
                g.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
            batch = prepare_batch(batch, config, generator=g)
            next_is_optimizer_step = ((state.step + 1) % max(grad_accum_steps, 1)) == 0
            sync_ctx = (
                state.model.no_sync()
                if world > 1 and not next_is_optimizer_step and hasattr(state.model, "no_sync")
                else contextlib.nullcontext()
            )
            # DDP no_sync must cover forward as well as backward.
            with sync_ctx:
                state, metrics = train_step(
                    state, encoder=encoder, batch=batch, config=config,
                    plan_encoder=plan_encoder,
                )
            if config.engineering_smoke_report:
                total_used, gpu_processes = _gpu_memory_snapshot()
                if total_used is not None:
                    smoke_peak_total_mib = max(smoke_peak_total_mib, total_used)
                ids = list(batch.get("sample_id") or batch.get("example_id") or [])
                smoke_sample_ids.extend(ids)
                plan_token_lengths = batch["plan_attention_mask"].sum(1).cpu()
                plan_tokens_per_slot = (
                    PLAN_VAE_SPAN
                    if getattr(config, "plan_source", "frozen_pool") == "span_vae"
                    else 4
                )
                k_values = (
                    (plan_token_lengths + plan_tokens_per_slot - 1)
                    // plan_tokens_per_slot
                ).tolist()
                smoke_records.append({
                    "global_step": global_step + 1,
                    "optimizer_step": optimizer_step_count + int(bool(metrics.get("optimizer_step"))),
                    "sample_ids": ids,
                    "response_valid_tokens": int(metrics["response_valid_tokens"]),
                    "plan_valid_slots": int(metrics["plan_valid_slots"]),
                    "plan_capacity_slots": int(metrics["plan_capacity_slots"]),
                    "K": k_values,
                    "loss": float(metrics["loss"]), "token_l2_loss": float(metrics["l2_loss"]),
                    "decoder_ce_loss": float(metrics["ce_loss"]),
                    "plan_l2_loss": float(metrics["plan_l2_loss"]),
                    "gradient_norm": float(metrics["gradient_norm"]),
                    "learning_rate": float(state.optimizer.param_groups[0]["lr"]),
                    "gpu_total_used_mib": total_used, "gpu_processes": gpu_processes,
                    "presentation_index": batch.get("presentation_index", torch.tensor([-1])).tolist(),
                    "schedule_seeds": {key: batch[key].tolist() for key in FormalStageBScheduleDataset.SEEDS if key in batch},
                    "group_mode": group_mode,
                    "plan_present": metrics["plan_present"],
                    "plan_time_equal_token": metrics["plan_clock_on_diagonal"],
                    "plan_time_all_zero": metrics["plan_time_all_zero"],
                    "register_no_thinking_target": metrics["register_no_thinking_target"],
                    "plan_mediation_rows": int(metrics["plan_mediation_rows"]),
                    "response_noise_sha256": metrics.get("response_noise_sha256"),
                    "token_time_sha256": metrics.get("token_time_sha256"),
                    "branch_sha256": metrics.get("branch_sha256"),
                    "plan_mask_sha256": metrics.get("plan_mask_sha256"),
                    "plan_input_sha256": metrics.get("plan_input_sha256"),
                })

            # Sync only on first step to measure torch.compile time;
            # float() on the loss below already forces a device-to-host sync.
            if is_first_step:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                log_for_0("First training step (torch.compile + execution) completed...")

            global_step += 1
            train_metrics.append(metrics)
            epoch_pbar.update(1)
            if metrics.get("optimizer_step", False):
                optimizer_step_count += 1

            if (metrics.get("optimizer_step", False)
                    and config.save_optimizer_steps
                    and optimizer_step_count in set(config.save_optimizer_steps)
                    and optimizer_step_count != config.max_optimizer_steps):
                save_checkpoint(
                    state, config.output_dir, optimizer_step_count,
                    hf_repo_id=config.hf_repo_id,
                )
                log_for_0(f"Saved planned optimizer checkpoint {optimizer_step_count}")

            if (config.max_optimizer_steps is not None
                    and optimizer_step_count >= config.max_optimizer_steps):
                state.epoch = epoch + (step_in_epoch + 1) / max(steps_per_epoch, 1)
                checkpoint_label = (optimizer_step_count if config.save_optimizer_steps
                                    else global_step)
                save_checkpoint(
                    state, config.output_dir, checkpoint_label,
                    hf_repo_id=config.hf_repo_id,
                )
                if config.engineering_smoke_report:
                    gathered_records = [None] * world if rank == 0 else None
                    gathered_ids = [None] * world if rank == 0 else None
                    if dist.is_initialized():
                        dist.gather_object(smoke_records, gathered_records, dst=0)
                        dist.gather_object(smoke_sample_ids, gathered_ids, dst=0)
                    else:
                        gathered_records, gathered_ids = [smoke_records], [smoke_sample_ids]
                if config.engineering_smoke_report and rank == 0:
                    smoke_records = [row for rows in gathered_records for row in rows]
                    smoke_sample_ids = [sid for ids in gathered_ids for sid in ids]
                    response_tokens = sum(row["response_valid_tokens"] for row in smoke_records)
                    plan_slots = sum(row["plan_valid_slots"] for row in smoke_records)
                    plan_capacity = sum(row["plan_capacity_slots"] for row in smoke_records)
                    k_values = [k for row in smoke_records for k in row["K"]]
                    payload = {
                        "scope": "engineering_smoke_train_diagnostic", "optimizer_steps": optimizer_step_count,
                        "micro_steps": len(smoke_records), "rows_read": len(smoke_sample_ids),
                        "unique_sample_ids": len(set(smoke_sample_ids)),
                        "response_valid_tokens": response_tokens, "valid_plan_slots": plan_slots,
                        "plan_mediation_rows": sum(
                            row["plan_mediation_rows"] for row in smoke_records
                        ),
                        "K_min": min(k_values), "K_mean": sum(k_values) / len(k_values),
                        "K_max": max(k_values),
                        "plan_padding_ratio": 1.0 - plan_slots / max(plan_capacity, 1),
                        "token_padding_ratio": 1.0 - response_tokens / max(len(smoke_sample_ids) * config.max_length, 1),
                        "project_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                        "project_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                        "gpu_total_peak_sampled_mib": smoke_peak_total_mib,
                        "all_metrics_finite": all(
                            all(np.isfinite(row[key]) for key in (
                                "loss", "token_l2_loss", "decoder_ce_loss", "plan_l2_loss",
                                "gradient_norm", "learning_rate")) for row in smoke_records
                        ),
                        "group_mode": group_mode,
                        "sample_order_sha256": hashlib.sha256("\n".join(smoke_sample_ids).encode()).hexdigest(),
                        "global_batch_grouping_sha256": hashlib.sha256(json.dumps([smoke_sample_ids[i:i+8] for i in range(0,len(smoke_sample_ids),8)],sort_keys=True).encode()).hexdigest(),
                        "schedule_rng_sha256": hashlib.sha256(json.dumps([row["schedule_seeds"] for row in smoke_records],sort_keys=True).encode()).hexdigest(),
                        "response_noise_hash": hashlib.sha256(json.dumps([row["response_noise_sha256"] for row in smoke_records]).encode()).hexdigest(),
                        "token_time_hash": hashlib.sha256(json.dumps([row["token_time_sha256"] for row in smoke_records]).encode()).hexdigest(),
                        "branch_hash": hashlib.sha256(json.dumps([row["branch_sha256"] for row in smoke_records]).encode()).hexdigest(),
                        "k_mask_hash": hashlib.sha256(json.dumps([(row["K"],row["plan_mask_sha256"]) for row in smoke_records],sort_keys=True).encode()).hexdigest(),
                    }
                    with open(os.path.join(config.output_dir, "engineering_smoke_metrics.jsonl"), "w") as handle:
                        for row in smoke_records: handle.write(json.dumps(row, sort_keys=True) + "\n")
                    with open(os.path.join(config.output_dir, "engineering_smoke_summary.json"), "w") as handle:
                        json.dump(payload, handle, indent=2, sort_keys=True); handle.write("\n")
                log_for_0(
                    f"Reached max_optimizer_steps={config.max_optimizer_steps}, "
                    f"optimizer_step_count={optimizer_step_count}, "
                    f"global_step={global_step}, saving checkpoint and exiting."
                )
                epoch_pbar.close()
                return

            if config.max_train_steps > 0 and global_step >= config.max_train_steps:
                state.epoch = epoch + (step_in_epoch + 1) / max(steps_per_epoch, 1)
                save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
                log_for_0(
                    f"Reached max_train_steps={config.max_train_steps}; "
                    f"saved checkpoint at step {global_step} and stopping."
                )
                epoch_pbar.close()
                return

            if global_step % config.log_freq == 0:
                stacked = torch.stack([
                    torch.stack([m["loss"] for m in train_metrics]).mean(),
                    torch.stack([m["l2_loss"] for m in train_metrics]).mean(),
                    torch.stack([m["ce_loss"] for m in train_metrics]).mean(),
                    torch.stack([m["plan_l2_loss"] for m in train_metrics]).mean(),
                    torch.stack([m["gradient_norm"] for m in train_metrics]).mean(),
                    torch.stack([m["response_valid_tokens"] for m in train_metrics]).sum(),
                    torch.stack([m["plan_valid_slots"] for m in train_metrics]).sum(),
                    torch.stack([m["plan_capacity_slots"] for m in train_metrics]).sum(),
                    torch.stack([m["plan_mediation_rows"] for m in train_metrics]).sum(),
                ])
                # Average each metric across DDP ranks before logging — done
                # once per log_freq so we never sync on every train step.
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
                    stacked = stacked / dist.get_world_size()
                avg_loss, avg_l2, avg_ce, avg_plan, avg_grad, response_tokens, plan_slots, plan_capacity, mediated_rows = (
                    float(x) for x in stacked.tolist()
                )
                # Running collator truncation totals (conditional Stage-B only; zero
                # elsewhere). A capped plan target once hid here unlogged for a full run.
                for key in _TRUNCATION_KEYS:
                    truncation_totals[key] += sum(
                        float(m[key]) for m in train_metrics if key in m)
                truncation_str = "/".join(str(int(truncation_totals[k])) for k in _TRUNCATION_KEYS)
                now = time.time()
                steps_per_sec = (global_step - last_log_step) / max(now - last_log_time, 1e-8)
                current_lr = state.optimizer.param_groups[0]["lr"]

                postfix_dict = {
                    "step": f"{global_step}", "loss": f"{avg_loss:.4f}",
                    "l2": f"{avg_l2:.4f}", "ce": f"{avg_ce:.4f}",
                    "plan": f"{avg_plan:.4f}",
                    "grad": f"{avg_grad:.4f}", "response_tokens": f"{int(response_tokens)}",
                    "plan_slots": f"{int(plan_slots)}",
                    "mediated_rows": f"{int(mediated_rows)}",
                    "plan_padding": f"{1.0 - plan_slots / max(plan_capacity, 1.0):.4f}",
                    "cut_p/r/t": truncation_str,
                    "sps": f"{steps_per_sec:.1f}", "lr": f"{current_lr:.2e}",
                }
                log_for_0(postfix_dict)
                epoch_pbar.set_postfix(**postfix_dict)

                if rank == 0:
                    tqdm.write(
                        f"INFO - engine - Step {global_step}: loss={avg_loss:.4f}, "
                        f"l2={avg_l2:.4f}, ce={avg_ce:.4f}, plan={avg_plan:.4f}, "
                        f"grad={avg_grad:.4f}, response_tokens={int(response_tokens)}, "
                        f"plan_slots={int(plan_slots)}, mediated_rows={int(mediated_rows)}, plan_padding="
                        f"{1.0 - plan_slots / max(plan_capacity, 1.0):.4f}, "
                        f"truncated(prompt/response/thinking)={truncation_str}, "
                        f"lr={current_lr:.2e}, steps/sec={steps_per_sec:.2f}"
                    )
                    if config.use_wandb and wandb is not None:
                        current_epoch_progress = epoch + (step_in_epoch + 1) / steps_per_epoch
                        try:
                            wandb.log({
                                "train_loss": avg_loss, "train_l2_loss": avg_l2,
                                "train_ce_loss": avg_ce, "train_plan_l2_loss": avg_plan,
                                "lr": current_lr,
                                "epoch": current_epoch_progress, "step": global_step,
                            }, step=global_step)
                        except Exception:
                            pass

                train_metrics = []
                last_log_step = global_step
                last_log_time = now

            # Intra-epoch checkpoint saving (fractional save_freq, e.g., 0.1 epoch)
            if 0 < config.save_freq < 1:
                progress = epoch + (global_step - epoch * steps_per_epoch) / steps_per_epoch
                if progress - last_save_epoch >= config.save_freq:
                    save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
                    log_for_0(f"Saved checkpoint at epoch {progress:.2f} (step {global_step})")
                    last_save_epoch = progress

        epoch_pbar.close()
        current_epoch = epoch + 1
        state.epoch = current_epoch

        if config.save_freq >= 1 and current_epoch % config.save_freq == 0:
            save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
            log_for_0(f"Saved checkpoint at epoch {current_epoch} (step {global_step})")

        if should_run_validation(config, current_epoch):
            validation_generator = torch.Generator(device="cpu").manual_seed(
                config.seed + 8_001_000 + current_epoch
            )
            validation_metrics = evaluate_validation_loss(
                state, encoder, plan_encoder, validation_dataloader,
                config, validation_generator, replicated_across_ranks=(world > 1),
            )
            log_for_0(validation_metrics)
            if rank == 0:
                tqdm.write(
                    "INFO - validation - "
                    + ", ".join(f"{key}={value}" for key, value in validation_metrics.items())
                )
                if config.use_wandb and wandb is not None:
                    try:
                        wandb.log(validation_metrics, step=global_step)
                    except Exception:
                        pass
            last_log_step = global_step
            last_log_time = time.time()

        if should_run_generation(config, current_epoch):
            run_generation(
                state=state, encoder=encoder, eval_dataset=eval_dataset,
                tokenizer=tokenizer, config=config, generator=g,
                local_batch_size=local_batch_size,
            )
            last_log_step = global_step
            last_log_time = time.time()

    log_for_0("\n" + "=" * 60)
    log_for_0("Final Checkpoint")
    log_for_0("=" * 60)
    save_checkpoint(state, config.output_dir, global_step, hf_repo_id=config.hf_repo_id)
    log_for_0(f"Final checkpoint saved to {config.output_dir}")
    if config.use_wandb and rank == 0 and wandb is not None:
        wandb.finish()


def main():
    """CLI entry point: parse args, load config, then run training."""
    args = parse_args()
    config = load_config_from_yaml(args.config)
    if args.config_override:
        config = apply_config_overrides(config, args.config_override)
        log_for_0(f"Applied {len(args.config_override)} config override(s)")
    run_training(config, force_cpu=args.use_cpu)


if __name__ == "__main__":
    main()
