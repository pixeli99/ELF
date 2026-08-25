import yaml
import os


class SamplingConfig:
    """Sampling configuration for generation."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        fields = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        for k in self.__class__.__annotations__:
            if k not in fields:
                fields[k] = getattr(self, k, None)
        items = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"SamplingConfig({items})"

    sampling_method: str = "ode"
    num_sampling_steps: list = [50]
    cfgs: list = [1]
    self_cond_cfg_scales: list = [1.0]
    time_schedule: str = "logit_normal"  # 'logit_normal' or 'uniform'
    sde_gamma: float = 0.0  # Per-step SDE churn fraction; 0.0 -> pure ODE. Used when sampling_method == "sde".
    # Ordered ELF two-clock sampling (only used when the model has a plan stream):
    #   "diagonal"       -> t_plan == t_tok
    #   "planning_first" -> t_plan = min(1, plan_lead_alpha * t_tok)  (plan clock leads with
    #                       alpha>1; alpha<1 gives a *lagging* plan — the falsification arm)
    #   "null"           -> t_plan == 0 forever (plan stays pure noise; register / no-plan probe)
    plan_trajectory: str = "diagonal"
    plan_lead_alpha: float = 2.0
    # Plan grid-CFG: guide the token velocity by extrapolating the plan-conditioned prediction
    # against a null forward at (x_plan = the run's initial noise plan, t_plan = 0).
    # 1.0 disables (no extra forward). Needs no dropout training: the t_plan≈0 region is
    # covered by the conditional-uniform plan clock during training.
    plan_cfg_scale: float = 1.0


# ============================================
# Configuration
# ============================================
class Config:
    # Dataset
    data_path: str = None
    eval_data_path: str = None
    max_length: int = 128
    max_input_length: int = None  # Max length for conditioning input (e.g., prompt or encoder input); None = no limit
    pad_token: str = "pad"  # "pad" or "eos" - which token to use for padding

    # Tokenizer
    tokenizer_name: str = None  # Defaults to encoder_model_name if not set

    # Encoder
    encoder_model_name: str = "t5-small"
    encoder_checkpoint: str = None
    latent_mean: float = 0.0
    latent_std: float = 1.0

    # Model architecture
    model: str = "ELF-B"
    bottleneck_dim: int = 128  # Bottleneck dimension for text projection
    num_time_tokens: int = 4  # Number of in-context time conditioning tokens
    num_self_cond_cfg_tokens: int = 4  # Number of in-context self-cond CFG tokens
    num_model_mode_tokens: int = 4  # If > 0, prepend learnable model-mode tokens that signal decoding mode
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    # Planning stream (Ordered ELF)
    # num_plan_slots = 0 -> vanilla ELF (no plan stream); all existing configs are unchanged.
    num_plan_slots: int = 0            # K planning slots (second stream); 0 disables the plan stream
    num_plan_time_tokens: int = 4      # In-context time tokens carrying the plan clock t_plan
    plan_resampler: str = "frozen_pool"  # How the plan target x0_plan is built: "frozen_pool" | "learnable"
    plan_source: str = "frozen_pool"  # "frozen_pool" | "thinking_mlp_4to1"
    plan_response_attention: str = "bidirectional"  # "bidirectional" | "causal_bottleneck"
    max_plan_slots: int = None  # Runtime cap for variable thinking slots; defaults to num_plan_slots
    thinking_data_path: str = None
    thinking_resampler_checkpoint: str = None
    thinking_split: str = "80_10_10"
    thinking_plan_add_special_tokens: bool = True
    formal_stage_b_manifest: str = None
    formal_stage_b_manifest_sha256: str = None
    formal_stage_b_schedule: str = None
    formal_stage_b_schedule_sha256: str = None
    # --- conditional Stage-B (prompt + thinking plan + response) ---
    # The prompt shares the `max_length` window with the response, so
    # condition_max_tokens is the guarantee that a response always has room:
    # with 1024 / 2048 the package's longest response (1006 tokens) always fits.
    conditional_train_manifest: str = None
    conditional_train_manifest_sha256: str = None
    conditional_rows: int = None       # schedule length; None uses the whole package
    conditional_master_seed: int = 42  # with the row count, this IS the schedule
    conditional_verify_shards: bool = True
    condition_max_tokens: int = 1024
    group_mode: str = "ordered"  # ordered | diagonal | register | vanilla
    frozen_thinking_encoder: str = None
    thinking_whitener_artifact: str = None
    frozen_stage_a_checkpoint_sha256: str = None
    frozen_encoder_sha256: str = None
    whitening_manifest_sha256: str = None
    save_optimizer_steps: list = None
    engineering_smoke_report: bool = False
    plan_loss_weight: float = 1.0      # lambda_plan: weight on the plan-stream velocity L2
    # --- plan clock training distribution (2D time grid coverage) ---
    # Science-arm default: t_plan | t ~ U[0,1] plus an atom at t_plan=1. Conditional-uniform makes
    # the training density along ANY monotone trajectory equal to f(t)*1, so diagonal /
    # planning-first / lagging comparisons are trained equally (no coverage confound). The atom is
    # needed because saturating lead trajectories DWELL at t_plan=1 for a finite fraction of steps
    # (measure zero under any continuous distribution).
    plan_time_schedule: str = "uniform"  # "uniform" (science arm) | "logit_normal" (match token clock; systems arm)
    plan_done_frac: float = 0.15       # Atom P(t_plan = 1): trains "denoise tokens given a finished plan"
    plan_diag_frac: float = 0.0        # Fraction forced onto t_plan == t_tok. Keep 0 for the science arm
                                       # (a diagonal atom would bias trajectory comparisons); >0 is a
                                       # systems knob for diagonal-specialist training (1.0 = rung 3)
    # --- frozen target-maker whitening (fit once before training, stored as model buffers) ---
    # The raw mean-pool target is badly scaled: on real t5-small latents its std is ~0.47 (info-
    # bearing centered part ~0.28) vs ~0.84 for tokens, under a shared noise scale — the plan would
    # resolve LATER than tokens in SNR terms on the diagonal. Whitening bakes the fix into the
    # frozen target maker (cf. LADD's unit-norm clamp / CCDD's SNR matching).
    plan_whiten: str = "zscore"        # "none" | "zscore" (per-dim standardize) | "pca" (whiten + project)
    plan_target_dim: int = 128         # PCA output dim per slot (only used when plan_whiten == "pca")
    plan_whiten_batches: int = 64      # Batches for the pre-training stats pass
    # --- register control arm ---
    # True register control: slots are structurally present but carry NO data-derived information
    # (input is pure noise at t_plan = 0) and get NO plan loss. Separates the "extra computation
    # tokens help" (ViT-registers) effect from the planning effect. NOTE: plan_loss_weight = 0.0
    # alone is NOT a register control — the noised pooled target still leaks in via the input.
    plan_register_only: bool = False

    # Denoiser objective
    denoiser_p_mean: float = 0.8
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 1.0
    t_eps: float = 5e-2
    time_schedule: str = "logit_normal"  # 'logit_normal' or 'uniform'

    # Decoder objective
    decoder_prob: float = 0.5  # Probability of decoder (CE) step vs denoiser (L2) step
    decoder_noise_scale: float = 1.0  # Scale of noise in logit-normal-noised latent for CE branch
    decoder_p_mean: float = 0.8  # Mean for logit-normal noise schedule in decoder objective
    decoder_p_std: float = 0.8  # Std for logit-normal noise schedule in decoder objective

    # Conditioning / CFG
    label_drop_prob: float = 0.0
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0

    # Training (optimizer + schedule)
    epochs: int = 200
    warmup_epochs: float = None
    warmup_steps: int = 5000
    warmup_optimizer_steps: int = None  # Explicit optimizer-step warmup; overrides micro-step fields.
    batch_size: int = None
    global_batch_size: int = 512
    lr: float = None
    blr: float = 5e-5
    min_lr: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 0.0
    optimizer: str = "muon"  # "adamw" or "muon"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    grad_accum_steps: int = 1  # Gradient accumulation steps (optimizer updates every K mini-batches)
    use_bf16: bool = True  # Use CUDA BF16 autocast for training/eval forward passes.
    use_compile: bool = False  # Wrap the eval/sampling model in torch.compile.
    gradient_checkpointing: bool = False  # Save activation memory by recomputing ELF blocks during backward.
    ddp_find_unused_parameters: bool = False
    ddp_replicated_optimizer: bool = False  # Full optimizer state on every rank for exact resume.
    max_train_steps: int = -1  # Debug only; if >0 stop training after this many global training steps.
    max_optimizer_steps: int = None

    # EMA
    ema_decay1: float = 0.9999

    # Sampling
    sampling_configs_path: str = None
    # Sampling configs sweep (list of SamplingConfig objects, loaded from YAML)
    sampling_configs: list = [SamplingConfig()]
    num_samples: int = 100
    # Eval-only paired trajectory ablation. When enabled, trajectories with the same
    # seed/rank/step-count/batch use the same RNG stream; trajectory identity is excluded.
    paired_trajectory_eval: bool = False
    paired_eval_base_seed: int = 42

    # PPL Evaluation
    online_eval: bool = True  # Enable PPL evaluation for generated samples
    eval_ppl_model: str = "gpt2-large"  # Model for PPL evaluation
    eval_ppl_batch_size: int = 64  # Batch size for PPL evaluation (adjusted to be divisible by device count)
    eval_ppl_max_length: int = 1024  # Max sequence length for PPL evaluation

    # Logging & Checkpointing
    log_freq: int = 100
    eval_freq: int = 10
    save_freq: float = 100  # Can be fractional (e.g., 0.1 for saving every 0.1 epoch)

    # Output
    output_dir: str = "./output_dir"
    hf_repo_id: str = None  # Optional HF repo id to mirror local outputs/checkpoints.
    resume: str = None
    init_from: str = None  # Optional model-only warm-start checkpoint/HF id; does not restore optimizer/step.

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "ELF"
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_tag: str = None
    wandb_resume: str = "allow"

    # Misc
    seed: int = 0
    num_workers: int = 8


def load_config_from_yaml(path: str) -> Config:
    """Load a YAML config and override defaults in Config."""
    config = Config()
    if not path or not os.path.isfile(path):
        return config

    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}
    base_path = cfg_dict.pop("base_config", None)
    if base_path:
        if not os.path.isabs(base_path):
            base_path = os.path.normpath(os.path.join(os.path.dirname(path), base_path))
        config = load_config_from_yaml(base_path)

    for key, value in cfg_dict.items():
        if key == "sampling_configs":
            continue  # handled below
        if hasattr(config, key):
            setattr(config, key, value)

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    return config


def apply_config_overrides(config: Config, overrides: list) -> Config:
    """Apply command-line config overrides to a Config object.

    Args:
        config: Config object to modify
        overrides: List of strings in format "field_name=value"

    Returns:
        Modified config object
    """
    if not overrides:
        return config

    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'field_name=value'")

        field_name, value_str = override.split("=", 1)
        field_name = field_name.strip()
        value_str = value_str.strip()

        if not hasattr(config, field_name):
            raise ValueError(f"Config has no field named '{field_name}'")

        original_value = getattr(config, field_name)
        original_type = type(original_value)

        # Allow setting a field back to None
        if value_str.lower() == "none":
            setattr(config, field_name, None)
            continue

        if original_value is None:
            # Use type annotation to infer the intended type
            annotated_type = config.__annotations__.get(field_name)
            if annotated_type == int:
                converted_value = int(value_str)
            elif annotated_type == float:
                converted_value = float(value_str)
            elif annotated_type == bool:
                converted_value = value_str.lower() in ("true", "1", "yes")
            else:
                converted_value = value_str
        elif original_type == bool:
            converted_value = value_str.lower() in ("true", "1", "yes")
        elif original_type == int:
            converted_value = int(value_str)
        elif original_type == float:
            converted_value = float(value_str)
        elif original_type == str:
            converted_value = value_str
        else:
            converted_value = value_str

        setattr(config, field_name, converted_value)

    return config


def load_sampling_configs(sampling_configs_path: str):
    """Return sampling configs, loading from sampling_configs_path if set."""
    with open(sampling_configs_path, "r") as f:
        entries = yaml.safe_load(f)
    return [SamplingConfig(**entry) for entry in entries]
