"""The plan stream: the one place a Stage-B group's plan tensors are built.

The four Stage-B groups differ only in what this module hands back to the
training step:

    vanilla   no plan tensors at all (the model has no plan slots)
    register  K noise slots at t_plan = 0 on every row, no supervision
    ordered   the real compressed-thinking plan on an independent clock
    diagonal  the same real plan, clock pinned to the token clock

Keeping that table in one file is the point. The two bugs this module replaces
both came from the same shape of mistake: the thinking -> plan target path
existed in three copies that drifted apart (one forgot to normalize the T5
latents), and the group semantics lived in four independent YAML switches that
had to be kept mutually consistent by hand (one of them special-cased the
diagonal clock and took that group off the diagonal).

RNG note: every draw here is ordered exactly as the training step needs it, and
`schedule_seed` is called at the same points, because the formal protocol
requires the four groups to consume identical noise for the same sample.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from utils.encoder_utils import encode_thinking_x0
from utils.plan_utils import apply_plan_whitening, build_thinking_plan_target
from utils.sampling_utils import add_noise, sample_timesteps

GROUP_MODES = ("ordered", "diagonal", "register", "vanilla")
PLAN_SOURCES = ("span_vae", "thinking_mlp_4to1", "frozen_pool")
THINKING_GROUP_SIZE = 4


@dataclass(frozen=True)
class GroupSpec:
    """The structural invariants of one Stage-B group, checked against config."""

    mode: str
    plan_enabled: bool
    supervised: bool
    register_only: bool


# The flag tuple each labelled group must carry, as
# (plan_register_only, plan_done_frac, plan_diag_frac, plan_loss_weight).
GROUP_PROTOCOL = {
    "ordered": (False, 0.15, 0.0, 1.0),
    "diagonal": (False, 0.0, 1.0, 1.0),
    "register": (True, 0.0, 0.0, 0.0),
}


def resolve_group(config) -> GroupSpec:
    """What kind of plan stream this step runs. Structure only, no protocol.

    Behaviour derives from the flags, not the label, so an exploratory config can
    sweep any knob it likes. The formal four-group protocol is a separate, much
    stricter statement -- see `assert_group_protocol`, which a formal run asserts
    once at launch.
    """
    mode = getattr(config, "group_mode", "ordered")
    if mode not in GROUP_MODES:
        raise ValueError(f"invalid group_mode: {mode}")
    if mode == "vanilla" and (int(config.num_plan_slots) != 0
                              or int(config.max_plan_slots or 0) != 0):
        raise ValueError("vanilla requires num_plan_slots=max_plan_slots=0")
    # A plan-labelled config with no slots is the legacy unconditional model: the
    # label is inert and the step must draw no plan RNG at all.
    plan_enabled = int(config.num_plan_slots) > 0
    register_only = plan_enabled and bool(config.plan_register_only)
    return GroupSpec(mode=mode, plan_enabled=plan_enabled,
                     supervised=plan_enabled and not register_only,
                     register_only=register_only)


def assert_group_protocol(config) -> GroupSpec:
    """Enforce the formal Stage-B definition of the group this run claims to be.

    A config labelled `diagonal` that leaves `plan_diag_frac` at 0 is not a
    diagonal run with unusual settings, it is a mislabelled ordered run. Four
    independent switches encoding one categorical choice is how that happens, so
    the launch path checks all four together.
    """
    group = resolve_group(config)
    if group.mode in GROUP_PROTOCOL:
        actual = (bool(config.plan_register_only), float(config.plan_done_frac),
                  float(config.plan_diag_frac), float(config.plan_loss_weight))
        if actual != GROUP_PROTOCOL[group.mode]:
            raise ValueError(f"group_mode={group.mode} protocol mismatch: {actual}")
    return group


def plan_slot_lengths(plan_token_mask: torch.Tensor) -> torch.Tensor:
    """K per row: the number of adjacent-4 groups the valid thinking tokens fill."""
    token_lengths = plan_token_mask.sum(dim=1)
    return torch.div(token_lengths + THINKING_GROUP_SIZE - 1, THINKING_GROUP_SIZE,
                     rounding_mode="floor")


def max_plan_slots(config) -> int:
    return int(getattr(config, "max_plan_slots", None) or config.num_plan_slots)


@torch.no_grad()
def compress_thinking_to_slots(input_ids, token_mask, encoder, plan_encoder, config):
    """thinking token ids -> unwhitened plan slots. The single canonical path.

    Callers that need whitened slots want `build_whitened_thinking_plan`; the
    whitener stats pass is the only legitimate caller of this raw form, since it
    is what fits the statistics in the first place.
    """
    if plan_encoder is None:
        raise ValueError("thinking_mlp_4to1 requires a frozen plan_encoder")
    latents = encode_thinking_x0(
        input_ids=input_ids, attention_mask=token_mask, encoder=encoder,
        latent_mean=config.latent_mean, latent_std=config.latent_std,
        use_bf16=bool(getattr(config, "use_bf16", True)) and input_ids.is_cuda,
    )
    return build_thinking_plan_target(
        latents, token_mask, plan_encoder, max_plan_slots=max_plan_slots(config),
    )


@torch.no_grad()
def build_vae_plan_target(input_ids, token_mask, encoder, plan_vae, config):
    """thinking token ids -> Plan-VAE posterior mean, trailing NULL slots as zeros.

    The returned mask is ALL ONES on purpose: nulls are generated content, not
    padding. The Stage-B model always runs the full K_MAX slots; at inference
    the denoiser itself decides how many slots resolve to content and how many
    to the zero null code -- the plan length is generated, never an input.
    """
    if plan_vae is None:
        raise ValueError("span_vae requires a frozen plan_vae")
    latents = encode_thinking_x0(
        input_ids=input_ids, attention_mask=token_mask, encoder=encoder,
        latent_mean=config.latent_mean, latent_std=config.latent_std,
        use_bf16=bool(getattr(config, "use_bf16", True)) and input_ids.is_cuda,
    )
    mu, _, active = plan_vae.posterior(latents, token_mask.bool())
    ones = torch.ones(mu.shape[:2], dtype=torch.bool, device=mu.device)
    return mu, ones, active


@torch.no_grad()
def build_whitened_thinking_plan(input_ids, token_mask, encoder, plan_encoder, model, config):
    """thinking token ids -> the whitened plan target the model is trained against."""
    raw, plan_mask = compress_thinking_to_slots(
        input_ids, token_mask, encoder, plan_encoder, config,
    )
    return apply_plan_whitening(model, raw, plan_mask), plan_mask


@dataclass
class PlanStream:
    """What the plan stream contributes to one training step.

    `x_plan_input` / `t_plan_input` go to the main forward. `aux_x` / `aux_t` go
    to the self-conditioning forwards, which are always denoiser-shaped.
    `x0_plan` is the supervision target, or None when the group has no plan loss.
    """

    x_plan_input: Optional[torch.Tensor] = None
    t_plan_input: Optional[torch.Tensor] = None
    plan_mask: Optional[torch.Tensor] = None
    x0_plan: Optional[torch.Tensor] = None
    plan_z: Optional[torch.Tensor] = None
    plan_t: Optional[torch.Tensor] = None
    supervised: bool = False

    @property
    def aux_x(self):
        return self.plan_z if self.supervised else self.x_plan_input

    @property
    def aux_t(self):
        return self.plan_t if self.supervised else self.t_plan_input


def sample_plan_clock(config, t, schedule_seed):
    """Draw t_plan for the batch. Consumes `plan_time_seed`."""
    batch_size, dtype, device = t.shape[0], t.dtype, t.device
    schedule_seed("plan_time_seed")
    if config.plan_time_schedule == "uniform":
        # Conditional-uniform (science arm): t_plan | t ~ U[0,1], so every monotone
        # inference trajectory -- leading, diagonal, lagging -- is trained equally
        # densely and no trajectory comparison is confounded by coverage.
        plan_t = torch.rand((batch_size,), dtype=dtype, device=device)
    elif config.plan_time_schedule == "logit_normal":
        plan_t = sample_timesteps(
            batch_size, P_mean=config.denoiser_p_mean, P_std=config.denoiser_p_std,
            time_schedule="logit_normal", device=device, dtype=dtype,
        )
    else:
        raise ValueError(f"Unknown plan_time_schedule: {config.plan_time_schedule!r}")

    if config.plan_done_frac > 0:
        # Atom at t_plan = 1: saturating lead trajectories dwell there for a finite
        # fraction of their steps, so it needs finite training mass.
        done = torch.rand((batch_size,), dtype=dtype, device=device) < config.plan_done_frac
        plan_t = torch.where(done, torch.ones_like(plan_t), plan_t)
    if config.plan_diag_frac > 0:
        on_diag = torch.rand((batch_size,), dtype=dtype, device=device) < config.plan_diag_frac
        plan_t = torch.where(on_diag, t, plan_t)
    return plan_t


def _register_stream(config, batch, model, t, plan_source, schedule_seed):
    """Slots that occupy the same compute as a plan but carry no information.

    The ViT-registers control: K and the mask match what the ordered group would
    have used for this sample, the content is pure noise at t_plan = 0 on every
    row, decoder rows included, and there is no plan loss.
    """
    batch_size, dtype, device = t.shape[0], t.dtype, t.device
    if plan_source == "thinking_mlp_4to1":
        lengths = plan_slot_lengths(batch["plan_attention_mask"].to(device))
        runtime_k = int(lengths.max())
        plan_mask = torch.arange(runtime_k, device=device)[None, :] < lengths[:, None]
    else:
        runtime_k = int(config.num_plan_slots)
        plan_mask = torch.ones((batch_size, runtime_k), dtype=torch.bool, device=device)

    schedule_seed("plan_noise_seed")
    noise = torch.randn((batch_size, runtime_k, model.plan_latent_dim),
                        dtype=dtype, device=device)
    x_plan = noise * config.denoiser_noise_scale * plan_mask.unsqueeze(-1).to(dtype)
    return PlanStream(x_plan_input=x_plan, t_plan_input=torch.zeros_like(t),
                      plan_mask=plan_mask, supervised=False)


def build_plan_stream(
    *, config, group, batch, model, encoder, plan_encoder, t, decoder_step_active,
    x0, loss_mask, schedule_seed,
):
    """Build every plan tensor this step needs, for whichever group is running."""
    if not group.plan_enabled:
        return PlanStream()

    plan_source = getattr(config, "plan_source", "frozen_pool")
    if plan_source not in PLAN_SOURCES:
        raise ValueError(f"Unknown plan_source: {plan_source!r}")
    if group.register_only:
        return _register_stream(config, batch, model, t, plan_source, schedule_seed)

    device, dtype = t.device, t.dtype
    if plan_source == "span_vae":
        x0_plan, plan_mask, _ = build_vae_plan_target(
            batch["plan_input_ids"].to(device, non_blocking=True).long(),
            batch["plan_attention_mask"].to(device, non_blocking=True).bool(),
            encoder, plan_encoder, config,
        )
        x0_plan = x0_plan.to(dtype)
        if bool(batch.get("diagnostic_zero_plan_content", False)):
            x0_plan = torch.zeros_like(x0_plan)
    elif plan_source == "thinking_mlp_4to1":
        x0_plan, plan_mask = build_whitened_thinking_plan(
            batch["plan_input_ids"].to(device, non_blocking=True).long(),
            batch["plan_attention_mask"].to(device, non_blocking=True).bool(),
            encoder, plan_encoder, model, config,
        )
        x0_plan = x0_plan.to(dtype)
        # Read-only fixed-noise diagnostics may strip plan content while keeping the
        # runtime K and mask. Without this explicit key training is unchanged.
        if bool(batch.get("diagnostic_zero_plan_content", False)):
            x0_plan = torch.zeros_like(x0_plan)
    else:
        x0_plan = model.build_plan_target(x0, loss_mask)  # legacy fixed-K target
        plan_mask = torch.ones(x0_plan.shape[:2], dtype=torch.bool, device=device)

    plan_t = sample_plan_clock(config, t, schedule_seed)
    schedule_seed("plan_noise_seed")
    plan_noise = torch.randn(x0_plan.shape, dtype=dtype, device=device)
    plan_z = add_noise(x0_plan, plan_noise, plan_t, config)
    plan_z = plan_z * plan_mask.unsqueeze(-1).to(plan_z.dtype)

    # Per-row mix, identical in shape to the token stream: decoder rows get a
    # finished plan at t_plan = 1, denoiser rows get the noised plan at t_plan.
    # This is also what keeps the diagonal group on the diagonal. Decoder rows run
    # the token clock at t = 1, so their plan clock must be 1 too -- and (t=1,
    # t_plan=1) is exactly the corner every generation decode uses. Special-casing
    # the group here would train diagonal decoder rows at (t=1, t_plan~U[0,1]) and
    # leave that one arm mismatched against its own inference protocol.
    gate_b1 = decoder_step_active.view(-1, 1)
    gate_b11 = decoder_step_active.view(-1, 1, 1)
    return PlanStream(
        x_plan_input=gate_b11 * x0_plan + (1.0 - gate_b11) * plan_z,
        t_plan_input=(gate_b1.view(-1) * torch.ones_like(t)
                      + (1.0 - gate_b1.view(-1)) * plan_t),
        plan_mask=plan_mask, x0_plan=x0_plan, plan_z=plan_z, plan_t=plan_t,
        supervised=True,
    )


def plan_loss(plan_out, stream: PlanStream, decoder_step_active):
    """Plan x-prediction MSE over supervised slots on denoiser rows.

    x-space rather than the tokens' v-space on purpose: v-MSE = x-MSE / (1-t)^2
    blows up by 1/t_eps^2 on the t_plan=1 atom rows, which the science-arm clock
    visits with finite probability. And with a correctly whitened target, a value
    of 1.0 is exactly the predict-the-mean baseline, so the metric doubles as the
    kill criterion -- which only holds while the plan target really is whitened,
    i.e. while the Stage-A latents are normalized on the way in.
    """
    per_slot = (plan_out - stream.x0_plan).square().mean(dim=-1)  # (B, K)
    denoiser_rows = (1.0 - decoder_step_active.view(-1, 1)).to(per_slot.dtype)
    weights = denoiser_rows * stream.plan_mask.to(per_slot.dtype)
    return (per_slot * weights).sum() / weights.sum().clamp_min(1.0)
