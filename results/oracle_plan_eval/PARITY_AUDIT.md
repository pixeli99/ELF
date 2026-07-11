# Oracle Plan Evaluation Parity Audit

The N=512 oracle-plan results produced before this audit are not interpretable. Their null
baseline used a different sampler from the official ordered evaluation.

## Root cause

The official strict-null result (`Gen-PPL 26.9372`) loaded
`src/configs/sampling_configs/ordered_sampling_configs.yml` and used:

- SDE sampling
- 32 steps
- `sde_gamma=1.5`
- logit-normal time schedule
- CFG 1
- self-conditioning CFG 3
- strict-null plan trajectory

The initial oracle script constructed a new `SamplingConfig` with ODE sampling, a uniform
time schedule, `sde_gamma=0`, and self-conditioning CFG 1. Those are materially different
dynamics; the resulting null `Gen-PPL 231.4575` was not a parity baseline.

The script now loads the official YAML through `configs.config.load_sampling_configs`, uses
the selected trajectory's sampler settings, and only replaces the trajectory for each oracle
mode. It also uses the official rank-0 paired seed formula and logs every effective sampling,
decode, tokenizer, and PPL parameter.

## Other audited paths

- Checkpoint: both paths load checkpoint `params` first, then overlay `ema_params1` while
  retaining model buffers from `params`.
- Initial latent: both use model parameter dtype, shape `[B, max_length, d_model]`, CUDA RNG,
  and `denoiser_noise_scale` from the training config.
- Decode: both call `_dlm_decode_batch`; strict null passes `t_plan=0`, other modes pass 1.
- Text: both call `mask_after_eos`, then tokenizer decode with `skip_special_tokens=True`.
- Empty samples: both exclude empty/whitespace-only strings from generative PPL.
- PPL: both use `utils.metrics_utils.Metrics`, GPT2-large by default, retokenization,
  `eval_ppl_max_length=1024`, and the requested PPL batch size.

## Acceptance check

Run the N=64 null parity smoke from `README.md`. Do not run or interpret the larger oracle
experiment until its Gen-PPL is close to the official strict-null result, allowing for normal
N=64 sampling variance.

The implementation was syntax-checked after this fix. An attempted N=32 acceptance run on
2026-07-11 stalled during remote CUDA/model initialization before checkpoint loading and was
terminated; it produced no result file. Therefore runtime parity remains an explicit gate, not
a claimed result.
