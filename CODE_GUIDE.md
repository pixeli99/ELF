# Code Guide

Each repository file covered by this handoff has one row. “Main chain” means the current common80k thinking-plan unconditional pipeline; upstream files are retained as references and are not silently promoted into that pipeline.

## Scripts

| File path | Purpose | Main input | Main output | When to run | Current main chain | Conditional change? |
|---|---|---|---|---|---|---|
| `scripts/launch.sh` | Upstream distributed train/eval launcher | upstream YAML and GPU count | upstream run | Legacy/reference runs | No | Reuse launcher mechanics |
| `scripts/eval_ppl.py` | Upstream standalone perplexity entry | generated text/checkpoint | PPL report | Legacy evaluation | No | Replace with task metric where appropriate |
| `scripts/run_thinking_mlp_4to1.sh` | Launch formal Stage-A MLP | canonical thinking manifest | MLP checkpoints | Stage A | Yes | No |
| `scripts/run_formal_thinking_whitener.sh` | Export encoder and compute all-train whitening | final MLP and train manifest | frozen encoder/whitener | After Stage A | Yes | Usually no |
| `scripts/run_stage_b_common80k_four_groups.sh` | Launch four independent common-schedule runs | four configs and schedule | four Stage-B runs | Stage-B training | Yes | Retrain four conditional groups |
| `scripts/run_stage_b_common80k_generation.sh` | Schedule/finalize fourteen generation arms | checkpoints and no-gold shapes | arm metrics/root manifest | Formal generation | Yes | Supply shared instruction condition |
| `scripts/run_stage_b_oracle_serial.sh` | Stable serial four-mode Oracle probe | Ordered checkpoint and donor map | mechanism report | Post-evaluation probe | Yes | Keep gold thinking privileged |

## Tools

| File path | Purpose | Main input | Main output | When to run | Current main chain | Conditional change? |
|---|---|---|---|---|---|---|
| `tools/train_formal_thinking_mlp.py` | Train nonlinear 4-to-1 autoencoder | thinking-only shards | Stage-A MLP | Stage A | Yes | No |
| `tools/export_frozen_thinking_mlp.py` | Export only the frozen MLP encoder | final MLP checkpoint | encoder artifact | After Stage A | Yes | No |
| `tools/compute_formal_thinking_whitener.py` | Compute FP64 train-only slot statistics | train thinking and encoder | whitener artifacts | After export | Yes | No |
| `tools/build_tpt_global_candidates.py` | Canonical exact-record/hash dependency used by heldout builder | accepted source manifests | canonical candidate metadata | Data preparation | Dependency | Extend schema carefully |
| `tools/build_stage_b_common_schedule.py` | Build deterministic presentation schedule | paired metadata | index-only 80k schedule | Before four runs | Yes | Include instruction identity |
| `tools/build_stage_b_heldout_manifests.py` | Build provenance-only Dev/Test and no-gold shapes | paired pool and train schedule | heldout indexes/shapes | Before evaluation | Yes | Include prompt-cluster gates |
| `tools/build_stage_b_oracle_donor_mapping.py` | Build exact-K unique Oracle donors | heldout indexes and shapes | donor mapping | Before Oracle probe | Yes | Keep thinking-only privilege |
| `tools/preflight_stage_b_common80k_generation.py` | Check identities and launch plan without GPU | configs/checkpoints/shapes | preflight gate | Before generation | Yes | Add instruction lock |
| `tools/eval_stage_b_common80k_generation.py` | Execute one formal generation arm | one EMA checkpoint and shapes | per-sample metrics | Worker process | Yes | Pass clean instruction prefix |
| `tools/finalize_stage_b_common80k_generation.py` | Validate and aggregate all arms | completed arm directories | root manifest/tables | After workers | Yes | Usually no |
| `tools/eval_stage_b_oracle_serial.py` | Execute serial null/self/matched/shuffled probe | Ordered EMA, thinking map | probe records | Mechanism analysis | Yes | Share instruction across modes |

## Source modules

| File path | Purpose | Main input | Main output | When used | Current main chain | Conditional change? |
|---|---|---|---|---|---|---|
| `src/configs/config.py` | Parse base/overlay training and sampling YAML | YAML | resolved config | Every entry | Yes | Add/version conditional fields only as needed |
| `src/eval.py` | Upstream evaluation orchestration | config/checkpoint/data | metrics | Evaluation | Shared | Preserve BLEU/ROUGE dispatch |
| `src/generation.py` | Upstream conditional and unconditional generation orchestration | model/data/sampling config | decoded samples | Evaluation | Shared | Reuse clean-prefix path |
| `src/plan_probes.py` | Legacy fixed-K response-derived graft/consistency probe | response latent and fixed slots | legacy probe metrics | Upstream reference only | No | Do not import into thinking-plan Oracle |
| `src/train.py` | Training entry and data/model setup | resolved config | checkpoints/logs | Training | Yes | Select conditional paired loader |
| `src/train_step.py` | Conditional prefix, flow/decoder branches and variable-K losses | token/plan batches | losses/gradients | Every update | Yes | Combine existing prefix with plan groups |
| `src/modules/layers.py` | Attention and RoPE primitives | mixed sequence/masks | hidden states | Model forward | Yes | Test instruction/plan/response visibility |
| `src/modules/model.py` | ELF model and variable-K plan interface | states/clocks/masks | flow predictions | Train/generation | Yes | Thread clean prefix without leaking thinking |
| `src/modules/t5_encoder.py` | Frozen T5 latent encoder | token IDs/masks | clean latents | Stage A/B | Shared | Reuse existing condition encoding |
| `src/modules/thinking_resampler.py` | Nonlinear adjacent-4 MLP | grouped T5 latents | plan slots/reconstruction | Stage A/plan build | Yes | No |
| `src/utils/checkpoint_utils.py` | Checkpoint resolution/save/load helpers | run paths/state | restored state | Train/eval | Yes | No |
| `src/utils/data_utils.py` | Upstream conditional plus formal paired loaders | datasets/manifests | token/mask batches | Train/eval | Yes | Preserve `condition_input_ids` and add paired thinking |
| `src/utils/encoder_utils.py` | Frozen-T5 encoding and condition-mask construction | IDs/lengths | latent/masks | Train/eval | Shared | Reuse clean-prefix masks |
| `src/utils/formal_thinking_mlp.py` | Stage-A dataset/model/grouping helpers | thinking records | MLP tensors | Stage A | Yes | No |
| `src/utils/generation_utils.py` | Solver-facing token/plan generation and decode | states/clocks/conditions | generated latent/IDs | Evaluation | Yes | Carry identical condition across groups |
| `src/utils/logging_utils.py` | Rank-aware logging | messages/rank | logs | Runtime | Shared | No |
| `src/utils/loss_utils.py` | Masked sum/count loss reductions | targets/predictions/masks | losses | Training | Yes | Preserve denominators |
| `src/utils/metrics_utils.py` | BLEU, ROUGE, GPT-2 NLL/Gen-PPL and token-frequency entropy | hypotheses/references | metrics | Evaluation | Shared | Add answer accuracy outside this module |
| `src/utils/muon_utils.py` | Muon/Adam optimizer support | model parameters | optimizer state | Training | Yes | No |
| `src/utils/plan_utils.py` | Thinking-plan grouping, whitening, clocks and masks | latents/masks | plan targets | Stage A/B | Yes | Keep gold thinking train-only |
| `src/utils/sampling_utils.py` | ODE/SDE steps and clean-condition restoration | states/time grid/condition | updated states | Generation | Yes | Reuse existing conditional restoration |
| `src/utils/stage_b_common80k_generation.py` | Condition matrix, deterministic noise and truth gates | shape metadata | arm protocol | Formal generation | Yes | Add instruction identity gate |
| `src/utils/stage_b_eval_runtime.py` | Shared current model/EMA and thinking-plan stack loader | config/checkpoint/artifacts | frozen runtime | Formal evaluators | Yes | Shared helper; no sampler rewrite |
| `src/utils/stage_b_oracle_content_probe.py` | Provenance, exact-K donor and Oracle input helpers | heldout indexes | validated mapping inputs | Serial Oracle | Yes | Shared helper only; no legacy consistency |
| `src/utils/stage_b_oracle_serial.py` | Four-mode serial protocol and paired summaries | four rollouts | probe validation | Serial Oracle | Yes | Share instruction across modes |
| `src/utils/thinking_tokenization.py` | T5 tokenizer/EOS contract | thinking/response text | IDs/masks | Data gates | Yes | Add instruction contract separately |
| `src/utils/tpt_million_data.py` | Canonical normalization, hashes and source adapters | source records | canonical metadata | Data preparation | Dependency | Extend schema without changing hashes |
| `src/utils/train_utils.py` | schedules/distributed/EMA training helpers | config/model | runtime utilities | Training | Yes | No |

## Configurations

| File path | Purpose | Main input | Main output | When used | Current main chain | Conditional change? |
|---|---|---|---|---|---|---|
| `src/configs/sampling_configs/cond_sampling_configs.yml` | Upstream conditional sampling | conditional checkpoint | WMT/XSum samples | Upstream reference | No | Starting reference |
| `src/configs/sampling_configs/register_sampling_configs.yml` | Upstream register sampling | register checkpoint | register samples | Upstream reference | No | Runtime-K adaptation already elsewhere |
| `src/configs/sampling_configs/uncond_sampling_configs.yml` | Upstream vanilla sampling | unconditional checkpoint | samples | Upstream reference | No | Reference only |
| `src/configs/sampling_configs/ordered_sampling_configs.yml` | Ordered alpha/NFE trajectories | Ordered checkpoint | trajectory arms | Formal protocol | Yes | Add identical instruction lock |
| `src/configs/sampling_configs/stage_b_common80k_generation_eval_v1.yml` | Formal fourteen-arm lock | four checkpoints/shapes | resolved evaluation | Formal generation | Yes | Version conditional evaluation |
| `src/configs/sampling_configs/stage_b_common80k_oracle_serial_v1.yml` | Serial four-mode lock | Ordered checkpoint/map | resolved probe | Oracle probe | Yes | Version conditional probe |
| `src/configs/training_configs/formal_thinking_mlp_4to1.yml` | Stage-A architecture/training lock | thinking view | MLP run | Stage A | Yes | No |
| `src/configs/training_configs/train_tpt_ELF-B_formal_mlp_variable_k_10k.yml` | Shared common80k Stage-B base | paired data/artifacts | resolved base | Four groups | Yes | Add conditional paired path |
| `src/configs/training_configs/train_stage_b_common80k_ordered_10k_v1.yml` | Ordered overlay | shared base | Ordered config | Stage B | Yes | Retrain with instruction |
| `src/configs/training_configs/train_stage_b_common80k_diagonal_10k_v1.yml` | Diagonal overlay | shared base | Diagonal config | Stage B | Yes | Retrain with instruction |
| `src/configs/training_configs/train_stage_b_common80k_register_10k_v1.yml` | Register overlay | shared base | Register config | Stage B | Yes | Retrain with instruction |
| `src/configs/training_configs/train_stage_b_common80k_vanilla_10k_v1.yml` | Vanilla overlay | shared base | Vanilla config | Stage B | Yes | Retrain with instruction |
| `src/configs/training_configs/train_de-en_ELF-B.yml` | Upstream WMT conditional training | source/target text | conditional checkpoint | Upstream reference | No | BLEU reference |
| `src/configs/training_configs/train_xsum_ELF-B.yml` | Upstream XSum conditional training | document/summary | conditional checkpoint | Upstream reference | No | ROUGE reference |
| `src/configs/training_configs/train_owt_ELF-B.yml` | Upstream vanilla OWT base | OWT latents | ELF-B checkpoint | Upstream reference | No | Initialization reference |
| `src/configs/training_configs/train_owt_ELF-B_ordered.yml` | Upstream fixed-K Ordered overlay | OWT latents | legacy Ordered run | Upstream reference | No | Do not use as common80k config |
| `src/configs/training_configs/train_owt_ELF-B_register.yml` | Upstream fixed-K Register overlay | OWT latents | legacy Register run | Upstream reference | No | Do not use as common80k config |
| `src/configs/training_configs/train_owt_ELF-L.yml` | Upstream large OWT model | OWT latents | ELF-L run | Upstream reference | No | Architecture reference |
| `src/configs/training_configs/train_owt_ELF-M.yml` | Upstream medium OWT model | OWT latents | ELF-M run | Upstream reference | No | Architecture reference |

## Tests

| File path | Purpose | Main input | Main output | When run | Current main chain | Conditional change? |
|---|---|---|---|---|---|---|
| `tests/test_causal_plan_attention.py` | Plan/response attention and padding checks | tiny ELF | assertions | CPU CI | Yes | Extend instruction regions |
| `tests/test_conditional_handoff.py` | Upstream conditional-path preservation and four-group contract | tiny conditional batches/configs | assertions | CPU CI | Shared | Update with conditional implementation |
| `tests/test_formal_stage_b_variable_k.py` | Variable-K masks/loss/freeze behavior | synthetic K batches | assertions | CPU CI | Yes | Add shared instruction |
| `tests/test_formal_thinking_mlp.py` | Stage-A architecture/grouping/checkpoint tests | synthetic latents | assertions | CPU CI | Yes | No |
| `tests/test_formal_thinking_whitener.py` | FP64 statistics and inverse tests | synthetic slots | assertions | CPU CI | Yes | No |
| `tests/test_stage_b_common80k_generation.py` | Fourteen-arm NFE/noise/mask/resume gates | mock shapes | assertions | CPU CI | Yes | Add instruction lock |
| `tests/test_stage_b_common_schedule.py` | Deterministic schedule and overlay checks | mock metadata | assertions | CPU CI | Yes | Include instruction identity |
| `tests/test_stage_b_oracle_serial.py` | Serial order/exact-K/freeze protocol | mock mapping | assertions | CPU CI | Yes | Keep gold thinking privileged |
| `tests/test_stage_b_thinking_plan.py` | Thinking target, whitening and runtime plan masks | synthetic thinking | assertions | CPU CI | Yes | No |

## Upstream conditional/legacy reference

The following files are retained unchanged to explain the inherited PyTorch implementation, not because common80k used them. WMT/XSum and `cond_sampling_configs.yml` demonstrate the existing conditional clean-prefix path. `src/plan_probes.py` is a legacy fixed-K, response-derived-target probe; the current thinking-plan Oracle must not import or call it.

| File | Reference role |
|---|---|
| `scripts/launch.sh` | Upstream train/eval launch mechanics. |
| `scripts/eval_ppl.py` | Upstream standalone PPL evaluation. |
| `src/configs/sampling_configs/cond_sampling_configs.yml` | Conditional WMT/XSum sampling reference. |
| `src/configs/sampling_configs/register_sampling_configs.yml` | Legacy fixed-slot register reference. |
| `src/configs/sampling_configs/uncond_sampling_configs.yml` | Legacy unconditional sampling reference. |
| `src/configs/training_configs/train_de-en_ELF-B.yml` | WMT conditional prefix and BLEU configuration. |
| `src/configs/training_configs/train_xsum_ELF-B.yml` | XSum conditional prefix and ROUGE configuration. |
| `src/configs/training_configs/train_owt_ELF-B.yml` | Upstream unconditional ELF-B base. |
| `src/configs/training_configs/train_owt_ELF-B_ordered.yml` | Legacy fixed-K Ordered training reference. |
| `src/configs/training_configs/train_owt_ELF-B_register.yml` | Legacy fixed-K Register training reference. |
| `src/configs/training_configs/train_owt_ELF-L.yml` | Upstream large architecture reference. |
| `src/configs/training_configs/train_owt_ELF-M.yml` | Upstream medium architecture reference. |
| `src/plan_probes.py` | Legacy response-derived fixed-K probe; forbidden dependency for current Oracle. |

## Conditional modification map

1. Audit and reuse `condition_input_ids`, condition masks, and the clean T5 prefix already implemented in `data_utils.py`, `encoder_utils.py`, `train_step.py`, `generation.py`, and `generation_utils.py`.
2. Attach variable-K thinking plans to that conditional path; do not invent a second instruction encoder unless the inherited path is proven insufficient.
3. Provide instruction/prompt at test time. Gold thinking constructs training plan targets only and must never enter test generation.
4. Feed the identical instruction to all groups: Vanilla instruction+response; Register instruction+fixed-noise-plan+response; Ordered instruction+evolving-plan+response; Diagonal instruction+diagonal-plan+response.
5. Update model/attention interfaces and position offsets while preserving clean-prefix and plan/response padding semantics.
6. Retrain all four groups on one fixed schedule; unconditional checkpoints are not conditional checkpoints.
7. Evaluate WMT with BLEU and XSum with ROUGE. Implement separate answer extraction and accuracy for GSM8K/MATH.
