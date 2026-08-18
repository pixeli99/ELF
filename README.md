# Ordered ELF Thinking-Plan Unconditional

This branch is the code-only handoff of the formal **unconditional** Ordered ELF thinking-plan pipeline. Generation currently receives no instruction or prompt. Training data consists of paired `thinking` and `response` records, but only the response is generated.

## Formal pipeline

Stage A constructs a variable-length plan:

```text
thinking
-> frozen T5-small latent sequence
-> adjacent groups of four valid tokens (zero-pad only the final group)
-> frozen nonlinear 2048 -> 6144 -> 512 MLP encoder
-> train-only diagonal whitener
-> plan [B, K, 512], K = ceil(valid_thinking_tokens / 4)
```

Stage B jointly models the variable-K plan stream and response-token stream with separate masks and clocks `t_plan` and `t`. Response generation is unconditional.

| Group | Meaning |
|---|---|
| Ordered | Real compressed-thinking plan, variable K; conditional `t_plan` with `plan_done_frac=0.15`. |
| Diagonal | Same real plan and masks; forces `t_plan=t`. |
| Register | Same K and plan-mask compute slots, but fixed Gaussian registers; no thinking plan or plan loss. |
| Vanilla | No plan tensor, plan mask, plan clock, or plan-token region. |

## Entrypoints

- Stage-A MLP: `scripts/run_thinking_mlp_4to1.sh`
- Stage-A export: `tools/export_frozen_thinking_mlp.py`
- Train-only whitener: `scripts/run_formal_thinking_whitener.sh`
- Common schedule: `tools/build_stage_b_common_schedule.py`
- Heldout split builder: `tools/build_stage_b_heldout_manifests.py`
- Exact-K Oracle donor mapping: `tools/build_stage_b_oracle_donor_mapping.py`
- Four Stage-B groups: `scripts/run_stage_b_common80k_four_groups.sh`
- Fourteen-arm generation evaluation: `scripts/run_stage_b_common80k_generation.sh`
- Stable serial Oracle matched/shuffled probe: `scripts/run_stage_b_oracle_serial.sh`

Paths in configs are repository-relative examples. Launchers accept environment overrides such as `PYTHON_BIN`, `HF_HOME`, `OUTPUT_ROOT`, `STAGE_A_ARTIFACT_DIR`, `COMMON_SCHEDULE`, `STAGE_B_RUN_ROOT`, `STAGE_B_SPLIT_MANIFEST`, `STAGE_B_SHAPE_DIR`, and `GPT2_LARGE_SNAPSHOT`. Artifact SHA locks remain part of the formal protocol.

## External evaluation inputs

Formal generation does not run without externally built, provenance-only inputs. `tools/build_stage_b_heldout_manifests.py` creates Dev/Test and a no-gold generation-shape manifest; `tools/build_stage_b_oracle_donor_mapping.py` creates the privileged exact-K Oracle donor mapping. Their data products are deliberately not committed. The generation-shape schema contains only `eval_id`, source audit hash, K/mask lengths, response length/mask length, and deterministic noise seeds. The Oracle mapping contains recipient/donor identities, K, hashes, and source pointers needed to retrieve **thinking only**; it must not expose gold response text to generation.

## Conditional handoff

Do not add an instruction encoder from scratch before auditing the inherited conditional path. Upstream already supports `condition_input_ids`, condition masks, and a clean frozen-T5 prefix in `data_utils.py`, `train_step.py`, `generation.py`, and `generation_utils.py`. The next implementation should attach variable-K thinking plans to that path:

- instruction/prompt is available at both training and test time;
- gold thinking is used only to construct the training plan target and is never a test-time input;
- all four groups receive exactly the same instruction condition;
- Vanilla is instruction + response with no plan;
- Register is instruction + fixed Gaussian plan + response;
- Ordered is instruction + evolving plan + response;
- Diagonal is instruction + diagonal-clock plan + response.

All four groups must be retrained. WMT uses BLEU and XSum uses ROUGE. GSM8K/MATH require a separate answer parser and exact/appropriate accuracy evaluator; the unconditional GPT-2 metric path is not sufficient. Do not treat this unconditional checkpoint as conditionally trained.

No dataset, manifest instance, checkpoint, model weight, generation output, or formal result is included. See `CODE_GUIDE.md` for the file-level map and `absolute_path_audit.txt` for removed machine bindings.
