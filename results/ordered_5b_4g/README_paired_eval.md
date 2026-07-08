# Ordered 5B Paired Trajectory Eval

This eval-only rerun is meant to make the ordered trajectory ablation stricter. It does not change training, model structure, loss, or checkpoint contents.

Checkpoint:

`outputs/elf_b-owt-ordered-5b-4g/checkpoint_152592`

Paired randomness:

- Enable with `paired_trajectory_eval=true`.
- Base seed is `paired_eval_base_seed=42`.
- For a fixed eval seed, rank, sampling step count, and batch index, all trajectories reuse the same RNG stream.
- The paired seed deliberately excludes sampling config index, trajectory name, and alpha.
- This pairs initial token noise, initial plan noise, random logit-normal time grids, and SDE noise increments.

Normal trajectory comparison:

- `diagonal`: `t_plan == t_tok`, endpoint `(1,1)`.
- `planning_first alpha=2`: `t_plan=min(1, 2*t_tok)`, endpoint `(1,1)`.
- `lagging alpha=0.5`: `t_plan=0.5*t_tok` during trajectory. The current eval design still decodes normal trajectories with a finished plan condition, `t_plan=1`. This is a design choice and should be reported as such; it is not changed here.

Null ablation/control:

- `null` is not a normal path to `(1,1)`.
- It deliberately measures no usable plan condition.
- It keeps the plan as pure noise and uses `t_plan=0` through denoising and final decode.

Run smoke:

```bash
bash scripts/eval_ordered_5b_paired_smoke.sh
```

Run full:

```bash
bash scripts/eval_ordered_5b_paired_full.sh
```

Audit full:

```bash
/mnt/niumiaohe/miniconda3/envs/elf/bin/python tools/audit_ordered_paired_eval.py
```
