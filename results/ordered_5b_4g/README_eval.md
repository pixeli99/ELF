# Ordered 5B 4-GPU Trajectory Eval

Checkpoint:

`outputs/elf_b-owt-ordered-5b-4g/checkpoint_152592`

This evaluates the same ordered checkpoint under four inference-time plan trajectories and two NFE budgets, using `src/configs/sampling_configs/ordered_sampling_configs.yml`:

- `diagonal`, alpha `1`
- `planning_first`, alpha `2`
- `lagging`, alpha `0.5`
- `null`, alpha `0`
- sampling steps `8` and `32`

Interpretation:

- Expected ordering: `planning_first alpha=2 >= diagonal alpha=1 > lagging alpha=0.5 / null alpha=0`.
- The 8-step gap should be larger than the 32-step gap, because trajectory effects should matter most under coarse integration.
- If all four trajectories tie, the ordered trajectory is not producing a measurable effect.
- Ordered must later be compared against the register control. If ordered does not beat register, the gain may come from the extra 16 computation tokens rather than planning order.

Run smoke:

```bash
bash scripts/eval_ordered_5b_smoke.sh
```

Run full:

```bash
bash scripts/eval_ordered_5b_full.sh
```

Collect metrics:

```bash
/mnt/niumiaohe/miniconda3/envs/elf/bin/python tools/collect_ordered_eval_metrics.py \
  --input_dir outputs/eval_ordered_5b_4g_full \
  --output_csv results/ordered_5b_4g/eval_trajectories_summary.csv
```

Plot:

```bash
/mnt/niumiaohe/miniconda3/envs/elf/bin/python tools/plot_ordered_eval_trajectories.py \
  --input_csv results/ordered_5b_4g/eval_trajectories_summary.csv \
  --output_dir results/ordered_5b_4g/plots
```
