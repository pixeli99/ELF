#!/usr/bin/env bash
set -uo pipefail

cd /home/greatwall/ELF_ordered/ELF

export PATH=/mnt/niumiaohe/miniconda3/envs/elf/bin:$PATH
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HOME=/workspace/greatwall/ELF_repro/hf_cache
export TRANSFORMERS_CACHE=/workspace/greatwall/ELF_repro/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

CKPT=outputs/elf_b-owt-ordered-5b-4g/checkpoint_152592
OUT=outputs/eval_ordered_5b_4g_paired_full_seed42_b32
LOG=logs/eval_ordered_5b_4g/eval_paired_full_seed42_b32.log

mkdir -p logs/eval_ordered_5b_4g "$OUT"

NGPU=4 bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-B_ordered.yml \
  --checkpoint_path "$CKPT" \
  --seed 42 \
  --config_override output_dir="$OUT" \
  --config_override sampling_configs_path=src/configs/sampling_configs/ordered_sampling_configs.yml \
  --config_override num_samples=1000 \
  --config_override global_batch_size=32 \
  --config_override eval_ppl_batch_size=4 \
  --config_override paired_trajectory_eval=true \
  --config_override paired_eval_base_seed=42 \
  --config_override use_wandb=false \
  2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo EXIT_CODE=$status >> "$LOG"
exit $status
