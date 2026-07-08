#!/usr/bin/env bash
set -uo pipefail

cd /home/greatwall/ELF_ordered/ELF

export PATH=/mnt/niumiaohe/miniconda3/envs/elf/bin:$PATH
export CUDA_VISIBLE_DEVICES=4,5,6,7
export HF_HOME=/workspace/greatwall/ELF_repro/hf_cache
export TRANSFORMERS_CACHE=/workspace/greatwall/ELF_repro/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

CKPT=/workspace/greatwall/ELF_repro/hf_cache/hub/models--embedded-language-flows--ELF-B-owt-torch/snapshots/146f84133c1389bfd4ef47f14ec7a955da22faa7/checkpoint_95085

OUTDIR=outputs/elf_b-owt-ordered-5b-4g
LOGDIR=logs/ordered_5b_4g
LOGFILE=$LOGDIR/train_ordered_4g_5b.log

mkdir -p "$LOGDIR"

echo "============================================================"
echo "Ordered ELF 5B run, 4 GPUs"
echo "CUDA_VISIBLE_DEVICES=4,5,6,7"
echo "NGPU=4"
echo "global_batch_size=32"
echo "grad_accum_steps=16"
echo "effective_batch=512"
echo "max_length=1024"
echo "max_optimizer_steps=9537"
echo "expected_tokens = 9537 * 512 * 1024 = 5,000,134,656"
echo "init_from=$CKPT"
echo "output_dir=$OUTDIR"
echo "============================================================"

set +e
NGPU=4 bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B_ordered.yml \
  --config_override init_from=$CKPT \
  --config_override output_dir=$OUTDIR \
  --config_override global_batch_size=32 \
  --config_override grad_accum_steps=16 \
  --config_override max_optimizer_steps=9537 \
  --config_override max_length=1024 \
  --config_override gradient_checkpointing=true \
  --config_override online_eval=false \
  --config_override use_wandb=false \
  2>&1 | tee "$LOGFILE"

status=${PIPESTATUS[0]}
echo EXIT_CODE=$status >> "$LOGFILE"
exit $status
