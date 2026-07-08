#!/usr/bin/env bash
set -uo pipefail

cd /home/greatwall/ELF_ordered/ELF

export CONDA_PREFIX=/mnt/niumiaohe/miniconda3/envs/elf
export CONDA_DEFAULT_ENV=elf
export PATH=/mnt/niumiaohe/miniconda3/envs/elf/bin:$PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/workspace/greatwall/ELF_repro/hf_cache
export TRANSFORMERS_CACHE=/workspace/greatwall/ELF_repro/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

CKPT=/workspace/greatwall/ELF_repro/hf_cache/hub/models--embedded-language-flows--ELF-B-owt-torch/snapshots/146f84133c1389bfd4ef47f14ec7a955da22faa7/checkpoint_95085

mkdir -p logs/ordered_full

echo "===== ENV CHECK ====="
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "python=$(which python)"
echo "pip=$(which pip)"
echo "torchrun=$(which torchrun)"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('n_gpu', torch.cuda.device_count())"
echo "====================="

set +e
NGPU=8 bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B_ordered.yml \
  --config_override init_from="$CKPT" \
  --config_override output_dir=outputs/elf_b-owt-ordered \
  --config_override global_batch_size=64 \
  --config_override grad_accum_steps=8 \
  --config_override gradient_checkpointing=true \
  --config_override use_wandb=false \
  2>&1 | tee logs/ordered_full/train_ordered_8g_b8.log

status=${PIPESTATUS[0]}
echo EXIT_CODE=$status >> logs/ordered_full/train_ordered_8g_b8.log
exit $status
