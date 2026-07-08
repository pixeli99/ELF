#!/usr/bin/env bash
set -uo pipefail

cd /home/greatwall/ELF_ordered/ELF

RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/ordered_full/debug_${RUN_ID}"
mkdir -p "$LOG_DIR"

MAIN_LOG="$LOG_DIR/main.log"
GPU_LOG="$LOG_DIR/gpu_watch.log"
HEART_LOG="$LOG_DIR/heartbeat.log"

exec > >(stdbuf -oL tee -a "$MAIN_LOG") 2>&1

log_stage() {
  echo
  echo "========== $(date '+%F %T') | $* =========="
  sync
}

log_stage "STAGE 00: script started"
echo "RUN_ID=$RUN_ID"
echo "PWD=$(pwd)"
echo "HOST=$(hostname)"
echo "USER=$USER"
echo "LOG_DIR=$LOG_DIR"

export CONDA_PREFIX=/mnt/niumiaohe/miniconda3/envs/elf
export CONDA_DEFAULT_ENV=elf
export PATH=/mnt/niumiaohe/miniconda3/envs/elf/bin:$PATH

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HF_HOME=/workspace/greatwall/ELF_repro/hf_cache
export TRANSFORMERS_CACHE=/workspace/greatwall/ELF_repro/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,ENV
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

CKPT=/workspace/greatwall/ELF_repro/hf_cache/hub/models--embedded-language-flows--ELF-B-owt-torch/snapshots/146f84133c1389bfd4ef47f14ec7a955da22faa7/checkpoint_95085

log_stage "STAGE 01: env check"
echo "CONDA_DEFAULT_ENV=$CONDA_DEFAULT_ENV"
echo "CONDA_PREFIX=$CONDA_PREFIX"
echo "PATH=$PATH"
echo "python=$(which python)"
echo "pip=$(which pip)"
echo "torchrun=$(which torchrun)"
python --version
pip --version
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('n_gpu', torch.cuda.device_count())"

log_stage "STAGE 02: system check"
uptime
free -h
nvidia-smi
nvidia-smi -L

log_stage "STAGE 03: checkpoint/output check"
echo "CKPT=$CKPT"
ls -lah "$CKPT" || true
echo "Existing output dir:"
ls -lah outputs/elf_b-owt-ordered 2>/dev/null || true
echo "Existing checkpoints:"
find outputs/elf_b-owt-ordered -maxdepth 1 -type d -name "checkpoint_*" -print 2>/dev/null || true

log_stage "STAGE 04: start background monitors"
(
  while true; do
    echo "===== $(date '+%F %T') ====="
    uptime
    free -h | head -3
    ps -u "$USER" -o pid,ppid,stat,etime,%cpu,%mem,cmd | grep -E "train.py|torchrun|launch.sh" | grep -v grep || true
    echo
    sleep 5
  done
) >> "$HEART_LOG" 2>&1 &
HEART_PID=$!

stdbuf -oL nvidia-smi \
  --query-gpu=timestamp,index,pstate,temperature.gpu,power.draw,memory.used,utilization.gpu,utilization.memory \
  --format=csv \
  -l 1 >> "$GPU_LOG" 2>&1 &
GPU_PID=$!

echo "HEART_PID=$HEART_PID"
echo "GPU_PID=$GPU_PID"
echo "HEART_LOG=$HEART_LOG"
echo "GPU_LOG=$GPU_LOG"
sync

cleanup() {
  status=$?
  echo
  echo "========== $(date '+%F %T') | SCRIPT EXIT status=$status =========="
  kill "$HEART_PID" "$GPU_PID" 2>/dev/null || true
  sync
  exit $status
}
trap cleanup EXIT

log_stage "STAGE 05: before torchrun launch"
echo "Launch command:"
echo "NGPU=8 bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B_ordered.yml ..."
sync

set +e
NGPU=8 bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B_ordered.yml \
  --config_override init_from="$CKPT" \
  --config_override output_dir=outputs/elf_b-owt-ordered \
  --config_override global_batch_size=64 \
  --config_override grad_accum_steps=8 \
  --config_override gradient_checkpointing=true \
  --config_override use_wandb=false

TRAIN_STATUS=$?
set -e

log_stage "STAGE 06: torchrun returned"
echo "TRAIN_STATUS=$TRAIN_STATUS"

log_stage "STAGE 07: final nvidia-smi"
nvidia-smi || true

exit "$TRAIN_STATUS"
