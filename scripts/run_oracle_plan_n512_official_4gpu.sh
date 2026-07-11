#!/usr/bin/env bash
set -u

cd /home/greatwall/ELF_ordered/ELF

export PATH=/mnt/niumiaohe/miniconda3/envs/elf/bin:$PATH
export HF_HOME=/workspace/greatwall/ELF_repro/hf_cache
export TRANSFORMERS_CACHE=/workspace/greatwall/ELF_repro/hf_cache/hub
export HF_ENDPOINT=https://hf-mirror.com
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

unset HF_HUB_OFFLINE
unset TRANSFORMERS_OFFLINE
unset HF_DATASETS_OFFLINE
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy

OUTBASE=results/oracle_plan_eval/n512_steps32_seed42_official_sampling_by_mode
mkdir -p "${OUTBASE}/logs"

run_mode () {
  GPU="$1"
  MODE="$2"

  echo "[$(date)] starting ${MODE} on GPU ${GPU}"

  CUDA_VISIBLE_DEVICES="${GPU}" python tools/eval_oracle_plan.py \
    --config_path src/configs/training_configs/train_owt_ELF-B_ordered.yml \
    --checkpoint_path outputs/elf_b-owt-ordered-5b-4g/checkpoint_152592 \
    --sampling_config_path src/configs/sampling_configs/ordered_sampling_configs.yml \
    --dataset embedded-language-flows/openwebtext-t5 \
    --num_samples 512 \
    --num_sampling_steps 32 \
    --modes "${MODE}" \
    --seed 42 \
    --paired_eval_base_seed 42 \
    --global_batch_size 4 \
    --eval_ppl_batch_size 4 \
    --out_dir "${OUTBASE}/${MODE}" \
    > "${OUTBASE}/logs/${MODE}.log" 2>&1

  status=$?
  echo "${status}" > "${OUTBASE}/logs/${MODE}.exit"
  echo "[$(date)] finished ${MODE} on GPU ${GPU}, exit=${status}"
  return "${status}"
}

run_mode 4 null &
run_mode 5 self_planning_first &
run_mode 6 oracle_matched &
run_mode 7 oracle_shuffled &

wait

echo "All oracle-plan jobs finished at $(date)"
for m in null self_planning_first oracle_matched oracle_shuffled; do
  echo "===== ${m} exit code ====="
  cat "${OUTBASE}/logs/${m}.exit"
done
