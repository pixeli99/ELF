#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
[[ ${1:-} == formal && $# -eq 1 ]] || { echo "usage: $0 formal" >&2; exit 2; }
PYTHON=${PYTHON:-/mnt/niumiaohe/miniconda3/envs/elf/bin/python}; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; : "${CUDA_VISIBLE_DEVICES:=7}"; export CUDA_VISIBLE_DEVICES
MANIFEST=${FORMAL_DATA_MANIFEST:?set FORMAL_DATA_MANIFEST}; ENCODER=${FROZEN_THINKING_MLP_ENCODER:?set FROZEN_THINKING_MLP_ENCODER}; OUT=/workspace/greatwall/ELF_ordered_outputs/formal_thinking_mlp_4to1_whitener
[[ ! -e $OUT ]] || { echo "refusing existing output: $OUT" >&2; exit 3; }
"$PYTHON" tools/compute_thinking_plan_whitener.py --config src/configs/training_configs/formal_thinking_mlp_4to1.yml --data_manifest "$MANIFEST" --frozen_encoder "$ENCODER" --output_dir "$OUT" 2>&1 | tee "${OUT}.console.log"
