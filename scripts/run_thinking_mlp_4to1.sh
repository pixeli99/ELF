#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
[[ $# -eq 1 && ( "$1" == smoke || "$1" == formal ) ]] || { echo "usage: $0 {smoke|formal}" >&2; exit 2; }
MODE=$1; PYTHON=${PYTHON:-python}; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
: "${CUDA_VISIBLE_DEVICES:=7}"; export CUDA_VISIBLE_DEVICES
if [[ $MODE == formal ]]; then MANIFEST=${FORMAL_DATA_MANIFEST:?set FORMAL_DATA_MANIFEST}; DEFAULT_OUT=$REPO_ROOT/outputs/formal_thinking_mlp_4to1; EXTRA=--formal
else MANIFEST=${SMOKE_DATA_MANIFEST:?set SMOKE_DATA_MANIFEST}; DEFAULT_OUT=$REPO_ROOT/outputs/formal_thinking_mlp_4to1_smoke; EXTRA=""; fi
OUT=${OUTPUT_DIR:-$DEFAULT_OUT}
[[ ! -e $OUT ]] || { echo "refusing existing output: $OUT" >&2; exit 3; }
LIMIT=(); [[ -z ${MAX_OPTIMIZER_STEPS:-} ]] || LIMIT=(--max_optimizer_steps "$MAX_OPTIMIZER_STEPS")
"$PYTHON" tools/train_formal_thinking_mlp.py --config src/configs/training_configs/formal_thinking_mlp_4to1.yml --data_manifest "$MANIFEST" --output_dir "$OUT" $EXTRA "${LIMIT[@]}" 2>&1 | tee "${OUT}.console.log"
