#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
[[ $# -eq 1 && ( $1 == smoke || $1 == formal ) ]] || { echo "usage: $0 {smoke|formal}" >&2; exit 2; }
MODE=$1
PYTHON=${PYTHON:-python}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
: "${CUDA_VISIBLE_DEVICES:=7}"; export CUDA_VISIBLE_DEVICES
DATA=${DATA_MANIFEST:-$REPO_ROOT/data/main_general_v2_train_canonical/manifest.json}
CKPT=${STAGE_A_CHECKPOINT:-$REPO_ROOT/artifacts/stage_a/final.pt}
ALLOWLIST=${ALLOWLIST_SAMPLES:-/tmp/formal_mlp_gpu7_allowlist_samples.json}
if [[ $MODE == smoke ]]; then OUT=${OUTPUT_DIR:-$REPO_ROOT/outputs/whitening_smoke_512_v1}; BS=8
else OUT=${OUTPUT_DIR:-$REPO_ROOT/outputs/whitening_train_all_v1}; BS=${BATCH_SIZE:-16}; fi
[[ ! -e $OUT ]] || { echo "refusing existing output: $OUT" >&2; exit 3; }
"$PYTHON" tools/compute_formal_thinking_whitener.py --mode "$MODE" --config src/configs/training_configs/formal_thinking_mlp_4to1.yml --data_manifest "$DATA" --checkpoint "$CKPT" --output_dir "$OUT" --allowlist_samples "$ALLOWLIST" --batch_size "$BS"
