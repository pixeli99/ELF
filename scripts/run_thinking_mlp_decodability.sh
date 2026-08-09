#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
[[ $# -eq 1 && ( "$1" == smoke || "$1" == formal ) ]] || { echo "usage: $0 {smoke|formal}" >&2; exit 2; }
MODE=$1; PYTHON=${PYTHON:-/mnt/niumiaohe/miniconda3/envs/elf/bin/python}; export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1; : "${CUDA_VISIBLE_DEVICES:=7}"; export CUDA_VISIBLE_DEVICES
MANIFEST_VAR=$([[ $MODE == formal ]] && echo FORMAL_DATA_MANIFEST || echo SMOKE_DATA_MANIFEST); MANIFEST=${!MANIFEST_VAR:?set $MANIFEST_VAR}; AE_CHECKPOINT=${THINKING_MLP_CHECKPOINT:?set THINKING_MLP_CHECKPOINT}; ELF_CHECKPOINT=${ELF_CHECKPOINT:-outputs/elf_b-owt-ordered-5b-4g/checkpoint_152592}; DEFAULT_OUT=/workspace/greatwall/ELF_ordered_outputs/thinking_mlp_decodability_${MODE}; OUT=${OUTPUT_DIR:-$DEFAULT_OUT}
[[ ! -e $OUT ]] || { echo "refusing existing output: $OUT" >&2; exit 3; }
"$PYTHON" tools/eval_thinking_mlp_token_decodability.py --elf_config src/configs/training_configs/train_owt_ELF-B_ordered.yml --elf_checkpoint "$ELF_CHECKPOINT" --autoencoder_checkpoint "$AE_CHECKPOINT" --data_manifest "$MANIFEST" --output_dir "$OUT" 2>&1 | tee "${OUT}.console.log"
