#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
MODE=${1:?smoke|formal}; [[ "$MODE" == smoke || "$MODE" == formal ]]
PY=${PYTHON_BIN:-python}
GPU_ID=${GPU_ID:-7}; RESUME=${RESUME:-0}
BASE=${OUTPUT_BASE:-$REPO_ROOT/outputs/stage_b_common80k_final_eval_v1/oracle_content_probe_single_gpu_serial_v1}
ROOT=$BASE/smoke_n4; N=4
[[ "$MODE" == formal ]] && ROOT=$BASE/formal_n1000 && N=1000
LOG=results/tpt_million_v1/logs/stage_b_oracle_serial_${MODE}_v1.log
mkdir -p "$(dirname "$LOG")"
ts(){ date -Iseconds; }; stage=START
trap 'rc=$?; printf "[%s] [serial_oracle] [FAIL] stage=%s source=%s line=%s command=%q exit=%s\n" "$(ts)" "$stage" "${BASH_SOURCE[0]}" "$LINENO" "$BASH_COMMAND" "$rc"; exit "$rc"' ERR
export PYTHONUNBUFFERED=1 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
stage=PRECHECK; [[ -x "$PY" ]]; [[ "$GPU_ID" =~ ^[0-9]+$ ]]
if [[ "$MODE" == formal ]]; then
  "$PY" - <<PY
import json
from pathlib import Path
p=Path('$BASE/smoke_n4/manifest.json')
assert p.exists() and json.loads(p.read_text()).get('complete') is True
PY
fi
stage=RUN
args=(--samples "$N" --output "$ROOT")
[[ "$RESUME" == 1 ]] && args+=(--resume)
printf '[%s] [serial_oracle] [START] mode=%s gpu=%s samples=%s output=%s resume=%s\n' "$(ts)" "$MODE" "$GPU_ID" "$N" "$ROOT" "$RESUME" | tee -a "$LOG"
set -o pipefail
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" -u tools/eval_stage_b_oracle_serial.py "${args[@]}" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
printf 'STAGE_B_ORACLE_SERIAL_%s_EXIT_CODE=%s\n' "${MODE^^}" "$rc" | tee -a "$LOG"
exit "$rc"
