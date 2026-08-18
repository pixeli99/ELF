#!/usr/bin/env bash
set -Eeuo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

MODE=${1:?usage: $0 smoke|formal|preflight}
TARGET_MODE=${TARGET_MODE:-smoke}
[[ "$MODE" == preflight ]] && PREFLIGHT_ONLY=1 && MODE=$TARGET_MODE
[[ "$MODE" == smoke || "$MODE" == formal ]] || { echo "invalid mode: $MODE"; exit 2; }
PREFLIGHT_ONLY=${PREFLIGHT_ONLY:-0}; RESUME=${RESUME:-0}
GPU_IDS=${GPU_IDS:-0,1,2,3}; MAX_PARALLEL=${MAX_PARALLEL:-4}
BASE=${OUTPUT_BASE:-$REPO_ROOT/outputs/stage_b_common80k_final_eval_v1}
SMOKE="$BASE/generation_smoke_n4_v1_rerun1"
ROOT="$SMOKE"; N=4
[[ "$MODE" == formal ]] && ROOT="$BASE/generation_formal_n1000_v1_rerun1" && N=1000
CURRENT_STAGE=BOOT; CURRENT_CONDITION=launcher

ts(){ date -Iseconds; }
log(){ printf '[%s] [%s] [%s] %s\n' "$(ts)" "${1:-launcher}" "${2:-INFO}" "${3:-}"; }
write_failure(){
  local rc=$1 cmd=$2 src=$3 line=$4
  [[ -d "$ROOT" ]] || return 0
  FAILURE_RC=$rc FAILURE_CMD=$cmd FAILURE_SRC=$src FAILURE_LINE=$line FAILURE_STAGE=$CURRENT_STAGE FAILURE_CONDITION=$CURRENT_CONDITION ROOT_PATH=$ROOT "$PY" - <<'PY' || true
import json,os,datetime,pathlib
r=pathlib.Path(os.environ['ROOT_PATH'])
p={'complete':False,'failed_stage':os.environ['FAILURE_STAGE'],'failed_condition':os.environ['FAILURE_CONDITION'],
'exit_code':int(os.environ['FAILURE_RC']),'command':os.environ['FAILURE_CMD'],'source_file':os.environ['FAILURE_SRC'],
'line_number':int(os.environ['FAILURE_LINE']),'timestamp':datetime.datetime.now().astimezone().isoformat(),
'worker_log':None,'completed_conditions':[],'pending_conditions':[]}
(r/'failure_report.json').write_text(json.dumps(p,indent=2,sort_keys=True)+'\n')
PY
}
on_err(){ local rc=$? line=$1 cmd=$2 src=$3; log "$CURRENT_CONDITION" FAIL "stage=$CURRENT_STAGE exit=$rc source=$src line=$line command=$cmd"; write_failure "$rc" "$cmd" "$src" "$line"; log launcher FAIL "GENERATION_EVALUATION_GATE_FAIL failure_report=$ROOT/failure_report.json"; exit "$rc"; }
trap 'on_err "$LINENO" "$BASH_COMMAND" "${BASH_SOURCE[0]}"' ERR

PY=${PYTHON_BIN:-python}
CURRENT_STAGE=PYTHON_ENV
[[ -x "$PY" ]] || { log launcher FAIL "stage=$CURRENT_STAGE expected=executable_python actual=$PY"; exit 2; }
export PYTHONUNBUFFERED=1 HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
CONDITIONS=(ordered_alpha2p0_steps8 ordered_alpha2p0_steps32 ordered_alpha1p0_steps8 ordered_alpha1p0_steps32 ordered_alpha0p5_steps8 ordered_alpha0p5_steps32 ordered_alpha0p0_steps8 ordered_alpha0p0_steps32 diagonal_alpha1p0_steps8 diagonal_alpha1p0_steps32 register_native_steps8 register_native_steps32 vanilla_native_steps8 vanilla_native_steps32)
log launcher START "mode=$MODE preflight_only=$PREFLIGHT_ONLY RESUME=$RESUME GPU_IDS=$GPU_IDS MAX_PARALLEL=$MAX_PARALLEL python=$PY"
log launcher ENV "HF_HOME=$HF_HOME HF_HUB_OFFLINE=$HF_HUB_OFFLINE TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE root=$ROOT smoke_root=$SMOKE"
log launcher CONDITIONS "${CONDITIONS[*]}"

CURRENT_STAGE=CPU_PREFLIGHT
"$PY" -u tools/preflight_stage_b_common80k_generation.py --mode "$MODE" --root "$ROOT" --resume "$RESUME" --gpu-ids "$GPU_IDS" --max-parallel "$MAX_PARALLEL"
if [[ "$PREFLIGHT_ONLY" == 1 ]]; then log launcher PASS "GENERATION_PREFLIGHT_GATE_PASS"; exit 0; fi

IFS=',' read -r -a GPUS <<<"$GPU_IDS"
[[ "$RESUME" == 1 || ! -e "$ROOT" ]] || { log launcher FAIL "stage=OUTPUT expected=absent actual=$ROOT"; exit 3; }
mkdir -p "$ROOT"
printf 'schema_version\tpid\tgpu\tcondition_id\tworker_log_work\tworker_log_final\n' >"$ROOT/pids.tsv"
printf 'condition_id\texit_code\tpid\tgpu\tstatus\n' >"$ROOT/exit_codes.tsv"
: >"$ROOT/progress.jsonl"
SMOKE_IDS="$ROOT/smoke_eval_ids.json"
if [[ "$MODE" == smoke ]]; then
  CURRENT_STAGE=SMOKE_SELECTION
  "$PY" -u - <<'PY' >"$SMOKE_IDS"
import json,sys;sys.path.insert(0,'src')
from utils.stage_b_common80k_generation import validate_shape_inputs,select_smoke
b='results/tpt_million_v1/stage_b_heldout_v2';r=validate_shape_inputs(b+'/split_manifest.json',b+'/stage_b_generation_shape_test_n1000_seed45_v2/manifest.json',b+'/stage_b_generation_shape_test_n1000_seed45_v2/shapes.jsonl')
print(json.dumps({'seed':42,'eval_ids':[x['eval_id'] for x in select_smoke(r)]},indent=2))
PY
else
  CURRENT_STAGE=SMOKE_PREREQUISITE
  "$PY" -u tools/finalize_stage_b_common80k_generation.py --root "$SMOKE" --expected-rows 4 --check-root-only
fi

PENDING=()
for cid in "${CONDITIONS[@]}"; do
  if [[ "$RESUME" == 1 ]] && "$PY" -u tools/finalize_stage_b_common80k_generation.py --root "$ROOT" --expected-rows "$N" --validate-arm "$cid"; then
    log "$cid" RESUME "SKIP manifest/hash identity valid"; printf '%s\t0\t0\t-1\tSKIPPED_VALID\n' "$cid" >>"$ROOT/exit_codes.tsv"
  else
    [[ ! -e "$ROOT/$cid" ]] || { log "$cid" FAIL "resume validation failed for existing arm"; exit 3; }
    PENDING+=("$cid")
  fi
done

failed=0; run_id=$$; completed=()
for ((offset=0; offset<${#PENDING[@]}; offset+=MAX_PARALLEL)); do
  PIDS=(); NAMES=(); GPUS_USED=(); LOGS=()
  for ((slot=0; slot<MAX_PARALLEL && offset+slot<${#PENDING[@]}; slot++)); do
    cid=${PENDING[$((offset+slot))]}; gpu=${GPUS[$slot]}; work="$ROOT/${cid}.work.${run_id}"; mkdir -p "$work"; worker_log="$work/worker.log"; args=(); [[ "$MODE" == smoke ]] && args+=(--smoke-ids "$SMOKE_IDS")
    CURRENT_STAGE=WORKER_START; CURRENT_CONDITION=$cid
    log "$cid" START "gpu=$gpu output=$ROOT/$cid work=$work worker_log=$worker_log"
    (set -o pipefail; CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u tools/eval_stage_b_common80k_generation.py --condition-id "$cid" --output "$ROOT/$cid" --work-dir "$work" --samples "$N" "${args[@]}" 2>&1 | sed -u "s/^/[$cid] /" | tee -a "$worker_log") &
    pid=$!; PIDS+=("$pid"); NAMES+=("$cid"); GPUS_USED+=("$gpu"); LOGS+=("$worker_log"); printf '2\t%s\t%s\t%s\t%s\t%s\n' "$pid" "$gpu" "$cid" "$worker_log" "$ROOT/$cid/worker.log" >>"$ROOT/pids.tsv"; log "$cid" PID "pid=$pid gpu=$gpu"
  done
  for i in "${!PIDS[@]}"; do
    CURRENT_STAGE=WAIT_WORKER; CURRENT_CONDITION=${NAMES[$i]}; set +e; wait "${PIDS[$i]}"; code=$?; set -e
    printf '%s\t%s\t%s\t%s\tFINISHED\n' "${NAMES[$i]}" "$code" "${PIDS[$i]}" "${GPUS_USED[$i]}" >>"$ROOT/exit_codes.tsv"
    log "${NAMES[$i]}" DONE "exit=$code pid=${PIDS[$i]} gpu=${GPUS_USED[$i]} worker_log=${LOGS[$i]}"
    if [[ $code -eq 0 ]]; then completed+=("${NAMES[$i]}"); else failed=1; fi
  done
  [[ $failed -eq 0 ]] || break
done

if [[ $failed -ne 0 ]]; then
  CURRENT_STAGE=WORKER_FAILURE; CURRENT_CONDITION=launcher
  write_failure 1 "one_or_more_workers_failed" "${BASH_SOURCE[0]}" "$LINENO"
  log launcher FAIL "GENERATION_EVALUATION_GATE_FAIL failure_report=$ROOT/failure_report.json"; exit 1
fi
CURRENT_STAGE=FINALIZE; CURRENT_CONDITION=launcher
FINAL_ARGS=(); [[ "$MODE" == formal ]] && FINAL_ARGS+=(--smoke-prerequisite "$SMOKE")
"$PY" -u tools/finalize_stage_b_common80k_generation.py --root "$ROOT" --expected-rows "$N" "${FINAL_ARGS[@]}"
log launcher PASS "GENERATION_EVALUATION_GATE_PASS root=$ROOT"
