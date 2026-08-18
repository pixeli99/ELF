#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"
MODE=${1:?usage: $0 smoke|formal}; [[ "$MODE" == smoke || "$MODE" == formal ]] || exit 2
IFS=',' read -r -a GPUS <<<"${GPU_IDS:-0,1,2,3}"; [[ ${#GPUS[@]} == 4 ]] || { echo "exactly four GPU IDs required";exit 2; }
PY=${PYTHON_BIN:-python}
SCHEDULE=${COMMON_SCHEDULE:-data/common_schedule_80k_seed42_v1.jsonl}
[[ $(sha256sum "$SCHEDULE"|cut -d' ' -f1) == eb840ba3dc78e90aa91f91c1543f47e30f4fe10260ac16c249320a56694dd766 ]] || exit 9
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
OUTPUT_ROOT=${OUTPUT_ROOT:-$REPO_ROOT/outputs}
STAGE_A_ARTIFACT_DIR=${STAGE_A_ARTIFACT_DIR:-$REPO_ROOT/artifacts/stage_a}
groups=(ordered register diagonal vanilla);pids=();names=();mkdir -p results/tpt_million_v1/logs
cleanup(){ for pid in "${pids[@]:-}";do kill "$pid" 2>/dev/null||true;done;for pid in "${pids[@]:-}";do wait "$pid" 2>/dev/null||true;done; }
trap cleanup INT TERM
EXIT=results/tpt_million_v1/logs/stage_b_common80k_${MODE}_exit_codes.tsv; : >"$EXIT"
for i in "${!groups[@]}";do
 group=${groups[$i]};config=src/configs/training_configs/train_stage_b_common80k_${group}_10k_v1.yml
 out=$OUTPUT_ROOT/elf_b_common80k_${group}_10k_v1; [[ "$MODE" == smoke ]] && out=${out%_10k_v1}_smoke20_v1
 log=results/tpt_million_v1/logs/stage_b_common80k_${group}_${MODE}_v1.log
 append_log=0
 if [[ "$MODE" == formal && -f "$out/checkpoint_10000" ]];then echo "$group already complete";continue;fi
 overrides=(--config_override "output_dir=$out"); [[ "$MODE" == smoke ]] && overrides+=(--config_override max_optimizer_steps=20 --config_override save_optimizer_steps='[20]' --config_override engineering_smoke_report=true)
 if [[ -e "$out" ]];then
  [[ ${RESUME:-0} == 1 ]] || { echo "refusing overwrite $out";exit 3; }
  overrides+=(--config_override init_from=null --config_override "resume=$out")
  append_log=1
 else
  [[ ! -e "$log" ]] || { echo "refusing overwrite $log";exit 3; }
 fi
 (
  before=$(sha256sum "$STAGE_A_ARTIFACT_DIR/frozen_encoder_v1.pt" "$STAGE_A_ARTIFACT_DIR/whitener_stageb_v1.pt")
  set +e
  if [[ $append_log == 1 ]];then
   CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY" src/train.py --config "$config" "${overrides[@]}" >>"$log" 2>&1
  else
   CUDA_VISIBLE_DEVICES=${GPUS[$i]} "$PY" src/train.py --config "$config" "${overrides[@]}" >"$log" 2>&1
  fi
  rc=$?;set -e
  after=$(sha256sum "$STAGE_A_ARTIFACT_DIR/frozen_encoder_v1.pt" "$STAGE_A_ARTIFACT_DIR/whitener_stageb_v1.pt")
  [[ "$before" == "$after" ]] || exit 91;exit "$rc"
 ) & pids+=($!);names+=("$group")
done
rc=0
for i in "${!pids[@]}";do code=0;wait "${pids[$i]}"||code=$?;printf '%s\t%s\n' "${names[$i]}" "$code" >>"$EXIT";[[ $code == 0 ]]||rc=1;done
exit "$rc"
