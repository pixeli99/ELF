#!/bin/bash
# Stage 0 floor check on one 16-GPU node: four arms, four GPUs each, then GSM8K.
#
# DLC hands the UserCommand to dash, not bash, and dash exits on `set -o pipefail` and
# silently skips `source`. So the UserCommand stays POSIX (cd / export / bash) and every
# bashism lives here.
#
#   ARMS           space separated config basenames under src/configs/training_configs
#   GPUS_PER_ARM   defaults to 4
#   STAGE          all (default) | train | eval | preflight
#
# Training auto-resumes from output_dir, so a preempted job picks up its own last
# checkpoint rather than warm starting from the pretrained weights again.
set -u

REPO=/cpfs01/shared/public/users/pengxiang.li/ELF
cd "$REPO" || { echo "FATAL: cannot cd to $REPO"; exit 1; }

ARMS=${ARMS:-"math_b_noreason math_b_cot math_l_noreason math_l_cot"}
GPUS_PER_ARM=${GPUS_PER_ARM:-4}
STAGE=${STAGE:-all}
RUN=${RUN:-stage0}
EVAL_BATCH=${EVAL_BATCH:-32}

LOGS=outputs/logs/$RUN
mkdir -p "$LOGS"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1          # every weight is staged on cpfs01; never reach the network
export TORCH_DIST_TIMEOUT_MINUTES=180

# The cluster image does not ship muon-optimizer, which the official configs select.
export PYTHONPATH="$REPO/assets/pydeps${PYTHONPATH:+:$PYTHONPATH}"

# DLC injects WORLD_SIZE / RANK / MASTER_ADDR / MASTER_PORT for the pytorchjob as a whole.
# This job runs several independent single-node launches inside one pod, so leaving them
# set makes every bare `python` process believe it is rank 0 of one group and race for one
# port: four eval processes then die with EADDRINUSE on the injected port 23456. torchrun
# sets these itself for its own children, so unsetting them costs the training nothing.
unset WORLD_SIZE RANK LOCAL_RANK MASTER_ADDR MASTER_PORT
unset GROUP_RANK ROLE_RANK LOCAL_WORLD_SIZE ROLE_WORLD_SIZE TORCHELASTIC_RUN_ID

fail() { echo "FATAL: $*" >&2; exit 1; }

banner() { echo; echo "=== $* ==="; date '+%F %T'; }

preflight() {
    banner "preflight"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || fail "no GPUs"
    local visible
    visible=$(nvidia-smi --list-gpus | wc -l)
    local needed=$(( $(echo "$ARMS" | wc -w) * GPUS_PER_ARM ))
    [ "$visible" -ge "$needed" ] || fail "need $needed GPUs for these arms, node shows $visible"

    # Trap 3 in the DLC notes: a path that resolves on the submitting host may simply not
    # exist on a worker. Assert every input up front so a failure names itself.
    [ -d assets/t5-small ] || fail "missing assets/t5-small (the staged encoder/tokenizer)"
    [ -f data/gsm8k_test.jsonl ] || fail "missing data/gsm8k_test.jsonl"
    for arm in $ARMS; do
        local config=src/configs/training_configs/$arm.yml
        [ -f "$config" ] || fail "missing $config"
        local data init
        data=$(grep -E '^data_path:' "$config" | awk '{print $2}')
        init=$(grep -E '^init_from:' "$config" | awk '{print $2}')
        [ -d "$data" ] || fail "$arm: missing dataset $data"
        [ -f "$init" ] || fail "$arm: missing warm start checkpoint $init"
        echo "  ok  $arm  data=$data  init=$init"
    done
    python -c "import torch, transformers, datasets; print(f'  torch {torch.__version__} transformers {transformers.__version__} datasets {datasets.__version__}')" \
        || fail "python imports failed"
    # muon lives in assets/pydeps, not in the image. A missing optimizer surfaces only once
    # the model is already built, which on the first attempt cost a whole 16-GPU job.
    python -c "import muon; print(f'  muon {muon.__file__}')" \
        || fail "cannot import muon: run scripts/stage_assets.sh"
    [ -z "${MASTER_PORT:-}" ] || fail "MASTER_PORT is still set; the unset above did not take"
}

devices_for() {  # index -> "0,1,2,3"
    local first=$(( $1 * GPUS_PER_ARM ))
    seq -s, "$first" $(( first + GPUS_PER_ARM - 1 ))
}

train_all() {
    banner "train"
    local index=0 pids="" names=""
    for arm in $ARMS; do
        local devices port
        devices=$(devices_for "$index")
        port=$(( 29500 + index ))
        echo "  launching $arm on GPUs $devices (master_port $port)"
        CUDA_VISIBLE_DEVICES=$devices torchrun \
            --nnodes=1 --nproc_per_node="$GPUS_PER_ARM" --master_port="$port" \
            src/train.py --config "src/configs/training_configs/$arm.yml" \
            > "$LOGS/$arm.train.log" 2>&1 &
        pids="$pids $!"
        names="$names $arm"
        index=$(( index + 1 ))
    done

    local failures=0 i=1
    for pid in $pids; do
        local arm
        arm=$(echo "$names" | cut -d' ' -f$(( i + 1 )))
        if wait "$pid"; then
            echo "  train ok       $arm"
        else
            echo "  train FAILED   $arm (see $LOGS/$arm.train.log)"
            tail -30 "$LOGS/$arm.train.log"
            failures=$(( failures + 1 ))
        fi
        i=$(( i + 1 ))
    done
    return $failures
}

eval_all() {
    banner "eval on GSM8K test"
    # test_generation_cond does not shard the eval set across ranks, so extra ranks would
    # regenerate the same rows. One GPU per arm, four arms at once.
    local index=0 pids="" names=""
    for arm in $ARMS; do
        local device=$(( index * GPUS_PER_ARM ))
        echo "  evaluating $arm on GPU $device"
        CUDA_VISIBLE_DEVICES=$device python src/eval.py \
            --config "src/configs/training_configs/$arm.yml" \
            --checkpoint_path "outputs/$arm" \
            --config_override eval_data_path=data/gsm8k_test.jsonl \
            --config_override num_samples=1319 \
            --config_override global_batch_size="$EVAL_BATCH" \
            --config_override output_dir="outputs/$arm/gsm8k" \
            > "$LOGS/$arm.eval.log" 2>&1 &
        pids="$pids $!"
        names="$names $arm"
        index=$(( index + 1 ))
    done

    local failures=0 i=1
    for pid in $pids; do
        local arm
        arm=$(echo "$names" | cut -d' ' -f$(( i + 1 )))
        if wait "$pid"; then
            echo "  eval ok        $arm"
        else
            echo "  eval FAILED    $arm (see $LOGS/$arm.eval.log)"
            tail -30 "$LOGS/$arm.eval.log"
            failures=$(( failures + 1 ))
        fi
        i=$(( i + 1 ))
    done

    banner "scores"
    for arm in $ARMS; do
        [ -d "outputs/$arm/gsm8k" ] || continue
        python tools/score_gsm8k.py \
            --generated "outputs/$arm/gsm8k" \
            --gold data/gsm8k_test.jsonl \
            --out "outputs/$arm/gsm8k/scored.jsonl" \
            --label "$arm" || failures=$(( failures + 1 ))
    done
    return $failures
}

preflight
rc=0
case "$STAGE" in
    train) train_all; rc=$? ;;
    eval)  eval_all;  rc=$? ;;
    preflight) echo "preflight only" ;;
    all)
        train_all; rc=$?
        # Evaluate whatever trained: a single failed arm should not cost the other three
        # their numbers, and every checkpoint on disk is still worth scoring.
        eval_all || rc=$(( rc + $? ))
        ;;
    *) fail "unknown STAGE=$STAGE" ;;
esac

banner "done"
echo "exit=$rc"
exit $rc
