#!/bin/bash
# Baseline experiment of the Isaac Lab generators (SoftMimicGen, DexMimicGen) on one GPU machine, end to end:
# for every task, the official source demonstrations, one generated pool of 1,000 successes with the official
# generator settings (method none, generation pool_isaac), the fixed evaluation scenes, a 1,000-demonstration image
# dataset, then three training seeds of the image Diffusion Policy, each validated on the 25 validation scenes at
# every checkpoint, selected, and tested on the 300 test scenes. Ends with a results bundle (scripts/collect_results.py).
#
# Two lanes keep the GPU busy: the generation lane runs the tasks one after another (sources, pool, resets, dataset),
# the policy lane trains and evaluates each task as soon as its dataset exists. Everything is resumable: finished
# pools, datasets, checkpoints, validations and evaluations are kept and skipped on a rerun.
#
#   CLUSTER=ada scripts/run_isaaclab_baseline.sh
# Variables: CLUSTER (ada), GPU (0), SOFT_TASKS ("rope towel jenga"), DEX_TASKS ("pick_place"), SEEDS ("202601 202602 202603"),
#   POLICY (diffusion_policy), SIZE (1000), VAL_PARALLEL (2: checkpoints validated at a time, each one Isaac Sim),
#   TEST_SHARDS (3: parallel shards of the 300 test scenes), SEED_PARALLEL (1: training runs at a time),
#   ONLY_GENERATION (1: stop after the datasets), BUNDLE_NAME (results folder name under <data root>/reports).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-ada}; GPU=${GPU:-0}; SOFT_TASKS=${SOFT_TASKS:-"rope towel jenga"}; DEX_TASKS=${DEX_TASKS:-"pick_place"}
SEEDS=${SEEDS:-"202601 202602 202603"}; POLICY=${POLICY:-diffusion_policy}; SIZE=${SIZE:-1000}
VAL_PARALLEL=${VAL_PARALLEL:-2}; TEST_SHARDS=${TEST_SHARDS:-3}; SEED_PARALLEL=${SEED_PARALLEL:-1}
SCENE=ours; METHOD=none; GENERATION=pool_isaac; OBSERVATION=image
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
LOG=$DATA_ROOT/logs/isaaclab_baseline; mkdir -p "$LOG"
BUNDLE_NAME=${BUNDLE_NAME:-isaaclab_baseline_$(date +%Y%m%d)}
log() { echo "[driver $(date '+%F %T')] $*" | tee -a "$LOG/driver.log"; }
fail() { log "$1 FAILED (see $2)"; echo "$1" >> "$LOG/FAILURES"; }
# generator/task pairs in run order (Soft first, then Dex; the two images are never needed at the same time in one lane)
JOBS=""
for t in $SOFT_TASKS; do JOBS="$JOBS softmimicgen:isaaclab-soft:$t"; done
for t in $DEX_TASKS; do JOBS="$JOBS dexmimicgen:isaaclab-dex:$t"; done
dataset_of() { echo "$DATA_ROOT/datasets/$1/$2/$SCENE/$METHOD/$GENERATION/$OBSERVATION/n${SIZE}_seed0.hdf5"; }
resets_of() { echo "$DATA_ROOT/resets/$1/$2/$SCENE/val25_test300_seed202600.json"; }
run_of() { echo "$DATA_ROOT/runs/$1/$2/$SCENE/$METHOD/$GENERATION/$OBSERVATION/$POLICY/seed$3"; }
M() { make -s -C "$REPO" "$@"; }

generation_lane() {
  for job in $JOBS; do
    IFS=: read -r gen image task <<< "$job"
    local common="GENERATOR=$gen IMAGE=$image TASK=$task CLUSTER=$CLUSTER"
    if [ -f "$(dataset_of $gen $task)" ] && [ -f "$(resets_of $gen $task)" ]; then log "[$gen/$task] dataset and resets exist, generation skipped"; continue; fi
    log "[$gen/$task] sources"
    M sources $common > "$LOG/sources_${gen}_$task.log" 2>&1 || { fail "sources $gen/$task" "$LOG/sources_${gen}_$task.log"; continue; }
    log "[$gen/$task] pool ($GENERATION: 4 chunks of 250 successes)"
    M generate $common SCENE=$SCENE METHOD=$METHOD GENERATION=$GENERATION > "$LOG/pool_${gen}_$task.log" 2>&1 \
      || { fail "pool $gen/$task" "$LOG/pool_${gen}_$task.log"; continue; }
    log "[$gen/$task] pool done: $(grep -h 'pool complete' "$LOG/pool_${gen}_$task.log" | tail -1 | cut -c1-100)"
    M resets $common SCENE=$SCENE > "$LOG/resets_${gen}_$task.log" 2>&1 || { fail "resets $gen/$task" "$LOG/resets_${gen}_$task.log"; continue; }
    M dataset $common SCENE=$SCENE METHOD=$METHOD GENERATION=$GENERATION OBSERVATION=$OBSERVATION OVERRIDES="size=$SIZE" \
      > "$LOG/dataset_${gen}_$task.log" 2>&1 || { fail "dataset $gen/$task" "$LOG/dataset_${gen}_$task.log"; continue; }
    log "[$gen/$task] dataset written: $(dataset_of $gen $task)"
  done
  log "generation lane done"
}

policy_task() {   # all seeds of one task, SEED_PARALLEL at a time; each seed has its own block of 40 ports
  local gen=$1 image=$2 task=$3 port=$4
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$SEED_PARALLEL" ]; do wait -n || true; done
    local run; run=$(run_of $gen $task $seed)
    if compgen -G "$run/eval/test_h*/*/eval_summary.json" > /dev/null; then log "[$gen/$task seed $seed] test result exists, skipped"; continue; fi
    log "[$gen/$task seed $seed] train, validate, select, test (ports $port..$((port + 39)))"
    SKIP=0; [ -f "$run/train_result.json" ] && SKIP=1
    CLUSTER=$CLUSTER GPU=$GPU GENERATOR=$gen IMAGE=$image SCENE=$SCENE METHOD=$METHOD GENERATION=$GENERATION OBSERVATION=$OBSERVATION \
      POLICY=$POLICY SEED=$seed SIZE=$SIZE VAL_PARALLEL=$VAL_PARALLEL TEST_SHARDS=$TEST_SHARDS SKIP_TRAIN=$SKIP \
      "$REPO/scripts/task_policy_pipeline.sh" $task $port > "$LOG/policy_${gen}_${task}_seed$seed.log" 2>&1 \
      || fail "policy $gen/$task seed $seed" "$LOG/policy_${gen}_${task}_seed$seed.log" &
    port=$((port + 40)); sleep 30
  done
  wait
}

policy_lane() {
  local port=8800
  for job in $JOBS; do
    IFS=: read -r gen image task <<< "$job"
    until [ -f "$(dataset_of $gen $task)" ] && [ -f "$(resets_of $gen $task)" ]; do
      grep -q "pool $gen/$task\|dataset $gen/$task\|sources $gen/$task\|resets $gen/$task" "$LOG/FAILURES" 2>/dev/null && { log "[$gen/$task] generation failed, policies skipped"; continue 2; }
      sleep 120
    done
    policy_task $gen $image $task $port
    port=$((port + 40 * 3))
  done
  log "policy lane done"
}

log "start: cluster $CLUSTER, GPU $GPU, jobs:$JOBS, seeds: $SEEDS, policy $POLICY"
rm -f "$LOG/FAILURES"
generation_lane &
if [ "${ONLY_GENERATION:-0}" = 1 ]; then wait; else policy_lane & wait; fi
log "collecting results into $DATA_ROOT/reports/$BUNDLE_NAME"
python3 "$REPO/scripts/collect_results.py" --data-root "$DATA_ROOT" --out "$DATA_ROOT/reports/$BUNDLE_NAME" --policy $POLICY --seeds $SEEDS \
  $(for job in $JOBS; do IFS=: read -r gen image task <<< "$job"; echo "--job $gen/$task/$SCENE/$METHOD/$GENERATION/$OBSERVATION"; done) \
  --logs "$LOG" > "$LOG/collect.log" 2>&1 && log "bundle: $DATA_ROOT/reports/$BUNDLE_NAME.tar.gz" || fail "collect" "$LOG/collect.log"
[ -f "$LOG/FAILURES" ] && log "FAILURES: $(tr '\n' ';' < "$LOG/FAILURES")"
echo "ISAACLAB_BASELINE_DONE $(date)" > "$LOG/DONE"; log "done"
