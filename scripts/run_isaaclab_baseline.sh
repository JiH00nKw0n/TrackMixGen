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
#   ONLY_GENERATION (1: stop after the datasets), BUNDLE_NAME (results folder name under <result root>/reports),
#   TMG_RESULTS_ROOT (persistent summaries and driver logs; defaults to the data root),
#   POLICIES (space-separated policies, defaults to POLICY), TMG_RETIRE_COMPLETED (1: verify all evaluations,
#   keep selected weights and evidence, remove completed task bulk data; requires SEQUENTIAL_TASKS=1).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-ada}; GPU=${GPU:-0}; SOFT_TASKS=${SOFT_TASKS-"rope towel jenga"}; DEX_TASKS=${DEX_TASKS-"pick_place"}
SEEDS=${SEEDS:-"202601 202602 202603"}; POLICY=${POLICY:-diffusion_policy}; SIZE=${SIZE:-1000}
POLICIES=${POLICIES:-$POLICY}
VAL_PARALLEL=${VAL_PARALLEL:-2}; TEST_SHARDS=${TEST_SHARDS:-3}; SEED_PARALLEL=${SEED_PARALLEL:-1}
SEQUENTIAL_TASKS=${SEQUENTIAL_TASKS:-0}
if [ "${TMG_RETIRE_COMPLETED:-0}" = 1 ] && [ "$SEQUENTIAL_TASKS" != 1 ]; then
  echo 'Storage retirement requires SEQUENTIAL_TASKS=1' >&2; exit 1
fi
SCENE=ours; METHOD=none; GENERATION=pool_isaac; OBSERVATION=image
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
RESULTS_ROOT=${TMG_RESULTS_ROOT:-$DATA_ROOT}
LOG=$RESULTS_ROOT/logs/isaaclab_baseline; mkdir -p "$LOG"
BUNDLE_NAME=${BUNDLE_NAME:-isaaclab_baseline_$(date +%Y%m%d)}
log() { echo "[driver $(date '+%F %T')] $*" | tee -a "$LOG/driver.log"; }
fail() { log "$1 FAILED (see $2)"; echo "$1" >> "$LOG/FAILURES"; }
# generator/task pairs in run order (Soft first, then Dex; the two images are never needed at the same time in one lane)
JOBS=""
for t in $SOFT_TASKS; do JOBS="$JOBS softmimicgen:isaaclab-soft:$t"; done
for t in $DEX_TASKS; do JOBS="$JOBS dexmimicgen:isaaclab-dex:$t"; done
ALL_JOBS=$JOBS
dataset_of() { echo "$DATA_ROOT/datasets/$1/$2/$SCENE/$METHOD/$GENERATION/$OBSERVATION/n${SIZE}_seed0.hdf5"; }
resets_of() { echo "$DATA_ROOT/resets/$1/$2/$SCENE/val25_test300_seed202600.json"; }
run_of() { echo "$DATA_ROOT/runs/$1/$2/$SCENE/$METHOD/$GENERATION/$OBSERVATION/$POLICY/seed$3"; }
M() { make -s -C "$REPO" "$@"; }

collect_bundle() {
  local args=() job gen image task
  for job in $ALL_JOBS; do
    IFS=: read -r gen image task <<< "$job"
    args+=(--job "$gen/$task/$SCENE/$METHOD/$GENERATION/$OBSERVATION")
  done
  for policy in $POLICIES; do
    python3 "$REPO/scripts/collect_results.py" --data-root "$DATA_ROOT" --out "$RESULTS_ROOT/reports/$BUNDLE_NAME/$policy" \
      --policy "$policy" --seeds $SEEDS "${args[@]}" --logs "$LOG" "$@" > "$LOG/collect_$policy.log" 2>&1 \
      || { fail "collect $policy" "$LOG/collect_$policy.log"; return 1; }
  done
}

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
  local POLICY seed
  for POLICY in $POLICIES; do
  for seed in $SEEDS; do
    while [ "$(jobs -rp | wc -l)" -ge "$SEED_PARALLEL" ]; do wait -n || true; done
    local run; run=$(run_of $gen $task $seed)
    if compgen -G "$run/eval/test_h*/*/eval_summary.json" > /dev/null; then log "[$gen/$task seed $seed] test result exists, skipped"; continue; fi
    log "[$gen/$task $POLICY seed $seed] train, validate, select, test (ports $port..$((port + 39)))"
    SKIP=0; [ -f "$run/train_result.json" ] && SKIP=1
    CLUSTER=$CLUSTER GPU=$GPU GENERATOR=$gen IMAGE=$image SCENE=$SCENE METHOD=$METHOD GENERATION=$GENERATION OBSERVATION=$OBSERVATION \
      POLICY=$POLICY SEED=$seed SIZE=$SIZE VAL_PARALLEL=$VAL_PARALLEL TEST_SHARDS=$TEST_SHARDS SKIP_TRAIN=$SKIP \
      "$REPO/scripts/task_policy_pipeline.sh" $task $port > "$LOG/policy_${gen}_${task}_${POLICY}_seed$seed.log" 2>&1 \
      || fail "policy $gen/$task $POLICY seed $seed" "$LOG/policy_${gen}_${task}_${POLICY}_seed$seed.log" &
    port=$((port + 40)); sleep 30
  done
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

log "start: cluster $CLUSTER, GPU $GPU, jobs:$JOBS, seeds: $SEEDS, policies $POLICIES"
if [ -f "$LOG/FAILURES" ]; then mv "$LOG/FAILURES" "$LOG/FAILURES.$(date +%Y%m%d_%H%M%S)"; fi
if [ -f "$LOG/DONE" ]; then mv "$LOG/DONE" "$LOG/DONE.$(date +%Y%m%d_%H%M%S)"; fi
if [ "$SEQUENTIAL_TASKS" = 1 ]; then
  for one_job in $ALL_JOBS; do
    JOBS=$one_job
    IFS=: read -r gen image task <<< "$one_job"
    marker="$DATA_ROOT/retired/$gen/$task.json"
    if [ -f "$marker" ]; then
      python3 - "$marker" "$POLICIES" "$SEEDS" <<'PY' || exit 1
import json, sys
r = json.load(open(sys.argv[1]))
if r['status'] != 'removed' or not set(sys.argv[2].split()) <= set(r.get('policies', [])) or not set(map(int, sys.argv[3].split())) <= set(r.get('seeds', [])):
    raise SystemExit('Retired data cannot be reused for new/incomplete policies or seeds')
PY
      log "[$gen/$task] completed and retired, preserved results reused"; continue
    fi
    generation_lane
    if [ "${ONLY_GENERATION:-0}" != 1 ]; then policy_lane; fi
    # Persist completed task summaries before starting the next task. Raw data and weights stay in DATA_ROOT.
    if collect_bundle --no-tar && [ "${TMG_RETIRE_COMPLETED:-0}" = 1 ] && [ "${ONLY_GENERATION:-0}" != 1 ]; then
      python3 "$REPO/scripts/retire_isaaclab_data.py" task --root "$DATA_ROOT" --generator "$gen" --task "$task" \
        --policies $POLICIES --seeds $SEEDS >> "$LOG/storage.log" 2>&1 || fail "retire $gen/$task" "$LOG/storage.log"
    fi
  done
  JOBS=$ALL_JOBS
else
  generation_lane &
  if [ "${ONLY_GENERATION:-0}" = 1 ]; then wait; else policy_lane & wait; fi
fi
log "collecting results into $RESULTS_ROOT/reports/$BUNDLE_NAME"
collect_bundle && log "bundles: $RESULTS_ROOT/reports/$BUNDLE_NAME"
if [ -f "$LOG/FAILURES" ]; then log "FAILURES: $(tr '\n' ';' < "$LOG/FAILURES")"; exit 1; fi
echo "ISAACLAB_BASELINE_DONE $(date)" > "$LOG/DONE"; log "done"
