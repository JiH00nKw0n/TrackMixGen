#!/bin/bash
# Mixup exploration, training stage (docs/MIXUP_EXPLORATION.md): low-dim Diffusion Policy on every mx_* dataset that
# scripts/run_mixup_lowdim.sh writes, for the generators in GENERATORS, three training seeds each. Every run is
# scripts/task_policy_pipeline.sh: train (100,000 updates), validate every checkpoint on the 25 validation scenes, select,
# test on the 300 test scenes.
#
# The script waits for the datasets: it sweeps the list of runs every 5 minutes and starts a run as soon as its dataset
# exists, POLICY_PARALLEL runs at a time on GPU, each with its own block of 40 server ports from PORT_BASE. A run whose
# test result exists is skipped (a rerun resumes). A run is dropped when the generation driver logged its pool or dataset
# as FAILED, or finished ("done") without writing its dataset. The fixed evaluation scenes of a task are drawn once, when
# its first dataset appears (CP-Gen draws them in the environment rebuilt from that dataset's env_args); a resets file that
# already exists, for example from the baseline driver, is used as it is.
#
#   CLUSTER=a100 GPU=6 scripts/run_mixup_train.sh
# Variables: CLUSTER (a100), GPU (0), GENERATORS ("mimicgen cpgen"), CONDITIONS (the six conditions of the 2026-10-30 decision: mx_none mx_top1 mx_top5
#   mx_success_a40 mx_transform mx_junction; the other mx_* files stay for the pools already made), SEEDS ("202601 202602
#   202603"), SIZE (500, decided 2026-10-30), POLICY_PARALLEL (2), VAL_PARALLEL (4), TEST_SHARDS (6), PORT_BASE (12000), MINI (1: the toy scale
#   of run_mixup_lowdim.sh MINI: pilot50 datasets of MINI_SIZE (10) demonstrations, 200 updates, 6 evaluation scenes per split).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-a100}; GPU=${GPU:-0}; SIZE=${SIZE:-500}
GENERATORS=${GENERATORS:-"mimicgen cpgen"}
CONDITIONS=${CONDITIONS:-"mx_none mx_top1 mx_top5 mx_success_a40 mx_transform mx_junction"}
SEEDS=${SEEDS:-"202601 202602 202603"}
POLICY=diffusion_policy; OBSERVATION=low_dim; SCENE=ours; SELECTION=ours; GENERATION=pool_robosuite_lowdim
POLICY_PARALLEL=${POLICY_PARALLEL:-2}; VAL_PARALLEL=${VAL_PARALLEL:-4}; TEST_SHARDS=${TEST_SHARDS:-6}; PORT_BASE=${PORT_BASE:-12000}
TRAIN_OVERRIDES=${TRAIN_OVERRIDES:-}; EVAL_EXTRA=${EVAL_EXTRA:-}; LOG_NAME=mixup_train; SUFFIX=""
if [ "${MINI:-0}" = 1 ]; then
  GENERATION=test/pilot50; SIZE=${MINI_SIZE:-10}; VAL_PARALLEL=1; TEST_SHARDS=2; LOG_NAME=mixup_train_mini; SUFFIX=_mini
  TRAIN_OVERRIDES="training.by_policy.diffusion_policy.low_dim.updates=200 training.updates_per_epoch=100 training.checkpoint_every_epochs=2 $TRAIN_OVERRIDES"
  EVAL_EXTRA="max_scenes=6 $EVAL_EXTRA"
fi
GEN_NAME=${GENERATION##*/}
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
RUNTIME_DATA_ROOT=$(make -s -C "$REPO" print-runtime-data-root CLUSTER="$CLUSTER")
LOG=$DATA_ROOT/logs/$LOG_NAME; mkdir -p "$LOG"
log() { echo "[train $(date '+%F %T')] $*" | tee -a "$LOG/driver.log"; }
M() { make -s -C "$REPO" "$@"; }
tasks_of() {
  if [ "$1" = cpgen ]; then echo "coffee threading mug_cleanup hammer_cleanup three_piece_assembly"
  else echo "hammer_cleanup mug_cleanup three_piece_assembly coffee threading square stack_three"; fi
}
image_of() { [ "$1" = cpgen ] && echo cpgen || echo robosuite; }
gen_log() { [ "$1" = cpgen ] && echo "$DATA_ROOT/logs/mixup_lowdim_cpgen$SUFFIX" || echo "$DATA_ROOT/logs/mixup_lowdim$SUFFIX"; }
dataset_of() { echo "$DATA_ROOT/datasets/$1/$2/$SCENE/$SELECTION/$3/$GEN_NAME/$OBSERVATION/n${SIZE}_seed0.hdf5"; }
resets_of() { echo "$DATA_ROOT/resets/$1/$2/$SCENE/val25_test300_seed202600.hdf5"; }
run_of() { echo "$DATA_ROOT/runs/$1/$2/$SCENE/$SELECTION/$3/$GEN_NAME/$OBSERVATION/$POLICY/seed$4"; }
tested() { compgen -G "$(run_of "$@")/eval/test_h*/*/eval_summary.json" > /dev/null; }
generation_gave_up() {   # the generation driver failed this pool or dataset, or finished without its dataset
  local glog; glog=$(gen_log $1)
  grep -qx "pool $2/$3.*\|dataset $2/$3\|pools $2 .*\|sources $2" "$glog/FAILURES" 2>/dev/null && return 0
  tail -n 1 "$glog/driver.log" 2>/dev/null | grep -q "\] done$" && [ ! -f "$(dataset_of $1 $2 $3)" ]
}

ensure_resets() {   # draw the task's fixed evaluation scenes once (lock folder against a parallel draw)
  local g=$1 t=$2 d=$3 lock="$LOG/resets_$1_$2.lock"
  [ -f "$(resets_of $g $t)" ] && return 0
  mkdir "$lock" 2>/dev/null || { while [ -d "$lock" ]; do sleep 30; done; [ -f "$(resets_of $g $t)" ]; return; }
  local overrides=""
  [ "$g" = cpgen ] && overrides="dataset=${d/#$DATA_ROOT/$RUNTIME_DATA_ROOT}"
  log "[$g/$t] drawing the evaluation scenes"
  M resets GENERATOR=$g IMAGE=$(image_of $g) TASK=$t SCENE=$SCENE CLUSTER=$CLUSTER OVERRIDES="$overrides" > "$LOG/resets_${g}_$t.log" 2>&1
  local rc=$?; rmdir "$lock"
  [ $rc = 0 ] && [ -f "$(resets_of $g $t)" ] || { log "[$g/$t] resets FAILED (see $LOG/resets_${g}_$t.log)"; echo "resets $g/$t" >> "$LOG/FAILURES"; return 1; }
}

declare -a SLOT_PID
for k in $(seq 0 $((POLICY_PARALLEL - 1))); do SLOT_PID[$k]=0; done
free_slot() {   # index of a slot whose run has ended, waiting until one has
  while :; do
    for k in "${!SLOT_PID[@]}"; do
      local p=${SLOT_PID[$k]}
      if [ "$p" = 0 ] || ! kill -0 "$p" 2>/dev/null; then echo "$k"; return; fi
    done
    sleep 30
  done
}

declare -A STARTED DROPPED
pending=()
for g in $GENERATORS; do for t in $(tasks_of $g); do for m in $CONDITIONS; do for s in $SEEDS; do pending+=("$g $t $m $s"); done; done; done; done
log "start: generators [$GENERATORS], conditions [$CONDITIONS], seeds [$SEEDS], ${#pending[@]} runs, GPU $GPU, $POLICY_PARALLEL at a time, ports from $PORT_BASE, generation $GENERATION, size $SIZE${MINI:+, MINI mode}"
while :; do
  left=0
  for job in "${pending[@]}"; do
    [ -n "${STARTED[$job]:-}${DROPPED[$job]:-}" ] && continue
    read -r g t m s <<< "$job"
    if tested $g $t $m $s; then STARTED[$job]=done; continue; fi
    d=$(dataset_of $g $t $m)
    if [ ! -f "$d" ]; then
      if generation_gave_up $g $t $m; then DROPPED[$job]=1; log "[$g/$t/$m seed $s] dropped: generation failed or ended without the dataset"
      else left=$((left + 1)); fi
      continue
    fi
    ensure_resets $g $t "$d" || { DROPPED[$job]=1; continue; }
    k=$(free_slot); port=$((PORT_BASE + 40 * k))
    run=$(run_of $g $t $m $s); skip=0; [ -f "$run/train_result.json" ] && skip=1
    log "[$g/$t/$m seed $s] train, validate, select, test (slot $k, ports $port..$((port + 39)))"
    CLUSTER=$CLUSTER GPU=$GPU GENERATOR=$g IMAGE=$(image_of $g) SCENE=$SCENE SELECTION=$SELECTION METHOD=$m GENERATION=$GENERATION \
      OBSERVATION=$OBSERVATION POLICY=$POLICY SEED=$s SIZE=$SIZE VAL_PARALLEL=$VAL_PARALLEL TEST_SHARDS=$TEST_SHARDS SKIP_TRAIN=$skip \
      TRAIN_OVERRIDES="$TRAIN_OVERRIDES" EXTRA="$EVAL_EXTRA" \
      "$REPO/scripts/task_policy_pipeline.sh" $t $port > "$LOG/run_${g}_${t}_${m}_seed$s.log" 2>&1 \
      || echo "run $g/$t/$m seed $s" >> "$LOG/FAILURES" &
    SLOT_PID[$k]=$!; STARTED[$job]=1
    sleep 30
  done
  [ "$left" = 0 ] && break
  sleep 300
done
wait
log "all runs started have ended; failures: $(cat "$LOG/FAILURES" 2>/dev/null | tr '\n' ';')"
log "done"
