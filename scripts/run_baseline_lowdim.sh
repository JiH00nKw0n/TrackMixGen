#!/bin/bash
# Baseline experiment on one machine: generator mimicgen, method none, scene ours, the seven tasks, low-dim
# observations, BC-RNN, one GPU.
#   Stage A (CPU): source demonstrations, then POOL_LANES pools at a time (each with the cluster's worker count):
#            a pool per task without images (attempts sized for about 1,100 successes at the measured success
#            rates), followed by that task's fixed evaluation scenes and its 1,000-demonstration dataset.
#   Stage B (GPU): every task at once through scripts/task_policy_pipeline.sh (train, validate checkpoints in
#            parallel, select, test in parallel shards), each task with its own block of 40 server ports.
# Everything is resumable: finished pools, datasets and checkpoints are kept and skipped on a rerun.
#   CLUSTER=a100 GPU=1 POOL_LANES=4 scripts/run_baseline_lowdim.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-a100}; GPU=${GPU:-1}; POOL_LANES=${POOL_LANES:-4}; SCENE=ours; METHOD=none; GENERATION=pool_robosuite_lowdim
SELECTION=${SELECTION:-ours}   # the 2026-10 exploration pools: a source chosen at random for every subtask
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
LOG=$DATA_ROOT/logs/baseline_lowdim; mkdir -p "$LOG"
TASKS="three_piece_assembly square stack_three mug_cleanup hammer_cleanup threading coffee"   # longest pools first
declare -A ATTEMPTS=( [coffee]=2500 [threading]=6250 [hammer_cleanup]=7000 [mug_cleanup]=8750 [stack_three]=9500 [square]=15000 [three_piece_assembly]=17000 )
log() { echo "[driver $(date +%H:%M:%S)] $*" | tee -a "$LOG/driver.log"; }

pool_lane() {   # one task: pool, resets, dataset
  local t=$1
  make -s -C "$REPO" generate TASK=$t SCENE=$SCENE SELECTION=$SELECTION METHOD=$METHOD GENERATION=$GENERATION CLUSTER=$CLUSTER \
    OVERRIDES="generation.attempts=${ATTEMPTS[$t]}" > "$LOG/pool_$t.log" 2>&1 || { log "pool $t FAILED"; echo "pool $t" >> "$LOG/FAILURES"; return 1; }
  log "pool $t done: $(grep -h 'pool complete' "$LOG/pool_$t.log" | tail -1 | cut -c1-80)"
  make -s -C "$REPO" resets TASK=$t SCENE=$SCENE CLUSTER=$CLUSTER > "$LOG/resets_$t.log" 2>&1 || { log "resets $t FAILED"; echo "resets $t" >> "$LOG/FAILURES"; }
  make -s -C "$REPO" dataset TASK=$t SCENE=$SCENE SELECTION=$SELECTION METHOD=$METHOD GENERATION=$GENERATION OBSERVATION=low_dim CLUSTER=$CLUSTER \
    OVERRIDES="size=1000" > "$LOG/dataset_$t.log" 2>&1 || { log "dataset $t FAILED"; echo "dataset $t" >> "$LOG/FAILURES"; }
}

log "stage A: sources, then pools in $POOL_LANES lanes ($CLUSTER)"
for t in $TASKS; do
  make -s -C "$REPO" sources TASK=$t CLUSTER=$CLUSTER > "$LOG/sources_$t.log" 2>&1 || { log "sources $t FAILED"; echo "sources $t" >> "$LOG/FAILURES"; }
done
log "sources done"
for t in $TASKS; do
  while [ "$(jobs -rp | wc -l)" -ge "$POOL_LANES" ]; do wait -n || true; done
  log "pool $t: ${ATTEMPTS[$t]} attempts"
  pool_lane $t &
done
wait
log "stage A done"

TRAIN_SIF=$DATA_ROOT/images/tmg_train.sif
until [ -f "$TRAIN_SIF" ]; do log "waiting for $TRAIN_SIF"; sleep 300; done
log "stage B: policies of every task at once on GPU $GPU"
port=8800
for t in $TASKS; do
  [ -f "$DATA_ROOT/datasets/mimicgen/$t/$SCENE/$SELECTION/$METHOD/${GENERATION##*/}/low_dim/n1000_seed0.hdf5" ] || { log "no dataset for $t, skipped"; continue; }
  CLUSTER=$CLUSTER GPU=$GPU SELECTION=$SELECTION "$REPO/scripts/task_policy_pipeline.sh" $t $port > "$LOG/policy_$t.log" 2>&1 &
  port=$((port + 40))
  sleep 60     # stagger the dataset loads and model builds
done
wait
grep -h "^RESULT" "$LOG"/policy_*.log | tee -a "$LOG/driver.log"
log "stage B done"; echo "BASELINE_DONE $(date)" > "$LOG/DONE"
