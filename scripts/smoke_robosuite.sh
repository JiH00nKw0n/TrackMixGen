#!/bin/bash
# Smoke test of one robosuite generator setting end to end: sources, a 200-attempt image pool (generation test/smoke200),
# the fixed evaluation scenes, a 10-demonstration dataset, a 200-update policy, validation and test on 6 scenes each.
# Same code path and folders as the real runs; about 30 to 60 minutes on a machine with many cores.
#   CLUSTER=a100 GPU=5 GENERATOR=mimicgen IMAGE=robosuite TASK=coffee SCENE=official_d1 SELECTION=official scripts/smoke_robosuite.sh
#   CLUSTER=a100 GPU=5 GENERATOR=cpgen IMAGE=cpgen TASK=coffee SCENE=official_wide scripts/smoke_robosuite.sh
# Variables: CLUSTER (local), GPU (0), GENERATOR (mimicgen), IMAGE (robosuite), TASK (coffee), SCENE (ours), SELECTION (official),
#   METHOD (none), POLICY (diffusion_policy), SEED (202601), PORT (first of 40 server ports, 8900), SIZE (10), UPDATES (200), SCENES (6).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-local}; GPU=${GPU:-0}; GENERATOR=${GENERATOR:-mimicgen}; IMAGE=${IMAGE:-robosuite}; TASK=${TASK:-coffee}
SCENE=${SCENE:-ours}; SELECTION=${SELECTION:-official}; METHOD=${METHOD:-none}; POLICY=${POLICY:-diffusion_policy}; SEED=${SEED:-202601}
PORT=${PORT:-8900}; SIZE=${SIZE:-10}; UPDATES=${UPDATES:-200}; SCENES=${SCENES:-6}
GENERATION=test/smoke200; GEN_NAME=smoke200; OBSERVATION=image
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
RUNTIME_DATA_ROOT=$(make -s -C "$REPO" print-runtime-data-root CLUSTER="$CLUSTER")
TAG="${GENERATOR}_${TASK}_${SCENE}_${SELECTION}"
LOG=$DATA_ROOT/logs/smoke/$TAG; mkdir -p "$LOG"
log() { echo "[smoke $TAG $(date '+%F %T')] $*" | tee -a "$LOG/driver.log"; }
fail() { log "$1 FAILED (see $2)"; echo "SMOKE_FAILED $TAG $1" >> "$LOG/driver.log"; exit 1; }
G="GENERATOR=$GENERATOR IMAGE=$IMAGE TASK=$TASK SCENE=$SCENE SELECTION=$SELECTION CLUSTER=$CLUSTER"
DATASET=$DATA_ROOT/datasets/$GENERATOR/$TASK/$SCENE/$SELECTION/$METHOD/$GEN_NAME/$OBSERVATION/n${SIZE}_seed0.hdf5
DATASET_C=$RUNTIME_DATA_ROOT/datasets/$GENERATOR/$TASK/$SCENE/$SELECTION/$METHOD/$GEN_NAME/$OBSERVATION/n${SIZE}_seed0.hdf5

log "start: $G method $METHOD policy $POLICY seed $SEED"
log "sources"; make -s -C "$REPO" sources $G > "$LOG/sources.log" 2>&1 || fail sources "$LOG/sources.log"
log "pool: 200 attempts ($GENERATION)"
make -s -C "$REPO" generate $G METHOD=$METHOD GENERATION=$GENERATION > "$LOG/pool.log" 2>&1 || fail pool "$LOG/pool.log"
log "$(grep -h 'pool complete' "$LOG/pool.log" | tail -1 | cut -c1-120)"
log "dataset ($SIZE demonstrations, image)"
make -s -C "$REPO" dataset $G METHOD=$METHOD GENERATION=$GENERATION OBSERVATION=$OBSERVATION OVERRIDES="size=$SIZE" > "$LOG/dataset.log" 2>&1 || fail dataset "$LOG/dataset.log"
log "fixed scenes"
RESET_OVERRIDES=""; [ "$GENERATOR" = cpgen ] && RESET_OVERRIDES="dataset=$DATASET_C"     # CP-Gen rebuilds its environment from the dataset's env_args
make -s -C "$REPO" resets $G OVERRIDES="$RESET_OVERRIDES" > "$LOG/resets.log" 2>&1 || fail resets "$LOG/resets.log"
log "policy: $POLICY, $UPDATES updates, $SCENES validation and test scenes (ports $PORT..$((PORT + 39)))"
CLUSTER=$CLUSTER GPU=$GPU GENERATOR=$GENERATOR IMAGE=$IMAGE SCENE=$SCENE SELECTION=$SELECTION METHOD=$METHOD GENERATION=$GENERATION OBSERVATION=$OBSERVATION \
  POLICY=$POLICY SEED=$SEED SIZE=$SIZE VAL_PARALLEL=1 TEST_SHARDS=2 \
  TRAIN_OVERRIDES="training.by_policy.$POLICY.image.updates=$UPDATES training.updates_per_epoch=100 training.checkpoint_every_epochs=2" EXTRA="max_scenes=$SCENES" \
  "$REPO/scripts/task_policy_pipeline.sh" $TASK $PORT > "$LOG/policy.log" 2>&1 || fail policy "$LOG/policy.log"
grep -h "^RESULT" "$LOG/policy.log" | tee -a "$LOG/driver.log"
log "done"; echo "SMOKE_DONE $TAG" >> "$LOG/driver.log"
