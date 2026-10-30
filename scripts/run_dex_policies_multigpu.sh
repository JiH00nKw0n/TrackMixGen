#!/bin/bash
# DexMimicGen baseline policies on one machine with several GPUs (AWS g4dn.12xlarge, 4x T4, 2026-10-30): every task x
# {Diffusion Policy, BC-RNN} x 3 seeds, each one scripts/task_policy_pipeline.sh run (train, validate every checkpoint on
# the 25 validation scenes, select, test on the 300 test scenes) pinned to one GPU. A job starts when its task's dataset
# and fixed scenes exist (scripts/run_dex_generation_multigpu.sh writes them) and a GPU has room.
#
# Measured on the T4s (2026-10-30): Diffusion Policy trains at about 58 examples/s in float32 (batch 64: about 30.6 h for
# the 100,000 updates), twice as fast with mixed precision at an unchanged loss curve; BC-RNN at about 100 examples/s
# (batch 16: about 4.4 h). Diffusion Policy therefore trains with training.amp=true here (docs/GENERATORS.md, DexMimicGen
# training on the T4s); BC-RNN keeps float32. RAM limits the machine more than GPU memory: an Isaac Sim process takes
# about 15 GB of the 186 GB and a training process holds its dataset's images in RAM.
#
# Scheduling: Diffusion Policy jobs first (the long ones), then BC-RNN. While the generation driver runs, at most
# PER_GPU_DURING_GEN jobs per GPU and only Diffusion Policy, whose training outlasts the generation, so no evaluation
# (Isaac Sim) starts beside the eight generation processes; afterwards PER_GPU jobs per GPU. A job starts only when
# MIN_FREE_GB of RAM are available. Each job validates one checkpoint at a time and tests in TEST_SHARDS shards.
#
#   CLUSTER=aws scripts/run_dex_policies_multigpu.sh
# Variables: CLUSTER (aws), GPUS ("0 1 2 3"), TASKS ("microwave bottle_door pick_place"), SEEDS ("202601 202602 202603"),
#   SIZE (1000), PER_GPU (2), PER_GPU_DURING_GEN (1), MIN_FREE_GB (40), VAL_PARALLEL (1), TEST_SHARDS (2), PORT_BASE (8800),
#   DP_OVERRIDES ("training.amp=true"), BC_OVERRIDES ("").
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-aws}; GPUS=${GPUS:-"0 1 2 3"}; TASKS=${TASKS:-"microwave bottle_door pick_place"}
SEEDS=${SEEDS:-"202601 202602 202603"}; SIZE=${SIZE:-1000}; PER_GPU=${PER_GPU:-2}; PER_GPU_DURING_GEN=${PER_GPU_DURING_GEN:-1}
MIN_FREE_GB=${MIN_FREE_GB:-40}; VAL_PARALLEL=${VAL_PARALLEL:-1}; TEST_SHARDS=${TEST_SHARDS:-2}; PORT_BASE=${PORT_BASE:-8800}
DP_OVERRIDES=${DP_OVERRIDES-"training.amp=true"}; BC_OVERRIDES=${BC_OVERRIDES-""}
GEN=dexmimicgen; SCENE=ours; SELECTION=official; METHOD=none; GENERATION=pool_isaac; OBSERVATION=image
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
LOG=$DATA_ROOT/logs/dex_policies; mkdir -p "$LOG"
log() { echo "[driver $(date -u '+%F %T')] $*" | tee -a "$LOG/driver.log"; }
dataset_of() { echo "$DATA_ROOT/datasets/$GEN/$1/$SCENE/$SELECTION/$METHOD/$GENERATION/$OBSERVATION/n${SIZE}_seed0.hdf5"; }
resets_of() { echo "$DATA_ROOT/resets/$GEN/$1/$SCENE/val25_test300_seed202600.json"; }
run_of() { echo "$DATA_ROOT/runs/$GEN/$1/$SCENE/$SELECTION/$METHOD/$GENERATION/$OBSERVATION/$2/seed$3"; }
gen_running() { pgrep -f "run_dex_generation_multigpu[.]sh" > /dev/null; }
free_gb() { awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo 2>/dev/null || echo 999; }

JOBS=()   # "task policy seed", Diffusion Policy first
for p in diffusion_policy bc_rnn; do for t in $TASKS; do for s in $SEEDS; do JOBS+=("$t $p $s"); done; done; done
declare -A PID_GPU=() PID_JOB=() STARTED=() LOGF=()
load_of() { local g=$1 n=0 pid; for pid in "${!PID_GPU[@]}"; do [ "${PID_GPU[$pid]}" = "$g" ] && n=$((n + 1)); done; echo $n; }

start_job() {   # index gpu
  local k=$1 g=$2 t p s run skip overrides
  read -r t p s <<< "${JOBS[$k]}"
  LOGF[$k]=$LOG/policy_${t}_${p}_seed$s.log
  run=$(run_of $t $p $s); skip=0; [ -f "$run/train_result.json" ] && skip=1
  overrides=$BC_OVERRIDES; [ "$p" = diffusion_policy ] && overrides=$DP_OVERRIDES
  log "[$t $p seed $s] start on GPU $g (ports $((PORT_BASE + 40 * k))..$((PORT_BASE + 40 * k + 39)), skip train $skip, overrides '$overrides')"
  CLUSTER=$CLUSTER GPU=$g ISAAC_GPU=$g GENERATOR=$GEN IMAGE=isaaclab-dex SCENE=$SCENE SELECTION=$SELECTION METHOD=$METHOD \
    GENERATION=$GENERATION OBSERVATION=$OBSERVATION POLICY=$p SEED=$s SIZE=$SIZE VAL_PARALLEL=$VAL_PARALLEL TEST_SHARDS=$TEST_SHARDS \
    SKIP_TRAIN=$skip TRAIN_OVERRIDES="$overrides" EXTRA="" \
    "$REPO/scripts/task_policy_pipeline.sh" $t $((PORT_BASE + 40 * k)) > "${LOGF[$k]}" 2>&1 &
  PID_GPU[$!]=$g; PID_JOB[$!]=$k; STARTED[$k]=1
}

log "start: GPUs $GPUS, tasks $TASKS, seeds $SEEDS, $PER_GPU jobs per GPU ($PER_GPU_DURING_GEN while generating), DP '$DP_OVERRIDES'"
while :; do
  for pid in "${!PID_GPU[@]}"; do      # reap finished jobs
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; code=$?; k=${PID_JOB[$pid]}
      [ $code = 0 ] && log "[${JOBS[$k]}] done: $(grep -h '^RESULT' "${LOGF[$k]}" | tail -1)" \
                    || { log "[${JOBS[$k]}] FAILED ($code)"; echo "${JOBS[$k]}" >> "$LOG/FAILURES"; }
      unset "PID_GPU[$pid]" "PID_JOB[$pid]"
    fi
  done
  pending=0
  for k in "${!JOBS[@]}"; do
    [ -n "${STARTED[$k]:-}" ] && continue
    read -r t p s <<< "${JOBS[$k]}"
    if compgen -G "$(run_of $t $p $s)/eval/test_h*/*/eval_summary.json" > /dev/null; then STARTED[$k]=1; log "[${JOBS[$k]}] test result exists, skipped"; continue; fi
    if ! { [ -f "$(dataset_of $t)" ] && [ -f "$(resets_of $t)" ]; }; then
      if ! gen_running; then STARTED[$k]=1; log "[${JOBS[$k]}] no dataset or scenes and generation has stopped, skipped"; echo "${JOBS[$k]} (no dataset)" >> "$LOG/FAILURES"; continue; fi
      pending=$((pending + 1)); continue
    fi
    pending=$((pending + 1))
    limit=$PER_GPU
    if gen_running; then limit=$PER_GPU_DURING_GEN; [ "$p" = diffusion_policy ] || continue; fi
    [ "$(free_gb)" -ge "$MIN_FREE_GB" ] || continue
    best=""; bestn=99
    for g in $GPUS; do n=$(load_of $g); [ "$n" -lt "$limit" ] && [ "$n" -lt "$bestn" ] && { best=$g; bestn=$n; }; done
    [ -n "$best" ] || continue
    start_job $k $best; pending=$((pending - 1)); sleep 60     # one start at a time, so RAM and GPU readings settle
  done
  [ "$pending" = 0 ] && [ "${#PID_GPU[@]}" = 0 ] && break
  sleep 60
done
log "done"
