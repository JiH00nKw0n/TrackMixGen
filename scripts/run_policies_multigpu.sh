#!/bin/bash
# Baseline policies of one generator on one machine with several GPUs: every task x {Diffusion Policy, BC-RNN} x 3 seeds,
# each one scripts/task_policy_pipeline.sh run (train, validate every checkpoint on the 25 validation scenes, select,
# test on the 300 test scenes) pinned to one GPU. A job starts when its task's dataset and fixed scenes exist (the
# generation driver writes them) and a GPU has room. Generalizes scripts/run_dex_policies_multigpu.sh (the AWS T4 run,
# whose defaults are kept here) to other generators and machines.
#
# Scheduling: Diffusion Policy jobs first (the long ones), then BC-RNN, at most PER_GPU jobs per GPU. While a process
# matching GEN_PATTERN runs (the generation driver), at most PER_GPU_DURING_GEN jobs per GPU, and with DP_ONLY_DURING_GEN=1
# only Diffusion Policy (on the T4 machine RAM cannot hold Isaac Sim evaluations beside the generation). A job starts only
# when MIN_FREE_GB of RAM are available. GPU_WAIT holds GPUs back until a log line appears, e.g.
# "6:/home/jihoonkwon/logs/rank_cpgen_fast.log:RANK_ALL_DONE" (A100 GPU 6 joins after the CP-Gen ranking).
#
#   CLUSTER=aws scripts/run_policies_multigpu.sh                                    (DexMimicGen on the AWS T4s)
#   CLUSTER=a100 GENERATOR=cpgen IMAGE=cpgen GENERATION=pool_robosuite_image TASKS="..." GPUS="5 6" PER_GPU=3 \
#     PER_GPU_DURING_GEN=3 DP_ONLY_DURING_GEN=0 MIN_FREE_GB=100 VAL_PARALLEL=3 TEST_SHARDS=4 DP_OVERRIDES="" \
#     GEN_PATTERN="run_mimicgen_baseline[.]sh" GPU_WAIT="6:/home/jihoonkwon/logs/rank_cpgen_fast.log:RANK_ALL_DONE" \
#     scripts/run_policies_multigpu.sh                                             (CP-Gen on A100 GPUs 5 and 6)
# Variables: CLUSTER (aws), GENERATOR (dexmimicgen), IMAGE (isaaclab-dex), GENERATION (pool_isaac), GPUS ("0 1 2 3"),
#   TASKS ("microwave bottle_door pick_place"), SEEDS ("202601 202602 202603"), SIZE (1000), PER_GPU (2),
#   PER_GPU_DURING_GEN (1), DP_ONLY_DURING_GEN (1), GEN_PATTERN ("run_dex_generation_multigpu[.]sh"), MIN_FREE_GB (40),
#   VAL_PARALLEL (1), TEST_SHARDS (2), PORT_BASE (8800), DP_OVERRIDES ("training.amp=true"), BC_OVERRIDES (""), GPU_WAIT ("").
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-aws}; GPUS=${GPUS:-"0 1 2 3"}; TASKS=${TASKS:-"microwave bottle_door pick_place"}
GEN=${GENERATOR:-dexmimicgen}; IMAGE=${IMAGE:-isaaclab-dex}; GENERATION=${GENERATION:-pool_isaac}
GEN_PATTERN=${GEN_PATTERN:-"run_dex_generation_multigpu[.]sh"}; DP_ONLY_DURING_GEN=${DP_ONLY_DURING_GEN:-1}; GPU_WAIT=${GPU_WAIT:-}
SEEDS=${SEEDS:-"202601 202602 202603"}; SIZE=${SIZE:-1000}; PER_GPU=${PER_GPU:-2}; PER_GPU_DURING_GEN=${PER_GPU_DURING_GEN:-1}
MIN_FREE_GB=${MIN_FREE_GB:-40}; VAL_PARALLEL=${VAL_PARALLEL:-1}; TEST_SHARDS=${TEST_SHARDS:-2}; PORT_BASE=${PORT_BASE:-8800}
DP_OVERRIDES=${DP_OVERRIDES-"training.amp=true"}; BC_OVERRIDES=${BC_OVERRIDES-""}
SCENE=ours; SELECTION=official; METHOD=none; OBSERVATION=image
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
LOG=$DATA_ROOT/logs/${GEN}_policies; mkdir -p "$LOG"
log() { echo "[driver $(date -u '+%F %T')] $*" | tee -a "$LOG/driver.log"; }
dataset_of() { echo "$DATA_ROOT/datasets/$GEN/$1/$SCENE/$SELECTION/$METHOD/$GENERATION/$OBSERVATION/n${SIZE}_seed0.hdf5"; }
resets_of() {   # robosuite resets store states (.hdf5); the Isaac Lab ones are a seed list (.json)
  local f=$DATA_ROOT/resets/$GEN/$1/$SCENE/val25_test300_seed202600
  [ -f "$f.hdf5" ] && echo "$f.hdf5" || echo "$f.json"; }
run_of() { echo "$DATA_ROOT/runs/$GEN/$1/$SCENE/$SELECTION/$METHOD/$GENERATION/$OBSERVATION/$2/seed$3"; }
gen_running() { pgrep -f "$GEN_PATTERN" > /dev/null; }
gpu_ready() {   # GPU_WAIT entries "gpu:file:text": that GPU only once the file holds the text
  local e g f txt
  for e in $GPU_WAIT; do IFS=: read -r g f txt <<< "$e"; [ "$g" = "$1" ] && ! grep -q "$txt" "$f" 2>/dev/null && return 1; done
  return 0; }
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
  CLUSTER=$CLUSTER GPU=$g ISAAC_GPU=$g GENERATOR=$GEN IMAGE=$IMAGE SCENE=$SCENE SELECTION=$SELECTION METHOD=$METHOD \
    GENERATION=$GENERATION OBSERVATION=$OBSERVATION POLICY=$p SEED=$s SIZE=$SIZE VAL_PARALLEL=$VAL_PARALLEL TEST_SHARDS=$TEST_SHARDS \
    SKIP_TRAIN=$skip TRAIN_OVERRIDES="$overrides" EXTRA="" \
    "$REPO/scripts/task_policy_pipeline.sh" $t $((PORT_BASE + 40 * k)) > "${LOGF[$k]}" 2>&1 &
  PID_GPU[$!]=$g; PID_JOB[$!]=$k; STARTED[$k]=1
}

log "start: $GEN on GPUs $GPUS${GPU_WAIT:+ (held: $GPU_WAIT)}, tasks $TASKS, seeds $SEEDS, $PER_GPU jobs per GPU ($PER_GPU_DURING_GEN while generating), DP '$DP_OVERRIDES'"
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
    if gen_running; then limit=$PER_GPU_DURING_GEN; [ "$DP_ONLY_DURING_GEN" = 1 ] && [ "$p" != diffusion_policy ] && continue; fi
    [ "$(free_gb)" -ge "$MIN_FREE_GB" ] || continue
    best=""; bestn=99
    for g in $GPUS; do gpu_ready $g || continue; n=$(load_of $g); [ "$n" -lt "$limit" ] && [ "$n" -lt "$bestn" ] && { best=$g; bestn=$n; }; done
    [ -n "$best" ] || continue
    start_job $k $best; pending=$((pending - 1)); sleep 60     # one start at a time, so RAM and GPU readings settle
  done
  [ "$pending" = 0 ] && [ "${#PID_GPU[@]}" = 0 ] && break
  sleep 60
done
log "done"
