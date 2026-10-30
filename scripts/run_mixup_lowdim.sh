#!/bin/bash
# Mixup exploration, generation stage (docs/MIXUP_EXPLORATION.md): low-dim pools of every mx_* condition for the tasks of
# one generator (GENERATOR=mimicgen | cpgen), scene ours, selection ours (every source; MimicGen picks one at random per
# subtask, CP-Gen one per attempt), then the SIZE-demonstration low-dim dataset of each pool. Training and evaluation are
# scripts/run_mixup_train.sh.
#
# Stage 1 (only for RANK_TASKS): the 3,000-attempt ranking pool (method none, generation rank_human_demos) of each task,
#   all tasks at once, then `tmg rank_sources`, which writes configs/experiment/generator/<generator>/ranking/<task>.yaml in
#   this checkout (the mixers and top-K read it). A ranking pool that is already finished (its DONE.json exists) is only
#   ranked. RANK_WAIT=1: never build a ranking pool here; wait until another process has finished it, then rank it (the CP-Gen
#   ranking pools are built by their own script on the planner GPU). A task whose ranking file already holds a 3,000-attempt
#   ranking is skipped.
# Stage 2: for every task and condition, one pool (generation pool_robosuite_lowdim) extended by 25 % steps until it holds
#   SIZE successes (a pool that already holds them, for example from the earlier 1,000-demonstration plan, is not generated again), then its low-dim dataset of SIZE demonstrations. One pool at a time, each with the cluster's workers
#   (CP-Gen: CPGEN_WORKERS chunk processes, because every chunk runs cuRobo on the GPU).
#
#   CLUSTER=a100 scripts/run_mixup_lowdim.sh
#   CLUSTER=a100 GPU=6 GENERATOR=cpgen RANK_WAIT=1 scripts/run_mixup_lowdim.sh
# Variables: CLUSTER (a100), GENERATOR (mimicgen), GPU (CUDA device of CP-Gen's planner; unset: the cluster default), TASKS
#   (MimicGen: hammer_cleanup mug_cleanup three_piece_assembly coffee threading square stack_three; CP-Gen: coffee threading
#   mug_cleanup hammer_cleanup three_piece_assembly), RANK_TASKS (MimicGen: coffee threading square stack_three; CP-Gen: all
#   its tasks), RANK_WAIT (0), CONDITIONS (the six conditions of the 2026-10-30 decision: mx_none mx_top1 mx_top5
#   mx_success_a40 mx_transform mx_junction; the other mx_* files stay for the pools already made), SIZE (500, decided
#   2026-10-30; the first attempt counts scale with it), RANK_WORKERS (12 per ranking pool), CPGEN_WORKERS
#   (24), SKIP_DATASET (1: pools only), MINI (1: toy scale for an end-to-end check: a MINI_ATTEMPTS-attempt pool (200) in
#   chunks of MINI_CHUNK (50) in the folder pilot50 without images, MINI_SIZE-demonstration datasets (10), never extended;
#   any committed or hand-made ranking file is accepted).
# Resumable: finished pools and datasets are kept and skipped; a failed step is logged in FAILURES and the next one runs.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-a100}; SIZE=${SIZE:-500}; GENERATOR=${GENERATOR:-mimicgen}
CONDITIONS=${CONDITIONS:-"mx_none mx_top1 mx_top5 mx_success_a40 mx_transform mx_junction"}
RANK_WORKERS=${RANK_WORKERS:-12}; RANK_WAIT=${RANK_WAIT:-0}
SCENE=ours; SELECTION=ours; GENERATION=pool_robosuite_lowdim; POOL_OVERRIDES=""
if [ "$GENERATOR" = cpgen ]; then
  IMAGE=cpgen; LOG_NAME=mixup_lowdim_cpgen
  TASKS=${TASKS:-"coffee threading mug_cleanup hammer_cleanup three_piece_assembly"}
  RANK_TASKS=${RANK_TASKS-$TASKS}
  # attempts for 1,000 successes (scaled to SIZE below): Coffee gave 23.5 % in the 2026-10-29 smoke pool; the other tasks start
  # at 5,000, as in the CP-Gen baseline driver
  declare -A ATTEMPTS=( [coffee]=4750 [threading]=5000 [mug_cleanup]=5000 [hammer_cleanup]=5000 [three_piece_assembly]=5000 )
  POOL_OVERRIDES="cluster.max_workers=${CPGEN_WORKERS:-24}"; RANK_WORKERS=${CPGEN_WORKERS:-24}
  [ -n "${GPU:-}" ] && export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU
else
  IMAGE=robosuite; LOG_NAME=mixup_lowdim
  TASKS=${TASKS:-"hammer_cleanup mug_cleanup three_piece_assembly coffee threading square stack_three"}
  RANK_TASKS=${RANK_TASKS-"coffee threading square stack_three"}
  # attempts for 1,000 successes (scaled to SIZE below): about 1.1x the attempts that gave 1,000 successes in the 2026-10
  # low-dim pools
  declare -A ATTEMPTS=( [coffee]=2500 [threading]=6250 [hammer_cleanup]=7000 [mug_cleanup]=8750 [stack_three]=9500 [square]=15000 [three_piece_assembly]=17000 )
fi
for task in "${!ATTEMPTS[@]}"; do ATTEMPTS[$task]=$(( (ATTEMPTS[$task] * SIZE / 1000 + 249) / 250 * 250 )); done   # whole chunks of 250
if [ "${MINI:-0}" = 1 ]; then
  GENERATION=test/pilot50; SIZE=${MINI_SIZE:-10}; LOG_NAME=${LOG_NAME}_mini
  POOL_OVERRIDES="generation.chunk=${MINI_CHUNK:-50} generation.protocol.camera_names=[] $POOL_OVERRIDES"
  for t in $TASKS; do ATTEMPTS[$t]=${MINI_ATTEMPTS:-200}; done
fi
GEN_NAME=${GENERATION##*/}
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
LOG=$DATA_ROOT/logs/$LOG_NAME; mkdir -p "$LOG"
log() { echo "[mixup $(date '+%F %T')] $*" | tee -a "$LOG/driver.log"; }
fail() { log "$1 FAILED (see $2)"; echo "$1" >> "$LOG/FAILURES"; }
M() { make -s -C "$REPO" "$@"; }
common() { echo "GENERATOR=$GENERATOR IMAGE=$IMAGE TASK=$1 SCENE=$SCENE SELECTION=$SELECTION CLUSTER=$CLUSTER"; }
ranking_file() { echo "$REPO/configs/experiment/generator/$GENERATOR/ranking/$1.yaml"; }
has_ranking() {   # a 3,000-attempt ranking file (MINI: any ranking file, so a check can run before the ranking pools end)
  if [ "${MINI:-0}" = 1 ]; then [ -f "$(ranking_file $1)" ]; else grep -q "3000 attempts" "$(ranking_file $1)" 2>/dev/null; fi
}
rank_pool_done() { [ -f "$DATA_ROOT/pools/$GENERATOR/$1/$SCENE/$SELECTION/none/rank_human_demos/DONE.json" ]; }
pool_successes() {   # successes recorded in the pool's DONE.json, 0 when unfinished
  python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('successes', 0))" \
    "$DATA_ROOT/pools/$GENERATOR/$1/$SCENE/$SELECTION/$2/$GEN_NAME/DONE.json" 2>/dev/null || echo 0
}

log "start: generator $GENERATOR, tasks [$TASKS], ranking first for [$RANK_TASKS] (wait for other builders: $RANK_WAIT), conditions [$CONDITIONS], size $SIZE, generation $GENERATION, code $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo archive)"

# stage 1: ranking pools in parallel, then the rankings
rank_task() {
  local t=$1
  if has_ranking $t; then log "[$t] ranking exists: $(ranking_file $t)"; return 0; fi
  M sources $(common $t) > "$LOG/sources_$t.log" 2>&1 || { fail "ranking $t (sources)" "$LOG/sources_$t.log"; return 1; }
  if ! rank_pool_done $t; then
    if [ "$RANK_WAIT" = 1 ]; then
      log "[$t] waiting for its ranking pool (built elsewhere)"
      until rank_pool_done $t; do sleep 600; done
    else
      M generate $(common $t) METHOD=none GENERATION=rank_human_demos OVERRIDES="cluster.max_workers=$RANK_WORKERS" > "$LOG/rank_pool_$t.log" 2>&1 \
        || { fail "ranking $t" "$LOG/rank_pool_$t.log"; return 1; }
    fi
  fi
  M rank-sources $(common $t) GENERATION=rank_human_demos > "$LOG/rank_sources_$t.log" 2>&1 \
    || { fail "ranking $t (rank_sources)" "$LOG/rank_sources_$t.log"; return 1; }
  log "[$t] ranking written: $(grep -m1 -o '3000 attempts' "$(ranking_file $t)" || echo '?')"
}
for t in $RANK_TASKS; do rank_task $t & done
wait

# stage 2: condition pools (extended until SIZE successes) and datasets
for t in $TASKS; do
  if ! has_ranking $t; then fail "pools $t (no 3,000-attempt ranking in $(ranking_file $t))" "$LOG/driver.log"; continue; fi
  M sources $(common $t) > "$LOG/sources_$t.log" 2>&1 || { fail "sources $t" "$LOG/sources_$t.log"; continue; }
  for m in $CONDITIONS; do
    attempts=${ATTEMPTS[$t]}
    if [ "$(pool_successes $t $m)" -ge "$SIZE" ]; then
      log "[$t/$m] pool already holds $(pool_successes $t $m) successes, generation skipped"   # never re-plan a finished pool smaller
    else while :; do
      log "[$t/$m] pool: $attempts attempts"
      M generate $(common $t) METHOD=$m GENERATION=$GENERATION OVERRIDES="generation.attempts=$attempts $POOL_OVERRIDES" >> "$LOG/pool_${t}_$m.log" 2>&1 \
        || { fail "pool $t/$m" "$LOG/pool_${t}_$m.log"; break; }
      succ=$(pool_successes $t $m)
      log "[$t/$m] pool: $succ successes in $attempts attempts"
      [ "$succ" -ge "$SIZE" ] && break
      [ "${MINI:-0}" = 1 ] && { fail "pool $t/$m (MINI: fewer than $SIZE successes in $attempts attempts)" "$LOG/pool_${t}_$m.log"; break; }
      attempts=$(( (attempts * 5 / 4 + 249) / 250 * 250 ))      # 25 % more, whole chunks of 250
    done; fi
    [ "$(pool_successes $t $m)" -ge "$SIZE" ] || continue
    [ "${SKIP_DATASET:-0}" = 1 ] && continue
    M dataset $(common $t) METHOD=$m GENERATION=$GENERATION OBSERVATION=low_dim OVERRIDES="size=$SIZE" > "$LOG/dataset_${t}_$m.log" 2>&1 \
      && log "[$t/$m] dataset written" || fail "dataset $t/$m" "$LOG/dataset_${t}_$m.log"
  done
done
log "done"
