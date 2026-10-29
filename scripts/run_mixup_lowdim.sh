#!/bin/bash
# Mixup exploration, generation stage (docs/MIXUP_EXPLORATION.md): low-dim pools of every mx_* condition for the
# MimicGen tasks, scene ours, selection ours (every source, one random pick per subtask), then the 1,000-demonstration
# low-dim dataset of each pool. Training and evaluation are a later stage.
#
# Stage 1 (only for RANK_TASKS): the 3,000-attempt ranking pool (method none, generation rank_human_demos) of each
#   task, all tasks at once, then `tmg rank_sources`, which writes configs/experiment/generator/mimicgen/ranking/<task>.yaml
#   in this checkout (the mixers and top-K read it). Tasks not in RANK_TASKS must already have a 3,000-attempt ranking.
# Stage 2: for every task and condition, one pool (generation pool_robosuite_lowdim) extended by 25 % steps until it holds
#   SIZE successes, then its low-dim dataset of SIZE demonstrations. One pool at a time, each with the cluster's workers.
#
#   CLUSTER=a100 scripts/run_mixup_lowdim.sh
# Variables: CLUSTER (a100), TASKS (hammer_cleanup mug_cleanup three_piece_assembly coffee threading square stack_three),
#   RANK_TASKS (coffee threading square stack_three), CONDITIONS (the ten mx_* files), SIZE (1000), WORKERS (cluster
#   max_workers), RANK_WORKERS (12 per ranking pool), SKIP_DATASET (1: pools only).
# Resumable: finished pools and datasets are kept and skipped; a failed step is logged in FAILURES and the next one runs.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-a100}; SIZE=${SIZE:-1000}
TASKS=${TASKS:-"hammer_cleanup mug_cleanup three_piece_assembly coffee threading square stack_three"}
RANK_TASKS=${RANK_TASKS-"coffee threading square stack_three"}
CONDITIONS=${CONDITIONS:-"mx_none mx_top1 mx_top3 mx_top5 mx_success_a20 mx_success_a40 mx_success_a90 mx_success_anone mx_transform mx_junction"}
RANK_WORKERS=${RANK_WORKERS:-12}
GENERATOR=mimicgen; IMAGE=robosuite; SCENE=ours; SELECTION=ours; GENERATION=pool_robosuite_lowdim
# first attempt count per task: about 1.1x the attempts that gave 1,000 successes in the 2026-10 low-dim pools
declare -A ATTEMPTS=( [coffee]=2500 [threading]=6250 [hammer_cleanup]=7000 [mug_cleanup]=8750 [stack_three]=9500 [square]=15000 [three_piece_assembly]=17000 )
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
LOG=$DATA_ROOT/logs/mixup_lowdim; mkdir -p "$LOG"
log() { echo "[mixup $(date '+%F %T')] $*" | tee -a "$LOG/driver.log"; }
fail() { log "$1 FAILED (see $2)"; echo "$1" >> "$LOG/FAILURES"; }
M() { make -s -C "$REPO" "$@"; }
common() { echo "GENERATOR=$GENERATOR IMAGE=$IMAGE TASK=$1 SCENE=$SCENE SELECTION=$SELECTION CLUSTER=$CLUSTER"; }
pool_successes() {   # successes recorded in the pool's DONE.json, 0 when unfinished
  python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('successes', 0))" \
    "$DATA_ROOT/pools/$GENERATOR/$1/$SCENE/$SELECTION/$2/$GENERATION/DONE.json" 2>/dev/null || echo 0
}

log "start: tasks [$TASKS], ranking first for [$RANK_TASKS], conditions [$CONDITIONS], size $SIZE, code $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo archive)"

# stage 1: ranking pools in parallel, then the rankings
for t in $RANK_TASKS; do
  ( M sources $(common $t) > "$LOG/sources_$t.log" 2>&1 \
    && M generate $(common $t) METHOD=none GENERATION=rank_human_demos OVERRIDES="cluster.max_workers=$RANK_WORKERS" > "$LOG/rank_pool_$t.log" 2>&1 \
    && M rank-sources $(common $t) GENERATION=rank_human_demos > "$LOG/rank_sources_$t.log" 2>&1 \
    && log "[$t] ranking written: $(grep -m1 -o '3000 attempts' "$REPO/configs/experiment/generator/$GENERATOR/ranking/$t.yaml" || echo '?')" ) \
    || fail "ranking $t" "$LOG/rank_pool_$t.log" &
done
wait

# stage 2: condition pools (extended until SIZE successes) and datasets
for t in $TASKS; do
  if ! grep -q "3000 attempts" "$REPO/configs/experiment/generator/$GENERATOR/ranking/$t.yaml" 2>/dev/null; then
    fail "pools $t (no 3,000-attempt ranking in configs/experiment/generator/$GENERATOR/ranking/$t.yaml)" "$LOG/driver.log"; continue
  fi
  M sources $(common $t) > "$LOG/sources_$t.log" 2>&1 || { fail "sources $t" "$LOG/sources_$t.log"; continue; }
  for m in $CONDITIONS; do
    attempts=${ATTEMPTS[$t]}
    while :; do
      log "[$t/$m] pool: $attempts attempts"
      M generate $(common $t) METHOD=$m GENERATION=$GENERATION OVERRIDES="generation.attempts=$attempts" >> "$LOG/pool_${t}_$m.log" 2>&1 \
        || { fail "pool $t/$m" "$LOG/pool_${t}_$m.log"; break; }
      succ=$(pool_successes $t $m)
      log "[$t/$m] pool: $succ successes in $attempts attempts"
      [ "$succ" -ge "$SIZE" ] && break
      attempts=$(( (attempts * 5 / 4 + 249) / 250 * 250 ))      # 25 % more, whole chunks of 250
    done
    [ "$(pool_successes $t $m)" -ge "$SIZE" ] || continue
    [ "${SKIP_DATASET:-0}" = 1 ] && continue
    M dataset $(common $t) METHOD=$m GENERATION=$GENERATION OBSERVATION=low_dim OVERRIDES="size=$SIZE" > "$LOG/dataset_${t}_$m.log" 2>&1 \
      && log "[$t/$m] dataset written" || fail "dataset $t/$m" "$LOG/dataset_${t}_$m.log"
  done
done
log "done"
