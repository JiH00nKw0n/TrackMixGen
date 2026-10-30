#!/bin/bash
# DexMimicGen baseline generation on one machine with several GPUs (AWS g4dn.12xlarge, 4x T4, 2026-10-30): for every task,
# one pool shared by one container per GPU (generation.shard, Isaac Sim renders on every GPU a container sees), then the
# fixed evaluation scenes and the 1,000-demonstration image dataset, in the background while the next task generates.
#
# Measured on the T4s (2026-10-30, test/pilot50 settings with cameras): two generation processes per GPU give the most
# attempts per hour (Pick Place about 147 per GPU per hour, Microwave about 266) at 8.3 GB of the 16 GB; a third adds
# nothing, and eight processes use up to 125 GB of the 186 GB RAM. Isaac Sim takes about 9 minutes to start, so a chunk
# is long (one chunk per process).
#
# Pool sizes: Pick Place keeps the official `guarantee` (attempts count successes): 1,000 successes in chunks of 125.
# Microwave and Bottle Door run a fixed number of attempts officially (AutoData `guarantee_success: false`); at their
# measured success rates (Microwave 75 % on the T4s, Bottle Door 55 % in the Isaac pilot) 1,000 attempts give fewer than
# the 1,000 successes a dataset needs, so they get 1,504 and 2,200 attempts.
#
#   CLUSTER=aws scripts/run_dex_generation_multigpu.sh
# Variables: CLUSTER (aws), GPUS ("0 1 2 3"), WORKERS (2 per GPU), TASKS ("microwave bottle_door pick_place": fastest first
#   so its training can start early), SIZE (1000).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-aws}; GPUS=${GPUS:-"0 1 2 3"}; WORKERS=${WORKERS:-2}; SIZE=${SIZE:-1000}
TASKS=${TASKS:-"microwave bottle_door pick_place"}
declare -A POOL=( [microwave]="generation.attempts=1504 generation.chunk=188"
                  [bottle_door]="generation.attempts=2200 generation.chunk=275"
                  [pick_place]="generation.attempts=1000 generation.chunk=125" )
C="GENERATOR=dexmimicgen IMAGE=isaaclab-dex SCENE=ours METHOD=none SELECTION=official GENERATION=pool_isaac CLUSTER=$CLUSTER"
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
LOG=$DATA_ROOT/logs/dex_generation; mkdir -p "$LOG"
log() { echo "[driver $(date -u '+%F %T')] $*" | tee -a "$LOG/driver.log"; }
n=$(set -- $GPUS; echo $#)
after_pool() {   # resets on the first GPU and the dataset, while the next task generates
  local t=$1 g=${GPUS%% *}
  ISAAC_GPU=$g make -s -C "$REPO" resets $C TASK=$t > "$LOG/resets_$t.log" 2>&1 || { log "[$t] resets FAILED"; echo "resets $t" >> "$LOG/FAILURES"; return 1; }
  make -s -C "$REPO" dataset $C TASK=$t OBSERVATION=image OVERRIDES="size=$SIZE" > "$LOG/dataset_$t.log" 2>&1 \
    || { log "[$t] dataset FAILED"; echo "dataset $t" >> "$LOG/FAILURES"; return 1; }
  log "[$t] resets and dataset written"
}
log "start: GPUs $GPUS, $WORKERS processes per GPU, tasks $TASKS"
for t in $TASKS; do
  log "[$t] pool: ${POOL[$t]}, $n shards"
  i=0; shards=""
  for g in $GPUS; do
    # the brackets are quoted for the shell that runs the make recipe
    ISAAC_GPU=$g make -s -C "$REPO" generate $C TASK=$t \
      OVERRIDES="${POOL[$t]} 'generation.shard=[$i,$n]' cluster.max_workers=$WORKERS" > "$LOG/pool_${t}_gpu$g.log" 2>&1 &
    shards="$shards $!"; i=$((i + 1)); sleep ${SHARD_STAGGER:-30}
  done
  wait $shards                       # only the shards: the previous task's resets and dataset keep running
  pool=$DATA_ROOT/pools/dexmimicgen/$t/ours/official/none/pool_isaac
  if [ -f "$pool/DONE.json" ]; then
    log "[$t] pool done: $(python3 -c "import json; d=json.load(open('$pool/DONE.json')); print(d['successes'], 'successes of', d['attempts'], 'attempts')")"
    after_pool $t &
  else
    log "[$t] pool FAILED (no DONE.json; see $LOG/pool_${t}_gpu*.log)"; echo "pool $t" >> "$LOG/FAILURES"
  fi
done
wait
log "done"
