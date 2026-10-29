#!/bin/bash
# Evaluate one checkpoint: start policy server(s) in the train image, play the fixed scenes in the simulator image,
# stop the servers. Host-side glue (both containers need the same data root). Variables:
#   RUN    run folder on the host (…/runs/<generator>/<task>/<scene>/<method>/<generation>/<observation>/<policy>/seed<k>)
#   CKPT   checkpoint folder on the host (RUN/checkpoints/<update>); default: the newest one
#   TASK, SCENE, OBSERVATION (image|low_dim, as trained), CLUSTER, SPLIT (val|test), EXTRA (extra tmg eval overrides)
#   GENERATOR (mimicgen), IMAGE (simulator image of the rollouts: robosuite, or cpgen for CP-Gen data)
#   PORT   first server port (default 8765); SHARDS (default 1): play the scenes in SHARDS parallel shards, each with
#          its own server on PORT, PORT+1, ... (the servers keep per-episode state, so one server serves one shard)
# The validation split appends a row to RUN/validation.jsonl (what `tmg select_checkpoint` reads).
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
: "${RUN:?RUN=<run folder> is required}"
: "${TASK:?TASK=<generator/task> is required}"
SCENE=${SCENE:-ours}; OBSERVATION=${OBSERVATION:-image}; CLUSTER=${CLUSTER:-local}; SPLIT=${SPLIT:-val}; GENERATOR=${GENERATOR:-mimicgen}
PORT=${PORT:-8765}; SHARDS=${SHARDS:-1}; EXTRA=${EXTRA:-}; IMAGE=${IMAGE:-robosuite}
if [ "$CLUSTER" = pod ] && [ -z "${TMG_RESOURCE_OWNER:-}" ] && [ -f "${TMG_RESULTS_ROOT:-}/reports/evaluation_capacity/selected.json" ]; then
  exec python3 "$REPO/scripts/pod_resource_guard.py" --kind eval --task "$TASK" --evaluations "$SHARDS" -- bash "$0"
fi
CKPT=${CKPT:-$(for d in "$RUN"/checkpoints/[0-9]*/; do [ -f "$d/checkpoint.json" ] && echo "${d%/}"; done | sort | tail -1)}
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
RUNTIME_DATA_ROOT=$(make -s -C "$REPO" print-runtime-data-root CLUSTER="$CLUSTER")
in_container() { echo "${1/#$DATA_ROOT/$RUNTIME_DATA_ROOT}"; }
CKPT_C=$(in_container "$CKPT"); RUN_C=$(in_container "$RUN")
TRAINING_SEED=$(basename "$RUN" | sed -E 's/^seed//')
PORTS=$(seq "$PORT" $((PORT + SHARDS - 1)))

STARTED_PORTS=""
stop_servers() { for p in $STARTED_PORTS; do make -s -C "$REPO" serve-stop SERVER_NAME="tmg-serve-$p" CLUSTER="$CLUSTER" >/dev/null 2>&1 || true; done; }
trap stop_servers EXIT
echo "[evaluate] $SHARDS server(s) for $CKPT on port(s) $(echo $PORTS | tr '\n' ' ')"
for p in $PORTS; do
  make -s -C "$REPO" serve-detached CLUSTER="$CLUSTER" CKPT="$CKPT_C" PORT="$p" SERVER_NAME="tmg-serve-$p"
  STARTED_PORTS="$STARTED_PORTS $p"
done
for p in $PORTS; do
  for i in $(seq 1 90); do
    if make -s -C "$REPO" serve-ready CLUSTER="$CLUSTER" PORT="$p" >/dev/null 2>&1; then break; fi
    sleep 2
    [ "$i" = 90 ] && { echo "[evaluate] server on port $p did not come up"; exit 1; }
  done
done
APPEND=""
[ "$SPLIT" = val ] && APPEND="append_validation=$RUN_C"
if [ "$SHARDS" -gt 1 ]; then
  k=0; pids=""
  for p in $PORTS; do
    make -s -C "$REPO" eval CLUSTER="$CLUSTER" IMAGE="$IMAGE" GENERATOR="$GENERATOR" TASK="$TASK" SCENE="$SCENE" OBSERVATION="$OBSERVATION" \
      OVERRIDES="checkpoint=$CKPT_C split=$SPLIT training_seed=$TRAINING_SEED evaluation.server.port=$p shard=[$k,$SHARDS] $EXTRA" &
    pids="$pids $!"; k=$((k + 1))
  done
  fail=0; for pid in $pids; do wait "$pid" || fail=1; done
  [ "$fail" = 0 ] || { echo "[evaluate] a shard failed"; exit 1; }
fi
# the merge run (every scene already played when sharded) writes the summary and the validation row
make -s -C "$REPO" eval CLUSTER="$CLUSTER" IMAGE="$IMAGE" GENERATOR="$GENERATOR" TASK="$TASK" SCENE="$SCENE" OBSERVATION="$OBSERVATION" \
  OVERRIDES="checkpoint=$CKPT_C split=$SPLIT training_seed=$TRAINING_SEED evaluation.server.port=$PORT $APPEND $EXTRA"
