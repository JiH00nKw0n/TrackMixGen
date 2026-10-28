#!/bin/bash
# One task's policy stage for the baseline experiment: train BC-RNN (low-dim) on the task's dataset, validate every
# checkpoint on the 25 validation scenes (VAL_PARALLEL checkpoints at a time), select the checkpoint, evaluate it
# on the test scenes (evaluation.test_scenes, 300) in TEST_SHARDS parallel shards.
#   scripts/task_policy_pipeline.sh <task> <first port>      (the task uses ports <first port> .. <first port>+39:
#   validation ports <first port>+<checkpoint index>, test shards from <first port>+20)
# Variables: CLUSTER (a100), GPU (CUDA device for the train image, 1), GENERATOR (mimicgen), IMAGE (simulator image of the
# rollouts, robosuite), METHOD (none), GENERATION (pool_lowdim), OBSERVATION (low_dim), POLICY (bc_rnn), SEED (1), SCENE (ours),
# SIZE (dataset demos, 1000), VAL_PARALLEL (5), TEST_SHARDS (6), SKIP_TRAIN (1: the run is already trained).
set -uo pipefail
TASK=$1; PORT0=$2
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-a100}; GPU=${GPU:-1}; METHOD=${METHOD:-none}; GENERATION=${GENERATION:-pool_lowdim}
OBSERVATION=${OBSERVATION:-low_dim}; POLICY=${POLICY:-bc_rnn}; SEED=${SEED:-1}; SCENE=${SCENE:-ours}
VAL_PARALLEL=${VAL_PARALLEL:-5}; TEST_SHARDS=${TEST_SHARDS:-6}; GENERATOR=${GENERATOR:-mimicgen}; IMAGE=${IMAGE:-robosuite}; SIZE=${SIZE:-1000}
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
RUN=$DATA_ROOT/runs/$GENERATOR/$TASK/$SCENE/$METHOD/$GENERATION/$OBSERVATION/$POLICY/seed$SEED
DATASET_C=/data/datasets/$GENERATOR/$TASK/$SCENE/$METHOD/$GENERATION/$OBSERVATION/n${SIZE}_seed0.hdf5
export GENERATOR IMAGE
# CP-Gen evaluates in the generation environment rebuilt from its dataset's env_args (robosuite 1.5); the Isaac Lab
# generators build their environment from the task id and need nothing from the dataset
[ "$GENERATOR" = cpgen ] && export EXTRA="dataset=$DATASET_C ${EXTRA:-}"
C="GENERATOR=$GENERATOR TASK=$TASK SCENE=$SCENE METHOD=$METHOD GENERATION=$GENERATION OBSERVATION=$OBSERVATION POLICY=$POLICY SEED=$SEED CLUSTER=$CLUSTER OVERRIDES=dataset.size=$SIZE"
log() { echo "[$TASK $(date +%H:%M:%S)] $*"; }
validated() { [ -f "$RUN/validation.jsonl" ] && grep -q "/$1\"" "$RUN/validation.jsonl"; }
port_busy() { python3 -c "import socket,sys; s=socket.socket(); s.settimeout(1); sys.exit(0 if s.connect_ex(('127.0.0.1', int(sys.argv[1])))==0 else 1)" "$1"; }

log "train -> $RUN"
if [ "${SKIP_TRAIN:-0}" = 1 ]; then log "train skipped (SKIP_TRAIN=1)"; else
make -s -C "$REPO" train $C || { log "TRAIN FAILED"; exit 1; }
fi
log "validate checkpoints ($VAL_PARALLEL at a time)"
i=0; pids=""
for d in "$RUN"/checkpoints/[0-9]*/; do
  d=${d%/}; [ -f "$d/checkpoint.json" ] || continue
  validated "$(basename "$d")" && continue
  while [ "$(jobs -rp | wc -l)" -ge "$VAL_PARALLEL" ]; do wait -n || true; done
  # one port per checkpoint (never reused while another validation may still hold it): PORT0 .. PORT0+19; a port
  # another process holds (a shared machine) is skipped
  port=$((PORT0 + i)); i=$((i + 1))
  while port_busy "$port"; do log "port $port is busy, skipping it"; port=$((port + 1)); i=$((i + 1)); done
  [ "$i" -le 20 ] || { log "more than 20 checkpoints or busy ports; the port block of 40 is too small"; exit 1; }
  make -s -C "$REPO" validate RUN="$RUN" CKPT="$d" GENERATOR=$GENERATOR TASK=$TASK SCENE=$SCENE OBSERVATION=$OBSERVATION CLUSTER=$CLUSTER PORT=$port \
    > "$RUN/validate_$(basename "$d").log" 2>&1 || log "VALIDATE $(basename "$d") FAILED (see $RUN/validate_$(basename "$d").log)" &
  sleep 20      # stagger server start-ups
done
wait
for d in "$RUN"/checkpoints/[0-9]*/; do d=${d%/}; [ -f "$d/checkpoint.json" ] && ! validated "$(basename "$d")" && { log "checkpoint $(basename "$d") has no validation row"; exit 1; }; done
log "select checkpoint"
make -s -C "$REPO" select-checkpoint $C || { log "SELECT FAILED"; exit 1; }
SEL=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['checkpoint'])" "$RUN/SELECTED.json")
SEL_HOST="${SEL/#\/data/$DATA_ROOT}"
# the test shards need TEST_SHARDS consecutive free ports from PORT0+20
tport=$((PORT0 + 20))
while :; do ok=1; for p in $(seq "$tport" $((tport + TEST_SHARDS - 1))); do port_busy "$p" && { ok=0; break; }; done; [ "$ok" = 1 ] && break; tport=$((tport + 1)); done
log "evaluate $(basename "$SEL_HOST") on the test scenes ($TEST_SHARDS shards, ports $tport..$((tport + TEST_SHARDS - 1)))"
make -s -C "$REPO" evaluate RUN="$RUN" CKPT="$SEL_HOST" GENERATOR=$GENERATOR TASK=$TASK SCENE=$SCENE OBSERVATION=$OBSERVATION CLUSTER=$CLUSTER SPLIT=test \
  PORT=$tport SHARDS=$TEST_SHARDS || { log "EVALUATE FAILED"; exit 1; }
python3 - "$RUN" "$SEL_HOST" <<'PY'
import json, sys, pathlib
run, sel = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
folders = sorted(list((run / "eval").glob("test_h*/" + sel.name + "/eval_summary.json"))
                 + list((run / "eval").glob("test/" + sel.name + "/eval_summary.json")), key=lambda p: p.stat().st_mtime)
summary = json.load(open(folders[-1]))
b = summary["by_split"].get("test") or summary["all"]
print(f"RESULT {run.parts[-7]} {sel.name} horizon {summary.get('horizon')}: test success {b['n_success']}/{b['n']} = {b['success_rate']:.3f} "
      f"(95% CI {b['ci_low']:.3f}-{b['ci_high']:.3f})")
PY
log "done"
