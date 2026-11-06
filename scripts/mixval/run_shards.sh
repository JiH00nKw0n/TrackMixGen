#!/bin/bash
# Start (or resume) every shard of a blend-ratio run as its own container.
#   scripts/mixval/run_shards.sh <mimicgen|cpgen> <run name> ["<gpu list>"] [first shard] [last shard]
# Needs DATA_ROOT (host data root, mounted at /data) and REPO (this checkout, mounted at /opt/tmg). The run folder
# /data/mixval/<task dir>/<run name> (task dir threading_mg for MimicGen, threading for CP-Gen) holds plan.json,
# template.json (a pool chunk spec of the task), mg_config.json (MimicGen only), scenes.txt (the scenes file inside the
# container) and shard_XX/. MimicGen runs on the CPU (robosuite image); CP-Gen needs a GPU for cuRobo, shard k on
# GPUS[k % len]. A shard whose container is already running is left alone; a finished shard exits at once.
set -euo pipefail
GEN=$1; RUN=$2; GPUS=(${3:-}); FIRST=${4:-0}
: "${DATA_ROOT:?}" "${REPO:?}"
case $GEN in
  mimicgen) TASKDIR=threading_mg; IMAGE=tmg/robosuite:latest; MODULE=tmg.mixval.mimicgen_entrypoint ;;
  cpgen)    TASKDIR=threading;    IMAGE=tmg/cpgen:latest;     MODULE=tmg.mixval.cpgen_entrypoint ;;
  *) echo "generator must be mimicgen or cpgen"; exit 2 ;;
esac
DIR=$DATA_ROOT/mixval/$TASKDIR/$RUN; C=/data/mixval/$TASKDIR/$RUN
N=$(ls -d $DIR/shard_* | wc -l); LAST=${5:-$((N - 1))}
SCENES=$(cat $DIR/scenes.txt)
ARGS="--template $C/template.json --scenes $SCENES"
[ $GEN = mimicgen ] && ARGS="$ARGS --mg-config $C/mg_config.json"
for k in $(seq $FIRST $LAST); do
  s=$(printf %02d $k); name=mixval_${RUN}_s$s
  docker ps --format '{{.Names}}' | grep -qx $name && { echo "$name already running"; continue; }
  if [ $GEN = cpgen ]; then
    gpu=${GPUS[$((k % ${#GPUS[@]}))]}
    EXTRA="--gpus device=$gpu -v $DATA_ROOT/cpgen_writable/curobo_task:/opt/src/curobo/src/curobo/content/configs/task
      -v $DATA_ROOT/cpgen_writable/mimicgen_mugs:/opt/src/mimicgen/mimicgen/models/robosuite/assets/shapenet_core/mugs"
  else
    gpu=cpu; EXTRA=""
  fi
  docker run -d --rm --name $name --network host $EXTRA -v $DATA_ROOT:/data -v $REPO:/opt/tmg \
    -e TMG_DATA_ROOT=/data -e HOME=/tmp -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 \
    --user $(id -u):$(id -g) --entrypoint bash $IMAGE -c \
    "python -m $MODULE $ARGS --shard $C/shard_$s >> $C/shard_$s/log.txt 2>&1" >/dev/null
  echo "$name ($gpu)"
  [ $GEN = cpgen ] && sleep 20 || sleep 1
done
