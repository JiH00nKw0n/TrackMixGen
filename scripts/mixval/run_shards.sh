#!/bin/bash
# Start (or resume) every shard of a blend-ratio run as its own CP-Gen container, shard k on GPU GPUS[k % len].
#   scripts/mixval/run_shards.sh <run name> "<gpu list>" [first shard] [last shard]
# Needs DATA_ROOT (host data root, mounted at /data) and REPO (this checkout, mounted at /opt/tmg). The run folder
# /data/mixval/threading/<run name> holds plan.json, template.json, the scenes file path in scenes.txt and shard_XX/.
# A shard whose container is already running is left alone; a finished shard exits at once (nothing left).
set -euo pipefail
RUN=$1; GPUS=($2); FIRST=${3:-0}
: "${DATA_ROOT:?}" "${REPO:?}"
DIR=$DATA_ROOT/mixval/threading/$RUN
N=$(ls -d $DIR/shard_* | wc -l); LAST=${4:-$((N - 1))}
SCENES=$(cat $DIR/scenes.txt)
for k in $(seq $FIRST $LAST); do
  s=$(printf %02d $k); name=mixval_${RUN}_s$s; gpu=${GPUS[$((k % ${#GPUS[@]}))]}
  docker ps --format '{{.Names}}' | grep -qx $name && { echo "$name already running"; continue; }
  docker run -d --rm --name $name --network host --gpus device=$gpu \
    -v $DATA_ROOT:/data -v $REPO:/opt/tmg \
    -v $DATA_ROOT/cpgen_writable/curobo_task:/opt/src/curobo/src/curobo/content/configs/task \
    -v $DATA_ROOT/cpgen_writable/mimicgen_mugs:/opt/src/mimicgen/mimicgen/models/robosuite/assets/shapenet_core/mugs \
    -e TMG_DATA_ROOT=/data -e HOME=/tmp -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
    --user $(id -u):$(id -g) --entrypoint bash tmg/cpgen:latest -c \
    "python -m tmg.mixval.entrypoint --template /data/mixval/threading/$RUN/template.json --shard /data/mixval/threading/$RUN/shard_$s --scenes $SCENES >> /data/mixval/threading/$RUN/shard_$s/log.txt 2>&1" >/dev/null
  echo "$name on GPU $gpu"
  sleep 20      # stagger the cuRobo warm-ups
done
