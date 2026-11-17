#!/usr/bin/env bash
set -eu
cd /home/gpuadmin/workspace/jihoonkwon/soft_source_demo_pilot_20261112/annotation_20261113
until test -s data/rope_annotations.json && test -s data/towel_annotations.json; do sleep 5; done
docker run --rm --name soft-source-pool-verify-20261113 --runtime=nvidia --gpus device=0 -e NVIDIA_VISIBLE_DEVICES=0 -e NVIDIA_DRIVER_CAPABILITIES=all --shm-size 8g -v "$PWD/data:/data" -v "$PWD/reader_check.py:/reader_check.py:ro" --entrypoint python tmg/isaaclab-soft:latest /reader_check.py > reader.log 2>&1
test -s data/reader_verified.json
