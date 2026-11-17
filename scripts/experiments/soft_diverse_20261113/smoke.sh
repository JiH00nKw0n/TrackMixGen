#!/usr/bin/env bash
set -eu
cd /home/gpuadmin/workspace/jihoonkwon/soft_source_demo_pilot_20261112/annotation_20261113
test -s data/reader_verified.json
for task in rope towel; do
    overrides="task.num_sources=10 generation.attempts=1 generation.chunk=1 cluster.workers=1 generation.chunk_retries=0"
    # The wider rope angles apply to evaluation; this input smoke test uses the
    # main generation scene for both tasks and has its own isolated output.
    make -s -C main generate DATA_ROOT="$PWD/data" ISAAC_GPU=0 IMAGE=isaaclab-soft GENERATOR=softmimicgen TASK="$task" CLUSTER=isaac SCENE=ours SELECTION=official METHOD=none GENERATION=test/pilot20 OVERRIDES="$overrides" > "smoke_${task}.log" 2>&1
done
