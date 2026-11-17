#!/bin/bash
set -eu -o pipefail
cd "$(dirname "$0")"
task=$1
case "$task" in rope) gpu=0; port=18500;; towel) gpu=2; port=19500;; *) exit 2;; esac
H=/scratch/t651a06/work/soft_diverse_20261113
R=/home/gpuadmin/workspace/jihoonkwon/soft_source_demo_pilot_20261112/annotation_20261113
h() { ssh -o BatchMode=yes -S "$HOME/.ssh/cm-hangang" hangang "$@"; }
i() { ssh -o BatchMode=yes isaac "$@"; }
dir=datasets/softmimicgen/$task/diverse20/official/none/pool_isaac/image
file=n1000_seed0.hdf5
until i "test -s '$R/data/$dir/n1000_seed0.DATASET.json' && test -s '$R/data/$dir/$file'"; do sleep 60; done
hash=$(i "sha256sum '$R/data/$dir/$file'" | cut -d' ' -f1)
until h "mkdir -p '$H/data/$dir'"; do sleep 60; done
until (i "tar -C '$R/data/$dir' -cf - ." | h "mkdir -p '$H/data/$dir.incoming' && tar -C '$H/data/$dir.incoming' -xf - && cd '$H/data/$dir.incoming' && echo '$hash  $file' | sha256sum -c - && mv * '$H/data/$dir/'"); do sleep 60; done
for policy in diffusion_policy bc_rnn; do
 for seed in 202601 202602 202603; do
  key="${task}_${policy}_${seed}"
  run="runs/softmimicgen/$task/diverse20/official/none/pool_isaac/image/$policy/seed$seed"
  if ! test -s "$key.job"; then
   id=$(h "cd '$H' && sbatch --parsable --job-name='sd_${task}_${seed}' -o '$H/logs/$key-%j.out' train.sbatch '$task' '$policy' '$seed'"); echo "$id" > "$key.job"
  fi
  until h "test -f '$H/logs/finished_$key'"; do
    state=$(h "sacct -nX -j $(cat "$key.job") -o State -P" 2>/dev/null || true)
    case "$state" in *FAILED*|*TIMEOUT*|*CANCELLED*|*OUT_OF_MEMORY*) echo "$key $state"; exit 1;; esac
    sleep 60
  done
  i "mkdir -p '$R/data/$run'"
  until (h "cd '$H/data/$run' && tar --exclude=training_state.pt -cf - ." | i "tar -C '$R/data/$run' -xf -"); do sleep 60; done
  i "cd '$R' && nohup flock '$R/logs/eval_$task.lock' env SKIP_TRAIN=1 TMG_DATA_ROOT='$R/data' CLUSTER=isaac GPU=$gpu ISAAC_GPU=$gpu GENERATOR=softmimicgen IMAGE=isaaclab-soft SCENE=diverse20 SELECTION=official METHOD=none GENERATION=pool_isaac OBSERVATION=image POLICY=$policy SEED=$seed SIZE=1000 VAL_PARALLEL=1 TEST_SHARDS=2 bash branch/scripts/task_policy_pipeline.sh $task $port >logs/eval_$key.log 2>&1 </dev/null &" </dev/null
 done
done
