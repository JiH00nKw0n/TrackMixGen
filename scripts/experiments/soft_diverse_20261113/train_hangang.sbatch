#!/bin/bash
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --comment=field=robotics;appl=pytorch
set -eu
cd /scratch/t651a06/work/soft_diverse_20261113
export TMG_DATA_ROOT="$PWD/data" PYTHONPATH="$PWD/branch" HF_HUB_OFFLINE=1
task=$1; policy=$2; seed=$3
/scratch/t651a06/work/tmg/native/envs/train/bin/python -m tmg train generator=softmimicgen task="$task" scene=diverse20 task.num_sources=10 selection=official method=none generation=pool_isaac cluster.name=hangang observation=image policy="$policy" seed="$seed" dataset.size=1000
touch "logs/finished_${task}_${policy}_${seed}"
