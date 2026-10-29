#!/bin/bash
# CP-Gen reproduction check on one task (Coffee) in our setting: scene ours, image observations, Diffusion Policy,
# one GPU. Steps: sources, a 20-attempt pilot (graft hash check, attempt time and success rate), the pool, the
# dataset, the fixed scenes drawn in the generation environment, then the policy stage (train, validate, select,
# test). CP-Gen plans with cuRobo on the GPU, so the pool runs WORKERS chunk processes on the one GPU.
#   CLUSTER=a100 GPU=2 WORKERS=4 ATTEMPTS=2500 scripts/run_cpgen_coffee.sh
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-a100}; GPU=${GPU:-2}; WORKERS=${WORKERS:-4}; ATTEMPTS=${ATTEMPTS:-2500}; SIZE=${SIZE:-1000}
TASK=coffee; SCENE=ours; METHOD=none; GENERATION=pool_robosuite_image; OBSERVATION=image; POLICY=${POLICY:-diffusion_policy}; PORT=${PORT:-9200}
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
LOG=$DATA_ROOT/logs/cpgen_coffee; mkdir -p "$LOG"
log() { echo "[cpgen $(date +%H:%M:%S)] $*" | tee -a "$LOG/driver.log"; }
G="GENERATOR=cpgen IMAGE=cpgen CLUSTER=$CLUSTER TASK=$TASK SCENE=$SCENE"

log "sources"
make -s -C "$REPO" sources $G > "$LOG/sources.log" 2>&1 || { log "sources FAILED"; exit 1; }
log "pilot: 20 attempts"
make -s -C "$REPO" generate $G METHOD=$METHOD GENERATION=test/pilot20 OVERRIDES="cluster.max_workers=$WORKERS" > "$LOG/pilot20.log" 2>&1 \
  || { log "pilot FAILED (see $LOG/pilot20.log)"; exit 1; }
python3 - "$DATA_ROOT/pools/cpgen/$TASK/$SCENE/$METHOD/pilot20" <<'PY' | tee -a "$LOG/driver.log"
import glob, json, sys
rows = [json.loads(l) for f in glob.glob(sys.argv[1] + "/chunks/*/records/attempts.jsonl") for l in open(f) if l.strip()]
el = [r.get("elapsed_s", 0) for r in rows]
print(f"pilot: {len(rows)} attempts, {sum(1 for r in rows if r.get('success'))} successes, "
      f"mean steps {sum(r['n_steps'] for r in rows)/max(len(rows),1):.0f}, mean seconds per attempt {sum(el)/max(len(el),1):.1f}")
PY
log "pool: $ATTEMPTS attempts with images, $WORKERS workers on GPU $GPU"
make -s -C "$REPO" generate $G METHOD=$METHOD GENERATION=$GENERATION OVERRIDES="generation.attempts=$ATTEMPTS cluster.max_workers=$WORKERS" > "$LOG/pool.log" 2>&1 \
  || { log "pool FAILED (see $LOG/pool.log)"; exit 1; }
log "$(grep -h 'pool complete' "$LOG/pool.log" | tail -1 | cut -c1-100)"
log "dataset ($SIZE demos, image)"
make -s -C "$REPO" dataset $G METHOD=$METHOD GENERATION=$GENERATION OBSERVATION=$OBSERVATION OVERRIDES="size=$SIZE" > "$LOG/dataset.log" 2>&1 \
  || { log "dataset FAILED (see $LOG/dataset.log)"; exit 1; }
DATASET_C=/data/datasets/cpgen/$TASK/$SCENE/$METHOD/${GENERATION##*/}/$OBSERVATION/n${SIZE}_seed0.hdf5
log "fixed scenes in the generation environment"
make -s -C "$REPO" resets $G OVERRIDES="dataset=$DATASET_C" > "$LOG/resets.log" 2>&1 || { log "resets FAILED (see $LOG/resets.log)"; exit 1; }
log "policy stage: $POLICY, image, GPU $GPU"
CLUSTER=$CLUSTER GPU=$GPU GENERATOR=cpgen IMAGE=cpgen METHOD=$METHOD GENERATION=$GENERATION OBSERVATION=$OBSERVATION POLICY=$POLICY SIZE=$SIZE \
  "$REPO/scripts/task_policy_pipeline.sh" $TASK $PORT > "$LOG/policy.log" 2>&1
grep -h "^RESULT" "$LOG/policy.log" | tee -a "$LOG/driver.log"
log "done"; echo "CPGEN_COFFEE_DONE $(date)" > "$LOG/DONE"
