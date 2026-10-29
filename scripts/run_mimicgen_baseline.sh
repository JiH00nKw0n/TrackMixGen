#!/bin/bash
# MimicGen baseline on one machine: the official generator with its official source selection (selection official, method none), scene ours,
# the seven tasks, image observations (agentview and eye-in-hand, 84x84), Diffusion Policy and BC-RNN, three training
# seeds. Stage A (CPU): per task the source demonstrations, an image pool that is extended until it holds at least SIZE
# successes, the fixed evaluation scenes and the SIZE-demonstration dataset; POOL_LANES tasks at a time, each pool with the
# cluster's worker count. Stage B (GPU): per task x policy x seed, scripts/task_policy_pipeline.sh (train, validate every
# checkpoint on the 25 validation scenes, select, test on the 300 test scenes), POLICY_PARALLEL runs at a time, each with its
# own block of 40 server ports. Everything is resumable: finished pools, datasets, trainings and test results are kept and
# skipped on a rerun. Results are bundled per policy under <data root>/reports/<BUNDLE_NAME>_<policy>.
#   CLUSTER=local GPU=0 scripts/run_mimicgen_baseline.sh
# Variables: CLUSTER (local), GPU (0: CUDA device of training and policy servers), TASKS (the seven), POLICIES
#   ("diffusion_policy bc_rnn"), SEEDS ("202601 202602 202603"), SIZE (1000), POOL_LANES (2: pools generated at the same time),
#   POLICY_PARALLEL (2: policy pipelines on the GPU at the same time), VAL_PARALLEL (4: checkpoints validated at a time per run),
#   TEST_SHARDS (6: parallel shards of the 300 test scenes), SELECTION (official: the generator's source selection rule; ours = a source
#   chosen at random for every subtask), ONLY_GENERATION (1: stop after the datasets), BUNDLE_NAME,
#   MINI (1: an end-to-end check at toy scale: a 200-attempt pool in the folder pilot50, 10-demonstration datasets, 200 training
#   updates, 6 evaluation scenes per split; same code path as the real run).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-local}; GPU=${GPU:-0}
TASKS=${TASKS:-"three_piece_assembly square stack_three mug_cleanup hammer_cleanup threading coffee"}   # longest pools first
POLICIES=${POLICIES:-"diffusion_policy bc_rnn"}; SEEDS=${SEEDS:-"202601 202602 202603"}; SIZE=${SIZE:-1000}
POOL_LANES=${POOL_LANES:-2}; POLICY_PARALLEL=${POLICY_PARALLEL:-2}; VAL_PARALLEL=${VAL_PARALLEL:-4}; TEST_SHARDS=${TEST_SHARDS:-6}
GENERATOR=mimicgen; IMAGE=robosuite; SCENE=ours; METHOD=none; OBSERVATION=image; SELECTION=${SELECTION:-official}
GENERATION=${GENERATION:-pool_robosuite_image}; GEN_OVERRIDES=${GEN_OVERRIDES:-}; TRAIN_OVERRIDES=${TRAIN_OVERRIDES:-}; EVAL_EXTRA=${EVAL_EXTRA:-}
# first attempt counts per task: about 1.1x the attempts that gave 1,000 successes in the 2026-10 low-dim pools (per-subtask random
# selection); the pool is extended by 25 % steps until it holds SIZE successes, so a lower success rate only costs time
declare -A ATTEMPTS=( [coffee]=2500 [threading]=6250 [hammer_cleanup]=7000 [mug_cleanup]=8750 [stack_three]=9500 [square]=15000 [three_piece_assembly]=17000 )
if [ "${MINI:-0}" = 1 ]; then
  GENERATION=test/pilot50; SIZE=10; VAL_PARALLEL=1; TEST_SHARDS=2; POLICY_PARALLEL=1
  GEN_OVERRIDES="generation.attempts=200 generation.chunk=50 $GEN_OVERRIDES"
  for t in $TASKS; do ATTEMPTS[$t]=200; done
  TRAIN_OVERRIDES="training.by_policy.diffusion_policy.image.updates=200 training.by_policy.diffusion_policy.low_dim.updates=200 training.by_policy.bc_rnn.image.updates=200 training.by_policy.bc_rnn.low_dim.updates=200 training.updates_per_epoch=100 training.checkpoint_every_epochs=2 $TRAIN_OVERRIDES"
  EVAL_EXTRA="max_scenes=6 $EVAL_EXTRA"
  BUNDLE_NAME=${BUNDLE_NAME:-mimicgen_mini_$(date +%Y%m%d)}
fi
GEN_NAME=${GENERATION##*/}
export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
LOG=$DATA_ROOT/logs/mimicgen_baseline; mkdir -p "$LOG"
BUNDLE_NAME=${BUNDLE_NAME:-mimicgen_baseline_$(date +%Y%m%d)}
log() { echo "[driver $(date '+%F %T')] $*" | tee -a "$LOG/driver.log"; }
fail() { log "$1 FAILED (see $2)"; echo "$1" >> "$LOG/FAILURES"; }
M() { make -s -C "$REPO" "$@"; }
dataset_of() { echo "$DATA_ROOT/datasets/$GENERATOR/$1/$SCENE/$SELECTION/$METHOD/$GEN_NAME/$OBSERVATION/n${SIZE}_seed0.hdf5"; }
resets_of() { echo "$DATA_ROOT/resets/$GENERATOR/$1/$SCENE/val25_test300_seed202600.json"; }
run_of() { echo "$DATA_ROOT/runs/$GENERATOR/$1/$SCENE/$SELECTION/$METHOD/$GEN_NAME/$OBSERVATION/$2/seed$3"; }
pool_successes() { python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(d.get('successes', 0))" "$DATA_ROOT/pools/$GENERATOR/$1/$SCENE/$SELECTION/$METHOD/$GEN_NAME/DONE.json" 2>/dev/null || echo 0; }

generation_task() {   # one task: sources, pool (extended until SIZE successes), resets, dataset
  local t=$1 attempts=${ATTEMPTS[$t]} common="GENERATOR=$GENERATOR IMAGE=$IMAGE TASK=$t CLUSTER=$CLUSTER SELECTION=$SELECTION"
  if [ -f "$(dataset_of $t)" ] && [ -f "$(resets_of $t)" ]; then log "[$t] dataset and resets exist, generation skipped"; return 0; fi
  log "[$t] sources"
  M sources $common > "$LOG/sources_$t.log" 2>&1 || { fail "sources $t" "$LOG/sources_$t.log"; return 1; }
  while :; do
    log "[$t] pool: $attempts attempts ($GENERATION)"
    M generate $common SCENE=$SCENE METHOD=$METHOD GENERATION=$GENERATION OVERRIDES="generation.attempts=$attempts $GEN_OVERRIDES" \
      > "$LOG/pool_$t.log" 2>&1 || { fail "pool $t" "$LOG/pool_$t.log"; return 1; }
    local succ; succ=$(pool_successes $t)
    log "[$t] pool: $succ successes in $attempts attempts"
    [ "$succ" -ge "$SIZE" ] && break
    [ "${MINI:-0}" = 1 ] && { fail "pool $t (MINI: fewer than $SIZE successes in $attempts attempts)" "$LOG/pool_$t.log"; return 1; }
    attempts=$(( (attempts * 5 / 4 + 249) / 250 * 250 ))         # 25 % more, rounded up to whole chunks of 250
  done
  M resets $common SCENE=$SCENE > "$LOG/resets_$t.log" 2>&1 || { fail "resets $t" "$LOG/resets_$t.log"; return 1; }
  M dataset $common SCENE=$SCENE METHOD=$METHOD GENERATION=$GENERATION OBSERVATION=$OBSERVATION OVERRIDES="size=$SIZE" \
    > "$LOG/dataset_$t.log" 2>&1 || { fail "dataset $t" "$LOG/dataset_$t.log"; return 1; }
  log "[$t] dataset written: $(dataset_of $t)"
}

generation_lane() {
  for t in $TASKS; do
    while [ "$(jobs -rp | wc -l)" -ge "$POOL_LANES" ]; do wait -n || true; done
    generation_task $t &
  done
  wait
  log "generation lane done"
}

policy_lane() {   # task x policy x seed, POLICY_PARALLEL pipelines at a time; a pipeline starts when its task's dataset exists
  local port=8800
  for t in $TASKS; do
    until [ -f "$(dataset_of $t)" ] && [ -f "$(resets_of $t)" ]; do
      grep -q "pool $t\|dataset $t\|sources $t\|resets $t" "$LOG/FAILURES" 2>/dev/null && { log "[$t] generation failed, policies skipped"; continue 2; }
      sleep 120
    done
    for policy in $POLICIES; do for seed in $SEEDS; do
      local run; run=$(run_of $t $policy $seed)
      if compgen -G "$run/eval/test_h*/*/eval_summary.json" > /dev/null; then log "[$t $policy seed $seed] test result exists, skipped"; continue; fi
      while [ "$(jobs -rp | wc -l)" -ge "$POLICY_PARALLEL" ]; do wait -n || true; done
      log "[$t $policy seed $seed] train, validate, select, test (ports $port..$((port + 39)))"
      local skip=0; [ -f "$run/train_result.json" ] && skip=1
      CLUSTER=$CLUSTER GPU=$GPU GENERATOR=$GENERATOR IMAGE=$IMAGE SCENE=$SCENE SELECTION=$SELECTION METHOD=$METHOD GENERATION=$GENERATION OBSERVATION=$OBSERVATION \
        POLICY=$policy SEED=$seed SIZE=$SIZE VAL_PARALLEL=$VAL_PARALLEL TEST_SHARDS=$TEST_SHARDS SKIP_TRAIN=$skip \
        TRAIN_OVERRIDES="$TRAIN_OVERRIDES" EXTRA="$EVAL_EXTRA" \
        "$REPO/scripts/task_policy_pipeline.sh" $t $port > "$LOG/policy_${t}_${policy}_seed$seed.log" 2>&1 \
        || fail "policy $t $policy seed $seed" "$LOG/policy_${t}_${policy}_seed$seed.log" &
      port=$((port + 40)); sleep 30
    done; done
  done
  wait
  log "policy lane done"
}

collect_bundles() {
  local policy args=()
  for t in $TASKS; do args+=(--job "$GENERATOR/$t/$SCENE/$SELECTION/$METHOD/$GEN_NAME/$OBSERVATION"); done
  for policy in $POLICIES; do
    python3 "$REPO/scripts/collect_results.py" --data-root "$DATA_ROOT" --out "$DATA_ROOT/reports/${BUNDLE_NAME}_$policy" \
      --policy "$policy" --seeds $SEEDS "${args[@]}" --logs "$LOG" > "$LOG/collect_$policy.log" 2>&1 \
      && log "bundle: $DATA_ROOT/reports/${BUNDLE_NAME}_$policy.tar.gz" || fail "collect $policy" "$LOG/collect_$policy.log"
  done
}

log "start: cluster $CLUSTER, GPU $GPU, tasks: $TASKS, policies: $POLICIES, seeds: $SEEDS, selection $SELECTION, generation $GENERATION, size $SIZE${MINI:+, MINI mode}"
if [ -f "$LOG/FAILURES" ]; then mv "$LOG/FAILURES" "$LOG/FAILURES.$(date +%Y%m%d_%H%M%S)"; fi
if [ -f "$LOG/DONE" ]; then mv "$LOG/DONE" "$LOG/DONE.$(date +%Y%m%d_%H%M%S)"; fi
generation_lane &
if [ "${ONLY_GENERATION:-0}" = 1 ]; then wait; else policy_lane & wait; fi
[ "${ONLY_GENERATION:-0}" = 1 ] || collect_bundles
if [ -f "$LOG/FAILURES" ]; then log "FAILURES: $(tr '\n' ';' < "$LOG/FAILURES")"; exit 1; fi
echo "MIMICGEN_BASELINE_DONE $(date)" > "$LOG/DONE"; log "done"
