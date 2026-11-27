#!/bin/bash
# SkillMix factor experiment (docs/SKILLMIX_FACTOR_EXPERIMENT.md): for every condition (a method config) and every task, a
# low-dim pool extended until it holds SIZE successes, the SIZE-demonstration low-dim dataset, and Diffusion Policy on
# three training seeds (train, validate every checkpoint, select, test on the 300 test scenes). Each condition is one run
# of scripts/run_mimicgen_baseline.sh with that method, generation pool_factor_lowdim and observation low_dim, so
# everything that driver does holds here: resumable (finished pools, datasets, trainings and test results are skipped),
# one log folder per condition (<data root>/logs/factor_<generator>_<method>), one result bundle per condition and policy.
#   CLUSTER=a100 GPU=6 GENERATOR=mimicgen TASKS="threading coffee" scripts/run_factor_experiment.sh
#   CLUSTER=a100 GPU=6 GENERATOR=mimicgen TASKS=threading CONDITIONS="skillmix_distcap skillmix_neff3" CHECK=1 scripts/run_factor_experiment.sh
# Variables:
#   CLUSTER (local), GPU (0), GENERATOR (mimicgen | cpgen), TASKS (required: the tasks of this generator),
#   CONDITIONS (every condition: none skillmix skillmix_select skillmix_top1 skillmix_uniform skillmix_distcap skillmix_neff3
#     skillmix_mix_grasp skillmix_mix_place; names of configs/experiment/method),
#   POLICY_PARALLEL (3: training and evaluation pipelines on the GPU at the same time, within one condition),
#   CONDITION_PARALLEL (1: conditions at the same time; each runs its own pools and POLICY_PARALLEL pipelines),
#   PORT_BASE (8800: first policy-server port. The k-th condition of CONDITIONS gets its own block of
#     40 x tasks x policies x seeds ports from PORT_BASE + k x block, so conditions never share a port; a second driver on
#     the same machine needs a base beyond this driver's last block),
#   SELECTION (ours: a source drawn for every subtask, the selection SkillMix is fitted and run with; also for `none`),
#   GENERATION (pool_factor_lowdim), POLICIES (diffusion_policy), SEEDS ("202601 202602 202603"), SIZE (1000),
#   CHECK (1: instead of the experiment, a 50-attempt pool per condition and task (generation test/factor_check50) and the
#     summary of its attempt records by scripts/mix_report.py: weights, effective sources and dropped sources per subtask.
#     Run it before the full pools to see that every condition mixes as intended; nothing is trained),
#   and every other variable of scripts/run_mimicgen_baseline.sh (POOL_LANES, VAL_PARALLEL, TEST_SHARDS, CPGEN_WORKERS,
#   ONLY_GENERATION, MINI, ...), which reach it through the environment.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
CLUSTER=${CLUSTER:-local}; GPU=${GPU:-0}; GENERATOR=${GENERATOR:-mimicgen}
: "${TASKS:?set TASKS to the tasks of $GENERATOR, e.g. TASKS=\"threading coffee\"}"
CONDITIONS=${CONDITIONS:-"none skillmix skillmix_select skillmix_top1 skillmix_uniform skillmix_distcap skillmix_neff3 skillmix_mix_grasp skillmix_mix_place"}
SCENE=ours; SELECTION=${SELECTION:-ours}; GENERATION=${GENERATION:-pool_factor_lowdim}; CHECK_GENERATION=test/factor_check50
POLICIES=${POLICIES:-diffusion_policy}; SEEDS=${SEEDS:-"202601 202602 202603"}; SIZE=${SIZE:-1000}
POLICY_PARALLEL=${POLICY_PARALLEL:-3}; CONDITION_PARALLEL=${CONDITION_PARALLEL:-1}; PORT_BASE=${PORT_BASE:-8800}
DATA_ROOT=$(make -s -C "$REPO" print-data-root CLUSTER="$CLUSTER")
LOG=$DATA_ROOT/logs/factor_${GENERATOR}; mkdir -p "$LOG"
log() { echo "[factor $(date '+%F %T')] $*" | tee -a "$LOG/driver.log"; }
count() { echo $#; }
PORT_BLOCK=$(( 40 * $(count $TASKS) * $(count $POLICIES) * $(count $SEEDS) ))   # ports of one condition (40 per pipeline)
PORT_LAST=$(( PORT_BASE + PORT_BLOCK * $(count $CONDITIONS) - 1 ))

for c in $CONDITIONS; do
  [ -f "$REPO/configs/experiment/method/$c.yaml" ] || { echo "unknown condition $c: no configs/experiment/method/$c.yaml" >&2; exit 2; }
done

run_condition() {   # run_condition <method> <first port>: the whole pipeline of one condition
  local c=$1 port=$2
  log "[$c] start (ports $port..$((port + PORT_BLOCK - 1)); log folder $DATA_ROOT/logs/factor_${GENERATOR}_$c)"
  CLUSTER=$CLUSTER GPU=$GPU GENERATOR=$GENERATOR TASKS="$TASKS" METHOD=$c SELECTION=$SELECTION GENERATION=$GENERATION \
    OBSERVATION=low_dim POLICIES="$POLICIES" SEEDS="$SEEDS" SIZE=$SIZE POLICY_PARALLEL=$POLICY_PARALLEL PORT_BASE=$port \
    RUN_NAME=factor_${GENERATOR}_$c "$REPO/scripts/run_mimicgen_baseline.sh" > "$LOG/condition_$c.out" 2>&1 \
    && log "[$c] done" || { log "[$c] FAILED (see $LOG/condition_$c.out)"; echo "$c" >> "$LOG/FAILURES"; }
}

check_condition() {   # check_condition <method>: 50 attempts per task and the mixing summary of their records
  local c=$1 t pool common
  for t in $TASKS; do
    common="GENERATOR=$GENERATOR TASK=$t CLUSTER=$CLUSTER"
    pool=$DATA_ROOT/pools/$GENERATOR/$t/$SCENE/$SELECTION/$c/${CHECK_GENERATION##*/}
    make -s -C "$REPO" sources $common > "$LOG/check_sources_$t.log" 2>&1 \
      && make -s -C "$REPO" generate $common SCENE=$SCENE SELECTION=$SELECTION METHOD=$c GENERATION=$CHECK_GENERATION \
           > "$LOG/check_${c}_$t.log" 2>&1 \
      || { log "[$c $t] check pool FAILED (see $LOG/check_${c}_$t.log)"; echo "check $c $t" >> "$LOG/FAILURES"; continue; }
    log "[$c $t] check pool $pool"
    python3 "$REPO/scripts/mix_report.py" --pool "$pool" 2>&1 | tee -a "$LOG/driver.log" "$LOG/check_${c}_$t.report"
  done
}

export SINGULARITYENV_CUDA_VISIBLE_DEVICES=$GPU CUDA_VISIBLE_DEVICES=$GPU
if [ -f "$LOG/FAILURES" ]; then mv "$LOG/FAILURES" "$LOG/FAILURES.$(date +%Y%m%d_%H%M%S)"; fi
log "start: cluster $CLUSTER, GPU $GPU, generator $GENERATOR, tasks: $TASKS, conditions: $CONDITIONS, selection $SELECTION, ports $PORT_BASE..$PORT_LAST${CHECK:+, CHECK mode}"
if [ "${CHECK:-0}" = 1 ]; then
  for c in $CONDITIONS; do check_condition "$c"; done
else
  index=0
  for c in $CONDITIONS; do
    while [ "$(jobs -rp | wc -l)" -ge "$CONDITION_PARALLEL" ]; do wait -n || true; done
    run_condition "$c" $((PORT_BASE + PORT_BLOCK * index)) &
    index=$((index + 1))
  done
  wait
fi
if [ -f "$LOG/FAILURES" ]; then log "FAILURES: $(tr '\n' ';' < "$LOG/FAILURES")"; exit 1; fi
log "done"
