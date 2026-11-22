#!/usr/bin/env bash
# Build a task's SkillMix rule file from scratch when it is missing:
#   configs/experiment/generator/<GENERATOR>/coefficients/<TASK>.yaml  (read by method skillmix)
# It is fitted (tmg fit_coefficients) on two pools of the same code, settings and chunk seeds, both selection ours (a source
# drawn for every subtask) and generation pool_ctrl (2,000 attempts, no images, chunks of 50; CP-Gen chunks of 25), the
# sizes of the pools the committed rule files were fitted on:
#   control      method none                                  (one source per subtask), folder none/pool_ctrl
#   calibration  method skillmix, mixing.rule=calibration     (random Dirichlet weights), folder skillmix_calibration/pool_calib
# A finished pool is reused; an unfinished one resumes its missing chunks. A rule file that exists is left as it is
# (delete it to refit).
#   GENERATOR=mimicgen TASK=threading SCENE=ours CLUSTER=a100 scripts/ensure_coefficients.sh
#   METHOD=<method config>: do nothing unless that method mixes with SkillMix's softmax rule (`make generate` uses this)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
if [ -n "${METHOD:-}" ]; then
  f="$REPO/configs/experiment/method/$METHOD.yaml"
  [ -f "$f" ] || exit 0
  grep -qE '^  type: skillmix' "$f" || exit 0
  grep -qE '^  rule: softmax' "$f" || exit 0
  case " ${OVERRIDES:-} " in *" method.mixing.rule=calibration "*) exit 0;; esac   # a calibration pool needs no rule file
fi
: "${GENERATOR:?}" "${TASK:?}"
SCENE=${SCENE:-ours}; CLUSTER=${CLUSTER:-local}; IMAGE=${IMAGE:-robosuite}
RULE="$REPO/configs/experiment/generator/$GENERATOR/coefficients/$TASK.yaml"
[ -f "$RULE" ] && exit 0
ARGS=(GENERATOR="$GENERATOR" IMAGE="$IMAGE" TASK="$TASK" SCENE="$SCENE" SELECTION=ours CLUSTER="$CLUSTER" GENERATION=pool_ctrl)
CHUNK=""; [ "$GENERATOR" = cpgen ] && CHUNK="generation.chunk=25"       # the CP-Gen pools of the committed rules
echo "[ensure_coefficients] $GENERATOR/$TASK: control pool (method none, 2,000 attempts)"
make -s -C "$REPO" generate "${ARGS[@]}" METHOD=none OVERRIDES="$CHUNK ${POOL_OVERRIDES:-}"
echo "[ensure_coefficients] $GENERATOR/$TASK: calibration pool (method skillmix, random weights, 2,000 attempts)"
make -s -C "$REPO" generate "${ARGS[@]}" METHOD=skillmix \
  OVERRIDES="generation.name=pool_calib method.mixing.rule=calibration method.name=skillmix_calibration $CHUNK ${POOL_OVERRIDES:-}"
make -s -C "$REPO" fit-coefficients "${ARGS[@]}"
