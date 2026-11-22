#!/usr/bin/env bash
# Build a task's SkillMix rule file from scratch when it is missing:
#   configs/experiment/generator/<GENERATOR>/coefficients/<TASK>.yaml  (read by method skillmix)
# It is fitted (tmg fit_coefficients) on two pools of the same code, settings and chunk seeds, both selection ours (a source
# drawn for every subtask) and generation pool_unmix (configs/experiment/generation/pool_unmix.yaml: 2,000 attempts, no
# images, the sizes of the pools the committed rule files were fitted on):
#   unmix        method none                                  (one source per subtask), folder none/pool_unmix
#   randmix  method skillmix, mixing.rule=randmix     (random Dirichlet weights), folder skillmix_randmix/pool_randmix
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
  case " ${OVERRIDES:-} " in *" method.mixing.rule=randmix "*) exit 0;; esac   # a randmix pool needs no rule file
fi
: "${GENERATOR:?}" "${TASK:?}"
SCENE=${SCENE:-ours}; CLUSTER=${CLUSTER:-local}
RULE="$REPO/configs/experiment/generator/$GENERATOR/coefficients/$TASK.yaml"
[ -f "$RULE" ] && exit 0
# IMAGE is passed on only when the caller set it: the Makefile reads the generator's image from its config
ARGS=(GENERATOR="$GENERATOR" ${IMAGE:+IMAGE="$IMAGE"} TASK="$TASK" SCENE="$SCENE" SELECTION=ours CLUSTER="$CLUSTER" GENERATION=pool_unmix)
echo "[ensure_coefficients] $GENERATOR/$TASK: unmix pool (method none)"
make -s -C "$REPO" generate "${ARGS[@]}" METHOD=none OVERRIDES="${POOL_OVERRIDES:-}"
echo "[ensure_coefficients] $GENERATOR/$TASK: randmix pool (method skillmix, random weights)"
make -s -C "$REPO" generate "${ARGS[@]}" METHOD=skillmix \
  OVERRIDES="generation.name=pool_randmix method.mixing.rule=randmix method.name=skillmix_randmix ${POOL_OVERRIDES:-}"
make -s -C "$REPO" fit-coefficients "${ARGS[@]}"
