#!/usr/bin/env bash
# Build a task's coefficient file from scratch when it is missing:
#   configs/experiment/generator/<GENERATOR>/coefficients/<TASK>.yaml  (read by the logistic mixer)
# It comes from one uniform-source baseline pool: method none, selection ours (a source drawn uniformly for every subtask),
# generation rank_human_demos (3,000 attempts). A finished pool is reused; an unfinished one resumes its missing chunks.
# A file that already exists is left as it is (delete it to refit).
#   GENERATOR=mimicgen TASK=threading SCENE=ours CLUSTER=a100 scripts/ensure_coefficients.sh
#   METHOD=<method config>: do nothing unless the method (its file and the files it lists under defaults) mixes with
#     mixing.type logistic (`make generate` uses this)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
method_files() {   # the method file and, recursively, the method files it names under defaults
  local f="$REPO/configs/experiment/method/$1.yaml"
  [ -f "$f" ] || return 0
  echo "$f"
  for d in $(awk '/^defaults:/{on=1;next} on&&/^  - /{print $2;next} on&&!/^  /{on=0}' "$f" | grep -v '^_self_$'); do method_files "$d"; done
}
if [ -n "${METHOD:-}" ]; then
  files=$(method_files "$METHOD")
  [ -n "$files" ] || exit 0
  grep -qE '^  type: logistic' $files || exit 0
  ! grep -qE '^mixing: \{enabled: false' "$REPO/configs/experiment/method/$METHOD.yaml" || exit 0
fi
: "${GENERATOR:?}" "${TASK:?}"
SCENE=${SCENE:-ours}; CLUSTER=${CLUSTER:-local}; IMAGE=${IMAGE:-robosuite}
COEF="$REPO/configs/experiment/generator/$GENERATOR/coefficients/$TASK.yaml"
[ -f "$COEF" ] && exit 0
ARGS=(GENERATOR="$GENERATOR" IMAGE="$IMAGE" TASK="$TASK" SCENE="$SCENE" SELECTION=ours CLUSTER="$CLUSTER" GENERATION=rank_human_demos)
echo "[ensure_coefficients] $GENERATOR/$TASK: building the uniform-source baseline pool (method none, 3,000 attempts)"
make -s -C "$REPO" generate "${ARGS[@]}" METHOD=none OVERRIDES="${POOL_OVERRIDES:-}"
make -s -C "$REPO" fit-coefficients "${ARGS[@]}"
