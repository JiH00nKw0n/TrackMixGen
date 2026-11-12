#!/usr/bin/env bash
# Build a task's ranking file and coefficient file from scratch when either is missing:
#   configs/experiment/generator/<GENERATOR>/ranking/<TASK>.yaml       (read by RankBasedSampling: top-1/3/5)
#   configs/experiment/generator/<GENERATOR>/coefficients/<TASK>.yaml  (read by the logistic mixer)
# Both come from one uniform-source baseline pool: method none, selection ours (a source drawn uniformly for every subtask),
# generation rank_human_demos (3,000 attempts). A finished pool is reused; an unfinished one resumes its missing chunks.
# A file that already exists is left as it is (delete it to refit).
#   GENERATOR=mimicgen TASK=threading SCENE=ours CLUSTER=a100 scripts/ensure_coefficients.sh
#   NEED=ranking | coefficients | both (default both): which files the caller needs
#   METHOD=<method config>: derive NEED from the method (its file and the files it lists under defaults): coefficients for
#     mixing.type logistic, ranking for sources.type rank_based; nothing to do for other methods (`make generate` uses this)
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
  want=""
  if grep -qE '^  type: logistic' $files && ! grep -qE '^mixing: \{enabled: false' "$REPO/configs/experiment/method/$METHOD.yaml"; then
    want=coefficients
  fi
  if grep -qE '^  type: rank_based' $files; then
    if [ -n "$want" ]; then want=both; else want=ranking; fi
  fi
  [ -n "$want" ] || exit 0
  NEED=$want
fi
: "${GENERATOR:?}" "${TASK:?}"
SCENE=${SCENE:-ours}; CLUSTER=${CLUSTER:-local}; IMAGE=${IMAGE:-robosuite}; NEED=${NEED:-both}
RANKING="$REPO/configs/experiment/generator/$GENERATOR/ranking/$TASK.yaml"
COEF="$REPO/configs/experiment/generator/$GENERATOR/coefficients/$TASK.yaml"
need_rank=0; need_coef=0
case "$NEED" in ranking|both) [ -f "$RANKING" ] || need_rank=1;; esac
case "$NEED" in coefficients|both) [ -f "$COEF" ] || need_coef=1;; esac
if [ $need_rank = 0 ] && [ $need_coef = 0 ]; then exit 0; fi
ARGS=(GENERATOR="$GENERATOR" IMAGE="$IMAGE" TASK="$TASK" SCENE="$SCENE" SELECTION=ours CLUSTER="$CLUSTER" GENERATION=rank_human_demos)
echo "[ensure_coefficients] $GENERATOR/$TASK: building the uniform-source baseline pool (method none, 3,000 attempts)"
make -s -C "$REPO" generate "${ARGS[@]}" METHOD=none OVERRIDES="${POOL_OVERRIDES:-}"
if [ $need_rank = 1 ]; then make -s -C "$REPO" rank-sources "${ARGS[@]}"; fi
if [ $need_coef = 1 ]; then make -s -C "$REPO" fit-coefficients "${ARGS[@]}"; fi
