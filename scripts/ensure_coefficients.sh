#!/usr/bin/env bash
# Build a task's missing SkillMix rule files from scratch:
#   configs/experiment/generator/<GENERATOR>/coefficients/softmax/<TASK>.yaml    (read by method skillmix)
#   configs/experiment/generator/<GENERATOR>/coefficients/sparsemax/<TASK>.yaml  (read by method skillmix_sparsemax)
# They are fitted (tmg fit_coefficients, one run for all the missing rules) on two pools of the same code, settings and
# chunk seeds, both selection ours (a source drawn for every subtask) and generation pool_unmix
# (configs/experiment/generation/pool_unmix.yaml: 2,000 attempts, no images, the sizes of the pools the committed rule
# files were fitted on):
#   unmix        method none                                  (one source per subtask), folder none/pool_unmix
#   randmix  method skillmix, mixing.rule=randmix     (random Dirichlet weights), folder skillmix_randmix/pool_randmix
# A finished pool is reused; an unfinished one resumes its missing chunks. Only the missing rule files are fitted: a rule
# file that exists is left as it is (delete it to refit).
#   GENERATOR=mimicgen TASK=threading SCENE=ours CLUSTER=a100 scripts/ensure_coefficients.sh     (every fitted rule)
#   METHOD=<method config> [OVERRIDES=<overrides of `make generate`>]: only the rule file that method reads; nothing for
#   a method that is not SkillMix, draws its weights (randmix) or names its rule file by a path (`make generate` uses this)
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
FITTED_RULES="softmax sparsemax"   # tmg.generation.method.skillmix.FITTED_RULES

# mixing.<key> ($1) of the method config $2: the file's own value (a key of its `mixing:` block, indented two spaces),
# else the value of the method files it inherits (its defaults list; the last one that has the key wins, as in Hydra)
file_value() {
  local key=$1 file="$REPO/configs/experiment/method/$2.yaml" value base inherited
  [ -f "$file" ] || return 0
  value=$(sed -nE "s/^  $key: *([^ #]+).*/\1/p" "$file" | head -1)
  if [ -z "$value" ]; then
    for base in $(sed -nE 's/^  - ([A-Za-z0-9_]+)$/\1/p' "$file"); do
      inherited=$(file_value "$key" "$base")
      value=${inherited:-$value}
    done
  fi
  echo "$value"
}

# mixing.<key> ($1) as `make generate METHOD=... OVERRIDES=...` composes it: the last override method.mixing.<key>=...,
# else the method config's value
mixing_value() {
  local key=$1 value words word
  value=$(file_value "$key" "$METHOD")
  read -ra words <<< "${OVERRIDES:-}"
  for word in ${words[@]+"${words[@]}"}; do
    case "$word" in "method.mixing.$key="*) value=${word#*=};; esac
  done
  echo "$value"
}

if [ -n "${METHOD:-}" ]; then
  [ "$(mixing_value type)" = skillmix ] || exit 0
  case "$(mixing_value coefficients)" in *.yaml) exit 0;; esac
  RULES=$(mixing_value rule)
else
  RULES=$FITTED_RULES
fi
: "${GENERATOR:?}" "${TASK:?}"
MISSING=()
for rule in $RULES; do
  case " $FITTED_RULES " in *" $rule "*) ;; *) continue;; esac
  [ -f "$REPO/configs/experiment/generator/$GENERATOR/coefficients/$rule/$TASK.yaml" ] || MISSING+=("$rule")
done
[ ${#MISSING[@]} -gt 0 ] || exit 0
SCENE=${SCENE:-ours}; CLUSTER=${CLUSTER:-local}
# IMAGE is passed on only when the caller set it: the Makefile reads the generator's image from its config
ARGS=(GENERATOR="$GENERATOR" ${IMAGE:+IMAGE="$IMAGE"} TASK="$TASK" SCENE="$SCENE" SELECTION=ours CLUSTER="$CLUSTER" GENERATION=pool_unmix)
echo "[ensure_coefficients] $GENERATOR/$TASK: unmix pool (method none)"
make -s -C "$REPO" generate "${ARGS[@]}" METHOD=none OVERRIDES="${POOL_OVERRIDES:-}"
echo "[ensure_coefficients] $GENERATOR/$TASK: randmix pool (method skillmix, random weights)"
make -s -C "$REPO" generate "${ARGS[@]}" METHOD=skillmix \
  OVERRIDES="generation.name=pool_randmix method.mixing.rule=randmix method.name=skillmix_randmix ${POOL_OVERRIDES:-}"
echo "[ensure_coefficients] $GENERATOR/$TASK: fit ${MISSING[*]}"
make -s -C "$REPO" fit-coefficients "${ARGS[@]}" OVERRIDES="rules=[$(IFS=,; echo "${MISSING[*]}")]"
