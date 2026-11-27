"""Derived SkillMix rule files: a task's fitted rule file changed by named transforms, for experiments that vary one
factor of the rule (docs/SKILLMIX_FACTOR_EXPERIMENT.md) without hand-edited copies.

A condition is a name and a list of transforms (``configs/experiment/derived_coefficients.yaml``, or given on the command
line of ``scripts/derive_coefficients.py``). Its file for a task is written to

    configs/experiment/generator/<generator>/coefficients/derived/<condition>/<task>.yaml

and read by a method whose config sets ``mixing.derived: <condition>`` (``tmg.generation.method.skillmix.rule_path``).
The file is the fitted one with the transformed values, a header that says so and a ``derived`` block: the condition, the
fitted file it came from with its sha256, and the transforms. ``is_current`` tells whether a derived file is still what
its fitted file and condition give (a refit makes it stale); the unit tests check every committed one.

Transforms (``TRANSFORMS``), each ``{type, subtasks, ...}``:

- ``set_d``: the spread effect ``d`` of the chosen subtasks becomes ``value``. ``d <= 0`` gives one source per subtask
  (no blending; which one is the method's ``mixing.single_source``), a large ``d`` nearly uniform weights
  (``softmax(s / d)``).
- ``boost_a``: ``by`` is added to the effect ``a`` of ``source`` in the chosen subtasks. With a large ``by`` and
  ``d <= 0`` that source is always the best one (a forced source).

``subtasks`` chooses the subtasks: ``all``, a list of subtask indices, ``{kind: <kind>}`` or ``{except_kind: <kind>}``.
Kinds are the task's ``subtask_kinds`` (``configs/experiment/generator/<generator>/stages/<task>.yaml``), one per subtask
of the rule file.
"""
from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import yaml

from tmg import CONFIGS_ROOT
from tmg.generation.fit_coefficients import write_rule
from tmg.generation.method.skillmix import SkillMixRule, rule_path
from tmg.generation.simulators.scenes import load_task_stages

CONDITIONS_FILE = "experiment/derived_coefficients.yaml"       # under the configs root


# subtasks -------------------------------------------------------------------------------------------------------------
def select_subtasks(selector, subtasks: Sequence[int], kinds: Sequence[str] | None) -> list[int]:
    """The subtask indices a transform's ``subtasks`` value chooses (module docstring) among ``subtasks`` (the rule
    file's, in order). ``kinds`` are the task's subtask kinds, needed only by the kind selectors."""
    subtasks = [int(k) for k in subtasks]
    if selector == "all":
        return subtasks
    if isinstance(selector, Mapping) and len(selector) == 1 and next(iter(selector)) in ("kind", "except_kind"):
        (how, kind), = selector.items()
        if kinds is None or len(kinds) != len(subtasks):
            raise ValueError(f"subtasks: {dict(selector)} needs one subtask kind per subtask of the rule file "
                             f"({len(subtasks)}); the task's stages file gives {kinds}")
        if kind not in kinds:
            raise ValueError(f"subtasks: {dict(selector)}: the task has no subtask of kind {kind!r} (kinds: {list(kinds)})")
        return [k for k, of_kind in zip(subtasks, kinds) if (of_kind == kind) == (how == "kind")]
    if isinstance(selector, Sequence) and not isinstance(selector, str):
        chosen = [int(k) for k in selector]
        unknown = sorted(set(chosen) - set(subtasks))
        if unknown:
            raise ValueError(f"subtasks: {unknown} are not subtasks of the rule file ({subtasks})")
        return chosen
    raise ValueError(f"subtasks must be 'all', a list of indices, {{kind: ...}} or {{except_kind: ...}}, got {selector!r}")


# transforms -----------------------------------------------------------------------------------------------------------
def set_d(block: dict, *, value: float) -> None:
    """``d`` of the subtask becomes ``value``."""
    block["d"] = float(value)


def boost_a(block: dict, *, source: int, by: float) -> None:
    """``by`` is added to the effect of ``source``."""
    if not 0 <= int(source) < len(block["a"]):
        raise ValueError(f"boost_a: source {source} is not one of the {len(block['a'])} sources")
    block["a"][int(source)] = float(block["a"][int(source)]) + float(by)


TRANSFORMS: dict[str, Callable] = {"set_d": set_d, "boost_a": boost_a}


def derive(data: Mapping, transforms: Sequence[Mapping], kinds: Sequence[str] | None = None) -> dict:
    """A copy of the rule-file mapping ``data`` with ``transforms`` applied in order (module docstring)."""
    out = copy.deepcopy(dict(data))
    blocks = {int(k): block for k, block in out["subtasks"].items()}
    for spec in transforms:
        options = dict(spec)
        kind, selector = options.pop("type", None), options.pop("subtasks", None)
        if kind not in TRANSFORMS:
            raise ValueError(f"unknown transform {kind!r}; known: {sorted(TRANSFORMS)}")
        for k in select_subtasks(selector, sorted(blocks), kinds):
            TRANSFORMS[kind](blocks[k], **options)
    out["subtasks"] = {k: blocks[k] for k in sorted(blocks)}
    return out


# conditions and files -------------------------------------------------------------------------------------------------
def load_conditions(root: Path | None = None) -> dict[str, list]:
    """``{condition: transforms}`` of ``configs/experiment/derived_coefficients.yaml``."""
    data = yaml.safe_load((Path(root or CONFIGS_ROOT) / CONDITIONS_FILE).read_text()) or {}
    return {str(name): list(entry["transforms"]) for name, entry in (data.get("conditions") or {}).items()}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _note(condition: str, source: str, transforms: Sequence[Mapping]) -> str:
    lines = [f"# DERIVED rule file, condition {condition!r}: not a fit. Written by scripts/derive_coefficients.py from",
             f"# configs/{source} with the transforms below; do not edit, derive it again after a refit.",
             *[f"#   {dict(t)}" for t in transforms],
             "# Everything else (a, g, c0 and the fit's statistics) is the fitted file's."]
    return "\n".join(lines) + "\n"


def derived_rule(generator: str, task: str, condition: str, transforms: Sequence[Mapping], *, rule: str,
                 root: Path | None = None) -> tuple[dict, str, dict]:
    """What the derived file of ``condition`` for ``generator``/``task`` holds: ``(rule mapping, header note, derived
    block)`` from the task's fitted ``rule`` file and stages file."""
    configs = Path(root or CONFIGS_ROOT)
    fitted = rule_path(f"{generator}/{task}", rule, configs)
    if not fitted.exists():
        raise FileNotFoundError(f"fitted rule file {fitted} not found (make coefficients GENERATOR={generator} TASK={task})")
    kinds = load_task_stages(generator, task, configs).get("subtask_kinds")
    data = derive(yaml.safe_load(fitted.read_text()), transforms, kinds)
    source = fitted.relative_to(configs).as_posix()
    block = {"condition": condition, "from": source, "from_sha256": sha256(fitted), "transforms": [dict(t) for t in transforms]}
    return data, _note(condition, source, transforms), block


def write_derived(generator: str, task: str, condition: str, transforms: Sequence[Mapping], *, rule: str = "softmax",
                  root: Path | None = None) -> Path:
    """Write the derived rule file of ``condition`` for ``generator``/``task`` (module docstring) and return its path."""
    data, note, block = derived_rule(generator, task, condition, transforms, rule=rule, root=root)
    path = rule_path(f"{generator}/{task}", rule, root, derived=condition)
    write_rule(path, rule=rule, generator=generator, task=task, basis=data["basis"], num_sources=data["num_sources"],
               blocks=data["subtasks"], note=note, extra={"derived": block})
    SkillMixRule.from_mapping(yaml.safe_load(path.read_text()), path=str(path))        # the mixer must be able to read it
    return path


def is_current(path: Path, root: Path | None = None) -> bool:
    """Whether the derived rule file ``path`` is what its recorded condition and transforms give from the fitted file as
    it is now (False after a refit or a hand edit)."""
    held = yaml.safe_load(Path(path).read_text())
    block = held["derived"]
    data, _note_text, expected = derived_rule(held["generator"], held["task"], block["condition"], block["transforms"],
                                              rule=held["rule"], root=root)
    return expected == block and data["subtasks"] == held["subtasks"]


def derived_files(root: Path | None = None) -> list[Path]:
    """Every derived rule file under the configs root."""
    return sorted(Path(root or CONFIGS_ROOT).glob("experiment/generator/*/coefficients/derived/*/*.yaml"))
