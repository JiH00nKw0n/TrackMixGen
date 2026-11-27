"""The stage record of an evaluation episode: the first step at which each stage signal turned on.

``tmg eval`` writes it into every ``episodes.jsonl`` row as ``term_first`` (``{signal: step index, -1 when never}``, the
key and format of the generation records, ``tmg.core.records``), and ``eval_summary.json`` reports per signal the share of
episodes that reached it (``tmg.evaluation.session.reach_rates``), so a failed episode says how far the policy got (grasped but did not insert, or never grasped).

The signals of a task come from its configuration, read every step through MimicGen's environment interface of the task
(``task.datagen_interface``), the object the generators read object poses and termination signals from:

- ``task.subtask_term_signals`` (MimicGen): the interface's subtask termination signals, the call the MimicGen runtime
  records per generation step (``get_subtask_term_signals``);
- ``stage_checks`` of the task's stages file (``configs/experiment/generator/<generator>/stages/<task>.yaml``, for
  generators without termination signals, CP-Gen): a list of checks, each ``{type, name, ...}``. ``type: object_lifted``
  with ``object`` (a name of the interface's object poses) and ``height`` (m) is on while the object is more than
  ``height`` above its height at the episode start.

A task with neither has no stage record (``build_stage_record`` returns None) and its rows carry no ``term_first``.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from tmg.core.records import SignalLog
from tmg.generation.simulators.robosuite import env_interface, interface_name
from tmg.generation.simulators.scenes import load_task_stages


class TermSignals:
    """The subtask termination signals of the task's MimicGen interface."""

    def __init__(self, interface):
        self.interface = interface

    def begin(self) -> None:
        """Nothing to remember at the episode start."""

    def read(self) -> dict:
        return {str(name): bool(value) for name, value in self.interface.get_subtask_term_signals().items()}


class ObjectLifted:
    """``name`` is on while ``object`` is more than ``height`` (m) above its height at the episode start."""

    def __init__(self, interface, *, name: str, object: str, height: float):       # the keys of a stage_checks entry
        known = sorted(interface.get_object_poses())
        if object not in known:
            raise ValueError(f"stage check {name!r}: object {object!r} is not an object of the task's interface {known}")
        self.interface, self.name, self.object, self.height = interface, str(name), str(object), float(height)
        self._start = None

    def _z(self) -> float:
        return float(np.asarray(self.interface.get_object_poses()[self.object])[2, 3])

    def begin(self) -> None:
        self._start = self._z()

    def read(self) -> dict:
        return {self.name: self._z() - self._start > self.height}


CHECKS: dict[str, Callable] = {"object_lifted": ObjectLifted}       # stage_checks[i].type -> signal source


class StageRecord:
    """The stage signals of one episode at a time: ``begin`` after the scene is restored, ``step`` after every
    environment step, ``first_on`` for the row."""

    def __init__(self, sources: list):
        self.sources = list(sources)
        self.log = SignalLog()

    def begin(self) -> None:
        self.log = SignalLog()
        for source in self.sources:
            source.begin()

    def step(self) -> None:
        signals = {}
        for source in self.sources:
            signals.update(source.read())
        self.log.add(signals)

    def first_on(self) -> dict[str, int]:
        return self.log.first_on()


def build_check(interface, spec: Mapping):
    """The signal source of one ``stage_checks`` entry."""
    options = dict(spec)
    kind = options.pop("type", None)
    if kind not in CHECKS:
        raise ValueError(f"stage_checks: unknown type {kind!r}; known: {sorted(CHECKS)}")
    return CHECKS[kind](interface, **options)


def build_stage_record(task: Mapping, env, checks: list | None = None) -> StageRecord | None:
    """The stage record of ``task`` on the robosuite environment ``env`` (module docstring) with the ``stage_checks``
    entries ``checks``; None when the task has no stage signals."""
    with_terms = any(signal is not None for signal in task.get("subtask_term_signals") or [])
    checks = list(checks or [])
    if not with_terms and not checks:
        return None
    interface = env_interface(interface_name(task), env)
    sources = [TermSignals(interface)] if with_terms else []
    return StageRecord(sources + [build_check(interface, spec) for spec in checks])


def stage_record_of(generator: str, task: Mapping, env) -> StageRecord | None:
    """``build_stage_record`` with the checks of the task's stages file."""
    return build_stage_record(task, env, load_task_stages(generator, str(task["name"])).get("stage_checks"))
