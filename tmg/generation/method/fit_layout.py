"""How ``tmg fit_coefficients`` reads a generator's records: the fit layout every generator adapter declares
(``SyntheticDataGenerator.fit_layout``).

One fit path (``tmg.generation.fit_coefficients.fit_blocks``) reads this declaration; a new generator plugs in by
declaring its layout, not by adding a branch to the fit:

- ``arms``: the generator's end effectors (``Arms``), the same ones its runtime builds the mixer with, so the fit's blocks,
  the rule file's keys and the mixer's lookups agree. The records must name exactly these end effectors.
- ``reached``: the name of the rule that says which subtasks an attempt reached (``fit_records.REACHED``):
  ``step_signals`` (MimicGen: the per-step termination signals up to the end of the subtask), ``term_window`` (the Isaac
  Lab generators: the subtask's recorded signal fired before the same arm's next subtask started), ``stops_at_failure``
  (CP-Gen: the attempt went on to the next subtask) or ``task_success`` (an environment that publishes no per-step
  subtask signal: every recorded subtask counts as reached and the task-success label alone decides; declared per task,
  ``task.fit.reached``).
- ``source_poses``: where the selected source's end-effector and carried-object poses of an unmixed attempt come from,
  the record itself (``records``) or the source file's ``datagen_info`` (MimicGen).
- ``features``: the features of the rule file (``success_model.FEATURES``, or ``HELD_FEATURES`` with a hold test).
- ``unversioned_held_rule``: the hold rule of records without a ``record_version`` (made before the rule was recorded).
- ``unversioned_donor_registration``: the settings the donors of a deformable subtask were registered with in such
  records (made when they were constants of the runtime and not recorded).
- ``sanity``: the limits of the records checks (``SanityLimits``, judged by ``fit_records.judge_sanity``).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field

from tmg.generation.method.interfaces import Arms, HeldRule
from tmg.generation.method.success_model import FEATURES, SOURCE_POSES, check_features


def binomial_tail(count: int, total: int, rate: float) -> float:
    """P(X >= count) for X ~ Binomial(total, rate)."""
    if count <= 0:
        return 1.0
    if rate <= 0.0:
        return 0.0
    if rate >= 1.0:
        return 1.0
    log_p, log_q = math.log(rate), math.log1p(-rate)
    terms = [math.lgamma(total + 1) - math.lgamma(i + 1) - math.lgamma(total - i + 1) + i * log_p + (total - i) * log_q
             for i in range(count, total + 1)]
    top = max(terms)
    return min(1.0, math.exp(top) * sum(math.exp(v - top) for v in terms))


def rate_exceeds(count: int, total: int, limit: float | None, confidence: float) -> bool:
    """True when ``count`` of ``total`` shows, with ``confidence``, that the true rate is above ``limit``: were the rate
    exactly ``limit``, a count this large would come up with probability below ``1 - confidence`` (the one-sided
    Clopper-Pearson lower bound of the rate is above the limit). A few unlucky attempts in a small sample therefore never
    trip it, while a failure of every attempt does at once (5 of 5 against a limit of 0.05). ``limit`` None: never."""
    if limit is None or total <= 0 or count <= 0:
        return False
    return binomial_tail(int(count), int(total), float(limit)) < 1.0 - float(confidence)


@dataclass(frozen=True)
class SanityLimits:
    """The limits of the records checks (generator config ``record_checks``), judged with ``rate_exceeds`` on the counts
    of everything read so far (a whole pool in the fit, the finished chunks of a pool in a chunk result), never on one
    chunk's share alone:

    - ``max_scene_error_rate``: attempts whose subtask scene could not be read;
    - ``max_unreached_success_rate``: successful attempts of a block that count as not having reached it (a successful
      attempt completed every subtask, so such an attempt means the reached rule misreads the records);
    - ``confidence``: how sure the counts must make it that the true rate is above the limit.

    A limit of None is reported only."""
    max_scene_error_rate: float | None = None
    max_unreached_success_rate: float | None = None
    confidence: float = 0.999

    @classmethod
    def from_mapping(cls, m: Mapping | None) -> SanityLimits:
        m = dict(m or {})
        unknown = sorted(set(m) - {"max_scene_error_rate", "max_unreached_success_rate", "confidence"})
        if unknown:
            raise ValueError(f"record_checks: unknown keys {unknown}")
        rate = lambda key: None if m.get(key) is None else float(m[key])     # noqa: E731
        return cls(max_scene_error_rate=rate("max_scene_error_rate"), max_unreached_success_rate=rate("max_unreached_success_rate"),
                   confidence=float(m.get("confidence", cls.confidence)))

    def scene_errors_exceeded(self, count: int, total: int) -> bool:
        return rate_exceeds(count, total, self.max_scene_error_rate, self.confidence)

    def unreached_exceeded(self, count: int, total: int) -> bool:
        return rate_exceeds(count, total, self.max_unreached_success_rate, self.confidence)


@dataclass(frozen=True)
class FitLayout:
    """A generator's fit layout (module docstring)."""
    arms: Arms
    reached: str
    source_poses: str
    features: tuple = FEATURES
    unversioned_held_rule: HeldRule | None = None
    unversioned_donor_registration: Mapping | None = None
    sanity: SanityLimits = field(default_factory=SanityLimits)

    def __post_init__(self):
        if self.source_poses not in SOURCE_POSES:
            raise ValueError(f"fit layout: source_poses must be one of {SOURCE_POSES}, got {self.source_poses!r}")
        object.__setattr__(self, "features", check_features(self.features))
        object.__setattr__(self, "arms", Arms.of(self.arms))
