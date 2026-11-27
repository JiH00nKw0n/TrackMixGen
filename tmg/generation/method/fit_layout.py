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
- ``sanity``: the limits of the sanity report (``fit_records.sanity_report``).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from tmg.generation.method.interfaces import Arms, HeldRule
from tmg.generation.method.success_model import FEATURES, SOURCE_POSES, check_features


@dataclass(frozen=True)
class SanityLimits:
    """``max_unreached_success_rate``: the largest share of a block's successful attempts that may count as not having
    reached it (a successful attempt completed every subtask, so such an attempt means the reached rule misreads the
    records); above it the report has an error. None: reported only. ``min_successes``: fewer successful attempts than
    this are too few to judge the share."""
    max_unreached_success_rate: float | None = None
    min_successes: int = 1

    @classmethod
    def from_mapping(cls, m: Mapping | None) -> SanityLimits:
        m = dict(m or {})
        rate = m.get("max_unreached_success_rate")
        return cls(max_unreached_success_rate=None if rate is None else float(rate), min_successes=int(m.get("min_successes", 1)))


@dataclass(frozen=True)
class FitLayout:
    """A generator's fit layout (module docstring)."""
    arms: Arms
    reached: str
    source_poses: str
    features: tuple = FEATURES
    unversioned_held_rule: HeldRule | None = None
    sanity: SanityLimits = field(default_factory=SanityLimits)

    def __post_init__(self):
        if self.source_poses not in SOURCE_POSES:
            raise ValueError(f"fit layout: source_poses must be one of {SOURCE_POSES}, got {self.source_poses!r}")
        object.__setattr__(self, "features", check_features(self.features))
        object.__setattr__(self, "arms", Arms.of(self.arms))
