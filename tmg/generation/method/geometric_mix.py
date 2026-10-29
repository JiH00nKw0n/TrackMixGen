"""Source mixing guided by scene geometry instead of the success ranking (mixup exploration axes 2 and 3).

Both mixers reuse ``GuidedMixer``'s flow and blend (event-aligned time warp, linear position, shortest-rotation
blend in the primary's reference-object frame) and change three steps: when a segment is mixed, which sources may
donate, and how much weight the primary keeps. Both need two poses of the current scene that the adapter puts into
``MixContext.extra``; without them the segment is left unchanged with the reason recorded.

- ``current_object_pose``: the reference object's pose in the current scene, the pose the generator transforms the
  segment to (MimicGen: ``cur_object_pose``).
- ``junction_pose``: the end-effector pose the generator interpolates from into this segment (MimicGen: the last
  target of the previous subtask, or the current end-effector pose on the first subtask).

Distances combine translation and rotation as ``d(T1, T2) = |p1 - p2| + rotation_weight * angle(R1, R2)``, with
``rotation_weight`` in metres per radian.

``TransformMixer`` (``mixing.type: transform``, axis 2: the transform is large). The transform size of source i is
``delta_i = d(O_i, O_cur)``, the distance between its reference-object pose at the segment start and the current
one. A segment is mixed when the primary's ``delta_a`` lies above the ``trigger_quantile`` of all sources' values;
the donor is the compatible source with the smallest ``delta_b < delta_a``, and the primary keeps
``w = clip(delta_b / (delta_a + delta_b), weight_min, weight_max)``, so the source that needs the smaller transform
gets the larger share.

``JunctionMixer`` (``mixing.type: junction``, axis 3: the subtasks do not join). The junction gap of source i is
``c_i = d(J, O_cur O_i^-1 E_i)``, the distance from the junction pose J to the source's first end-effector pose E_i
after the generator's transform. A segment is mixed when ``c_a`` lies above the ``trigger_quantile``; the donor is
the compatible source with the smallest ``c_b < c_a``. Only the head of the segment is blended: on the primary's
timeline the primary keeps ``w(t) = 1 - (1 - w0) * max(0, 1 - t / head_steps)`` with
``w0 = clip(c_b / (c_a + c_b), weight_min, weight_max)``, so the segment starts near the donor's start (close to J)
and returns to the primary after ``head_steps`` steps.

Donors are every other source (``donors: all``) or only the high-success group of the ranking (``donors: good``).
The choice is deterministic (smallest distance, lower index on ties), so no random draw is taken.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from tmg.core.poses import invert, rotation_angle
from tmg.generation.method.guided_mix import GuidedMixConfig, GuidedMixer, _Donor
from tmg.generation.method.interfaces import MixContext
from tmg.generation.method.ranking import load_ranking

DEFAULT_ROTATION_WEIGHT = 0.1          # metres per radian: one radian of object rotation counts like 10 cm
DEFAULT_TRIGGER_QUANTILE = 0.5         # mix when the primary is in the upper half of the sources
DEFAULT_WEIGHT_MIN, DEFAULT_WEIGHT_MAX = 0.2, 0.8
DEFAULT_HEAD_STEPS = 20                # junction blend length (steps of the primary's segment)


@dataclass(frozen=True)
class GeometricMixConfig(GuidedMixConfig):
    """``GuidedMixConfig`` plus the values of the geometric mixers (see the module docstring)."""
    donors: str = "all"                                   # all | good
    rotation_weight: float = DEFAULT_ROTATION_WEIGHT
    trigger_quantile: float = DEFAULT_TRIGGER_QUANTILE
    weight_min: float = DEFAULT_WEIGHT_MIN
    weight_max: float = DEFAULT_WEIGHT_MAX
    head_steps: int = DEFAULT_HEAD_STEPS

    @classmethod
    def from_mapping(cls, cfg: Mapping) -> GeometricMixConfig:
        base = GuidedMixConfig.from_mapping(cfg)
        out = cls(**{f: getattr(base, f) for f in base.__dataclass_fields__},
                  donors=str(cfg.get("donors", "all")),
                  rotation_weight=float(cfg.get("rotation_weight", DEFAULT_ROTATION_WEIGHT)),
                  trigger_quantile=float(cfg.get("trigger_quantile", DEFAULT_TRIGGER_QUANTILE)),
                  weight_min=float(cfg.get("weight_min", DEFAULT_WEIGHT_MIN)),
                  weight_max=float(cfg.get("weight_max", DEFAULT_WEIGHT_MAX)),
                  head_steps=int(cfg.get("head_steps", DEFAULT_HEAD_STEPS)))
        if out.donors not in ("all", "good"):
            raise ValueError(f"mixing.donors must be all or good, got {out.donors!r}")
        if not 0 <= out.weight_min <= out.weight_max <= 1:
            raise ValueError("need 0 <= mixing.weight_min <= mixing.weight_max <= 1")
        if out.head_steps < 1:
            raise ValueError("mixing.head_steps must be at least 1")
        return out


def pose_distance(t1: np.ndarray, t2: np.ndarray, rotation_weight: float) -> float:
    """``|p1 - p2| + rotation_weight * angle(R1, R2)`` of two 4x4 poses (metres, radians)."""
    t1, t2 = np.asarray(t1, dtype=float), np.asarray(t2, dtype=float)
    return float(np.linalg.norm(t1[:3, 3] - t2[:3, 3]) + rotation_weight * rotation_angle(t1[:3, :3], t2[:3, :3]))


class _GeometricMixer(GuidedMixer):
    """Shared steps of the two geometric mixers; subclasses define ``kind`` and ``_distances``."""
    kind = ""
    needs: tuple[str, ...] = ()

    def __init__(self, *, ranking, chunk_seed: int, config: GeometricMixConfig):
        super().__init__(ranking=ranking, chunk_seed=chunk_seed, config=config)

    @classmethod
    def from_config(cls, cfg: Mapping, *, chunk_seed: int):
        return cls(ranking=load_ranking(str(cfg["ranking"])), chunk_seed=chunk_seed,
                   config=GeometricMixConfig.from_mapping(cfg))

    def describe(self) -> dict:
        c = self.config
        return {**super().describe(), "type": self.kind, "donors": c.donors, "rotation_weight": c.rotation_weight,
                "trigger_quantile": c.trigger_quantile, "weight_min": c.weight_min, "weight_max": c.weight_max,
                **({"head_steps": c.head_steps} if self.kind == "junction" else {})}

    def _identity_reason(self, context: MixContext, primary: int, good: set[int], bad: set[int]) -> str | None:
        if not context.has_reference_object:
            return "subtask has no reference object"
        missing = [k for k in self.needs if context.extra.get(k) is None]
        if missing:
            return f"no {', '.join(missing)} in the mix context"
        return None

    def _donor_pool(self, context: MixContext, primary: int, good: set[int]) -> list[int]:
        pool = sorted(good) if self.config.donors == "good" else range(context.num_sources)
        return [b for b in pool if b != primary]

    def _distances(self, context: MixContext) -> np.ndarray:
        raise NotImplementedError

    def _weight(self, d_a: float, d_b: float) -> float:
        share = d_b / (d_a + d_b) if d_a + d_b > 0 else 0.5
        return float(np.clip(share, self.config.weight_min, self.config.weight_max))

    def _choose(self, context: MixContext, primary: int, candidates: list[_Donor], call: int, record: dict):
        d = self._distances(context)
        threshold = float(np.quantile(d, self.config.trigger_quantile))
        record[self.kind] = {"distances": [round(float(v), 4) for v in d], "threshold": round(threshold, 4)}
        if not d[primary] > threshold:
            record["identity_reason"] = f"primary {self.kind} distance not above the trigger quantile"
            return None, None
        closer = [c for c in candidates if d[c.index] < d[primary]]
        if not closer:
            record["identity_reason"] = f"no compatible donor with a smaller {self.kind} distance"
            return None, None
        donor = min(closer, key=lambda c: (d[c.index], c.index))
        w = self._weight(float(d[primary]), float(d[donor.index]))
        record[self.kind]["weight_a"] = round(w, 4)
        return donor, self._schedule(w)

    def _schedule(self, w: float) -> float | Callable[[int], np.ndarray]:
        """The primary's share: one number here; the junction mixer returns a per-step schedule."""
        return w


class TransformMixer(_GeometricMixer):
    """Axis 2 (the transform is large), see the module docstring."""
    kind = "transform"
    needs = ("current_object_pose",)

    def _distances(self, context: MixContext) -> np.ndarray:
        current = np.asarray(context.extra["current_object_pose"], dtype=float)
        rw = self.config.rotation_weight
        return np.array([pose_distance(o, current, rw) for o in context.src_object_poses])


class JunctionMixer(_GeometricMixer):
    """Axis 3 (the subtasks do not join), see the module docstring."""
    kind = "junction"
    needs = ("current_object_pose", "junction_pose")

    def _distances(self, context: MixContext) -> np.ndarray:
        current = np.asarray(context.extra["current_object_pose"], dtype=float)
        junction = np.asarray(context.extra["junction_pose"], dtype=float)
        rw = self.config.rotation_weight
        out = []
        for o, (eef_start, _targets, _grippers) in zip(context.src_object_poses, context.src_segments):
            start_now = current @ invert(np.asarray(o, dtype=float)) @ np.asarray(eef_start, dtype=float)
            out.append(pose_distance(junction, start_now, rw))
        return np.array(out)

    def _schedule(self, w: float) -> float | Callable[[int], np.ndarray]:
        """``w(t) = 1 - (1 - w) * max(0, 1 - t / head_steps)``: ``w`` at the first step, 1 from ``head_steps`` on."""
        head = self.config.head_steps
        return lambda n: 1.0 - (1.0 - w) * np.maximum(0.0, 1.0 - np.arange(n) / head)
