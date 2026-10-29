"""Guided source mixing: a primary source from the low-success group is blended with a compatible donor
from the high-success group. The groups come from the task's source ranking (``configs/experiment/generator/<generator>/ranking/<task>.yaml``, see ``tmg.generation.method.ranking``) split at run time by ``method.mixing.good_fraction``
(or ``good_count``); the ranking may differ per subtask. Mixing happens in the primary's reference-object
frame, so the generator's own scene transform (which uses the primary's reference pose) applies unchanged
afterwards.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from tmg.core.poses import invert, transform_all
from tmg.generation.method.event_mix import align_segments, anchor_rotation_degrees, blend_poses, mix_segments
from tmg.generation.method.grippers import SignGripper, build_detector
from tmg.generation.method.interfaces import MixContext, MixedSegment
from tmg.generation.method.ranking import SourceRanking, load_ranking

DEFAULT_WEIGHT_A = 0.8
DEFAULT_MAX_ANCHOR_ROTATION_DEG = 20.0
DEFAULT_SEED_OFFSET = 817263
DEFAULT_GOOD_FRACTION = 0.5


@dataclass(frozen=True)
class GuidedMixConfig:
    """The values of a ``method.mixing`` block that ``GuidedMixer`` reads (``ranking`` is loaded separately)."""
    weight_a: float = DEFAULT_WEIGHT_A                             # share of the primary source in the blend
    max_anchor_rotation_deg: float = DEFAULT_MAX_ANCHOR_ROTATION_DEG  # donor bound at the anchors (start, close events, end)
    seed_offset: int = DEFAULT_SEED_OFFSET                         # seed material of the private donor-choice RNG
    good_fraction: float | None = DEFAULT_GOOD_FRACTION            # top share of the ranking counted as high-success
    good_count: int | None = None                                  # explicit count instead of the fraction
    gripper_events: Mapping | None = None                          # detector config ({type: sign} when None)

    @classmethod
    def from_mapping(cls, cfg: Mapping) -> GuidedMixConfig:
        """Read the block with the defaults above for missing keys."""
        return cls(weight_a=cfg.get("weight_a", DEFAULT_WEIGHT_A),
                   max_anchor_rotation_deg=cfg.get("max_anchor_rotation_deg", DEFAULT_MAX_ANCHOR_ROTATION_DEG),
                   seed_offset=cfg.get("seed_offset", DEFAULT_SEED_OFFSET),
                   good_fraction=cfg.get("good_fraction", DEFAULT_GOOD_FRACTION), good_count=cfg.get("good_count"),
                   gripper_events=cfg.get("gripper_events"))


class _Donor(NamedTuple):
    """A high-success source whose segment is compatible with the primary's, in its own object frame."""
    index: int
    poses: np.ndarray            # (T_b, 4, 4) targets in the donor's reference-object frame
    grippers: np.ndarray         # (T_b, G) raw gripper commands
    series: np.ndarray           # (T_b,) scalar event series of the commands
    eef_start: np.ndarray        # (4, 4) first eef pose in the donor's object frame
    angles: list[float]          # anchor rotation differences to the primary (deg)


class GuidedMixer:
    """The ``SegmentMixer`` of TrackMix (see the module docstring).

    A primary source in the ``bad`` group is blended (``weight_a`` : ``1 - weight_a``) with a donor from the
    ``good`` group whose gripper events match and whose anchor rotations differ by at most
    ``max_anchor_rotation_deg``; otherwise the segment is returned unchanged with the reason recorded.
    """
    def __init__(self, *, ranking: SourceRanking, chunk_seed: int, config: GuidedMixConfig | None = None, **values):
        """Args:
            ranking: the task's source ranking (task-wide or per subtask).
            chunk_seed: seeds the private donor-choice RNG together with ``seed_offset`` and the mix call index.
            config: the mixing values; ``values`` may instead give its fields one by one (``weight_a``,
                ``max_anchor_rotation_deg``, ``seed_offset``, ``good_fraction``, ``good_count``, ``gripper_events``).
                ``gripper_events`` is the open/closed rule of the gripper commands (``{type: sign}`` for one scalar
                command per step, the default; ``{type: closure, columns: [...]}`` for many-joint hands such as the
                GR1's). Events found by this rule align the two segments; with the closure rule the mixed hand commands
                are the time-warped blend of both sources' joint commands (same phase at every step).
        """
        if config is None:
            config = GuidedMixConfig(**values)
        elif values:
            raise TypeError(f"give either config or the single values, not both: {sorted(values)}")
        self.config = config
        self.ranking = ranking
        self.detector = build_detector(config.gripper_events)
        self.good_fraction = None if config.good_fraction is None else float(config.good_fraction)
        self.good_count = None if config.good_count is None else int(config.good_count)
        self.weight_a = float(config.weight_a)
        self.max_anchor_deg = float(config.max_anchor_rotation_deg)
        self.chunk_seed = int(chunk_seed)
        self.seed_offset = int(config.seed_offset)
        self.calls = 0
        self._groups: dict[int, tuple[set[int], set[int]]] = {}

    def groups(self, subtask_ind: int) -> tuple[set[int], set[int]]:
        """``(good, bad)`` source sets of ``subtask_ind`` under the configured split (cached per subtask)."""
        if subtask_ind not in self._groups:
            ranked = self.ranking.for_subtask(subtask_ind)
            self._groups[subtask_ind] = ranked.split(good_fraction=self.good_fraction, good_count=self.good_count)
        return self._groups[subtask_ind]

    @classmethod
    def from_config(cls, cfg: Mapping, *, chunk_seed: int) -> GuidedMixer:
        """Build from the ``method.mixing`` config block: ``ranking`` names
        ``configs/experiment/generator/<generator>/ranking/<task>``."""
        return cls(ranking=load_ranking(str(cfg["ranking"])), chunk_seed=chunk_seed,
                   config=GuidedMixConfig.from_mapping(cfg))

    def describe(self) -> dict:
        """JSON-able description of the mixer settings for logs and provenance."""
        return {"type": "guided", "weight_a": self.weight_a, "max_anchor_rotation_deg": self.max_anchor_deg,
                "good_fraction": self.good_fraction, "good_count": self.good_count, "gripper_events": self.detector.name,
                "ranking": self.ranking.describe()}

    # events ------------------------------------------------------------------------------------------
    def _scalar_commands(self, commands: np.ndarray) -> bool:
        """True when the commands are one scalar sign command per step, the official robosuite convention, so the raw
        column itself is the event series and the blended scalar is the mixed command."""
        return isinstance(self.detector, SignGripper) and commands.reshape(len(commands), -1).shape[1] == 1

    def _event_series(self, commands) -> np.ndarray:
        """The commands as the scalar series the event functions read: the raw first column for the sign rule (the
        official robosuite convention, zeros carry the state), otherwise +1 closed / -1 open from the detector."""
        g = np.asarray(commands, dtype=float).reshape(len(commands), -1)
        if self._scalar_commands(g):
            return g[:, 0]
        return np.where(self.detector.states(g), 1.0, -1.0)

    # one subtask --------------------------------------------------------------------------------------
    def _identity_reason(self, context: MixContext, primary: int, good: set[int], bad: set[int]) -> str | None:
        """Why the segment is left unchanged before any donor is looked at, or None when mixing may go ahead."""
        if primary in good:
            return "primary is a high-success source"
        if primary not in bad:
            return "primary not ranked"
        if not context.has_reference_object:
            return "subtask has no reference object"
        if self.weight_a >= 1:
            return "weight_a is 1"
        return None

    def _donor_pool(self, context: MixContext, primary: int, good: set[int]) -> list[int]:
        """The sources that may donate, before the compatibility check: the high-success group (ascending index)."""
        return sorted(good)

    def _candidates(self, context: MixContext, pool, pa: np.ndarray, sa: np.ndarray) -> list[_Donor]:
        """The sources of ``pool`` (in its order) whose events match the primary's and whose anchor rotations stay
        within ``max_anchor_rotation_deg``; ``pa`` and ``sa`` are the primary's object-frame targets and event series."""
        candidates = []
        for b in pool:
            eef_b, targets_b, gripper_b = context.src_segments[b]
            inv_b = invert(np.asarray(context.src_object_poses[b], dtype=float))
            pb = transform_all(inv_b, np.asarray(targets_b, dtype=float))
            gb = np.asarray(gripper_b, dtype=float)
            sb = self._event_series(gb)
            angles = anchor_rotation_degrees(pa, sa, pb, sb)
            if angles is None or max(angles) > self.max_anchor_deg + 1e-6:
                continue
            candidates.append(_Donor(b, pb, gb, sb, inv_b @ np.asarray(eef_b, dtype=float), angles))
        return candidates

    def _blend_grippers(self, ga: np.ndarray, gb: np.ndarray, grip: np.ndarray, meta: dict, weight_a=None) -> np.ndarray:
        """The mixed gripper commands in the primary's layout: the blended scalar for sign commands, otherwise both
        sources' joint commands time-warped to the mixed timeline and blended with the pose weights (``weight_a``:
        one number or one per mixed step; ``self.weight_a`` when None)."""
        w = self.weight_a if weight_a is None else weight_a
        ga2, gb2 = ga.reshape(len(ga), -1), gb.reshape(len(gb), -1)
        if self._scalar_commands(ga2):
            return grip.reshape(-1) if ga.ndim == 1 else grip.reshape(-1, 1)      # the blended scalar command
        # many-joint hands: both sources' joint commands, time-warped to the mixed timeline (same open/closed phase
        # at every step, checked by mix_segments) and blended with the same weights as the poses
        ia, ib = np.asarray(meta["source_a_index"]), np.asarray(meta["source_b_index"])
        wc = np.asarray(w, dtype=float).reshape(-1, 1) if np.ndim(w) else w
        gripper = wc * ga2[ia] + (1 - wc) * gb2[ib]
        if ga.ndim == 1:
            gripper = gripper.reshape(-1)
        return gripper

    def mix(self, context: MixContext) -> MixedSegment:
        """Mix one subtask segment (see the class docstring).

        Args:
            context: the segment of the selected source and every source's segment, in world frame, with the
                reference-object poses that define each source's object frame.

        Returns:
            ``MixedSegment`` with the (possibly) blended first eef pose, targets and gripper commands, expressed in
            the primary source's world frame, and a record of what happened.
        """
        call = self.calls
        self.calls += 1
        a = int(context.primary)
        good, bad = self.groups(int(context.subtask_ind))
        record = {"call": call, "subtask": int(context.subtask_ind), "source_a": a, "weight_a": self.weight_a,
                  "mixed": False, "good": sorted(good)}
        identity = MixedSegment(context.primary_eef_start, context.primary_targets, context.primary_gripper, record)
        reason = self._identity_reason(context, a, good, bad)
        if reason is not None:
            record["identity_reason"] = reason
            return identity
        obj_a = np.asarray(context.src_object_poses[a], dtype=float)
        inv_a = invert(obj_a)
        pa = transform_all(inv_a, context.primary_targets)
        ga = np.asarray(context.primary_gripper, dtype=float)
        sa = self._event_series(ga)
        candidates = self._candidates(context, self._donor_pool(context, a, good), pa, sa)
        record["eligible_donors"] = [c.index for c in candidates]
        if not candidates:
            record["identity_reason"] = "no compatible donor"
            return identity
        donor, weight = self._choose(context, a, candidates, call, record)
        if donor is None:
            return identity                                   # _choose wrote the identity reason
        eef_a_obj = inv_a @ np.asarray(context.primary_eef_start, dtype=float)
        if not callable(weight):
            mixed_obj, grip, meta = mix_segments(pa, sa, donor.poses, donor.series, weight)
            eef_start = obj_a @ blend_poses(eef_a_obj[None], donor.eef_start[None], weight)[0]
            gripper = self._blend_grippers(ga, donor.grippers, grip, meta, weight)
        else:
            mixed_obj, gripper, meta = self._blend_schedule(pa, sa, ga, donor, weight)
            eef_start = obj_a @ blend_poses(eef_a_obj[None], donor.eef_start[None], float(meta["weights"][0]))[0]
        mixed = transform_all(obj_a, mixed_obj)
        record.update(mixed=True, source_b=int(donor.index), anchor_rotation_degrees=donor.angles, alignment=meta)
        return MixedSegment(eef_start, mixed, gripper, record)

    def _choose(self, context: MixContext, primary: int, candidates: list[_Donor], call: int, record: dict):
        """``(donor, weight_a)`` for a segment that may be mixed, or ``(None, None)`` after writing
        ``record["identity_reason"]``. ``weight_a`` is one number, or a callable ``n -> (n,) weights`` for a weight
        that changes along the mixed segment. Here: a uniform draw among the candidates from the private RNG
        ``SeedSequence([chunk_seed, call, seed_offset])`` and the fixed ``weight_a``."""
        rng = np.random.default_rng(np.random.SeedSequence([self.chunk_seed, call, self.seed_offset]))
        return candidates[int(rng.integers(len(candidates)))], self.weight_a

    def _blend_schedule(self, pa: np.ndarray, sa: np.ndarray, ga: np.ndarray, donor: _Donor, schedule):
        """Blend with a weight that changes along the segment: the timeline is the primary's (events at its own
        steps) and ``schedule(n)`` gives the primary's share at each of the n steps."""
        aligned_a, aligned_b, ia, ib, align = align_segments(pa, sa, donor.poses, donor.series, 1.0)
        w = np.asarray(schedule(len(aligned_a)), dtype=float)
        mixed = blend_poses(aligned_a, aligned_b, w)
        grip = w * sa[ia] + (1 - w) * donor.series[ib]
        meta = dict(weights=[round(float(v), 4) for v in w], **align)
        return mixed, self._blend_grippers(ga, donor.grippers, grip, meta, w), meta
