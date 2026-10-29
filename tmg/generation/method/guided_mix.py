"""Guided source mixing: a primary source from the low-success group is blended with a compatible donor
from the high-success group. The groups come from the task's source ranking (``configs/experiment/generator/<generator>/ranking/<task>.yaml``, see ``tmg.generation.method.ranking``) split at run time by ``method.mixing.good_fraction``
(or ``good_count``); the ranking may differ per subtask. Mixing happens in the primary's reference-object
frame, so the generator's own scene transform (which uses the primary's reference pose) applies unchanged
afterwards.
"""
from __future__ import annotations

import numpy as np

from tmg.core.poses import invert
from tmg.generation.method.event_mix import anchor_rotation_degrees, blend_poses, mix_segments
from tmg.generation.method.grippers import SignGripper, build_detector
from tmg.generation.method.interfaces import MixContext, MixedSegment
from tmg.generation.method.ranking import SourceRanking, load_ranking


class GuidedMixer:
    """The ``SegmentMixer`` of TrackMix (see the module docstring).

    A primary source in the ``bad`` group is blended (``weight_a`` : ``1 - weight_a``) with a donor from the
    ``good`` group whose gripper events match and whose anchor rotations differ by at most
    ``max_anchor_rotation_deg``; otherwise the segment is returned unchanged with the reason recorded.
    """
    def __init__(self, *, ranking: SourceRanking, weight_a: float, max_anchor_rotation_deg: float, chunk_seed: int,
                 seed_offset: int, good_fraction: float | None = 0.5, good_count: int | None = None, gripper_events=None):
        """Args:
            ranking: the task's source ranking (task-wide or per subtask).
            weight_a: share of the primary source in the blend.
            max_anchor_rotation_deg: donor eligibility bound at the anchors (start, close events, end).
            chunk_seed, seed_offset: seed the private donor-choice RNG with the mix call index.
            good_fraction, good_count: how many top-ranked sources count as high-success (see ``RankedSources.split``).
            gripper_events: the open/closed rule of the gripper commands (``{type: sign}`` for one scalar command per
                step, the default; ``{type: closure, columns: [...]}`` for many-joint hands such as the GR1's). Events
                found by this rule align the two segments; with the closure rule the mixed hand commands are the
                time-warped blend of both sources' joint commands (same phase at every step).
        """
        self.ranking = ranking
        self.detector = build_detector(gripper_events)
        self.good_fraction = None if good_fraction is None else float(good_fraction)
        self.good_count = None if good_count is None else int(good_count)
        self.weight_a = float(weight_a)
        self.max_anchor_deg = float(max_anchor_rotation_deg)
        self.chunk_seed = int(chunk_seed)
        self.seed_offset = int(seed_offset)
        self.calls = 0
        self._groups: dict[int, tuple[set[int], set[int]]] = {}

    def groups(self, subtask_ind: int) -> tuple[set[int], set[int]]:
        """``(good, bad)`` source sets of ``subtask_ind`` under the configured split (cached per subtask)."""
        if subtask_ind not in self._groups:
            ranked = self.ranking.for_subtask(subtask_ind)
            self._groups[subtask_ind] = ranked.split(good_fraction=self.good_fraction, good_count=self.good_count)
        return self._groups[subtask_ind]

    @classmethod
    def from_config(cls, cfg: dict, *, chunk_seed: int) -> GuidedMixer:
        """Build from the ``method.mixing`` config block: ``ranking`` names ``configs/ranking/<generator>/<task>``."""
        ranking = load_ranking(str(cfg["ranking"]))
        return cls(ranking=ranking, weight_a=cfg.get("weight_a", 0.8),
                   max_anchor_rotation_deg=cfg.get("max_anchor_rotation_deg", 20.0), chunk_seed=chunk_seed,
                   seed_offset=cfg.get("seed_offset", 817263), good_fraction=cfg.get("good_fraction", 0.5),
                   good_count=cfg.get("good_count"), gripper_events=cfg.get("gripper_events"))

    def describe(self) -> dict:
        """JSON-able description of the mixer settings for logs and provenance."""
        return {"type": "guided", "weight_a": self.weight_a, "max_anchor_rotation_deg": self.max_anchor_deg,
                "good_fraction": self.good_fraction, "good_count": self.good_count, "gripper_events": self.detector.name,
                "ranking": self.ranking.describe()}

    def _event_series(self, commands) -> np.ndarray:
        """The commands as the scalar series the event functions read: the raw first column for the sign rule (the
        official robosuite convention, zeros carry the state), otherwise +1 closed / -1 open from the detector."""
        g = np.asarray(commands, dtype=float).reshape(len(commands), -1)
        if isinstance(self.detector, SignGripper) and g.shape[1] == 1:
            return g[:, 0]
        return np.where(self.detector.states(g), 1.0, -1.0)

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
        if a in good:
            record["identity_reason"] = "primary is a high-success source"
            return identity
        if a not in bad:
            record["identity_reason"] = "primary not ranked"
            return identity
        if not context.has_reference_object:
            record["identity_reason"] = "subtask has no reference object"
            return identity
        if self.weight_a >= 1:
            record["identity_reason"] = "weight_a is 1"
            return identity
        obj_a = np.asarray(context.src_object_poses[a], dtype=float)
        inv_a = invert(obj_a)
        pa = np.einsum("ij,tjk->tik", inv_a, context.primary_targets)
        ga = np.asarray(context.primary_gripper, dtype=float)
        sa = self._event_series(ga)
        candidates = []
        for b in sorted(good):
            eef_b, targets_b, gripper_b = context.src_segments[b]
            inv_b = invert(np.asarray(context.src_object_poses[b], dtype=float))
            pb = np.einsum("ij,tjk->tik", inv_b, np.asarray(targets_b, dtype=float))
            gb = np.asarray(gripper_b, dtype=float)
            sb = self._event_series(gb)
            angles = anchor_rotation_degrees(pa, sa, pb, sb)
            if angles is None or max(angles) > self.max_anchor_deg + 1e-6:
                continue
            candidates.append((b, pb, gb, sb, inv_b @ np.asarray(eef_b, dtype=float), angles))
        record["eligible_donors"] = [c[0] for c in candidates]
        if not candidates:
            record["identity_reason"] = "no compatible donor"
            return identity
        rng = np.random.default_rng(np.random.SeedSequence([self.chunk_seed, call, self.seed_offset]))
        b, pb, gb, sb, eef_b_obj, angles = candidates[int(rng.integers(len(candidates)))]
        mixed_obj, grip, meta = mix_segments(pa, sa, pb, sb, self.weight_a)
        mixed = np.einsum("ij,tjk->tik", obj_a, mixed_obj)
        eef_a_obj = inv_a @ np.asarray(context.primary_eef_start, dtype=float)
        eef_start = obj_a @ blend_poses(eef_a_obj[None], eef_b_obj[None], self.weight_a)[0]
        ga2, gb2 = ga.reshape(len(ga), -1), gb.reshape(len(gb), -1)
        if isinstance(self.detector, SignGripper) and ga2.shape[1] == 1:
            gripper = grip.reshape(-1) if ga.ndim == 1 else grip.reshape(-1, 1)      # the blended scalar command
        else:
            # many-joint hands: both sources' joint commands, time-warped to the mixed timeline (same open/closed phase
            # at every step, checked by mix_segments) and blended with the same weights as the poses
            ia, ib = np.asarray(meta["source_a_index"]), np.asarray(meta["source_b_index"])
            gripper = self.weight_a * ga2[ia] + (1 - self.weight_a) * gb2[ib]
            if ga.ndim == 1:
                gripper = gripper.reshape(-1)
        record.update(mixed=True, source_b=int(b), anchor_rotation_degrees=angles, alignment=meta)
        return MixedSegment(eef_start, mixed, gripper, record)
