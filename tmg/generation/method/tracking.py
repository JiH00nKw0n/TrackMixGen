"""Error-adaptive path following (the "tracking" technique).

One follower covers the whole trajectory a generator executes for one subtask: the bridge (interpolation
and fixed) waypoints and the transformed skill waypoints, flattened into one path w_0..w_{n-1}, as the
2026-10 reference implementation (tracking_reference.py) did. Every end effector of the robot shares the
same path variable s: a bimanual trajectory keeps the relative timing of the demonstration (the hand that
releases does so when the other hand is where the demonstration had it). p_e(s) interpolates each end
effector's positions linearly and rotations by slerp.

The official executor advances one waypoint per control step (ds = 1). The follower advances by
ds = max(ds_min, 1 / (1 + alpha_p * e_pos + alpha_r * e_rot)), where e_pos (m) and e_rot (rad) are the
largest errors over the end effectors between the current targets p_e(s) and the measured poses. Gains depend
on the region of the current waypoint: 'approach' for the ``approach_window`` waypoints before (and at) each
gripper open/close event of any end effector, 'bridge' for bridge waypoints when the config names such gains,
'default' otherwise (an approach window wins over bridge). Gripper events are never skipped: s is clamped
onto the next event index. When the path has used as many extra steps as it has waypoints
(``extra_steps_cap: segment``) the follower advances to the next integer waypoint (base speed).

Random-number accounting is the runtime's job; the follower only reports which original waypoints a step
newly covered (``new_slots``), so the runtime can consume exactly one global draw per original waypoint.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from tmg.core.poses import pose_gap
from tmg.generation.method.event_mix import sample_poses
from tmg.generation.method.grippers import GripperEventDetector, SignGripper, build_detector, event_indices

EPS = 1e-9
KIND_BRIDGE, KIND_SKILL = 0, 1


def advance_inverse(e_pos: float, e_rot: float, alpha_p: float, alpha_r: float, ds_min: float) -> float:
    """ds = max(ds_min, 1 / (1 + alpha_p * e_pos + alpha_r * e_rot)); equals 1 at zero error."""
    assert 0.0 < ds_min <= 1.0 and alpha_p >= 0.0 and alpha_r >= 0.0
    return max(ds_min, 1.0 / (1.0 + alpha_p * e_pos + alpha_r * e_rot))


def clamp_advance(s: float, ds: float, events, s_end: float) -> float:
    """s + ds, capped at s_end and clamped onto the first gripper event strictly ahead of s."""
    s_new = min(float(s) + float(ds), float(s_end))
    ahead = np.asarray([e for e in events if e > s + EPS], dtype=float)
    if ahead.size and s_new >= ahead[0] - EPS:
        s_new = float(ahead[0])
    return s_new


def region_table(n: int, events, window: int, kinds=None) -> list[str]:
    """Region label per waypoint: ``bridge`` for bridge kinds, ``approach`` for the ``window`` waypoints before
    (and at) each event (wins over bridge), ``rest`` otherwise.
    """
    region = ["rest"] * n
    if kinds is not None:
        for k, kind in enumerate(kinds):
            if int(kind) == KIND_BRIDGE:
                region[k] = "bridge"
    for e in events:
        for k in range(max(0, int(e) - int(window)), min(n, int(e) + 1)):
            region[k] = "approach"
    return region


@dataclass
class EndEffectorPath:
    """One end effector's share of a path: poses (n,4,4) and gripper commands (n,G)."""
    name: str
    poses: np.ndarray
    grippers: np.ndarray

    def __post_init__(self):
        """Coerce arrays and validate the pose shape."""
        self.poses = np.asarray(self.poses, dtype=float)
        self.grippers = np.asarray(self.grippers, dtype=float).reshape(len(self.poses), -1)
        if self.poses.ndim != 3 or self.poses.shape[1:] != (4, 4):
            raise ValueError(f"{self.name}: poses must have shape (n, 4, 4)")


class ErrorAdaptiveFollower:
    """The ``PathFollower`` of TrackMix (see the module docstring for the rule)."""
    def __init__(self, *, ds_min: float, regions: dict, approach_window: int, extra_steps_cap: str,
                 detector: GripperEventDetector | None = None):
        """Args:
            ds_min: smallest advance per step (keeps the target moving).
            regions: gains per region name (``default`` required; ``approach``, ``bridge`` optional).
            approach_window: waypoints before an event that use the approach gains.
            extra_steps_cap: ``segment`` (fall back to base speed once extra steps reach the path length) or ``none``.
            detector: gripper event detector (default: sign rule).
        """
        self.ds_min = float(ds_min)
        self.regions = {k: {"alpha_p": float(v["alpha_p"]), "alpha_r": float(v["alpha_r"])}
                        for k, v in dict(regions).items()}
        if "default" not in self.regions:
            raise KeyError("tracking.regions needs a 'default' entry")
        self.window = int(approach_window)
        if extra_steps_cap not in ("segment", "none"):
            raise ValueError("tracking.extra_steps_cap must be 'segment' or 'none'")
        self.cap = extra_steps_cap
        self.detector = detector or SignGripper()
        self.eefs: list[EndEffectorPath] = []

    @classmethod
    def factory(cls, cfg: dict):
        """Return a zero-argument constructor for the ``method.tracking`` config block; the runtime makes one
        follower per trajectory. ``factory.describe()`` gives the settings for logs.
        """
        cfg = dict(cfg)

        def make():
            return cls(ds_min=cfg.get("ds_min", 0.25), regions=cfg["regions"],
                       approach_window=cfg.get("approach_window", 10),
                       extra_steps_cap=cfg.get("extra_steps_cap", "segment"),
                       detector=build_detector(cfg.get("gripper_events")))

        make.describe = lambda: {"type": "error_adaptive", **{k: v for k, v in cfg.items() if k != "enabled"}}
        return make

    # one path ---------------------------------------------------------------------------------------
    def start_path(self, eefs: list[EndEffectorPath], noises, kinds, previous_states: dict | None = None):
        """``noises`` and ``kinds`` are per waypoint; ``previous_states`` maps an end effector name to its last
        known gripper state before this path (None for the first path of an attempt)."""
        if not eefs:
            raise ValueError("a path needs at least one end effector")
        lengths = {e.name: len(e.poses) for e in eefs}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"end effectors have different path lengths: {lengths}")
        self.eefs = list(eefs)
        self.n = len(eefs[0].poses)
        if self.n == 0:
            raise ValueError("empty path")
        self._noises = list(noises)
        self.kinds = [int(k) for k in kinds]
        if len(self._noises) != self.n or len(self.kinds) != self.n:
            raise ValueError("noises and kinds must have one entry per waypoint")
        previous_states = dict(previous_states or {})
        self.events_by_eef = {e.name: event_indices(self.detector, e.grippers, previous_states.get(e.name))
                              for e in eefs}
        self.events = sorted({k for ev in self.events_by_eef.values() for k in ev})
        self.region = region_table(self.n, self.events, self.window, self.kinds)
        self.s_end = float(self.n - 1)
        self.s = None
        self.steps = 0
        self.extra = 0
        self.next_slot = 0
        self._finished = False

    def finished(self) -> bool:
        """True once s reached the last waypoint."""
        return self._finished

    def targets(self, s: float) -> dict:
        """Target pose of every end effector at path variable ``s`` (linear position, slerp rotation)."""
        return {e.name: sample_poses(e.poses, np.asarray([s]))[0] for e in self.eefs}

    def grippers_at(self, k: int) -> dict:
        """Gripper command of every end effector at waypoint index ``k``."""
        return {e.name: np.array(e.grippers[k]) for e in self.eefs}

    def final_gripper_states(self) -> dict:
        """Gripper state of each end effector at the end of the path (for the next path's event detection)."""
        return {e.name: bool(self.detector.states(e.grippers)[-1]) for e in self.eefs}

    def gains(self, k: int) -> dict:
        """Gains of the region of waypoint ``k``."""
        return self.regions.get(self.region[k], self.regions["default"])

    def error(self, s: float, measured: dict) -> tuple[float, float]:
        """Largest position and rotation error over the end effectors between p_e(s) and the measured poses."""
        targets = self.targets(s)
        e_pos = e_rot = 0.0
        for name, target in targets.items():
            if name not in measured:
                raise KeyError(f"measured pose of end effector {name!r} missing")
            p, r = pose_gap(target, measured[name])
            e_pos, e_rot = max(e_pos, p), max(e_rot, r)
        return e_pos, e_rot

    def next_target(self, measured: dict):
        """Advance once. Returns (targets by eef, grippers by eef, noise scale of waypoint k, record) with
        record = {s_before, s_after, ds_proposed, ds_actual, k, region, capped, new_slots, e_pos, e_rot, kind}."""
        if self._finished:
            raise IndexError("path already finished")
        before = self.s
        k_before = 0 if before is None else int(math.floor(before + EPS))
        e_pos, e_rot = self.error(0.0 if before is None else before, measured)
        capped = self.cap == "segment" and self.extra >= self.n
        if before is None:
            s, ds_proposed = 0.0, 0.0
        elif capped:
            s = min(float(math.floor(before + EPS) + 1), self.s_end)
            ds_proposed = s - before
        else:
            g = self.gains(k_before)
            ds_proposed = advance_inverse(e_pos, e_rot, g["alpha_p"], g["alpha_r"], self.ds_min)
            s = clamp_advance(before, ds_proposed, self.events, self.s_end)
        k = int(math.floor(s + EPS))
        new_slots = list(range(self.next_slot, k + 1))
        self.next_slot = max(self.next_slot, k + 1)
        self.steps += 1
        self.extra = self.steps - (k + 1)
        self._finished = s >= self.s_end - EPS
        if self._finished:
            new_slots += list(range(self.next_slot, self.n))
            self.next_slot = self.n
        self.s = s
        record = {"s_before": -1.0 if before is None else float(before), "s_after": float(s),
                  "ds_proposed": float(ds_proposed), "ds_actual": float(s - (0.0 if before is None else before)),
                  "k": k, "region": self.region[k_before], "capped": bool(capped), "new_slots": new_slots,
                  "e_pos": float(e_pos), "e_rot": float(e_rot), "kind": self.kinds[k]}
        return self.targets(s), self.grippers_at(k), self._noises[k], record
