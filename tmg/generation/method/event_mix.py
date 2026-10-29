"""Event-aligned mixing of two subtask segments (pure numpy/scipy; no simulator).

Ported from the 2026-10 Soft/Dex experiment code (event_aligned_mix.py) with the same semantics: the two
segments must have the same ordered gripper open/close events; event times are blended with ``weight_a``,
each interval between events is time-warped separately, positions are blended linearly and rotations by a
fraction of the shortest rotation from A to B. ``weight_a`` 1 or 0 returns the corresponding segment
unchanged. Gripper state means the COMMANDED open/close (positive = close), not measured contact.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tmg.generation.method.grippers import (  # noqa: F401  IncompatibleGripperEvents and gripper_states are re-exported
    IncompatibleGripperEvents,
    change_indices,
    gripper_states,
)


def events(commands) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(states, event step indices, labels 'close'/'open'). An event index is the first step of the new state."""
    states = gripper_states(commands)
    change = change_indices(states)
    labels = ["close" if states[i] else "open" for i in change]
    return states, change, labels


def _matching_events(gripper_a, gripper_b):
    """Events of both command sequences when they share the initial state and the event order, else None.

    Returns ``(states_a, events_a, states_b, events_b, labels)``; ``labels`` is the common 'close'/'open' sequence.
    """
    sa, ea, la = events(gripper_a)
    sb, eb, lb = events(gripper_b)
    if bool(sa[0]) != bool(sb[0]) or la != lb:
        return None
    return sa, ea, sb, eb, la


def _check_poses(poses, n: int) -> np.ndarray:
    """Validate a ``(n, 4, 4)`` pose array and return it as float."""
    poses = np.asarray(poses, dtype=float)
    if poses.shape != (n, 4, 4):
        raise ValueError(f"expected poses of shape ({n}, 4, 4), got {poses.shape}")
    return poses


def sample_poses(poses: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Pose at fractional indices: linear translation, slerp rotation."""
    poses = np.asarray(poses, dtype=float)
    n = len(poses)
    times = np.clip(np.asarray(times, dtype=float), 0, n - 1)
    out = np.tile(np.eye(4), (len(times), 1, 1))
    for k in range(3):
        out[:, k, 3] = np.interp(times, np.arange(n), poses[:, k, 3])
    if n == 1:
        out[:, :3, :3] = poses[0, :3, :3]
    else:
        out[:, :3, :3] = Slerp(np.arange(n), Rotation.from_matrix(poses[:, :3, :3]))(times).as_matrix()
    return out


def blend_poses(a: np.ndarray, b: np.ndarray, weight_a: float) -> np.ndarray:
    """Linear translation blend and shortest-rotation blend of two aligned pose arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("aligned pose arrays must have equal shapes")
    out = np.tile(np.eye(4), (len(a), 1, 1))
    out[:, :3, 3] = weight_a * a[:, :3, 3] + (1 - weight_a) * b[:, :3, 3]
    ra = Rotation.from_matrix(a[:, :3, :3])
    rb = Rotation.from_matrix(b[:, :3, :3])
    delta = ra.inv() * rb
    out[:, :3, :3] = (ra * Rotation.from_rotvec((1 - weight_a) * delta.as_rotvec())).as_matrix()
    return out


def mix_segments(poses_a, gripper_a, poses_b, gripper_b, weight_a: float):
    """Return (mixed poses, mixed gripper commands, alignment metadata).

    Inputs are the target poses of a subtask segment, each expressed in its own frozen reference-object
    frame, and the per-step gripper commands of shape (N,) or (N, 1).
    """
    if not np.isfinite(weight_a) or not 0 <= weight_a <= 1:
        raise ValueError("weight_a must lie in [0, 1]")
    if weight_a in (0, 1):
        p, g = (poses_a, gripper_a) if weight_a == 1 else (poses_b, gripper_b)
        return np.array(p, dtype=float), np.array(g, dtype=float), {
            "identity_source": "A" if weight_a == 1 else "B", "weight_a": float(weight_a), "samples": len(p)}
    ga = np.asarray(gripper_a, dtype=float)
    gb = np.asarray(gripper_b, dtype=float)
    if ga.ndim not in (1, 2) or gb.ndim not in (1, 2):
        raise ValueError("gripper commands must have shape (N,) or (N, 1)")
    if (ga.ndim == 2 and ga.shape[1] != 1) or (gb.ndim == 2 and gb.shape[1] != 1):
        raise ValueError("expected one gripper command per control step")
    pa = _check_poses(poses_a, len(ga))
    pb = _check_poses(poses_b, len(gb))
    matched = _matching_events(ga, gb)
    if matched is None:
        raise IncompatibleGripperEvents("open/close event order or initial commanded state differs")
    sa, ea, sb, eb, la = matched
    ka = np.r_[0, ea, len(pa)].astype(float)
    kb = np.r_[0, eb, len(pb)].astype(float)
    desired = weight_a * ka + (1 - weight_a) * kb
    knots = np.rint(desired).astype(int)
    if np.any(np.diff(knots) <= 0):
        raise IncompatibleGripperEvents("rounding collapses a gripper event interval")
    n = int(knots[-1])
    times = np.arange(n)
    ta = np.interp(times, knots, ka)
    tb = np.interp(times, knots, kb)
    mixed = blend_poses(sample_poses(pa, ta), sample_poses(pb, tb), weight_a)
    ia = np.minimum(np.floor(ta + 1e-9).astype(int), len(ga) - 1)
    ib = np.minimum(np.floor(tb + 1e-9).astype(int), len(gb) - 1)
    if not np.array_equal(sa[ia], sb[ib]):
        raise IncompatibleGripperEvents("warped commands disagree on the event phase")
    grip = weight_a * ga.reshape(-1)[ia] + (1 - weight_a) * gb.reshape(-1)[ib]
    if ga.ndim == 2:
        grip = grip[:, None]
    meta = dict(weight_a=float(weight_a), samples=n, source_a_samples=len(pa), source_b_samples=len(pb),
                event_labels=la, source_a_event_steps=ea.tolist(), source_b_event_steps=eb.tolist(),
                desired_event_steps=desired[1:-1].tolist(), actual_event_steps=knots[1:-1].tolist(),
                initial_commanded_closed=bool(sa[0]), source_a_index=ia.tolist(), source_b_index=ib.tolist())
    return mixed, grip, meta


def anchor_rotation_degrees(poses_a, gripper_a, poses_b, gripper_b) -> list[float] | None:
    """Rotation differences (deg) between the two segments at the anchors (start, every close event, end), or
    None when the gripper event structure differs."""
    matched = _matching_events(gripper_a, gripper_b)
    if matched is None:
        return None
    _, ea, _, eb, la = matched
    a = np.asarray(poses_a, dtype=float)
    b = np.asarray(poses_b, dtype=float)
    pairs = [(0, 0), (len(a) - 1, len(b) - 1)] + [(int(i), int(j)) for i, j, label in zip(ea, eb, la)
                                                   if label == "close"]
    out = []
    for i, j in pairs:
        cos = (np.sum(a[i, :3, :3] * b[j, :3, :3]) - 1) / 2
        out.append(float(np.degrees(np.arccos(np.clip(cos, -1, 1)))))
    return out
