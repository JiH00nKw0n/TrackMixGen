"""Event-aligned mixing of subtask segments (pure numpy/scipy; no simulator).

Ported from the 2026-10 Soft/Dex experiment code (event_aligned_mix.py) with the same semantics: the segments must have
the same ordered gripper open/close events; event times are the weighted mean of the segments' event steps, each interval
between events is time-warped separately, positions are blended linearly and rotations by the weighted rotation mean
(``mix_many``, SkillMix). ``align_segments`` puts two segments on one timeline (CP-Gen puts each retained source on the
primary's). Gripper state means the COMMANDED open/close (positive = close), not measured contact.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from tmg.generation.method.grippers import (  # IncompatibleGripperEvents and gripper_states are also re-exported
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


BLEND_PARTS = ("both", "position", "rotation")


def _check_commands(gripper_a, gripper_b) -> tuple[np.ndarray, np.ndarray]:
    ga = np.asarray(gripper_a, dtype=float)
    gb = np.asarray(gripper_b, dtype=float)
    if ga.ndim not in (1, 2) or gb.ndim not in (1, 2):
        raise ValueError("gripper commands must have shape (N,) or (N, 1)")
    if (ga.ndim == 2 and ga.shape[1] != 1) or (gb.ndim == 2 and gb.shape[1] != 1):
        raise ValueError("expected one gripper command per control step")
    return ga, gb


def align_segments(poses_a, gripper_a, poses_b, gripper_b, weight_a: float):
    """Put two segments on one timeline whose gripper events lie at the ``weight_a`` blend of both sources' event
    steps (``weight_a`` 1 keeps A's timeline, 0 B's). Each interval between events is time-warped separately.

    Returns ``(poses_a_on_timeline, poses_b_on_timeline, index_a, index_b, meta)``: both pose arrays have the mixed
    length; ``index_a``/``index_b`` are each source's step at every mixed step (for the gripper commands);
    ``meta`` holds the event steps. Raises ``IncompatibleGripperEvents`` when the event structures differ.
    """
    ga, gb = _check_commands(gripper_a, gripper_b)
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
    ia = np.minimum(np.floor(ta + 1e-9).astype(int), len(ga) - 1)
    ib = np.minimum(np.floor(tb + 1e-9).astype(int), len(gb) - 1)
    if not np.array_equal(sa[ia], sb[ib]):
        raise IncompatibleGripperEvents("warped commands disagree on the event phase")
    meta = dict(samples=n, source_a_samples=len(pa), source_b_samples=len(pb),
                event_labels=la, source_a_event_steps=ea.tolist(), source_b_event_steps=eb.tolist(),
                desired_event_steps=desired[1:-1].tolist(), actual_event_steps=knots[1:-1].tolist(),
                initial_commanded_closed=bool(sa[0]), source_a_index=ia.tolist(), source_b_index=ib.tolist())
    return sample_poses(pa, ta), sample_poses(pb, tb), ia, ib, meta


def mix_many(poses, series, weights, parts: str = "both"):
    """Convex blend of any number of segments: ``(mixed poses, mixed event series, alignment metadata)``.

    ``poses[k]`` are segment k's target poses in its own frozen reference-object frame, ``series[k]`` its scalar event
    series (positive = close), and ``weights`` one non-negative share per segment that sums to 1. Segments with weight 0
    are ignored; with one non-zero weight that segment is returned unchanged. The gripper events of the kept segments
    must share the initial state and the order (``IncompatibleGripperEvents`` otherwise). Each event is placed at the
    weighted mean of the segments' event steps and every interval between events is time-warped separately; positions
    are the weighted mean, rotations the weighted rotation mean (``Rotation.mean``), and the event series the weighted
    mean. ``parts`` keeps the rotations (``position``) or the positions (``rotation``) of the largest-weight segment.
    """
    if parts not in BLEND_PARTS:
        raise ValueError(f"parts must be one of {BLEND_PARTS}, got {parts!r}")
    w = np.asarray(weights, dtype=float).reshape(-1)
    if len(w) != len(poses) or len(w) != len(series):
        raise ValueError(f"{len(w)} weights for {len(poses)} pose arrays and {len(series)} event series")
    if not np.all(np.isfinite(w)) or np.any(w < 0) or abs(w.sum() - 1) > 1e-6:
        raise ValueError("weights must be finite, non-negative and sum to 1")
    keep = [k for k in range(len(w)) if w[k] > 0]
    anchor = int(np.argmax(w))
    if len(keep) == 1:
        return np.array(poses[anchor], dtype=float), np.array(series[anchor], dtype=float).reshape(-1), {
            "identity_source": anchor, "weights": w.tolist(), "samples": len(poses[anchor])}
    segs = []
    for k in keep:
        g = np.asarray(series[k], dtype=float).reshape(-1)
        segs.append((k, _check_poses(poses[k], len(g)), g))
    base = events(segs[0][2])
    for k, _, g in segs[1:]:
        if _matching_events(segs[0][2], g) is None:
            raise IncompatibleGripperEvents(f"segment {k}: open/close event order or initial commanded state differs from segment {segs[0][0]}")
    wk = np.array([w[k] for k, _, _ in segs])
    knots_k = [np.r_[0, events(g)[1], len(p)].astype(float) for _, p, g in segs]
    desired = sum(wi * kn for wi, kn in zip(wk, knots_k))
    knots = np.rint(desired).astype(int)
    if np.any(np.diff(knots) <= 0):
        raise IncompatibleGripperEvents("rounding collapses a gripper event interval")
    n = int(knots[-1])
    times = np.arange(n)
    aligned, index, grip = [], [], np.zeros(n)
    for (k, p, g), kn, wi in zip(segs, knots_k, wk):
        t = np.interp(times, knots, kn)
        i = np.minimum(np.floor(t + 1e-9).astype(int), len(g) - 1)
        aligned.append(sample_poses(p, t))
        index.append(i)
        grip += wi * g[i]
    states = [gripper_states(g)[i] for (_, _, g), i in zip(segs, index)]
    if any(not np.array_equal(states[0], s) for s in states[1:]):
        raise IncompatibleGripperEvents("warped commands disagree on the event phase")
    out = np.tile(np.eye(4), (n, 1, 1))
    a = keep.index(anchor)
    out[:, :3, 3] = sum(wi * al[:, :3, 3] for wi, al in zip(wk, aligned))
    for t in range(n):
        out[t, :3, :3] = Rotation.from_matrix(np.stack([al[t, :3, :3] for al in aligned])).mean(weights=wk).as_matrix()
    if parts == "position":
        out[:, :3, :3] = aligned[a][:, :3, :3]
    if parts == "rotation":
        out[:, :3, 3] = aligned[a][:, :3, 3]
    meta = dict(samples=n, weights=w.tolist(), segments=[k for k, _, _ in segs], event_labels=base[2],
                event_steps={str(k): events(g)[1].tolist() for k, _, g in segs},
                desired_event_steps=desired[1:-1].tolist(), actual_event_steps=knots[1:-1].tolist(),
                source_index={str(k): i.tolist() for (k, _, _), i in zip(segs, index)})
    return out, grip, meta
