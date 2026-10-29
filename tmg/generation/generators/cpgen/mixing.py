"""Source mixing for CP-Gen (mixup exploration): transform each source to the current scene first, then mix.

CP-Gen does not replay a source segment. For constraint k of the attempt it re-solves the annotated keypoint
constraint of the attempt's source demonstration (the primary, A) for the current object poses: for every step t of
the constraint window ``compute_X_W_eef_des`` gives the desired end-effector pose E_A(t) in the world, inverse
kinematics turns it into a joint configuration q(t), and cuRobo plans from the current configuration to q(0).

Mixing replaces E_A(t) by a blend with the same constraint of a second source (the donor, B), also solved for the
current scene:

1. Decision. A ``MixContext`` is built from the sources' recorded data (``source_context``) and the configured
   mixer's ``decide`` picks the donor and the primary's share exactly as it does for MimicGen (success group,
   transform size or junction gap, anchor-rotation compatibility). The reference object of constraint k is
   ``obj_names[0]`` for a robot-object constraint (the object the hand approaches) and ``obj_names[1]`` for an
   object-object constraint (the object the held object is placed on). Source i contributes its end-effector site
   poses over the window (``src_obs``), its reference-object pose at the window's first step and its gripper commands.
   The current scene gives the reference object's pose and the current end-effector pose, from which cuRobo plans.
2. Donor solve. The donor's constraint k is solved for the current scene by the official ``compute_X_W_eef_des`` at
   every step u of its window, with the symmetry transforms CP-Gen chose for the primary, giving E_B(u).
3. Blend. The donor is time-warped onto the primary's timeline: its gripper events are moved to the primary's event
   steps and each interval between events is stretched linearly (``event_mix.align_segments`` with weight 1). At each
   step t of the primary, ``E(t) = blend(E_A(t), E_B(warp(t)), w(t))``: positions ``w E_A + (1 - w) E_B`` and the
   rotation ``R_A exp((1 - w) log(R_A^T R_B))``. ``w(t)`` is the mixer's share (one number for the success and
   transform mixers, the head schedule for the junction mixer).
4. CP-Gen continues unchanged from E(t): its inverse kinematics, the gripper commands and post-constraint behaviour
   of the primary, and the cuRobo plan to the configuration of E(0).

The primary's timeline is kept (MimicGen's constant-weight mix instead moves the events to the blend of both
sources' event steps) because CP-Gen solves one configuration per step of the primary's window and copies the
primary's gripper command of that step.

Object-object constraints: CP-Gen overwrites the gripper command of every solved step of such a constraint with 1
(closed, the held object stays welded), so their event series is all closed here too.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from tmg.generation.method.event_mix import align_segments, blend_poses
from tmg.generation.method.interfaces import MixContext


def reference_object(constraint) -> str:
    """The object whose pose a constraint is solved relative to (module docstring)."""
    names = list(constraint.obj_names or [])
    if constraint.constraint_type == "object-object":
        return names[1]
    return names[0]


def window_length(constraint) -> int:
    """Steps CP-Gen solves for the constraint (its ``timesteps``)."""
    return len(constraint.timesteps)


def gripper_series(constraint) -> np.ndarray:
    """``(n, 1)`` gripper commands CP-Gen executes over the window (all closed for object-object constraints)."""
    n = window_length(constraint)
    if constraint.constraint_type == "object-object":
        return np.ones((n, 1))
    return np.asarray(constraint.src_gripper_action, dtype=float).reshape(-1, 1)[:n]


def site_poses(constraint, quat2mat: Callable) -> np.ndarray:
    """``(n, 4, 4)`` recorded end-effector site poses of the source over the window (``robot0_eef_pos`` and the
    x, y, z, w quaternion ``robot0_eef_quat_site``, read with CP-Gen's own ``quat2mat``)."""
    n = window_length(constraint)
    pos = np.asarray(constraint.src_obs["robot0_eef_pos"], dtype=float)[:n]
    quat = np.asarray(constraint.src_obs["robot0_eef_quat_site"], dtype=float)[:n]
    out = np.tile(np.eye(4), (n, 1, 1))
    out[:, :3, 3] = pos
    out[:, :3, :3] = np.stack([quat2mat(q) for q in quat])
    return out


def current_site_pose(obs: dict, quat2mat: Callable) -> np.ndarray:
    """The current end-effector site pose, the pose cuRobo plans from."""
    out = np.eye(4)
    out[:3, 3] = np.asarray(obs["robot0_eef_pos"], dtype=float).reshape(3)
    out[:3, :3] = quat2mat(np.asarray(obs["robot0_eef_quat_site"], dtype=float).reshape(4))
    return out


def source_context(sequences, constraint_idx: int, primary: int, obs: dict, quat2mat: Callable) -> MixContext:
    """The ``MixContext`` of constraint ``constraint_idx`` with source ``primary`` selected (module docstring, step 1).

    ``sequences`` are the policy's constraint sequences (one per source demonstration)."""
    constraints = [seq[constraint_idx] for seq in sequences]
    ref = reference_object(constraints[primary])
    segments, object_poses = [], []
    for c in constraints:
        targets = site_poses(c, quat2mat)
        segments.append((targets[0].copy(), targets, gripper_series(c)))
        object_poses.append(np.asarray(c.src_obj_pose[ref], dtype=float)[0])
    eef, targets, grippers = segments[primary]
    extra = {"current_object_pose": np.asarray(obs[f"{ref}_pose"], dtype=float),
             "junction_pose": current_site_pose(obs, quat2mat)}
    return MixContext(subtask_ind=int(constraint_idx), primary=int(primary), primary_eef_start=eef, primary_targets=targets,
                      primary_gripper=grippers, src_object_poses=object_poses, src_segments=segments,
                      num_sources=len(constraints), has_reference_object=True, rng=np.random.default_rng(0), extra=extra)


def weights_of(weight, n: int) -> np.ndarray:
    """The primary's share at each of ``n`` steps: a constant, or the mixer's schedule ``weight(n)``."""
    if callable(weight):
        w = np.asarray(weight(n), dtype=float).reshape(-1)
    else:
        w = np.full(n, float(weight))
    if len(w) != n or np.any(~np.isfinite(w)) or np.any((w < 0) | (w > 1)):
        raise ValueError(f"mix weights must be {n} values in [0, 1]")
    return w


class MixPlan:
    """The donor's solved poses on the primary's timeline and the primary's share per step (module docstring, step 3)."""

    def __init__(self, primary_gripper, donor_poses, donor_gripper, weight):
        primary_gripper = np.asarray(primary_gripper, dtype=float).reshape(-1, 1)
        n = len(primary_gripper)
        placeholder = np.tile(np.eye(4), (n, 1, 1))     # weight 1 keeps the primary's timeline; only B's warp is used
        _a, self.donor, _ia, ib, meta = align_segments(placeholder, primary_gripper, donor_poses,
                                                        np.asarray(donor_gripper, dtype=float).reshape(-1, 1), 1.0)
        self.weights = weights_of(weight, n)
        self.meta = {"samples": int(meta["samples"]), "source_b_samples": int(meta["source_b_samples"]),
                     "event_labels": meta["event_labels"], "source_a_event_steps": meta["source_a_event_steps"],
                     "source_b_event_steps": meta["source_b_event_steps"], "source_b_index": [int(v) for v in ib],
                     "weights": [round(float(v), 4) for v in self.weights]}

    def __len__(self) -> int:
        return len(self.weights)

    def pose(self, t: int, primary_pose: np.ndarray) -> np.ndarray:
        """The mixed pose of step ``t`` from the primary's solved pose of that step."""
        if not 0 <= t < len(self):
            raise IndexError(f"step {t} outside the primary's window of {len(self)} steps")
        return blend_poses(np.asarray(primary_pose, dtype=float)[None], self.donor[t][None], float(self.weights[t]))[0]
