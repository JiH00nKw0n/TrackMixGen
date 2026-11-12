"""Logistic source mixing for CP-Gen: solve every retained source for the current scene first, then blend.

CP-Gen does not replay a source segment. For constraint k of the attempt it re-solves the annotated keypoint constraint
of the attempt's source demonstration (the primary) for the current object poses: for every step t of the constraint
window ``compute_X_W_eef_des`` gives the desired end-effector pose E_primary(t), inverse kinematics turns it into a joint
configuration, and cuRobo plans from the current configuration to the first one.

With the logistic mixer (``mixing.type: logistic``) the runtime replaces E_primary(t) by a weighted blend of the same
constraint of several sources, each solved for the current scene:

1. Prediction. For every source j the success model (the task's coefficient file, ``tmg.generation.method.success_model``)
   predicts the probability that constraint k succeeds with source j from the scene features (x and y distance and
   rotation angle between the constraint's reference object now and in source j at its window start) and, from the
   second constraint on, the in-hand features (position distance and rotation angle between the previous constraint's
   reference object in the end-effector frame now and in source j). These are the quantities the runtime records for
   every constraint, so the fit and the mixing read the same values.
2. Weights. The retained set is the sources whose prediction is at least the mean over all sources and whose gripper
   events match the primary's (only those can be put on the primary's timeline); weights are the predictions normalized
   over the retained set (``logistic_mix.above_mean_weights`` with the compatible sources).
3. Solve. Every retained source other than the primary is solved for the current scene by the official
   ``compute_X_W_eef_des`` at every step of its own window, with the symmetry transforms CP-Gen chose for the primary.
4. Blend on the primary's timeline. Each retained source is time-warped onto the primary's window (gripper events moved
   to the primary's event steps, linear stretching between them, ``event_mix.align_segments``). At each step t the
   blended pose has the weighted mean position and the weighted mean rotation (``Rotation.mean`` with the weights). The
   primary keeps its timeline, gripper commands and post-constraint behaviour; CP-Gen's inverse kinematics and cuRobo
   then run on the blended poses.

The reference object of constraint k is ``obj_names[0]`` for a robot-object constraint and ``obj_names[1]`` for an
object-object constraint (the object the held one is placed on). Object-object constraints keep the gripper closed in
CP-Gen, so their event series is all closed.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.spatial.transform import Rotation

from tmg.generation.method.event_mix import _matching_events, align_segments
from tmg.generation.method.logistic_mix import above_mean_weights
from tmg.generation.method.success_model import in_hand, inhand_features, scene_features


def reference_object(constraint) -> str:
    """The object a constraint is solved relative to (module docstring)."""
    names = list(constraint.obj_names or [])
    if getattr(constraint, "constraint_type", None) == "object-object" and len(names) > 1:
        return names[1]
    return names[0]


def window_length(constraint) -> int:
    """Steps CP-Gen solves for the constraint (its ``timesteps``)."""
    return len(constraint.timesteps)


def gripper_series(constraint) -> np.ndarray:
    """``(n, 1)`` gripper commands CP-Gen executes over the window (all closed for object-object constraints)."""
    n = window_length(constraint)
    if getattr(constraint, "constraint_type", None) == "object-object":
        return np.ones((n, 1))
    return np.asarray(constraint.src_gripper_action, dtype=float).reshape(-1, 1)[:n]


def site_pose(pos, quat_xyzw) -> np.ndarray:
    """4x4 pose from a position and an (x, y, z, w) quaternion (robosuite's convention)."""
    out = np.eye(4)
    out[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    out[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float).reshape(4)).as_matrix()
    return out


def _quat_key(d) -> str:
    return "robot0_eef_quat_site" if "robot0_eef_quat_site" in d else "robot0_eef_quat"


def features(constraints: list, obs: dict, carried: str | None):
    """``(scene (J,3), inhand (J,2) or None)`` of every source for one constraint (module docstring, step 1). ``carried``
    is the previous constraint's reference object (None for the first constraint)."""
    ref = reference_object(constraints[0])
    current = np.asarray(obs[f"{ref}_pose"], dtype=float)
    scene = np.array([scene_features(current, np.asarray(c.src_obj_pose[ref], dtype=float)[0]) for c in constraints])
    if carried is None:
        return scene, None
    if f"{carried}_pose" not in obs:
        return scene, np.zeros((len(constraints), 2))
    now = in_hand(site_pose(obs["robot0_eef_pos"], obs[_quat_key(obs)]), np.asarray(obs[f"{carried}_pose"], dtype=float))
    rows = []
    for c in constraints:
        src_obs = getattr(c, "src_obs", None) or {}
        src_obj = getattr(c, "src_obj_pose", None) or {}
        if carried not in src_obj or "robot0_eef_pos" not in src_obs:
            rows.append(np.zeros(2))          # the fit used 0 where the source pose of the carried object is unknown
            continue
        src = in_hand(site_pose(np.asarray(src_obs["robot0_eef_pos"])[0], np.asarray(src_obs[_quat_key(src_obs)])[0]),
                      np.asarray(src_obj[carried], dtype=float)[0])
        rows.append(inhand_features(now, src))
    return scene, np.array(rows)


def decide(model, sequences: list, idx: int, primary: int, obs: dict, carried: str | None) -> tuple[dict, np.ndarray]:
    """Predictions and weights of constraint ``idx`` (module docstring, steps 1 and 2): ``(record, weights (J,))``."""
    constraints = [seq[idx] for seq in sequences]
    scene, inhand = features(constraints, obs, carried)
    sub = model.for_subtask(idx)
    if inhand is None and any(f in sub.slopes for f in ("inhand_position", "inhand_rotation")):
        inhand = np.zeros((len(constraints), 2))
    p = sub.predict(scene, inhand)
    series = [gripper_series(c)[:, 0] for c in constraints]
    compatible = [j for j in range(len(constraints)) if j == primary or _matching_events(series[primary], series[j]) is not None]
    weights = above_mean_weights(p, compatible)
    record = {"subtask": int(idx), "source_a": int(primary), "rule": "above_mean",
              "predicted": [round(float(v), 4) for v in p], "weights": [round(float(v), 4) for v in weights],
              "incompatible": [j for j in range(len(constraints)) if j not in compatible],
              "scene_features": np.round(scene, 4).tolist(), "carried_object": carried}
    if inhand is not None:
        record["inhand_features"] = np.round(inhand, 4).tolist()
    return record, weights


class MultiMixPlan:
    """The retained sources' solved poses on the primary's timeline and their weights (module docstring, step 4)."""

    def __init__(self, primary_gripper, others: dict, weights: dict):
        """``others`` maps a source index to ``(solved poses (n_j,4,4), gripper series (n_j,1))``; ``weights`` maps every
        retained source (the primary under key ``"primary"``) to its weight."""
        primary_gripper = np.asarray(primary_gripper, dtype=float).reshape(-1, 1)
        n = len(primary_gripper)
        placeholder = np.tile(np.eye(4), (n, 1, 1))
        self.n = n
        self.w_primary = float(weights.get("primary", 0.0))
        self.aligned, self.w, self.meta = {}, {}, {"samples": int(n), "sources": {}}
        for j, (poses, grip) in others.items():
            _a, warped, _ia, ib, meta = align_segments(placeholder, primary_gripper, np.asarray(poses, dtype=float),
                                                       np.asarray(grip, dtype=float).reshape(-1, 1), 1.0)
            self.aligned[j] = np.asarray(warped, dtype=float)
            self.w[j] = float(weights[j])
            self.meta["sources"][str(j)] = {"weight": round(float(weights[j]), 4), "source_samples": int(meta["source_b_samples"]),
                                            "event_steps": meta["source_b_event_steps"]}
        self.meta["primary_weight"] = round(self.w_primary, 4)

    def pose(self, t: int, primary_pose: np.ndarray) -> np.ndarray:
        """The blended pose of step ``t`` from the primary's solved pose of that step."""
        if not 0 <= t < self.n:
            raise IndexError(f"step {t} outside the primary's window of {self.n} steps")
        poses = ([np.asarray(primary_pose, dtype=float)] if self.w_primary > 0 else []) + [self.aligned[j][t] for j in self.aligned]
        w = np.array(([self.w_primary] if self.w_primary > 0 else []) + [self.w[j] for j in self.aligned], dtype=float)
        w = w / w.sum()
        out = np.eye(4)
        out[:3, 3] = np.einsum("i,ij->j", w, np.stack([p[:3, 3] for p in poses]))
        out[:3, :3] = Rotation.from_matrix(np.stack([p[:3, :3] for p in poses])).mean(weights=w).as_matrix()
        return out


def solve_sources(official, policy, sequences: list, idx: int, sources: list, obs: dict, transforms: dict) -> dict:
    """Solve constraint ``idx`` of each source in ``sources`` for the current scene (module docstring, step 3)."""
    out = {}
    for j in sources:
        c = sequences[j][idx]
        poses = np.stack([official.compute_X_W_eef_des(c, obs, policy.robot_configuration, policy.eef_configuration,
                                                       src_t_idx=u, ignore_scaling=policy.ignore_scaling, **transforms)
                          for u in range(window_length(c))])
        out[j] = (poses, gripper_series(c))
    return out


def quat_reader(official) -> Callable:
    return official.quat2mat
