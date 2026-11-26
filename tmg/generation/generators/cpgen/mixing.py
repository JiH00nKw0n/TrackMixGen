"""SkillMix for CP-Gen: solve every retained source for the current scene first, then blend.

CP-Gen does not replay a source segment. For constraint k of the attempt it re-solves the annotated keypoint constraint
of the attempt's source demonstration (the primary) for the current object poses: for every step t of the constraint
window ``compute_X_W_eef_des`` gives the desired end-effector pose E_primary(t), inverse kinematics turns it into a joint
configuration, and cuRobo plans from the current configuration to the first one.

With SkillMix (``mixing.type: skillmix``) the runtime replaces E_primary(t) by a weighted blend of the same constraint of
several sources, each solved for the current scene:

1. Features. For every source j the scene features (x and y distance and rotation angle between the constraint's
   reference object now and in source j at its window start) and, from the second constraint on, the in-hand features
   (position distance and rotation angle between the previous constraint's reference object in the end-effector frame
   now and in source j; 0 where unknown). These are the quantities the runtime records for every constraint, so the fit
   and the mixing read the same values.
2. Weights (``SkillMixer.decide`` of ``tmg.generation.method.skillmix``, the same call MimicGen's mixing makes). The
   sources whose gripper events match the primary's (the source CP-Gen drew for the attempt, as in MimicGen) are
   compatible. ``mixing.rule: softmax``: the scores ``s_j = a_j + g . phi_j`` of the task's rule file and the softmax of
   ``s / d`` over the compatible sources (one source alone when ``d <= 0``, ``mixing.single_source``).
   ``mixing.rule: randmix``: random Dirichlet weights over the compatible sources, drawn once per constraint and attempt
   (the randmix pools of the fit).
3. Symmetry alignment and solve. A source can be recorded in a different but equivalent representation of the same
   grasp: the reference object may be symmetric (the constraint's ``symmetries``, e.g. the hammer turned 180 degrees about
   its z axis) and the parallel gripper is symmetric under 180 degrees about its approach axis (CP-Gen's ``reflect_eef``).
   CP-Gen's ``find_best_transform`` picks one combination of these transforms for the primary (the one whose desired
   pose at the window start is closest to the current end-effector pose). Each other retained source picks its own
   combination from the same candidate list, the one whose desired pose at the window start is closest to the primary's
   desired pose there (as MimicGen aligns to the primary; same score as ``find_best_transform``: 5 per metre plus 1 per
   radian; the compositions of two annotated object symmetries are candidates too, e.g. Hammer Cleanup's placement needs
   the hammer turned about z and y at once). Every retained source is then solved for the current scene by the official
   ``compute_X_W_eef_des`` at every step of its own window. Without the alignment two sources that hold the hammer from
   opposite sides are averaged into a pose in the middle of the hammer (Hammer Cleanup pilot, 2026-11-13).
4. Blend on the weighted timeline (``event_mix.mix_many``, the blend MimicGen uses). Each gripper event is placed at the
   weighted mean of the sources' event steps and the window length is the weighted mean of their lengths; every interval
   is time-warped separately; at each step the blended pose has the weighted mean position and the weighted mean
   rotation. One retained source is returned unchanged (its own window, its own speed), as in MimicGen. CP-Gen's step
   loop then runs on a copy of the anchor's (the retained source with the largest weight) constraint whose per-step data are resampled onto the blended timeline (``window_on``), so the
   number of solved steps, the gripper commands and the gripper opening follow the blended timeline; the desired pose of
   every step is the blended one (``MixedWindow.pose``), and CP-Gen's inverse kinematics and cuRobo run on it. With one
   retained source this is exactly CP-Gen's own solve of that source.

The reference object of constraint k is ``obj_names[0]`` for a robot-object constraint and ``obj_names[1]`` for an
object-object constraint (the object the held one is placed on). Object-object constraints keep the gripper closed in
CP-Gen, so their event series is all closed.
"""
from __future__ import annotations

import copy

import numpy as np
from scipy.spatial.transform import Rotation

from tmg.generation.method.event_mix import events, mix_many
from tmg.generation.method.skillmix import gripper_flip, pose_distance, symmetry_label, symmetry_score
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


def quat_key(d) -> str:
    """The end-effector site quaternion's key in an observation (or source observation) mapping."""
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
    now = in_hand(site_pose(obs["robot0_eef_pos"], obs[quat_key(obs)]), np.asarray(obs[f"{carried}_pose"], dtype=float))
    rows = []
    for c in constraints:
        src_obs = getattr(c, "src_obs", None) or {}
        src_obj = getattr(c, "src_obj_pose", None) or {}
        if carried not in src_obj or "robot0_eef_pos" not in src_obs:
            rows.append(np.zeros(2))          # the fit used 0 where the source pose of the carried object is unknown
            continue
        src = in_hand(site_pose(np.asarray(src_obs["robot0_eef_pos"])[0], np.asarray(src_obs[quat_key(src_obs)])[0]),
                      np.asarray(src_obj[carried], dtype=float)[0])
        rows.append(inhand_features(now, src))
    return scene, np.array(rows)


def decide(mixer, sequences: list, idx: int, primary: int, obs: dict, carried: str | None) -> tuple[dict, np.ndarray]:
    """Features and weights of constraint ``idx`` (module docstring, steps 1 and 2) under ``mixer`` (a ``SkillMixer``):
    ``(record, weights (J,))``."""
    constraints = [seq[idx] for seq in sequences]
    scene, inhand = features(constraints, obs, carried)
    if inhand is None and idx >= 1:
        inhand = np.zeros((len(constraints), 2))     # from the second constraint on the in-hand features are recorded (0)
    series = [gripper_series(c)[:, 0] for c in constraints]
    weights, decided = mixer.decide(idx, primary, scene, inhand, series)
    return {"subtask": int(idx), "source_a": int(primary), **decided, "carried_object": carried}, weights


class MixedWindow:
    """The retained sources' solved poses blended on the weighted timeline (module docstring, step 4)."""

    def __init__(self, solved: dict, weights: dict, anchor: int):
        """``solved`` maps every retained source to ``(solved poses (n_j,4,4), gripper series (n_j,1))``, ``weights`` maps it
        to its weight (sum 1) and ``anchor`` is the retained source whose constraint CP-Gen's step loop runs on."""
        keep = list(solved)
        w = np.array([float(weights[j]) for j in keep])
        poses = [np.asarray(solved[j][0], dtype=float) for j in keep]
        series = [np.asarray(solved[j][1], dtype=float).reshape(-1) for j in keep]
        self.poses, _grip, meta = mix_many(poses, series, w / w.sum())
        self.n = len(self.poses)
        self.anchor = int(anchor)
        a = keep.index(self.anchor)
        if "source_index" in meta:
            self.anchor_index = np.asarray(meta["source_index"][str(a)], dtype=int)
        else:                                               # one retained source: its own steps
            self.anchor_index = np.arange(self.n)
        self.meta = {"samples": int(self.n), "anchor": self.anchor,
                     "sources": {str(j): {"weight": round(float(weights[j]), 4), "source_samples": int(len(poses[i])),
                                          "event_steps": events(series[i])[1].tolist()} for i, j in enumerate(keep)}}
        if "actual_event_steps" in meta:
            self.meta["actual_event_steps"] = meta["actual_event_steps"]

    def pose(self, t: int) -> np.ndarray:
        """The blended desired end-effector pose of step ``t`` of the blended window."""
        if not 0 <= t < self.n:
            raise IndexError(f"step {t} outside the blended window of {self.n} steps")
        return self.poses[t]


STEP_FIELDS = ("timesteps", "src_obj_pose", "src_obs", "src_obj_transform", "src_obj_geoms_size", "src_action", "src_state",
               "src_gripper_action", "keypoints_obj_frame", "keypoints_robot_link_frame")


def _resample(value, index: np.ndarray, n: int):
    """``value`` indexed by ``index`` along its first axis when that axis is the window (length ``n``); dictionaries are
    resampled per entry; anything else (a callable, per-demo data) is kept."""
    if isinstance(value, dict):
        return {k: _resample(v, index, n) for k, v in value.items()}
    if isinstance(value, np.ndarray) and value.ndim >= 1 and len(value) == n:
        return value[index].copy()
    if isinstance(value, (list, tuple)) and len(value) == n:
        return [value[int(i)] for i in index]
    return value


def window_on(constraint, index: np.ndarray):
    """A copy of ``constraint`` whose per-step data (``STEP_FIELDS``, first axis = the window) are resampled at the source
    steps ``index`` (one per step of the blended window), so CP-Gen's step loop runs ``len(index)`` steps."""
    n = window_length(constraint)
    index = np.asarray(index, dtype=int)
    if index.size and (index.min() < 0 or index.max() >= n):
        raise ValueError(f"source steps {index.min()}..{index.max()} outside the window of {n} steps")
    out = copy.copy(constraint)
    for name in STEP_FIELDS:
        if getattr(constraint, name, None) is not None:
            setattr(out, name, _resample(getattr(constraint, name), index, n))
    out.timesteps = [constraint.timesteps[int(i)] for i in index]
    return out


def symmetry_candidates(official, constraint) -> list[tuple[str, dict]]:
    """Every ``(X_obj0_transf, X_obj1_transf, X_eef_transf)`` combination ``find_best_transform`` tries for the
    constraint, in its loop order (eef outer, obj1 inner), each with a label such as ``"obj0:z180+eef:z180"``."""
    lists = {"obj0": [("", np.eye(4))], "obj1": [("", np.eye(4))], "eef": [("", np.eye(4))]}
    for sym in getattr(constraint, "symmetries", None) or []:
        if sym.type not in lists:
            continue
        if sym.type == "obj0" and getattr(sym, "skip_default", False):
            lists["obj0"] = []
        mats = official.generate_se3_transforms(sym.transforms)
        lists[sym.type] += [(symmetry_label(sym.type, p, k), np.asarray(T, dtype=float))
                            for k, (p, T) in enumerate(zip(sym.transforms, mats))]
    for kind in ("obj0", "obj1"):                    # compositions of two annotated symmetries are symmetries too
        base = [(label, t) for label, t in lists[kind] if label]
        for la, ta in base:
            for lb, tb in base:
                t = ta @ tb
                if la != lb and not any(np.allclose(t, u, atol=1e-9) for _l, u in lists[kind]):
                    lists[kind].append((f"{la}*{lb.split(':', 1)[1]}", t))
    if getattr(constraint, "constraint_type", None) == "robot-object" and getattr(constraint, "reflect_eef", True):
        lists["eef"].append(("eef:z180", gripper_flip()))
    out = []
    for le, te in lists["eef"]:
        for l0, t0 in lists["obj0"]:
            for l1, t1 in lists["obj1"]:
                out.append(("+".join(x for x in (l0, l1, le) if x) or "identity",
                            dict(X_obj0_transf=t0, X_obj1_transf=t1, X_eef_transf=te)))
    return out


def _same(t: dict, u: dict) -> bool:
    return all(np.allclose(np.asarray(t[k] if t[k] is not None else np.eye(4)), np.asarray(u[k] if u[k] is not None else np.eye(4)))
               for k in ("X_obj0_transf", "X_obj1_transf", "X_eef_transf"))


def _window_start(official, policy, c, obs: dict, transforms: dict) -> np.ndarray:
    return np.asarray(official.compute_X_W_eef_des(c, obs, policy.robot_configuration, policy.eef_configuration, src_t_idx=0,
                                                   ignore_scaling=policy.ignore_scaling, **transforms), dtype=float)


def align_sources(official, policy, sequences: list, idx: int, primary: int, sources: list, obs: dict,
                  primary_transforms: dict) -> tuple[dict, dict]:
    """Per source, the symmetry combination whose desired pose at the window start is closest to the primary's
    (module docstring, step 3). Returns ``(transforms per source, record per source)``."""
    reference = _window_start(official, policy, sequences[primary][idx], obs, primary_transforms)
    chosen, info = {}, {}
    for j in sources:
        c = sequences[j][idx]
        best, before = None, None
        for label, t in symmetry_candidates(official, c):
            dp, dr = pose_distance(_window_start(official, policy, c, obs, t), reference)
            score = symmetry_score(dp, dr)
            if best is None or score < best[0]:
                best = (score, label, t, dp, dr)
            if before is None and _same(t, primary_transforms):
                before = (dp, dr)
        _score, label, t, dp, dr = best
        chosen[j] = t
        info[str(j)] = {"symmetry": label, "pos_cm": round(100 * dp, 2), "rot_deg": round(float(np.degrees(dr)), 1)}
        if before is not None:
            info[str(j)]["primary_symmetry_pos_cm"] = round(100 * before[0], 2)
            info[str(j)]["primary_symmetry_rot_deg"] = round(float(np.degrees(before[1])), 1)
    return chosen, info


def solve_sources(official, policy, sequences: list, idx: int, sources: list, obs: dict, transforms: dict) -> dict:
    """Solve constraint ``idx`` of each source in ``sources`` for the current scene (module docstring, step 3) with that
    source's symmetry combination ``transforms[j]``."""
    out = {}
    for j in sources:
        c = sequences[j][idx]
        poses = np.stack([official.compute_X_W_eef_des(c, obs, policy.robot_configuration, policy.eef_configuration,
                                                       src_t_idx=u, ignore_scaling=policy.ignore_scaling, **transforms[j])
                          for u in range(window_length(c))])
        out[j] = (poses, gripper_series(c))
    return out
