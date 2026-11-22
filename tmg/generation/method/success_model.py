"""The per-source features SkillMix scores, and the rows of a SkillMix fit (``tmg fit_coefficients``).

Features of source demonstration j for subtask k of an attempt (``scene_features``, ``inhand_features``):

- ``dx_kj``, ``dy_kj`` (m): the x and y offsets (world frame, the table plane) between the reference object of subtask k
  in the current scene and in source j at the start of its subtask-k segment (absolute values);
- ``rot_kj`` (rad): the angle of the relative rotation between those two object orientations;
- ``inhand_pos_kj`` (m), ``inhand_rot_kj`` (rad): from the second subtask on (index k >= 1), the position distance and the
  rotation angle between the in-hand pose now and in source j at its segment start. The in-hand pose is the pose of the
  previous subtask's reference object in the end-effector frame (for a grasp-then-place task: the grasped object). Both
  are 0 for subtask 0, and 0 where the source pose of the carried object is not known (a CP-Gen constraint that does
  not involve it), at fitting and at mixing time alike.

Rows of the fit (``tmg.generation.fit_coefficients``) are the attempts that reached subtask k, labelled with the attempt's
TASK success, not with subtask k's own success (2026-11-17): a subtask that almost always succeeds by itself (a grasp)
can still decide whether a later subtask succeeds through the in-hand pose it leaves, and only the task label sees that.
In CP-Gen Threading the rule fitted on the grasp's own success kept the two grasp sources whose task success was 1 % and
7 %. For the last subtask both labels are the same.

Which subtasks an attempt reached comes from the records. MimicGen: for every subtask but the last, the subtask's
termination signal (``task.subtask_term_signals``, MimicGen's ``subtask_term_signal`` of the task template, recorded per
step as ``term__<signal>``) turned on at some step up to the end of that subtask's execution; the last subtask succeeds
when the attempt succeeds (``subtask_outcomes``). CP-Gen stops an attempt at its first failed subtask (constraint), so
there subtask k succeeded when the attempt went on to subtask k + 1 (``reached_outcomes``). An attempt reaches subtask k
when every earlier subtask succeeded.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from tmg.core.poses import rotation_angle

SCENE_FEATURES = ("dx", "dy", "rotation")
INHAND_FEATURES = ("inhand_position", "inhand_rotation")
FEATURES = SCENE_FEATURES + INHAND_FEATURES
UNITS = {"dx": "m", "dy": "m", "rotation": "rad", "inhand_position": "m", "inhand_rotation": "rad"}


# features ---------------------------------------------------------------------------------------------------------
def pose12(v) -> np.ndarray:
    """A 4x4 pose from the 12-number record layout (position, then the row-major rotation), or a 4x4 as is."""
    v = np.asarray(v, dtype=float)
    if v.shape == (4, 4):
        return v
    out = np.eye(4)
    out[:3, 3], out[:3, :3] = v[:3], v[3:12].reshape(3, 3)
    return out


def scene_features(current_pose, source_pose) -> np.ndarray:
    """``(|dx|, |dy|, rotation)`` between the current and a source's reference-object pose (4x4 world poses)."""
    c, s = pose12(current_pose), pose12(source_pose)
    return np.array([abs(c[0, 3] - s[0, 3]), abs(c[1, 3] - s[1, 3]), float(rotation_angle(c[:3, :3], s[:3, :3]))])


def in_hand(eef_pose, object_pose) -> np.ndarray:
    """The object's pose in the end-effector frame (4x4)."""
    e, o = pose12(eef_pose), pose12(object_pose)
    return np.linalg.inv(e) @ o


def inhand_features(current_in_hand, source_in_hand) -> np.ndarray:
    """``(position distance, rotation angle)`` between two in-hand poses (4x4, end-effector frame)."""
    c, s = np.asarray(current_in_hand, dtype=float), np.asarray(source_in_hand, dtype=float)
    return np.array([float(np.linalg.norm(c[:3, 3] - s[:3, 3])), float(rotation_angle(c[:3, :3], s[:3, :3]))])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# outcomes and rows -----------------------------------------------------------------------------------------------
def subtask_outcomes(attempt: Mapping, steps: Mapping, signals: Sequence) -> list[bool]:
    """Success of every subtask of one attempt (module docstring). ``steps`` holds the attempt's per-step arrays
    (``seg_of_step``, ``seg_sub`` and ``term__<signal>``); ``signals`` the termination signal of every subtask but the last
    (None for the last)."""
    num = len(signals)
    seg_sub = np.asarray(steps["seg_sub"]).astype(int)
    sub_of_step = seg_sub[np.asarray(steps["seg_of_step"]).astype(int)] if len(seg_sub) else np.zeros(0, dtype=int)
    out = []
    for k, signal in enumerate(signals):
        if k == num - 1 or signal is None:
            out.append(bool(attempt["success"]) if k == num - 1 else False)
            continue
        key = f"term__{signal}"
        if key not in steps:
            raise KeyError(f"attempt {attempt.get('attempt')}: no per-step signal {key!r} in the record")
        fired = np.asarray(steps[key]).astype(bool)
        upto = sub_of_step <= k
        out.append(bool(np.any(fired[: len(upto)] & upto[: len(fired)])))
    return out


def source_in_hand(source_infos: Sequence[Mapping], j: int, start: int, carried: str) -> np.ndarray:
    """Source j's in-hand pose of ``carried`` at step ``start`` (``source_infos[j]``: eef_pose (T,4,4), object_poses)."""
    info = source_infos[j]
    return in_hand(info["eef_pose"][start], info["object_poses"][carried][start])


def _inhand_row(s: Mapping, j: int, carried: str, source_infos) -> np.ndarray:
    """In-hand features of one subtask record. The source's in-hand pose comes from the record itself when it carries the
    source's end-effector and object poses (CP-Gen: ``src_eef_pose``, ``src_objects``), else from ``source_infos``
    (MimicGen ``datagen_info``). When the carried object's source pose is not available (a CP-Gen constraint that does not
    involve it), both features are 0 for that row, and the model uses 0 for it at mixing time as well."""
    objects = s.get("objects") or {}
    if carried not in objects:
        return np.zeros(2)
    now = in_hand(s["eef_pose"], objects[carried])
    if "src_eef_pose" in s:
        src_objects = s.get("src_objects") or {}
        if carried not in src_objects:
            return np.zeros(2)
        return inhand_features(now, in_hand(s["src_eef_pose"], src_objects[carried]))
    if source_infos is None:
        raise ValueError("the in-hand features of subtask >= 1 need the source demonstrations' datagen_info")
    return inhand_features(now, source_in_hand(source_infos, j, int(s["src_subtask_start"]), carried))


def reached_outcomes(attempt: Mapping, num_subtasks: int) -> list[bool]:
    """Subtask success of a generator that stops an attempt at its first failed subtask (CP-Gen): subtask k succeeded when
    the attempt went on to subtask k + 1; the last subtask succeeded when the attempt succeeded."""
    n = len(attempt.get("subtasks") or [])
    return [n > k + 1 if k < num_subtasks - 1 else bool(attempt["success"]) and n >= num_subtasks for k in range(num_subtasks)]


def design_rows(attempts: Sequence[Mapping], outcomes: Sequence[Sequence[bool]], k: int, num_sources: int,
                source_infos: Sequence[Mapping] | None) -> tuple[np.ndarray, np.ndarray, str | None]:
    """The rows and labels of subtask k: one row per attempt that reached it, labelled with the attempt's task success
    (module docstring). Columns: J source indicators, the three scene features, and for k >= 1 the two in-hand features."""
    rows, ys, carried = [], [], None
    for a, out in zip(attempts, outcomes):
        if not all(out[:k]):
            continue
        subs = sorted(a["subtasks"], key=lambda s: int(s["subtask_ind"]))
        if len(subs) <= k:
            continue
        s = subs[k]
        j = int(s["src_demo_ind"])
        x = np.zeros(num_sources)
        x[j] = 1.0
        parts = [x, scene_features(s["cur_obj_pose"], s["src_obj_pose"])]
        if k >= 1:
            carried = str(subs[k - 1]["object_name"])
            parts.append(_inhand_row(s, j, carried, source_infos))
        rows.append(np.concatenate(parts))
        ys.append(float(bool(a["success"])))           # label: task success of an attempt that reached subtask k
    return np.array(rows), np.array(ys), carried
