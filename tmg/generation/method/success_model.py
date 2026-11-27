"""The per-source features SkillMix scores, from a ``SubtaskScene`` (the mixer) or from an attempt record (the fit).

Features of source demonstration j for subtask k of an attempt (``source_features``):

- ``dx``, ``dy`` (m): the x and y offsets (world frame, the table plane) between the reference object of subtask k in the
  current scene and in source j at the start of its subtask-k segment (absolute values);
- ``rotation`` (rad): the angle of the relative rotation between those two object orientations;
- ``inhand_position`` (m), ``inhand_rotation`` (rad): from the end effector's second subtask on, the position distance and
  the rotation angle between the in-hand pose now and in source j at its segment start. The in-hand pose is the pose of
  the previous subtask's reference object (``SubtaskScene.carried_object``) in the end-effector frame. Both are 0 for the
  first subtask and 0 where one of the two in-hand poses does not exist: the pose is not known (a CP-Gen constraint that
  does not involve the object) or the hand does not hold the object (the hold test of the Isaac Lab runtimes);
- ``inhand_mismatch`` (0 or 1; only in the feature set of a generator with a hold test, ``HELD_FEATURES``): 1 when exactly
  one of the two hands holds the carried object (now, or in source j), else 0. Without it, "source j does not hold the
  object" and "source j holds it exactly as now" would both be the in-hand features (0, 0). With it (0, 0, 0) means the
  in-hand state is the same (the same pose, or nothing in either hand) and a mismatch has its own slope.

A rule file lists the features of its blocks (``features``): MimicGen and CP-Gen use ``FEATURES`` (five), generators with a
hold test ``HELD_FEATURES`` (six). The mixer scores with the rule file's list; the fit uses the list of the generator's
fit layout and writes it.

Rows of the fit (``tmg.generation.fit_coefficients``) are the attempts that reached subtask k, labelled with the attempt's
TASK success, not with subtask k's own success (2026-11-17): a subtask that almost always succeeds by itself (a grasp)
can still decide whether a later subtask succeeds through the in-hand pose it leaves, and only the task label sees that.
In CP-Gen Threading the rule fitted on the grasp's own success kept the two grasp sources whose task success was 1 % and
7 %. For the last subtask both labels are the same. Which subtasks an attempt reached is decided by the reached rule of
the generator's fit layout (``tmg.generation.method.fit_records``).

``scene_of_record`` rebuilds the ``SubtaskScene`` of a subtask record against its selected source, so a row of an unmixed
attempt (``design_rows``) and the mixer's features come from the same function. Where the selected source's poses come
from is the fit layout's ``source_poses``: the record itself (``src_eef_pose``, ``src_objects``: CP-Gen and the Isaac Lab
generators) or the source file's ``datagen_info`` (MimicGen).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from tmg.core.poses import rotation_angle
from tmg.generation.method.interfaces import HeldRule, HeldTest, SubtaskScene

SCENE_FEATURES = ("dx", "dy", "rotation")
INHAND_FEATURES = ("inhand_position", "inhand_rotation")
FEATURES = SCENE_FEATURES + INHAND_FEATURES                 # generators without a hold test (MimicGen, CP-Gen)
HELD_FEATURE = "inhand_mismatch"
HELD_FEATURES = FEATURES + (HELD_FEATURE,)                  # generators with a hold test (the Isaac Lab generators)
UNITS = {"dx": "m", "dy": "m", "rotation": "rad", "inhand_position": "m", "inhand_rotation": "rad",
         HELD_FEATURE: "0 or 1"}
SOURCE_POSES = ("records", "datagen_info")                  # FitLayout.source_poses


def check_features(names) -> tuple[str, ...]:
    """``names`` as a tuple, after checking that every one is a known feature."""
    names = tuple(str(n) for n in names)
    unknown = [n for n in names if n not in UNITS]
    if unknown or len(set(names)) != len(names):
        raise ValueError(f"unknown or repeated SkillMix features {unknown or list(names)}; known: {list(UNITS)}")
    return names


# poses ----------------------------------------------------------------------------------------------------------------
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


# features of a scene --------------------------------------------------------------------------------------------------
def current_in_hand(scene: SubtaskScene) -> np.ndarray | None:
    """The carried object's pose in the hand now, or None (nothing carried, pose unknown, not held)."""
    if scene.carried_object is None or scene.carried_pose is None:
        return None
    return in_hand(scene.eef_pose, scene.carried_pose)


def source_in_hand(scene: SubtaskScene, j: int) -> np.ndarray | None:
    """The carried object's pose in source j's hand at its segment start, or None."""
    if scene.carried_object is None or scene.source_carried_poses is None or scene.source_carried_poses[j] is None:
        return None
    return in_hand(scene.source_eef_poses[j], scene.source_carried_poses[j])


def held_flags(scene: SubtaskScene) -> tuple[bool, list[bool]]:
    """``(the hand holds the carried object now, per source: it holds it at its segment start)``: where the in-hand pose
    exists."""
    return current_in_hand(scene) is not None, [source_in_hand(scene, j) is not None for j in range(scene.num_sources)]


def source_features(scene: SubtaskScene, names: Sequence[str] = FEATURES) -> np.ndarray:
    """The features ``names`` of every source against the scene, ``(J, len(names))`` (module docstring)."""
    J = scene.num_sources
    current = scene.current_object_pose
    place = np.array([scene_features(current, scene.source_object_poses[j]) for j in range(J)]).reshape(J, len(SCENE_FEATURES))
    hand = np.zeros((J, len(INHAND_FEATURES)))
    mismatch = np.zeros((J, 1))
    now = current_in_hand(scene)
    if scene.carried_object is not None:
        for j in range(J):
            src = source_in_hand(scene, j)
            if now is not None and src is not None:
                hand[j] = inhand_features(now, src)
            mismatch[j, 0] = float((now is None) != (src is None))
    return select_features(place, hand, mismatch, names)


def select_features(place, hand, mismatch, names: Sequence[str]) -> np.ndarray:
    """The columns ``names`` of the scene features (J, 3), the in-hand features (J, 2) and the mismatch flag (J, 1)."""
    every = np.hstack([place, hand, mismatch])
    order = SCENE_FEATURES + INHAND_FEATURES + (HELD_FEATURE,)
    names = check_features(names)
    if names == order[:every.shape[1] - 1]:
        return np.hstack([place, hand])
    return every[:, [order.index(n) for n in names]]


def mix_record_features(mix: Mapping, names: Sequence[str], num_sources: int) -> np.ndarray:
    """The features ``names`` of every source from a mix record (``scene_features``, ``inhand_features`` and, with a hold
    test, ``held``: ``{now, sources}``), ``(J, len(names))``: what ``source_features`` gave the mixer."""
    place = np.asarray(mix["scene_features"], dtype=float)
    hand = np.asarray(mix["inhand_features"], dtype=float) if mix.get("inhand_features") else np.zeros((num_sources, len(INHAND_FEATURES)))
    mismatch = np.zeros((num_sources, 1))
    held = mix.get("held")
    if held:
        mismatch[:, 0] = [float(bool(held["now"]) != bool(s)) for s in held["sources"]]
    return select_features(place, hand, mismatch, names)


# rows of unmixed attempts ---------------------------------------------------------------------------------------------
def datagen_in_hand_poses(source_infos: Sequence[Mapping], j: int, start: int, carried: str) -> tuple[np.ndarray, np.ndarray]:
    """Source j's end-effector pose and pose of ``carried`` at step ``start`` (``source_infos[j]``: eef_pose (T,4,4),
    object_poses), from MimicGen's ``datagen_info``."""
    info = source_infos[j]
    return info["eef_pose"][start], info["object_poses"][carried][start]


def scene_of_record(s: Mapping, carried: str | None, *, source_poses: str = "records",
                    source_infos: Sequence[Mapping] | None = None) -> SubtaskScene:
    """The ``SubtaskScene`` of subtask record ``s`` against its selected source (one source, index 0). ``carried`` is the
    reference object of the end effector's previous subtask (None for its first). The carried object's pose now is
    ``objects[carried]`` (absent: not known or not held); the selected source's comes from the record (``source_poses``
    "records": ``src_eef_pose`` and ``src_objects[carried]``, absent: not known or not held) or from ``source_infos``
    ("datagen_info")."""
    if source_poses not in SOURCE_POSES:
        raise ValueError(f"source_poses must be one of {SOURCE_POSES}, got {source_poses!r}")
    if "src_obj_pose" not in s:
        raise ValueError(f"subtask record {s.get('eef')}/{s.get('subtask_ind')} lacks src_obj_pose: the pool was made before "
                         f"the runtime recorded the subtask scene; regenerate it")
    carried_pose = (s.get("objects") or {}).get(carried) if carried is not None else None
    src_eef = src_carried = None
    if carried is not None and source_poses == "records":
        if carried_pose is not None and "src_eef_pose" not in s:
            raise ValueError("the in-hand features of subtask >= 1 need the selected source's poses in the record "
                             "(src_eef_pose, src_objects)")
        src_eef, src_carried = s.get("src_eef_pose"), (s.get("src_objects") or {}).get(carried)
    elif carried_pose is not None:                  # datagen_info: read only next to a current in-hand pose
        if source_infos is None:
            raise ValueError("the in-hand features of subtask >= 1 need the source demonstrations' datagen_info")
        src_eef, src_carried = datagen_in_hand_poses(source_infos, int(s["src_demo_ind"]), int(s["src_subtask_start"]), carried)
    rule = HeldRule.from_mapping(s.get("held_rule"))
    held = None
    if rule is not None and s.get("held_distance") is not None and s.get("held") is not None:
        held = HeldTest(rule=rule, deformable=bool(s["held"]["deformable"]), now_m=float(s["held_distance"]["now"]),
                        sources_m=[float(s["held_distance"]["source"])])
    return SubtaskScene(subtask=int(s["subtask_ind"]), eef=s.get("eef"), reference_object=s.get("object_name"),
                        current_object_pose=s["cur_obj_pose"], source_object_poses=[s["src_obj_pose"]],
                        eef_pose=s.get("eef_pose"), source_eef_poses=[src_eef], carried_object=carried,
                        carried_pose=carried_pose, source_carried_poses=[src_carried], held=held, held_rule=rule,
                        object_frame=s.get("object_frame"))


def eef_subtasks(attempt: Mapping, eef: str | None = None) -> list[dict]:
    """The subtask records of an attempt by subtask index; with ``eef``, those of that end effector."""
    return sorted((s for s in attempt["subtasks"] if eef is None or s.get("eef") == eef), key=lambda s: int(s["subtask_ind"]))


def design_rows(attempts: Sequence[Mapping], outcomes: Sequence[Sequence[bool]], k: int, num_sources: int, *,
                eef: str | None = None, features: Sequence[str] = FEATURES, source_poses: str = "records",
                source_infos: Sequence[Mapping] | None = None) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """The rows of subtask k from single-source attempts: one row per attempt that reached it, labelled with the attempt's
    task success (module docstring). Returns ``(X, y, kept)``: the columns of ``X`` are the J source indicators and the
    ``features`` of the selected source, ``kept`` the index in ``attempts`` of every row. ``eef``: the end effector whose
    subtasks are numbered k (None: every subtask record of the attempt)."""
    rows, ys, kept = [], [], []
    for index, (a, out) in enumerate(zip(attempts, outcomes)):
        if not all(out[:k]):
            continue
        subs = eef_subtasks(a, eef)
        if len(subs) <= k:
            continue
        s = subs[k]
        x = np.zeros(num_sources)
        x[int(s["src_demo_ind"])] = 1.0
        carried = str(subs[k - 1]["object_name"]) if k >= 1 else None
        scene = scene_of_record(s, carried, source_poses=source_poses, source_infos=source_infos)
        rows.append(np.concatenate([x, source_features(scene, features)[0]]))
        ys.append(float(bool(a["success"])))           # label: task success of an attempt that reached subtask k
        kept.append(index)
    return np.array(rows).reshape(len(rows), num_sources + len(tuple(features))), np.array(ys), kept
