"""Isaac Lab side of the generator adapters (SoftMimicGen, DexMimicGen): scene ranges, hand layout, measured poses.

Three concerns that the robosuite layer has no equivalent for:

- **Scene ranges.** Isaac Lab mimic environments draw object placements in a reset event whose ``pose_range``
  parameter maps axis names (``x``, ``y``, ``z``, ``roll``, ``pitch``, ``yaw``) to ``(lo, hi)``. A scene file
  (``configs/experiment/generator/<generator>/scene/<name>.yaml``) lists, per task and per reset event, the
  ranges to write into that parameter; ``apply_scene`` edits the environment config before the environment is
  built. A scene with ``override_bounds: false`` (``official``) leaves the config untouched.
- **Hand layout.** The GR1 action is ``[left arm 7, right arm 7, 22 hand joints]`` and the controller's hand
  joint order interleaves the hands (``L_* x5, R_* x5, L_* x5, R_* x5, L_thumb, R_thumb``). The official
  ``actions[:, 14:25]`` / ``actions[:, 25:]`` split therefore mixes both hands. ``HandLayout`` maps each hand
  to its joint indices by the ``L_`` / ``R_`` name prefixes read from the action term at run time.
- **Measured poses.** The follower's error must be measured on the link the controller targets. The GR1 IK
  tasks target ``*_hand_pitch_link`` while the observations ``*_eef_pos`` / ``*_eef_quat`` read
  ``*_hand_roll_link``; ``measured_eef_poses`` reads the configured link's world pose from the robot's body
  state (minus the environment origin), so the error is zero when the controller converged.

Everything here is importable without Isaac Sim; the functions that touch an environment receive it as an
argument and only use attributes of Isaac Lab's ``ManagerBasedRLMimicEnv``.
"""
from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

AXES = ("x", "y", "z", "roll", "pitch", "yaw")


class HandLayout:
    """Physical-hand grouping of a flat hand-joint action block by joint-name prefix."""

    def __init__(self, joint_names: Sequence[str], *, arm_dim: int, prefixes: Mapping[str, str] | None = None):
        """Args:
            joint_names: controller order of the hand joints (``action_term._hand_joint_names``).
            arm_dim: number of leading arm values in the action before the hand block (14 for GR1).
            prefixes: ``{hand name: joint-name prefix}``.
        """
        self.names = list(joint_names)
        self.arm_dim = int(arm_dim)
        prefixes = dict(prefixes) if prefixes is not None else {"left": "L_", "right": "R_"}
        self.indices = {hand: [i for i, n in enumerate(self.names) if n.startswith(prefix)] for hand, prefix in prefixes.items()}
        covered = sorted(i for ids in self.indices.values() for i in ids)
        if covered != list(range(len(self.names))):
            raise ValueError(f"hand joint names are not split by the prefixes {prefixes}: {self.names}")
        if any(not ids for ids in self.indices.values()):
            raise ValueError("every hand needs at least one joint")

    @property
    def hands(self) -> list[str]:
        return list(self.indices)

    def split(self, actions):
        """``{hand: actions[..., hand joints]}`` of an action batch of width ``arm_dim + len(names)``."""
        if actions.shape[-1] != self.arm_dim + len(self.names):
            raise ValueError(f"action width {actions.shape[-1]} != {self.arm_dim} + {len(self.names)}")
        return {hand: actions[..., [self.arm_dim + i for i in ids]] for hand, ids in self.indices.items()}

    def assemble(self, fingers: Mapping):
        """Inverse of ``split`` for the hand block only: ``(..., len(names))`` in controller order."""
        first = next(iter(fingers.values()))
        shape = (*first.shape[:-1], len(self.names))
        out = np.empty(shape, dtype=first.dtype) if isinstance(first, np.ndarray) else first.new_empty(shape)
        for hand, ids in self.indices.items():
            out[..., ids] = fingers[hand]
        return out

    def describe(self) -> dict:
        return {"names": self.names, "indices": self.indices, "arm_dim": self.arm_dim}


def pose_range_of(entry: Mapping) -> dict:
    """``{axis: (lo, hi)}`` of a scene range entry (unlisted axes are left to the official config)."""
    out = {}
    for axis in AXES:
        if axis in entry:
            lo, hi = (float(v) for v in entry[axis])
            if lo > hi:
                raise ValueError(f"range {axis}: lo > hi")
            out[axis] = (lo, hi)
    if not out:
        raise ValueError(f"a scene entry needs at least one of {AXES}")
    return out


RANGE_PARAMS = ("pose_range", "partial_pose_range")


def ranges_of(entry: Mapping) -> dict:
    """``{parameter name: {axis: (lo, hi)}}`` of a scene event entry.

    A flat entry (``{x: [lo, hi], ...}``) is the event's ``pose_range``; a nested entry lists the parameters by name
    (``pose_range`` and, for Soft's rope, ``partial_pose_range``, the extra transform of one half of the rope).
    """
    nested = {k: entry[k] for k in RANGE_PARAMS if k in entry}
    if nested:
        return {k: pose_range_of(v) for k, v in nested.items()}
    return {"pose_range": pose_range_of(entry)}


def has_ranges(entry: Mapping) -> bool:
    """Whether a scene event entry lists any range, nested (``pose_range: {...}``) or flat (``x: [lo, hi]``)."""
    return any(k in entry for k in RANGE_PARAMS) or any(k in entry for k in AXES)


def validate_isaaclab_scene(scene: Mapping) -> None:
    """Check ``bounds: {task: {event name: entry}}`` of an Isaac Lab scene file.

    An entry either widens an existing reset event (its ``pose_range`` / ``partial_pose_range``) or, with
    ``add: true``, adds a reset event the official config lacks: ``func`` (``module:attribute``), ``asset`` (scene
    entity name) and optional ``pose_range`` / ``params`` (extra parameters of the function).
    """
    for task, events in (scene.get("bounds") or {}).items():
        for name, entry in events.items():
            if entry.get("add"):
                if ":" not in str(entry.get("func", "")) or not entry.get("asset"):
                    raise ValueError(f"added event {task}/{name} needs func 'module:attribute' and asset")
                if has_ranges(entry):
                    ranges_of(entry)
            else:
                ranges_of(entry)


def import_attr(path: str):
    """The object named by a ``module:attribute`` path (the form scene and task files use for functions)."""
    module, attr = str(path).split(":", 1)
    return getattr(importlib.import_module(module), attr)


_import_attr = import_attr      # older name, still imported by the adapters


def _add_event(env_cfg, event: str, entry: Mapping) -> dict:
    """Append the reset event ``event`` described by an ``add: true`` entry to ``env_cfg.events``; returns what was added."""
    if getattr(env_cfg.events, event, None) is not None:
        raise KeyError(f"environment config already has a reset event {event!r}; drop 'add' to widen it")
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    params = {"asset_cfg": SceneEntityCfg(str(entry["asset"]))}
    params.update(dict(entry.get("params") or {}))
    if has_ranges(entry):
        params.update(ranges_of(entry))
    setattr(env_cfg.events, event, EventTermCfg(func=import_attr(entry["func"]), mode="reset", params=params))
    return {"added": entry["func"], **{k: v for k, v in params.items() if k != "asset_cfg"}}


def _widen_event(term, entry: Mapping) -> dict:
    """Overwrite the listed axes of the existing event term's range parameters; returns the parameters as they now are
    (the bare ``pose_range`` mapping when that is the only parameter)."""
    applied = {}
    for param, ranges in ranges_of(entry).items():
        current = dict(term.params.get(param) or {})
        current.update(ranges)
        term.params[param] = current
        applied[param] = current
    if list(applied) == ["pose_range"]:
        return applied["pose_range"]
    return applied


def apply_scene(env_cfg, scene: Mapping, task_name: str) -> dict:
    """Write the scene's ranges into the environment config's reset events.

    Args:
        env_cfg: the Isaac Lab environment config (``env_cfg.events.<name>.params['pose_range']``).
        scene: the generator's scene dict; ``override_bounds`` false leaves ``env_cfg`` unchanged, and so does a
            task the scene lists no bounds for (its official reset distribution is kept).
        task_name: key of ``scene['bounds']``.

    Returns:
        ``{event name: applied ranges}`` (empty when nothing was overridden). Added events (entries with
        ``add: true``) are appended after the official reset events, so they run after them at every reset.
    """
    if not scene.get("override_bounds", False):
        return {}
    bounds = (scene.get("bounds") or {}).get(task_name)
    if bounds is None:
        return {}
    applied = {}
    for event, entry in bounds.items():
        if entry.get("add"):
            applied[event] = _add_event(env_cfg, event, entry)
            continue
        term = getattr(env_cfg.events, event, None)
        if term is None:
            raise KeyError(f"environment config has no reset event {event!r} (scene {scene.get('name')!r})")
        applied[event] = _widen_event(term, entry)
    return applied


def to_numpy(x) -> np.ndarray:
    """A numpy array of a torch tensor (detached, on the CPU); a numpy array or sequence passes through ``np.asarray``."""
    return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)


def pose_from_pos_quat(pos, quat_wxyz) -> np.ndarray:
    """4x4 pose from a position and a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = np.asarray(pos, dtype=float)
    return out


def measured_eef_poses(env, env_id: int, link_names: Mapping[str, str]) -> dict[str, np.ndarray]:
    """World pose (4x4, environment-origin frame) of each end effector's configured link.

    Args:
        env: Isaac Lab environment with ``scene['robot']`` (an articulation) and ``scene.env_origins``.
        env_id: environment index.
        link_names: ``{eef name: body name}`` (for GR1 the IK target links ``left/right_hand_pitch_link``).
    """
    robot = env.scene["robot"]
    body_names = list(robot.data.body_names)
    origin = np.asarray(to_numpy(env.scene.env_origins[env_id]), dtype=float)
    out = {}
    for eef, body in link_names.items():
        if body not in body_names:
            raise KeyError(f"robot has no body {body!r} for end effector {eef!r}; bodies: {body_names[:8]}...")
        b = body_names.index(body)
        pos = to_numpy(robot.data.body_pos_w[env_id, b]) - origin
        quat = to_numpy(robot.data.body_quat_w[env_id, b])
        out[eef] = pose_from_pos_quat(pos, quat)
    return out


def observed_eef_poses(env, env_id: int, eef_names: Sequence[str]) -> dict[str, np.ndarray]:
    """The environment's own ``get_robot_eef_pose`` for each end effector (the observation link)."""
    return {eef: np.asarray(to_numpy(env.get_robot_eef_pose(eef_name=eef, env_ids=[env_id])[0]), dtype=float)
            for eef in eef_names}


def isolate_controller_scratch(env_cfg, folder) -> dict:
    """Give every action controller that converts a USD to a URDF its own output folder under ``folder``.

    Isaac Lab's Pink IK controller configs write the converted robot URDF to a fixed ``urdf_output_dir`` (``/tmp/urdf``
    in the GR1 tasks). Chunk processes of one pool run in the same container and start together, so two of them write
    and read the same file at once and one reads a half-written URDF ("does not contain a valid URDF model",
    2026-10-29, Pick Place ranking pool with 3 workers). Returns ``{action name: new folder}``; actions without such a
    controller are left alone.
    """
    actions = getattr(env_cfg, "actions", None)
    moved = {}
    for name, action_cfg in list(vars(actions).items()) if actions is not None else []:
        controller = getattr(action_cfg, "controller", None)
        if controller is None or not hasattr(controller, "urdf_output_dir"):
            continue
        target = Path(folder) / "urdf" / str(name)
        target.mkdir(parents=True, exist_ok=True)
        controller.urdf_output_dir = str(target)
        moved[str(name)] = str(target)
    return moved
