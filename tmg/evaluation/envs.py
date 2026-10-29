"""Which simulator a condition evaluates in, and the robosuite environment settings shared by the evaluators.

``is_isaaclab`` and ``isaaclab_task_id`` read the composed config: Isaac Lab generators (SoftMimicGen, DexMimicGen)
have no stored scene states and play the task id of their camera environment. ``ROBOSUITE_EVAL_KWARGS`` are the
robosuite settings every evaluation environment shares (``rollout.make_env``, ``resets.ENV_KWARGS``,
``adjusted_env_meta`` below): object observations on, no episode termination by the environment, no reward shaping.

The MimicGen path builds the robosuite environment itself (``rollout.make_env``: the registered scene class with
MimicGen's data-collection settings, OSC delta actions). Generators that run on another robosuite version or
controller, CP-Gen in particular (robosuite 1.5, absolute end-effector pose actions through an IK controller),
must be evaluated in the environment they generated with. That environment is described by the ``env_args``
json robomimic stores in every demonstration file and ``tmg dataset`` copies into ``data.attrs``; this module
rebuilds it through robomimic's ``create_env_from_metadata`` with only the scene class, the cameras and the
image size replaced. Such evaluations run in the generator's own image (``make eval IMAGE=cpgen``).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

ROBOSUITE_EVAL_KWARGS = {"use_object_obs": True, "ignore_done": True, "control_freq": 20, "reward_shaping": False}


def is_isaaclab(plain: dict) -> bool:
    """Whether the composed config's generator simulates in Isaac Lab (scenes are seeds, not stored states)."""
    return str(plain["generator"].get("simulator", "")) == "isaaclab"


def isaaclab_task_id(task: dict) -> str:
    """The Isaac Lab task id the evaluation plays: the camera environment ``task.camera_env`` when the official task
    records no images, else ``task.env_name``."""
    return str(task.get("camera_env") or task["env_name"])


def env_meta_from_dataset(path: Path) -> dict:
    """The ``env_args`` json of a common hdf5 (``tmg dataset``) or of a robomimic demonstration file."""
    import h5py

    with h5py.File(path, "r") as f:
        raw = f["data"].attrs.get("env_args", "{}")
    meta = json.loads(raw if isinstance(raw, str) else raw.decode())
    if not meta.get("env_kwargs"):
        raise ValueError(f"{path}: env_args has no env_kwargs; cannot rebuild the generation environment")
    return meta


def adjusted_env_meta(env_meta: dict, env_name: str, cameras: list[str], image_size) -> dict:
    """A copy of ``env_meta`` with our scene class as ``env_name`` and the evaluation cameras.

    Everything else (controller configuration, control frequency, object observations, robot) stays as the
    generator had it, so the policy sees the environment it was trained for.
    """
    meta = copy.deepcopy(env_meta)
    meta["env_name"] = str(env_name)
    kw = meta.setdefault("env_kwargs", {})
    kw.pop("env_name", None)
    cameras = [str(c) for c in (cameras or [])]
    kw["use_camera_obs"] = bool(cameras)
    kw["has_offscreen_renderer"] = bool(cameras)
    kw["has_renderer"] = False
    for key in ("use_object_obs", "ignore_done", "reward_shaping"):     # not control_freq: the generator's stays
        kw[key] = ROBOSUITE_EVAL_KWARGS[key]
    if cameras:
        h, w = (int(v) for v in image_size)
        kw["camera_names"] = cameras
        kw["camera_heights"] = h
        kw["camera_widths"] = w
    return meta


def make_env_from_meta(env_meta: dict, env_name: str, observation: dict):
    """robomimic ``EnvRobosuite`` of ``adjusted_env_meta`` (robomimic's observation tables initialised for the
    observation config's keys, as ``create_for_data_processing`` would do)."""
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.obs_utils as ObsUtils

    cameras = [str(c) for c in (observation.get("cameras") or [])]
    low_dim = [str(k) for k in (observation.get("robot_state") or [])] + [str(k) for k in (observation.get("object_state") or [])]
    ObsUtils.initialize_obs_utils_with_obs_specs({"obs": {"low_dim": low_dim, "rgb": [f"{c}_image" for c in cameras]}})
    meta = adjusted_env_meta(env_meta, env_name, cameras, observation.get("image_size") or (84, 84))
    return EnvUtils.create_env_from_metadata(env_meta=meta, use_image_obs=bool(cameras), render_offscreen=bool(cameras), render=False)
