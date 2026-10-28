"""Episodes of ``tmg eval`` for the Isaac Lab generators (SoftMimicGen, DexMimicGen).

Runs inside the generator's image (``tmg/isaaclab-soft``, ``tmg/isaaclab-dex``) with the GPU. The environment is
built the way Isaac Lab's own ``scripts/imitation_learning/robomimic/play.py`` builds it: ``parse_env_cfg`` of the
task id (the camera environment ``task.camera_env`` when the official task records no images, else
``task.env_name``), observation terms kept as a dict, the time-out termination removed, the recorder removed, the
success termination taken out of the config and called by hand. Each scene of the seed file (``resets.py``):

1. ``env.seed(seed)``, ``torch.manual_seed``, ``numpy.random.seed`` and ``random.seed`` with the scene seed, then
   ``env.reset()``: the reset events draw the placement from the seeded generator, so a scene index gives the same
   placement in every run (same evaluation seed), whatever the policy or the condition;
2. the policy server is told ``reset`` with the episode's policy seed, then asked for an action every control
   step from the observation (``observation.images.<cam>`` (3, H, W) uint8 from the camera terms,
   ``observation.state`` = the task's robot state terms concatenated);
3. the success term is evaluated after every ``env.step`` (never at reset: SoftMimicGen's rope term fixes the rope
   ends at its first call, see ``task.success_term``), and the episode ends at success, at a termination or at the
   task horizon.

Every episode row also stores the initial placement read after the reset (rigid objects: position and quaternion in
the environment frame; deformable objects: mean nodal position), for later analysis of success against the scene.
Isaac Sim is started once per process and closed when the interpreter exits (``atexit``), after ``tmg eval`` has
written its summary and validation row.
"""
from __future__ import annotations

import atexit
import importlib
import random
import time
import traceback
from collections.abc import Callable

import numpy as np

from tmg.data.images import resize_frames
from tmg.evaluation.client import PolicyClient
from tmg.evaluation.rollout import check_server, episode_seed, policy_input
from tmg.generation.generators.autodata.runtime import reset_success_term
from tmg.generation.simulators.isaaclab import apply_scene


def launch_app(generator_cfg: dict, device: str):
    """Start Isaac Sim headless with cameras (the generator's ``kit_args``); returns the simulation app."""
    import argparse

    extra = [str(a) for a in (generator_cfg.get("extra_args") or [])]
    if "--enable_pinocchio" in extra:
        import pinocchio  # noqa: F401  (before AppLauncher, as the official scripts do)
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser()
    AppLauncher.add_app_launcher_args(parser)
    known = set(parser._option_string_actions)
    # Isaac Lab 2.x: --headless and --enable_cameras flags; Isaac Lab 3.0 (AutoData stack): --viz none, and cameras are a
    # constructor argument of AppLauncher (as AutoData's own script passes them)
    argv = (["--headless"] if "--headless" in known else ["--viz", "none"]) + ["--device", str(device)]
    kwargs = {}
    if "--enable_cameras" in known:
        argv.append("--enable_cameras")
    else:
        kwargs["enable_cameras"] = True
    if generator_cfg.get("kit_args"):
        argv += ["--kit_args", str(generator_cfg["kit_args"])]
    args = parser.parse_args(argv)
    app = AppLauncher(args, **kwargs).app
    atexit.register(app.close)
    return app


def register_arena_environment(generator_cfg: dict, task: dict, device: str, seed: int | None = None) -> None:
    """AutoData tasks: register the Arena environment through Arena's callback, as AutoData's own script does
    (``--task <name> --device <d> --num_envs 1 --enable_cameras`` plus the task's ``env_args``)."""
    import sys

    callback = importlib.import_module(str(generator_cfg["external_callback"]).rsplit(".", 1)[0])
    fn = getattr(callback, str(generator_cfg["external_callback"]).rsplit(".", 1)[1])
    argv = ["--task", str(task["env_name"]), "--device", str(device), "--num_envs", "1", "--enable_cameras"]
    if seed is not None:
        argv += ["--seed", str(int(seed))]
    argv += [str(a) for a in (task.get("env_args") or [])]
    saved = sys.argv
    try:
        sys.argv = [saved[0], *argv]
        remaining = fn()
        if remaining:
            raise RuntimeError(f"the Arena registration callback did not consume {remaining}")
    finally:
        sys.argv = saved
    print(f"[tmg] Arena environment {task['env_name']} registered ({argv})", flush=True)


def import_task_modules(generator_cfg: dict, task: dict) -> None:
    """Register the generator's environments (``<package root>.envs``, the task packages, the camera module)."""
    if generator_cfg.get("external_callback"):      # AutoData / Arena: registered by register_arena_environment
        return
    root = str(generator_cfg["package"]).split(".")[0]          # softmimicgen | isaaclab_mimic
    importlib.import_module(f"{root}.envs")
    if "--enable_pinocchio" in [str(a) for a in (generator_cfg.get("extra_args") or [])]:
        importlib.import_module(f"{root}.envs.pinocchio_envs")
        importlib.import_module("isaaclab_tasks.manager_based.manipulation.pick_place")
    for name in ("isaaclab_tasks", "softmimicgen_tasks"):
        try:
            importlib.import_module(name)
        except ImportError:
            pass
    if task.get("camera_module"):
        importlib.import_module(str(task["camera_module"]))


def build_env(plain: dict, device: str):
    """The evaluation environment and the success term (see the module docstring)."""
    import inspect

    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    task = plain["task"]
    autodata = bool(plain["generator"].get("external_callback"))
    task_id = str(task.get("camera_env") or task["env_name"])
    env_cfg = parse_env_cfg(task_id, device=str(device), num_envs=1)
    env_cfg.observations.policy.concatenate_terms = False
    success_term = env_cfg.terminations.success
    env_cfg.terminations.success = None
    if autodata:
        # as AutoData's setup_env_config: no terminations at all, and the Pink IK action with the shared URDF conversion
        env_cfg.terminations = None
        try:
            from autodata_interfaces.env.pink_ik_action import PinkInverseKinematicsActionSharedUrdf
            from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsAction
        except ImportError:
            PinkInverseKinematicsActionSharedUrdf = PinkInverseKinematicsAction = None
        if PinkInverseKinematicsAction is not None:
            for action_cfg in vars(env_cfg.actions).values():
                if getattr(action_cfg, "class_type", None) in (PinkInverseKinematicsAction,
                                                                 f"{PinkInverseKinematicsAction.__module__}:{PinkInverseKinematicsAction.__name__}"):
                    action_cfg.class_type = PinkInverseKinematicsActionSharedUrdf
    else:
        env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    if task.get("eval_episode_length_s"):
        env_cfg.episode_length_s = max(float(env_cfg.episode_length_s), float(task["eval_episode_length_s"]))
    applied = apply_scene(env_cfg, plain["scene"], str(task["name"]))
    print(f"[tmg] scene {plain['scene'].get('name')}: {'ranges applied ' + str(applied) if applied else 'official ranges kept'}", flush=True)
    env = gym.make(task_id, cfg=env_cfg).unwrapped
    if inspect.isclass(success_term.func):        # class-based term (Arena's TaskSuccessTerm): one instance for the whole run
        from tmg.generation.generators.autodata.runtime import initialize_success_term

        initialize_success_term(success_term, env)
        print(f"[tmg] success term instantiated: {type(success_term.func).__name__}", flush=True)
    return env, success_term, task_id


def seed_everything(env, seed: int) -> None:
    import torch

    env.seed(int(seed))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    random.seed(int(seed))


def _to_numpy(x) -> np.ndarray:
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x)


def initial_placement(env) -> dict:
    """Object poses right after the reset, in the environment frame (json-serializable)."""
    out = {}
    try:
        origin = _to_numpy(env.scene.env_origins[0]).astype(float)
        for name, obj in getattr(env.scene, "rigid_objects", {}).items():
            pos = _to_numpy(obj.data.root_pos_w[0]).astype(float) - origin
            quat = _to_numpy(obj.data.root_quat_w[0]).astype(float)
            out[name] = {"pos": [round(float(v), 5) for v in pos], "quat_wxyz": [round(float(v), 5) for v in quat]}
        for name, obj in getattr(env.scene, "deformable_objects", {}).items():
            nodes = _to_numpy(obj.data.nodal_pos_w[0]).astype(float) - origin
            out[name] = {"nodal_mean": [round(float(v), 5) for v in nodes.mean(axis=0)],
                         "nodal_min": [round(float(v), 5) for v in nodes.min(axis=0)], "nodal_max": [round(float(v), 5) for v in nodes.max(axis=0)]}
        for name, obj in getattr(env.scene, "articulations", {}).items():   # Arena's microwave and refrigerator (doors are joints)
            if name == "robot":
                continue
            pos = _to_numpy(obj.data.root_pos_w[0]).astype(float) - origin
            quat = _to_numpy(obj.data.root_quat_w[0]).astype(float)
            out[name] = {"pos": [round(float(v), 5) for v in pos], "quat_wxyz": [round(float(v), 5) for v in quat],
                         "joint_pos": [round(float(v), 5) for v in _to_numpy(obj.data.joint_pos[0]).astype(float)]}
    except Exception as e:  # noqa: BLE001  the placement is a diagnostic; a missing attribute must not fail the episode
        out["error"] = repr(e)
    return out


def observation_input(obs_dict: dict, observation: dict) -> dict:
    """Policy input from one environment's observation dict (groups of tensors with a leading batch of 1).

    Cameras are looked up in the ``policy`` group first and then in any other group (Arena records its head camera in a
    ``camera_obs`` group); frames larger than ``observation.image_size`` are resized with the dataset's resize.
    """
    policy = obs_dict.get("policy", obs_dict)
    flat = {}
    for cam in observation.get("cameras") or []:
        group = policy if cam in policy else next((g for g in obs_dict.values() if isinstance(g, dict) and cam in g), None)
        if group is None:
            raise KeyError(f"camera {cam!r} is in no observation group ({list(obs_dict)})")
        frame = _to_numpy(group[cam])[0]
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
        if observation.get("image_size"):
            frame = resize_frames(frame, observation["image_size"])
        flat[cam] = frame
    for key in observation["robot_state"]:
        flat[key] = _to_numpy(policy[key])[0].astype(np.float32)
    return policy_input(flat, observation, image_key="{cam}")


def run_episode(env, success_term, policy: PolicyClient, scene: dict, task: dict, observation: dict, *, horizon: int, seed: int) -> dict:
    """Play one scene; returns ``{"success", "steps", "error", "initial"}``."""
    import torch

    try:
        seed_everything(env, int(scene["seed"]))
        obs_dict, _ = env.reset()
        reset_success_term(success_term, [0])        # class-based terms keep per-environment progress state
        initial = initial_placement(env)
        policy.reset(seed, task.get("instruction"))
        width = int(env.action_manager.total_action_dim)
        success, steps = False, 0
        for steps in range(1, horizon + 1):
            action = policy.act(observation_input(obs_dict, observation))
            if action.size != width:
                raise RuntimeError(f"policy action has {action.size} values, the environment takes {width}")
            actions = torch.as_tensor(np.asarray(action, dtype=np.float32), device=env.device).view(1, width)
            obs_dict, _, terminated, truncated, _ = env.step(actions)
            if bool(success_term.func(env, **success_term.params)[0]):
                success = True
                break
            if bool(terminated) or bool(truncated):
                break
        return {"success": bool(success), "steps": int(steps), "error": None, "initial": initial}
    except Exception:  # noqa: BLE001  simulator or transport failures count as failures and are recorded
        return {"success": False, "steps": 0, "error": traceback.format_exc(), "initial": {}}


def play_isaaclab(plain: dict, todo: list[dict], checkpoint: str, horizon: int, record: Callable[[dict], None]) -> None:
    """Play the scenes of ``todo`` and hand each episode row to ``record`` (see the module docstring)."""
    task, obs_cfg, ev = plain["task"], plain["observation"], plain["evaluation"]
    device = str((plain.get("cluster") or {}).get("device", "cuda:0"))
    launch_app(plain["generator"], device)           # stays up until the interpreter exits
    if plain["generator"].get("external_callback"):
        register_arena_environment(plain["generator"], task, device, seed=int(ev["seed"]))
    import_task_modules(plain["generator"], task)
    env, success_term, task_id = build_env(plain, device)
    print(f"[tmg] Isaac Lab environment {task_id} on {device}; success term {getattr(success_term.func, '__name__', success_term.func)} "
          f"called after every step", flush=True)
    url = f"ws://{ev['server']['host']}:{ev['server']['port']}"
    try:
        with PolicyClient(url) as policy:
            info = policy.info()
            check_server(info, checkpoint)
            if info["cameras"] != list(obs_cfg.get("cameras") or []):
                raise RuntimeError(f"policy cameras {info['cameras']} differ from observation={obs_cfg['name']} ({obs_cfg.get('cameras')})")
            if bool(info.get("env_state_dim", 0)) != bool(obs_cfg.get("object_state")):
                raise RuntimeError(f"policy env_state_dim {info.get('env_state_dim', 0)} does not match observation={obs_cfg['name']}")
            for s in todo:
                seed = episode_seed(int(ev["seed"]), int(plain.get("training_seed", 0)), s["index"])
                t0 = time.time()
                r = run_episode(env, success_term, policy, s, task, obs_cfg, horizon=horizon, seed=seed)
                record({"reset_index": s["index"], "split": s["split"], "scene_seed": int(s["seed"]), **r, "policy_seed": seed,
                        "checkpoint": checkpoint, "seconds": round(time.time() - t0, 2)})
    finally:
        env.close()
