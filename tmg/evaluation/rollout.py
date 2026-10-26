"""``tmg eval``: play the fixed scenes of a (task, scene) with a served checkpoint.

The environment is robomimic's ``EnvRobosuite`` wrapper around the registered scene class, with the cameras
and image size of the observation config (``image``: the task's cameras; ``low_dim``: none, so nothing is rendered),
so observations are produced exactly as MimicGen produced the training images (same renderer, same vertical flip). Each episode:

1. ``env.reset_to({"states", "model"})`` restores one stored scene and ``sync_controllers`` re-anchors the arm
   controllers to it (robomimic leaves them at the goal of the fresh reset that precedes the restore);
2. the policy server is told ``reset`` with the episode's policy seed (``SeedSequence([eval seed, training
   seed, reset index])``), then asked for an action every control step from the observation dict
   (``observation.images.<cam>`` (3, H, W) uint8, ``observation.state`` = the robot state keys and, low-dim,
   ``observation.environment_state`` = the object state keys, all from the observation config);
3. the episode ends at task success or at the task horizon.

``episodes.jsonl`` gets one line per episode (``reset_index``, ``split``, ``success``, ``steps``,
``policy_seed``, ``checkpoint``, ``error``); a rerun plays only the missing reset indices. ``shard=[k, n]`` plays
only the scenes with reset index modulo n equal to k, so n processes (each with its own policy server) can
share one checkpoint's evaluation; they append to the same file, and a final run without ``shard`` writes the
summary. ``eval_summary.json``
holds the success rate per split with a Wilson 95% interval. The output folder is
``<run folder>/eval/<split>/<checkpoint name>`` unless ``output`` is given, so evaluations of different checkpoints
never share a file. With ``append_validation`` the summary of the
``val`` split is also appended as one row to the training run's ``validation.jsonl`` (what
``tmg select_checkpoint`` reads).
"""
from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path

import numpy as np

from tmg.config import dump_yaml, resolved
from tmg.core.chunks import write_json
from tmg.core.provenance import write_provenance
from tmg.evaluation.client import PolicyClient
from tmg.evaluation.resets import read_resets, resets_path
from tmg.generation.simulators.robosuite import load_scene_chain, register_env_class

EPISODES = "episodes.jsonl"
SUMMARY = "eval_summary.json"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of a binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def eval_output(checkpoint: Path, split: str, horizon: int | None = None) -> Path:
    """``<run folder>/eval/<split>_h<horizon>/<checkpoint name>`` for a checkpoint under ``<run folder>/checkpoints/``
    (``<split>`` alone when no horizon is given). Evaluations at different step limits never share a folder."""
    checkpoint = Path(checkpoint)
    run_dir = checkpoint.parent.parent if checkpoint.parent.name == "checkpoints" else checkpoint.parent
    folder = f"{split}_h{int(horizon)}" if horizon is not None else str(split)
    return run_dir / "eval" / folder / checkpoint.name


def make_env(env_name: str, observation: dict, robot: str, env_meta: dict | None = None):
    """The evaluation environment. With ``env_meta`` (the dataset's ``env_args``; CP-Gen) the generation
    environment is rebuilt from it (``tmg.evaluation.envs``). Otherwise: robomimic ``EnvRobosuite`` built exactly as MimicGen builds it for data collection
    (``create_for_data_processing``): it registers the cameras as rgb observations in robomimic's modality table,
    so ``get_observation`` returns the same vertically flipped uint8 (H, W, 3) images the generation wrote. With no
    cameras (low-dim) robomimic creates the environment without an offscreen renderer."""
    if env_meta is not None:
        from tmg.evaluation.envs import make_env_from_meta

        return make_env_from_meta(env_meta, env_name, observation)
    from robomimic.envs.env_robosuite import EnvRobosuite
    from robosuite.controllers import load_controller_config

    cameras = list(observation.get("cameras") or [])
    h, w = (int(v) for v in observation["image_size"]) if cameras else (84, 84)
    return EnvRobosuite.create_for_data_processing(
        env_name, camera_names=cameras, camera_height=h, camera_width=w, reward_shaping=False,
        robots=robot, controller_configs=load_controller_config(default_controller="OSC_POSE"), ignore_done=True,
        use_object_obs=True, control_freq=20)


def policy_input(obs: dict, observation: dict) -> dict:
    """Policy input from a robomimic observation dict (images HWC uint8 -> CHW; state keys concatenated)."""
    out = {}
    for cam in observation.get("cameras") or []:
        img = np.asarray(obs[f"{cam}_image"])
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        out[f"observation.images.{cam}"] = np.ascontiguousarray(img.transpose(2, 0, 1))
    out["observation.state"] = np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in observation["robot_state"]])
    object_keys = list(observation.get("object_state") or [])
    if object_keys:
        out["observation.environment_state"] = np.concatenate([np.asarray(obs[k], dtype=np.float32).reshape(-1) for k in object_keys])
    return out


def select_scenes(scenes: list[dict], *, limit=None, shard=None) -> list[dict]:
    """The scenes one ``tmg eval`` process plays: the first ``limit`` (check runs), then shard ``k`` of ``n``
    (``shard=[k, n]``: the scenes whose reset index modulo n equals k). Shards of one checkpoint append to the
    same ``episodes.jsonl`` (one line per episode); a final run without ``shard`` finds every scene done and
    writes the summary."""
    if limit is not None:
        scenes = scenes[:int(limit)]
    if shard is not None:
        k, n = (int(v) for v in shard)
        if not 0 <= k < n:
            raise ValueError(f"shard must be [k, n] with 0 <= k < n, got {shard}")
        scenes = [s for s in scenes if int(s["index"]) % n == k]
    return scenes


def check_server(info: dict, checkpoint: str) -> None:
    """The server must serve exactly the requested checkpoint. A server that failed to bind its port (another
    server still holds it) leaves the client talking to that other server; without this check the episodes
    would be recorded under the wrong checkpoint, as happened on 2026-10-27 when validation ports were reused."""
    served = str(info.get("checkpoint", "")).rstrip("/")
    if served != str(checkpoint).rstrip("/"):
        raise RuntimeError(f"the policy server serves {served!r}, not the requested {checkpoint!r}; "
                           "another server probably holds the port")


def episode_seed(eval_seed: int, training_seed: int, reset_index: int) -> int:
    return int(np.random.SeedSequence([eval_seed, training_seed, reset_index]).generate_state(1)[0])


def sync_controllers(env) -> None:
    """Re-anchor the robosuite arm controllers to the restored state.

    robomimic's ``reset_to`` first does a fresh ``reset()`` (which draws new initial-joint noise and sets the
    controller goal to that pose) and then overwrites the simulator state with the stored one without touching
    the controllers. The OSC controller therefore keeps a goal pose that belongs to the discarded reset, and the
    first commanded deltas are applied relative to the wrong end-effector pose. Generation never hits this (it
    plays from a fresh ``reset()``), so without this call every evaluation episode starts with a jump that the
    training data does not contain. The fix mirrors what ``Robot.reset`` does after setting the joints: initial
    joints, controller state and goal from the restored simulator state."""
    robosuite_env = getattr(env, "env", env)
    for robot in getattr(robosuite_env, "robots", []):
        controller = getattr(robot, "controller", None)
        if controller is None or not hasattr(controller, "reset_goal"):
            continue                 # robosuite 1.5 composite controllers (CP-Gen, absolute actions): no stale delta goal
        controller.update_initial_joints(robot.sim.data.qpos[robot._ref_joint_pos_indexes])
        controller.update(force=True)
        controller.reset_goal()


def run_episode(env, policy: PolicyClient, scene: dict, task: dict, observation: dict, *, horizon: int, seed: int) -> dict:
    """Play one scene; returns ``{"success", "steps", "error"}``."""
    try:
        obs = env.reset_to({"states": scene["states"], "model": scene["model"]})
        sync_controllers(env)
        obs = env.get_observation()
        policy.reset(seed, task.get("instruction"))
        success, steps = False, 0
        for steps in range(1, horizon + 1):
            action = policy.act(policy_input(obs, observation))
            obs, _, _, _ = env.step(action)
            if env.is_success()["task"]:
                success = True
                break
        return {"success": bool(success), "steps": int(steps), "error": None}
    except Exception:  # noqa: BLE001  simulator or transport failures count as failures and are recorded
        return {"success": False, "steps": 0, "error": traceback.format_exc()}


def summarize(rows: list[dict]) -> dict:
    """Success rate with Wilson interval overall and per split."""
    def block(rs):
        n, k = len(rs), sum(int(r["success"]) for r in rs)
        lo, hi = wilson(k, n)
        return {"n": n, "n_success": k, "success_rate": (k / n if n else None), "ci_low": lo, "ci_high": hi,
                "mean_steps": (float(np.mean([r["steps"] for r in rs])) if rs else None), "errors": sum(1 for r in rs if r.get("error"))}
    out = {"all": block(rows), "by_split": {}}
    for split in sorted({r["split"] for r in rows}):
        out["by_split"][split] = block([r for r in rows if r["split"] == split])
    return out


def evaluate(cfg) -> dict:
    """Run ``tmg eval`` (see the module docstring); returns the summary."""
    plain = resolved(cfg)
    task, scene = plain["task"], load_scene_chain(plain["scene"], plain["generator"]["name"])
    obs_cfg = plain["observation"]
    ev = plain["evaluation"]
    horizon = int(task["horizon"])
    output = Path(plain["output"]) if plain.get("output") else eval_output(Path(plain["checkpoint"]), str(plain.get("split")), horizon)
    output.mkdir(parents=True, exist_ok=True)
    dump_yaml(plain, output / "resolved_config.yaml")
    write_provenance(output, extra={"command": "eval", "output": str(output)})
    split = plain.get("split")
    scenes = read_resets(resets_path(plain), split if split != "all" else None)
    if not scenes:
        raise FileNotFoundError(f"no scenes of split {split!r} in {resets_path(plain)}")
    shard = plain.get("shard")
    scenes = select_scenes(scenes, limit=plain.get("max_scenes"), shard=shard)
    done = {}
    log = output / EPISODES
    checkpoint = str(plain["checkpoint"])
    if log.exists():
        for line in log.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("checkpoint") == checkpoint:
                    done[r["reset_index"]] = r
                else:
                    raise RuntimeError(f"{log} holds episodes of another checkpoint ({r.get('checkpoint')}); "
                                       f"use another output folder for {checkpoint}")
    todo = [s for s in scenes if s["index"] not in done]
    print(f"[tmg] eval {checkpoint} on {len(scenes)} scenes ({len(todo)} to play) of {task['name']}/{scene['name']}", flush=True)
    if todo:
        env_name = register_env_class(task, scene)
        env_meta = None
        if plain.get("dataset"):
            from tmg.evaluation.envs import env_meta_from_dataset

            env_meta = env_meta_from_dataset(Path(plain["dataset"]))
            print(f"[tmg] environment from the dataset's env_args ({plain['dataset']})", flush=True)
        env = make_env(env_name, obs_cfg, str(plain.get("robot", "Panda")), env_meta)
        url = f"ws://{ev['server']['host']}:{ev['server']['port']}"
        with PolicyClient(url) as policy:
            info = policy.info()
            check_server(info, checkpoint)
            if info["cameras"] != list(obs_cfg.get("cameras") or []):
                raise RuntimeError(f"policy cameras {info['cameras']} differ from observation={obs_cfg['name']} ({obs_cfg.get('cameras')})")
            wants_env = bool(obs_cfg.get("object_state"))
            if bool(info.get("env_state_dim", 0)) != wants_env:
                raise RuntimeError(f"policy env_state_dim {info.get('env_state_dim', 0)} does not match observation={obs_cfg['name']}; "
                                   "evaluate with the observation the checkpoint was trained with")
            with open(log, "a") as f:
                for s in todo:
                    seed = episode_seed(int(ev["seed"]), int(plain.get("training_seed", 0)), s["index"])
                    t0 = time.time()
                    r = run_episode(env, policy, s, task, obs_cfg, horizon=horizon, seed=seed)
                    row = {"reset_index": s["index"], "split": s["split"], **r, "policy_seed": seed, "checkpoint": checkpoint,
                           "seconds": round(time.time() - t0, 2)}
                    f.write(json.dumps(row) + "\n")
                    f.flush()
                    done[s["index"]] = row
                    print(f"[tmg]   reset {s['index']} ({s['split']}): {'success' if r['success'] else 'fail'} in {r['steps']} steps", flush=True)
        env.env.close()
    rows = [done[s["index"]] for s in scenes]
    summary = {"checkpoint": checkpoint, "task": task["name"], "scene": scene["name"], "split": split, "horizon": horizon, **summarize(rows),
               "policy_seed_rule": "SeedSequence([evaluation.seed, training_seed, reset_index])"}
    if shard is not None:
        print(f"[tmg] eval shard {list(shard)} done: {summary['all']['n_success']}/{summary['all']['n']} successes "
              f"(no summary written; run once more without shard to merge)", flush=True)
        return summary
    write_json(output / SUMMARY, summary)
    print(f"[tmg] eval done: {summary['all']['n_success']}/{summary['all']['n']} successes", flush=True)
    run_dir = plain.get("append_validation")
    if run_dir:
        block = summary["by_split"].get("val") or summary["all"]
        with open(Path(run_dir) / "validation.jsonl", "a") as f:
            f.write(json.dumps({"checkpoint": checkpoint, "success_rate": block["success_rate"], "n": block["n"],
                                "n_success": block["n_success"], "horizon": horizon, "eval_output": str(output)}) + "\n")
    return summary
