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
summary. ``eval_summary.json`` holds the success rate per split with a Wilson 95% interval. The output folder is
``<run folder>/eval/<split>_h<horizon>/<checkpoint name>`` unless ``output`` is given, so evaluations of different
checkpoints or step limits never share a file. With ``append_validation`` the summary of the ``val`` split is also
appended as one row to the training run's ``validation.jsonl`` (what ``tmg select_checkpoint`` reads).

The loop itself (server checks, per-episode seeds, episode rows) is ``tmg.evaluation.session``; this module adds the
robosuite environment and episode. Isaac Lab generators (SoftMimicGen, DexMimicGen) take the same command and write
the same files; their episodes are played by ``tmg.evaluation.isaaclab_rollout`` (Isaac Sim in the generator's
image, scenes fixed by seeds).
"""
from __future__ import annotations

import traceback
from pathlib import Path

from tmg.config import dump_yaml, resolved
from tmg.core.io_json import append_jsonl, read_jsonl, write_json
from tmg.core.layout import EPISODES, RESOLVED_CONFIG, SUMMARY, VALIDATION_JSONL, eval_output
from tmg.core.provenance import write_provenance
from tmg.evaluation.client import PolicyClient
from tmg.evaluation.envs import ROBOSUITE_EVAL_KWARGS, is_isaaclab
from tmg.evaluation.resets import read_resets, resets_path
from tmg.evaluation.session import (  # noqa: F401  re-exported: tests and scripts import them from here
    Scene,
    check_server,
    episode_seed,
    play_scenes,
    policy_input,
    select_scenes,
    summarize,
    wilson,
)
from tmg.generation.simulators.robosuite import load_scene_chain, register_env_class

__all__ = ["EPISODES", "SUMMARY", "eval_output", "wilson", "summarize", "episode_seed", "check_server", "select_scenes",
           "policy_input", "make_env", "sync_controllers", "run_episode", "append_validation_row", "evaluate"]

POLICY_SEED_RULE = "SeedSequence([evaluation.seed, training_seed, reset_index])"


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
        env_name, camera_names=cameras, camera_height=h, camera_width=w,
        robots=robot, controller_configs=load_controller_config(default_controller="OSC_POSE"), **ROBOSUITE_EVAL_KWARGS)


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


def run_episode(env, policy: PolicyClient, reset: Scene, task: dict, observation: dict, *, horizon: int, seed: int) -> dict:
    """Play one stored scene; returns ``{"success", "steps", "error"}``."""
    try:
        obs = env.reset_to({"states": reset["states"], "model": reset["model"]})
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


def _load_done(log: Path, checkpoint: str) -> dict[int, dict]:
    """The episodes of ``checkpoint`` already in ``episodes.jsonl`` by reset index (a rerun plays only the others).

    Raises:
        RuntimeError: the file holds episodes of another checkpoint (two evaluations must never share a folder).
    """
    done = {}
    for row in read_jsonl(log):
        if row.get("checkpoint") != checkpoint:
            raise RuntimeError(f"{log} holds episodes of another checkpoint ({row.get('checkpoint')}); "
                               f"use another output folder for {checkpoint}")
        done[row["reset_index"]] = row
    return done


def _play_robosuite(plain: dict, task: dict, scene: dict, todo: list[Scene], checkpoint: str, horizon: int, record) -> None:
    """Build the robosuite evaluation environment and play ``todo`` through the shared loop (``session.play_scenes``)."""
    obs_cfg, ev = plain["observation"], plain["evaluation"]
    env_name = register_env_class(task, scene)
    env_meta = None
    if plain.get("dataset"):
        from tmg.evaluation.envs import env_meta_from_dataset

        env_meta = env_meta_from_dataset(Path(plain["dataset"]))
        print(f"[tmg] environment from the dataset's env_args ({plain['dataset']})", flush=True)
    env = make_env(env_name, obs_cfg, str(plain.get("robot", "Panda")), env_meta)

    def run_one(policy, reset, seed):
        return run_episode(env, policy, reset, task, obs_cfg, horizon=horizon, seed=seed)

    play_scenes(todo, server=ev["server"], checkpoint=checkpoint, obs_cfg=obs_cfg, eval_seed=ev["seed"],
                training_seed=plain.get("training_seed", 0), run_one=run_one, record=record)
    env.env.close()


def append_validation_row(run_dir: Path, *, checkpoint: str, summary: dict, horizon: int, output: Path) -> dict:
    """Append the ``val`` block of ``summary`` (the whole summary when it has no ``val`` split) to the run's
    ``validation.jsonl``; returns the row written (what ``tmg select_checkpoint`` reads)."""
    block = summary["by_split"].get("val") or summary["all"]
    row = {"checkpoint": checkpoint, "success_rate": block["success_rate"], "n": block["n"], "n_success": block["n_success"],
           "horizon": horizon, "eval_output": str(output)}
    append_jsonl(Path(run_dir) / VALIDATION_JSONL, row)
    return row


def evaluate(cfg) -> dict:
    """Run ``tmg eval`` (see the module docstring); returns the summary."""
    plain = resolved(cfg)
    task = plain["task"]
    scene = plain["scene"] if is_isaaclab(plain) else load_scene_chain(plain["scene"], plain["generator"]["name"])
    horizon = int(task["horizon"])
    output = Path(plain["output"]) if plain.get("output") else eval_output(Path(plain["checkpoint"]), str(plain.get("split")), horizon)
    output.mkdir(parents=True, exist_ok=True)
    dump_yaml(plain, output / RESOLVED_CONFIG)
    write_provenance(output, extra={"command": "eval", "output": str(output)})
    split = plain.get("split")
    scenes = read_resets(resets_path(plain), split if split != "all" else None)
    if not scenes:
        raise FileNotFoundError(f"no scenes of split {split!r} in {resets_path(plain)}")
    shard = plain.get("shard")
    scenes = select_scenes(scenes, limit=plain.get("max_scenes"), shard=shard)
    log = output / EPISODES
    checkpoint = str(plain["checkpoint"])
    done = _load_done(log, checkpoint)
    todo = [s for s in scenes if s["index"] not in done]
    print(f"[tmg] eval {checkpoint} on {len(scenes)} scenes ({len(todo)} to play) of {task['name']}/{scene['name']}", flush=True)

    def record(row: dict) -> None:
        append_jsonl(log, row)
        done[row["reset_index"]] = row
        print(f"[tmg]   reset {row['reset_index']} ({row['split']}): {'success' if row['success'] else 'fail'} in {row['steps']} steps", flush=True)

    if todo and is_isaaclab(plain):
        from tmg.evaluation.isaaclab_rollout import play_isaaclab

        play_isaaclab(plain, todo, checkpoint, horizon, record)
    elif todo:
        _play_robosuite(plain, task, scene, todo, checkpoint, horizon, record)
    rows = [done[s["index"]] for s in scenes]
    summary = {"checkpoint": checkpoint, "task": task["name"], "scene": scene["name"], "split": split, "horizon": horizon, **summarize(rows),
               "policy_seed_rule": POLICY_SEED_RULE}
    if shard is not None:
        print(f"[tmg] eval shard {list(shard)} done: {summary['all']['n_success']}/{summary['all']['n']} successes "
              f"(no summary written; run once more without shard to merge)", flush=True)
        return summary
    write_json(output / SUMMARY, summary)
    print(f"[tmg] eval done: {summary['all']['n_success']}/{summary['all']['n']} successes", flush=True)
    if plain.get("append_validation"):
        append_validation_row(Path(plain["append_validation"]), checkpoint=checkpoint, summary=summary, horizon=horizon,
                              output=output)
    return summary
