"""``tmg resets``: the fixed evaluation scenes of a (task, scene).

The environment of the registered scene class is reset ``val + test`` times from ``seed``; each initial state
is stored with the model xml of that reset (MimicGen places fixtures such as drawers by editing the model,
so the flat simulator state alone does not restore them) and the object positions. Layout of
``<paths.resets>/<generator>/<task>/<scene>/val<V>_test<T>_seed<seed>.hdf5``::

    attrs: env_name, task, scene, seed, n_val, n_test, env_kwargs (json)
    reset_<i>/states    (S,) float64     flat MuJoCo state after the reset
    reset_<i>/model     bytes            model xml of that reset
    reset_<i>/attrs: split ("val" | "test"), object_xy (json), reach_rejects

Rollouts restore a scene with robomimic's ``EnvRobosuite.reset_to({"states": ..., "model": ...})``.

Isaac Lab generators (``generator.simulator == "isaaclab"``) have no stored states: a scene is a seed. The file is
``<paths.resets>/<generator>/<task>/<scene>/val<V>_test<T>_seed<seed>.json`` with one entry per scene index
``{"index", "split", "seed"}`` where ``seed = SeedSequence([evaluation.seed, index])``. The rollout seeds the
environment, torch, numpy and ``random`` with it right before ``env.reset()``, so the same evaluation seed gives the
same validation and test scenes in every run, on every machine, for every policy and condition (the reset events
draw placements from the seeded torch generator). Writing the file needs no simulator.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from tmg.config import dump_yaml, resolved
from tmg.core.provenance import write_provenance
from tmg.generation.simulators.robosuite import (
    SCENE_META_ATTR,
    load_scene_chain,
    make_object_xy,
    register_env_class,
)

ENV_KWARGS = {"has_renderer": False, "has_offscreen_renderer": False, "use_camera_obs": False, "use_object_obs": True,
              "ignore_done": True, "control_freq": 20, "reward_shaping": False}


def is_isaaclab(plain: dict) -> bool:
    """Whether the composed config's generator simulates in Isaac Lab (scenes are seeds, not stored states)."""
    return str(plain["generator"].get("simulator", "")) == "isaaclab"


def resets_path(plain: dict) -> Path:
    ev = plain["evaluation"]
    ext = "json" if is_isaaclab(plain) else "hdf5"
    return Path(plain["paths"]["resets"]) / plain["generator"]["name"] / plain["task"]["name"] / plain["scene"]["name"] / \
        f"val{int(ev['val_scenes'])}_test{int(ev['test_scenes'])}_seed{int(ev['seed'])}.{ext}"


def scene_seed(eval_seed: int, index: int) -> int:
    """The reset seed of scene ``index``: ``SeedSequence([evaluation.seed, index])``, fixed for all runs."""
    return int(np.random.SeedSequence([int(eval_seed), int(index)]).generate_state(1)[0])


def isaaclab_scenes(eval_seed: int, n_val: int, n_test: int) -> list[dict]:
    """``[{"index", "split", "seed"}]`` of the validation scenes (first) and the test scenes."""
    return [{"index": i, "split": "val" if i < n_val else "test", "seed": scene_seed(eval_seed, i)} for i in range(n_val + n_test)]


def make_isaaclab_resets(plain: dict, path: Path) -> dict:
    """Write the seed file of an Isaac Lab task (see the module docstring)."""
    task, ev = plain["task"], plain["evaluation"]
    n_val, n_test, seed = int(ev["val_scenes"]), int(ev["test_scenes"]), int(ev["seed"])
    out = {"env_name": str(task.get("camera_env") or task["env_name"]), "task": task["name"], "scene": plain["scene"]["name"],
           "seed": seed, "n_val": n_val, "n_test": n_test,
           "rule": "scene seed = SeedSequence([evaluation.seed, index]); the rollout calls env.seed, torch.manual_seed, "
                   "numpy.random.seed and random.seed with it right before env.reset()",
           "scenes": isaaclab_scenes(seed, n_val, n_test)}
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(".json.part")
    part.write_text(json.dumps(out, indent=1))
    part.replace(path)
    dump_yaml(plain, path.with_suffix(".resolved_config.yaml"))
    write_provenance(path.parent, extra={"command": "resets", "output": str(path)})
    print(f"[tmg] resets {path}: {n_val} validation + {n_test} test scene seeds of {out['env_name']}")
    return {"path": str(path), "n_val": n_val, "n_test": n_test, "env_name": out["env_name"]}


def make_resets(cfg) -> dict:
    """Draw and store the scenes (see the module docstring); an existing file is kept.

    Returns:
        ``{"path", "n_val", "n_test", "env_name"}``.
    """
    plain = resolved(cfg)
    path = resets_path(plain)
    if path.exists() and not plain.get("overwrite", False):
        print(f"[tmg] resets exist, keeping them: {path}")
        if path.suffix == ".json":
            meta = json.loads(path.read_text())
            return {"path": str(path), "n_val": int(meta["n_val"]), "n_test": int(meta["n_test"]), "env_name": str(meta["env_name"])}
        with h5py.File(path, "r") as f:
            return {"path": str(path), "n_val": int(f.attrs["n_val"]), "n_test": int(f.attrs["n_test"]), "env_name": str(f.attrs["env_name"])}
    if is_isaaclab(plain):
        return make_isaaclab_resets(plain, path)
    task, scene = plain["task"], load_scene_chain(plain["scene"], plain["generator"]["name"])
    ev = plain["evaluation"]
    path.parent.mkdir(parents=True, exist_ok=True)
    env_name = register_env_class(task, scene)
    if plain.get("dataset"):
        # the generation environment of another generator (CP-Gen): rebuilt from the dataset's env_args
        from tmg.evaluation.envs import adjusted_env_meta, env_meta_from_dataset, make_env_from_meta

        meta = env_meta_from_dataset(Path(plain["dataset"]))
        env = make_env_from_meta(meta, env_name, {"cameras": [], "robot_state": task.get("robot_state", []), "object_state": []}).env
        kwargs = adjusted_env_meta(meta, env_name, [], None)["env_kwargs"]
        print(f"[tmg] environment from the dataset's env_args ({plain['dataset']})", flush=True)
    else:
        import robosuite
        from robosuite.controllers import load_controller_config

        kwargs = dict(ENV_KWARGS, robots=str(plain.get("robot", "Panda")), controller_configs=load_controller_config(default_controller="OSC_POSE"))
        env = robosuite.make(env_name, **kwargs)
    interface = str(task.get("datagen_interface") or f"MG_{task['env_prefix']}")
    object_xy = make_object_xy(interface)
    n_val, n_test, seed = int(ev["val_scenes"]), int(ev["test_scenes"]), int(ev["seed"])
    np.random.seed(seed)
    part = path.with_suffix(".hdf5.part")
    with h5py.File(part, "w") as f:
        f.attrs["env_name"], f.attrs["task"], f.attrs["scene"], f.attrs["seed"] = env_name, task["name"], scene["name"], seed
        f.attrs["n_val"], f.attrs["n_test"] = n_val, n_test
        f.attrs["env_kwargs"] = json.dumps(kwargs, default=str)
        for i in range(n_val + n_test):
            env.reset()
            g = f.create_group(f"reset_{i}")
            g.create_dataset("states", data=np.asarray(env.sim.get_state().flatten(), dtype=np.float64))
            g.create_dataset("model", data=np.frombuffer(env.sim.model.get_xml().encode(), dtype=np.uint8), compression="gzip")
            g.attrs["split"] = "val" if i < n_val else "test"
            g.attrs["object_xy"] = json.dumps({k: [float(x) for x in v] for k, v in object_xy(env).items()})
            meta = getattr(env, SCENE_META_ATTR, None)
            g.attrs["reach_rejects"] = int(meta["n_reject"]) if meta else 0
    env.close()
    part.replace(path)
    dump_yaml(plain, path.with_suffix(".resolved_config.yaml"))
    write_provenance(path.parent, extra={"command": "resets", "output": str(path)})
    print(f"[tmg] resets {path}: {n_val} validation + {n_test} test scenes of {env_name}")
    return {"path": str(path), "n_val": n_val, "n_test": n_test, "env_name": env_name}


def read_resets(path: Path, split: str | None = None) -> list[dict]:
    """``[{"index", "split", "states", "model"}]`` of a robosuite resets file, or ``[{"index", "split", "seed"}]`` of an
    Isaac Lab seed file, optionally one split only."""
    path = Path(path)
    if path.suffix == ".json":
        scenes = json.loads(path.read_text())["scenes"]
        return [dict(s) for s in scenes if split is None or s["split"] == split]
    out = []
    with h5py.File(path, "r") as f:
        keys = sorted((k for k in f if k.startswith("reset_")), key=lambda k: int(k.split("_")[1]))
        for k in keys:
            g = f[k]
            if split is not None and g.attrs["split"] != split:
                continue
            out.append({"index": int(k.split("_")[1]), "split": str(g.attrs["split"]), "states": np.asarray(g["states"]),
                        "model": bytes(np.asarray(g["model"])).decode()})
    return out
