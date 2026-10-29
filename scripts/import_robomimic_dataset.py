"""Import a robomimic / MimicGen demonstration file as a tmg low-dim dataset.

    python scripts/import_robomimic_dataset.py --src core/three_piece_assembly_d2.hdf5 --dst <datasets>/.../n1000_seed0.hdf5 \
        --task three_piece_assembly --generation official [--size 1000] [--seed 0]

Writes the common hdf5 that ``tmg train`` reads (see tmg/data/dataset.py): per demo ``obs/<key>`` for the robot
state keys and object state keys, their concatenations ``obs/robot_state`` and ``obs/object_state``, ``actions``
(the file's actions, which MimicGen's released datasets store as executed), ``actions_executed`` and
``actions_pre_noise`` (both equal to ``actions``: the official files have no pre-noise record), ``states`` and
``src_demo_inds`` when present. Images are not imported (low-dim only). Used to separate our training and
evaluation pipeline from our data generation: a policy trained on the official dataset and evaluated with our
evaluator measures the pipeline alone.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from tmg.core.layout import indexed_keys

ROBOT_STATE = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
OBJECT_STATE = ["object"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--generation", default="official")
    parser.add_argument("--scene", default="ours")
    parser.add_argument("--size", type=int, default=None, help="number of demos to keep (default: all)")
    parser.add_argument("--seed", type=int, default=0, help="selection seed when --size is below the demo count")
    parser.add_argument("--robot-state", nargs="+", default=ROBOT_STATE)
    parser.add_argument("--object-state", nargs="+", default=OBJECT_STATE)
    args = parser.parse_args(argv)

    dst = Path(args.dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    total = 0
    with h5py.File(args.src, "r") as src, h5py.File(part, "w") as out:
        demos = indexed_keys(src["data"].keys())
        if args.size is not None and args.size < len(demos):
            demos = [demos[i] for i in sorted(np.random.default_rng(args.seed).choice(len(demos), args.size, replace=False))]
        data = out.create_group("data")
        for j, key in enumerate(demos):
            demo = src["data"][key]
            g = data.create_group(f"demo_{j}")
            obs = g.create_group("obs")
            for group, keys in (("robot_state", args.robot_state), ("object_state", args.object_state)):
                parts = []
                for k in keys:
                    arr = np.asarray(demo["obs"][k], dtype=np.float32)
                    obs.create_dataset(k, data=arr)
                    parts.append(arr.reshape(arr.shape[0], -1))
                obs.create_dataset(group, data=np.concatenate(parts, axis=1))
            actions = np.asarray(demo["actions"], dtype=np.float32)
            g.create_dataset("actions", data=actions)
            g.create_dataset("actions_executed", data=actions)
            g.create_dataset("actions_pre_noise", data=actions)
            if "states" in demo:
                g.create_dataset("states", data=np.asarray(demo["states"], dtype=np.float64))
            if "src_demo_inds" in demo:
                g.create_dataset("src_demo_inds", data=np.asarray(demo["src_demo_inds"], dtype=np.int64))
            g.attrs["source_demo"], g.attrs["num_samples"] = key, int(len(actions))
            total += int(len(actions))
        data.attrs["env_args"] = src["data"].attrs.get("env_args", "{}")
        data.attrs["total"] = total
        data.attrs["cameras"] = json.dumps([])
        data.attrs["robot_state_keys"] = json.dumps(list(args.robot_state))
        data.attrs["object_state_keys"] = json.dumps(list(args.object_state))
        for k, v in dict(task=args.task, generator="mimicgen", scene=args.scene, method="none", generation=args.generation,
                         observation="low_dim", label="executed", seed=args.seed, imported_from=str(args.src)).items():
            data.attrs[k] = v if isinstance(v, (int, float)) else str(v)
        out.create_group("mask").create_dataset("train", data=np.array([f"demo_{j}".encode() for j in range(len(demos))]))
    part.replace(dst)
    print(f"[tmg] imported {len(demos)} demos, {total} steps -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
