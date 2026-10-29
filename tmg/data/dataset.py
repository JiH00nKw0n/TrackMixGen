"""``tmg dataset``: one training dataset (common hdf5) from the successful attempts of a generation pool.

Input is a finished pool of (generator, task, scene, method, generation) and an ``observation`` choice: ``image``
stores the cameras and the robot state, ``low_dim`` stores the robot state and the task's object state keys and
no images (it can be built from a pool generated without cameras). Every chunk holds the generator's
own demonstration file with the successful attempts in order (MimicGen: ``<experiment>/demo.hdf5``; Isaac Lab
generators: ``generated.hdf5``) and our attempt records (``records/attempts.jsonl`` and ``attempt_%06d.npz``, all
attempts). Where each generator keeps images, actions and states, and whether the pre-noise labels come from
the records (Isaac Lab, whose recorder only stores executed actions), is the adapter's ``demo_layout()``
(``DemoLayout``). The k-th demonstration of a chunk is the k-th successful attempt of that chunk; the writer
checks that their step counts agree, so a mismatch is an error rather than a silently shifted label.

Output layout (``<output>``, robomimic-compatible plus our extras)::

    data.attrs: env_args (json of the generator's env meta), total, task, generator, scene, method, observation,
                cameras (json list), robot_state_keys (json list), object_state_keys (json list),
                label (which action the policy learns)
    data/demo_<j>/
        obs/<camera>_image      (T, H, W, 3) uint8      cameras of the observation config, as rendered during generation
        obs/<key>               (T, d) float32          every robot state and object state key of the observation config
        obs/robot_state         (T, D) float32          the robot state keys concatenated in config order
        obs/object_state        (T, E) float32          the object state keys concatenated (absent for ``image``)
        actions                 (T, A) float32          the pool's action label (method.labels decides pre-noise or executed)
        actions_executed        (T, A) float32          what the generator executed (noisy, clipped)
        actions_pre_noise       (T, A) float32          clipped pre-noise arm action + gripper
        states                  (T, S) float64          simulator states (for replay and for the first-state reset)
        subtask_of_step         (T,)   int16            subtask index of each step (from the segment table)
        src_demo_inds           (K,)   int64            source demonstration chosen for each subtask
        attrs: chunk, attempt, num_samples, pool
    mask/train                  (N,)   bytes            every demo key (one split; validation is by rollout)

Selection is uniform without replacement among all successes of the pool (``seed``). The sidecar
``DATASET.json`` lists the chosen (chunk, demo, attempt) triples and the pool's ``DONE.json`` summary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

from tmg.config import dump_yaml, resolved
from tmg.core.chunks import CHUNKS_DIR, POOL_DONE, chunk_is_done, read_json, write_json
from tmg.core.provenance import write_provenance
from tmg.core.records import RECORDS_DIR, read_attempts
from tmg.data.images import resize_frames

SIDECAR = "DATASET.json"


@dataclass(frozen=True)
class DemoLayout:
    """Where a generator's chunk output keeps what the dataset needs (``adapter.demo_layout()``)."""

    demo_glob: str                    # glob under the chunk folder of the demonstration hdf5
    image_key: str                    # obs key pattern of a camera, with ``{cam}`` (MimicGen: "{cam}_image")
    actions_key: str = "actions"      # dataset of the generator's action labels
    states_key: str | None = "states" # simulator states, or None when the generator keeps none in that layout
    labels_from_records: bool = False # pre-noise labels in ``records/labels_%06d.npz`` instead of the attempt npz
    images_from_records: bool = False # camera frames in ``records/images_%06d.npz`` (AutoData: the recorder keeps no camera group)
    success_only_file: bool = True    # the file lists successes in attempt order

    @classmethod
    def from_mapping(cls, m: dict) -> DemoLayout:
        return cls(**{k: m[k] for k in ("demo_glob", "image_key", "actions_key", "states_key", "labels_from_records", "images_from_records",
                                        "success_only_file") if k in m})


MIMICGEN_LAYOUT = DemoLayout(demo_glob="*/demo.hdf5", image_key="{cam}_image")


class DatasetError(RuntimeError):
    """The pool cannot provide the requested dataset (too few successes, inconsistent records)."""


def chunk_demo_file(chunk: Path, layout: DemoLayout = MIMICGEN_LAYOUT) -> Path | None:
    """The generator's demonstration file of a chunk, or None when the chunk has no success."""
    hits = sorted(chunk.glob(layout.demo_glob))
    return hits[0] if hits else None


def successes_of_chunk(chunk: Path) -> list[dict]:
    """Attempt records of the chunk's successful attempts, in attempt order (= demonstration order)."""
    return [r for r in read_attempts(chunk / RECORDS_DIR) if r.get("success")]


def candidates(pool: Path, layout: DemoLayout = MIMICGEN_LAYOUT) -> list[dict]:
    """Every success of the pool as ``{chunk, demo_key, attempt, n_steps}``; demo order checked against the records.

    Raises:
        DatasetError: a chunk's demonstration count or step counts disagree with its attempt records.
    """
    out = []
    for chunk in sorted(p for p in (pool / CHUNKS_DIR).glob("seed*") if chunk_is_done(p)):
        records = successes_of_chunk(chunk)
        demo_file = chunk_demo_file(chunk, layout)
        if not records:
            continue
        if demo_file is None:
            raise DatasetError(f"{chunk.name}: {len(records)} successful attempts but no demo.hdf5")
        with h5py.File(demo_file, "r") as f:
            keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))
            if len(keys) != len(records):
                raise DatasetError(f"{chunk.name}: demo.hdf5 has {len(keys)} demos, records list {len(records)} successes")
            for key, rec in zip(keys, records):
                n = int(f["data"][key][layout.actions_key].shape[0])
                if n != int(rec["n_steps"]):
                    raise DatasetError(f"{chunk.name}/{key}: {n} steps in demo.hdf5, {rec['n_steps']} in the records "
                                       f"(attempt {rec['attempt']})")
                out.append({"chunk": chunk.name, "demo_file": str(demo_file), "demo_key": key,
                            "attempt": int(rec["attempt"]), "n_steps": n, "src_demo_inds": rec.get("src_demo_inds")})
    return out


def select(cands: list[dict], size: int, seed: int) -> list[dict]:
    """``size`` candidates drawn uniformly without replacement with ``np.random.default_rng(seed)``."""
    if len(cands) < size:
        raise DatasetError(f"pool has {len(cands)} successes, dataset needs {size}")
    rng = np.random.default_rng(int(seed))
    idx = np.sort(rng.choice(len(cands), size=size, replace=False))
    return [cands[i] for i in idx]


def _subtask_of_step(z) -> np.ndarray:
    """Subtask index per step from the segment arrays of an attempt npz."""
    seg_sub = np.asarray(z["seg_sub"], dtype=np.int64)
    seg_of_step = np.asarray(z["seg_of_step"], dtype=np.int64)
    valid = (seg_of_step >= 0) & (seg_of_step < len(seg_sub))
    out = np.full(len(seg_of_step), -1, dtype=np.int16)
    out[valid] = seg_sub[seg_of_step[valid]]
    return out


def write_dataset(path: Path, chosen: list[dict], *, pool: Path, observation: dict, meta: dict,
                  layout: DemoLayout = MIMICGEN_LAYOUT) -> dict:
    """Write the common hdf5 (layout in the module docstring) atomically (``.part`` then rename).

    Args:
        path: destination file.
        chosen: selected candidates (``select``).
        pool: the pool folder (records are read from ``<pool>/chunks/<chunk>/records``).
        observation: the observation config (``cameras``, ``robot_state``, ``object_state``).
        meta: values for ``data.attrs`` besides ``env_args``/``total``.
        layout: where the generator keeps images, actions, states and labels.

    Returns:
        ``{"demos": N, "steps": total steps, "image_shape": [...] or None}``.
    """
    cameras = [str(c) for c in (observation.get("cameras") or [])]
    state_keys = [str(k) for k in (observation.get("robot_state") or [])]
    object_keys = [str(k) for k in (observation.get("object_state") or [])]
    if not cameras and not object_keys:
        raise DatasetError(f"observation {observation.get('name')!r} has neither cameras nor object_state keys: the policy "
                           "would see only the robot state (low_dim needs task.object_state)")
    part = path.with_suffix(path.suffix + ".part")
    path.parent.mkdir(parents=True, exist_ok=True)
    total_steps = 0
    image_shape = None
    env_args = None
    with h5py.File(part, "w") as out:
        data = out.create_group("data")
        open_files: dict[str, h5py.File] = {}
        try:
            for j, c in enumerate(chosen):
                src = open_files.get(c["demo_file"])
                if src is None:
                    src = open_files[c["demo_file"]] = h5py.File(c["demo_file"], "r")
                    if env_args is None:
                        env_args = src["data"].attrs.get("env_args")
                demo = src["data"][c["demo_key"]]
                z = np.load(Path(pool) / CHUNKS_DIR / c["chunk"] / RECORDS_DIR / f"attempt_{c['attempt']:06d}.npz")
                g = data.create_group(f"demo_{j}")
                obs = g.create_group("obs")
                frames_npz = None
                if cameras and layout.images_from_records:
                    frames_path = Path(pool) / CHUNKS_DIR / c["chunk"] / RECORDS_DIR / f"images_{c['attempt']:06d}.npz"
                    if not frames_path.exists():
                        raise DatasetError(f"{c['chunk']}/{c['demo_key']}: no recorded images {frames_path.name}")
                    frames_npz = np.load(frames_path)
                for cam in cameras:
                    key = layout.image_key.format(cam=cam)
                    if frames_npz is not None:
                        if cam not in frames_npz:
                            raise DatasetError(f"{c['chunk']}/{c['demo_key']}: camera {cam} not in the recorded images")
                        arr = np.asarray(frames_npz[cam])
                    else:
                        stored = next((k for k in (key, f"camera_obs/{cam}") if k in demo["obs"]), None)
                        if stored is None:
                            raise DatasetError(f"{c['chunk']}/{c['demo_key']}: camera {cam} not in the pool; generate with "
                                               f"generation.protocol.camera_names set (e.g. generation=test/pilot20_images)")
                        arr = np.asarray(demo["obs"][stored])
                    if len(arr) != c["n_steps"]:
                        raise DatasetError(f"{c['chunk']}/{c['demo_key']}: {len(arr)} image frames vs {c['n_steps']} recorded steps")
                    if observation.get("image_size") and tuple(arr.shape[1:3]) != tuple(int(v) for v in observation["image_size"]):
                        arr = resize_frames(arr, observation["image_size"])     # e.g. Arena's 512x512 head camera -> 128x128
                    image_shape = list(arr.shape[1:])
                    obs.create_dataset(key, data=arr.astype(np.uint8), compression="gzip", compression_opts=1,
                                       chunks=(1, *arr.shape[1:]))
                for group, keys in (("robot_state", state_keys), ("object_state", object_keys)):
                    if group == "object_state" and not keys:
                        continue
                    parts = []
                    for key in keys:
                        if key not in demo["obs"]:
                            raise DatasetError(f"{c['chunk']}/{c['demo_key']}: {group} key {key} not in the pool")
                        arr = np.asarray(demo["obs"][key], dtype=np.float32)
                        if key not in obs:
                            obs.create_dataset(key, data=arr)
                        parts.append(arr.reshape(arr.shape[0], -1))
                    obs.create_dataset(group, data=np.concatenate(parts, axis=1) if parts else np.zeros((c["n_steps"], 0), np.float32))
                actions = np.asarray(demo[layout.actions_key], dtype=np.float32)
                if layout.labels_from_records:
                    lab = np.load(Path(pool) / CHUNKS_DIR / c["chunk"] / RECORDS_DIR / f"labels_{c['attempt']:06d}.npz")
                    a_exec, a_label = np.asarray(lab["actions_executed"], dtype=np.float32), np.asarray(lab["actions_pre_noise"], dtype=np.float32)
                    if meta.get("label") == "pre_noise":
                        actions = a_label                 # the Isaac Lab recorder only stores executed actions
                else:
                    a_exec, a_label = np.asarray(z["a_exec"], dtype=np.float32), np.asarray(z["a_label"], dtype=np.float32)
                g.create_dataset("actions", data=actions)
                if len(a_exec) != len(actions):
                    raise DatasetError(f"{c['chunk']}/{c['demo_key']}: {len(a_exec)} recorded steps vs {len(actions)} actions")
                g.create_dataset("actions_executed", data=a_exec)
                g.create_dataset("actions_pre_noise", data=a_label)
                if layout.states_key and layout.states_key in demo:
                    g.create_dataset("states", data=np.asarray(demo[layout.states_key], dtype=np.float64))
                g.create_dataset("subtask_of_step", data=_subtask_of_step(z))
                g.create_dataset("src_demo_inds", data=np.asarray(c["src_demo_inds"] if c["src_demo_inds"] is not None
                                                                  else demo["src_demo_inds"], dtype=np.int64))
                g.attrs["chunk"], g.attrs["attempt"] = c["chunk"], int(c["attempt"])
                g.attrs["num_samples"], g.attrs["pool"] = int(c["n_steps"]), str(pool)
                total_steps += int(c["n_steps"])
        finally:
            for f in open_files.values():
                f.close()
        data.attrs["env_args"] = env_args if env_args is not None else "{}"
        data.attrs["total"] = total_steps
        data.attrs["cameras"] = json.dumps(cameras)
        data.attrs["robot_state_keys"] = json.dumps(state_keys)
        data.attrs["object_state_keys"] = json.dumps(object_keys)
        for k, v in meta.items():
            data.attrs[k] = v if isinstance(v, (int, float)) else str(v)
        out.create_group("mask").create_dataset("train", data=np.array([f"demo_{j}".encode() for j in range(len(chosen))]))
    part.replace(path)
    return {"demos": len(chosen), "steps": total_steps, "image_shape": image_shape}


def build_dataset(cfg) -> dict:
    """Run ``tmg dataset`` (see the module docstring).

    Args:
        cfg: composed ``dataset`` config (``pool``, ``output``, ``size``, ``seed``, ``overwrite``).

    Returns:
        The sidecar summary written to ``DATASET.json`` next to the file.
    """
    plain = resolved(cfg)
    pool, output = Path(plain["pool"]), Path(plain["output"])
    sidecar = output.with_name(output.stem + "." + SIDECAR)
    if output.exists() and not plain.get("overwrite", False):
        print(f"[tmg] dataset exists, keeping it: {output}")
        return read_json(sidecar) if sidecar.exists() else {"path": str(output)}
    if not (pool / POOL_DONE).exists():
        raise DatasetError(f"pool {pool} is not finished (no {POOL_DONE}); finished chunks would be used only partially")
    from tmg.generation.generators import make_generator

    layout = DemoLayout.from_mapping(make_generator(plain).demo_layout())
    cands = candidates(pool, layout)
    chosen = select(cands, int(plain["size"]), int(plain["seed"]))
    meta = {"task": plain["task"]["name"], "generator": plain["generator"]["name"], "scene": plain["scene"]["name"],
            "method": plain["method"]["name"], "generation": plain["generation"]["name"],
            "observation": plain["observation"]["name"],
            "label": str(plain["method"].get("labels", "executed")), "seed": int(plain["seed"])}
    stats = write_dataset(output, chosen, pool=pool, observation=plain["observation"], meta=meta, layout=layout)
    summary = {"path": str(output), "pool": str(pool), "candidates": len(cands), **stats, **meta,
               "chosen": [{k: c[k] for k in ("chunk", "demo_key", "attempt", "n_steps")} for c in chosen],
               "pool_done": read_json(pool / POOL_DONE)}
    write_json(sidecar, summary)
    dump_yaml(plain, output.with_name(output.stem + ".resolved_config.yaml"))
    write_provenance(output.parent, extra={"command": "dataset", "output": str(output)})
    print(f"[tmg] dataset {output}: {stats['demos']} demos, {stats['steps']} steps, images {stats['image_shape']} "
          f"from {len(cands)} successes")
    return summary
