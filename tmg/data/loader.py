"""In-memory frame dataset shared by every policy backend.

The common hdf5 (``tmg.data.dataset``) is loaded once into RAM (``TrajectoryStore``): images stay uint8
(T, H, W, 3) per demo, robot state and actions float32. ``FrameDataset`` indexes every (demo, step) pair
and builds one LeRobot-style item per step:

    observation.images.<camera>   (n_obs, 3, H, W) uint8        frames at ``obs_delta`` offsets, edge-padded
    observation.state             (n_obs, D) float32            the robot state
    observation.environment_state (n_obs, E) float32            the object state; only in low-dim datasets
    action                        (horizon, A) float32          actions at ``action_delta`` offsets, edge-padded
    action_is_pad                 (horizon,) bool               True where the offset fell outside the demo
    index, episode_index, frame_index                           ints (for logging)
    task                          str                           the task instruction, when the dataset was given one (π0)

Offsets follow LeRobot's ``observation_delta_indices`` / ``action_delta_indices`` convention (0 is the
current step; negative offsets look back). Padding repeats the first or last frame of the demo, as
LeRobot's dataset does. Images are converted to float in [0, 1] by the training loop, not here, so the
whole dataset fits in memory as uint8 (1,000 demos x 200 steps x 2 cameras x 84x84x3 is about 8.5 GB).

``collate`` stacks items into torch tensors and ``to_device`` moves a batch; ``compute_stats`` gives the
per-key statistics the LeRobot normalizers expect (``min``, ``max``, ``mean``, ``std``; images per channel on
the [0, 1] scale with shape (3, 1, 1)). numpy only; torch is imported lazily so the module is testable
without it.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np


@dataclass
class TrajectoryStore:
    """Every demo of a common hdf5 held in RAM."""

    images: dict[str, list[np.ndarray]]          # camera -> per-demo (T, H, W, 3) uint8
    state: list[np.ndarray]                      # per-demo (T, D) float32
    actions: list[np.ndarray]                    # per-demo (T, A) float32
    lengths: np.ndarray                          # (N,) steps per demo
    cameras: list[str]
    attrs: dict = field(default_factory=dict)
    env_state: list[np.ndarray] | None = None    # per-demo (T, E) float32 object state (low-dim datasets), else None

    @classmethod
    def load(cls, path: Path, *, cameras: Sequence[str] | None = None, action_key: str = "actions",
             demos: Iterable[str] | None = None) -> TrajectoryStore:
        """Read a dataset file.

        Args:
            path: the common hdf5.
            cameras: cameras to load (default: all listed in ``data.attrs['cameras']``).
            action_key: ``actions`` (the method's label), ``actions_executed`` or ``actions_pre_noise``.
            demos: demo keys to load (default: all, in index order).
        """
        images: dict[str, list[np.ndarray]] = {}
        state, actions, lengths = [], [], []
        env_state: list[np.ndarray] | None = None
        with h5py.File(path, "r") as f:
            data = f["data"]
            attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in data.attrs.items()}
            cams = list(cameras) if cameras is not None else list(json.loads(attrs.get("cameras", "[]")))
            keys = list(demos) if demos is not None else sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
            if json.loads(attrs.get("object_state_keys", "[]")):
                env_state = []
            for cam in cams:
                images[cam] = []
            for key in keys:
                g = data[key]
                for cam in cams:
                    images[cam].append(np.asarray(g["obs"][f"{cam}_image"], dtype=np.uint8))
                state.append(np.asarray(g["obs"]["robot_state"], dtype=np.float32))
                if env_state is not None:
                    env_state.append(np.asarray(g["obs"]["object_state"], dtype=np.float32))
                actions.append(np.asarray(g[action_key], dtype=np.float32))
                lengths.append(len(actions[-1]))
        return cls(images=images, state=state, actions=actions, lengths=np.asarray(lengths, dtype=np.int64),
                   cameras=cams, attrs=attrs, env_state=env_state)

    @property
    def num_demos(self) -> int:
        return len(self.lengths)

    @property
    def num_frames(self) -> int:
        return int(self.lengths.sum())

    @property
    def state_dim(self) -> int:
        return int(self.state[0].shape[1]) if self.state else 0

    @property
    def action_dim(self) -> int:
        return int(self.actions[0].shape[1]) if self.actions else 0

    @property
    def env_state_dim(self) -> int:
        """Width of the object state, 0 when the dataset has none (image datasets)."""
        return int(self.env_state[0].shape[1]) if self.env_state else 0

    @property
    def image_shape(self) -> tuple[int, int, int] | None:
        """(H, W, 3) of the first camera, or None without cameras."""
        if not self.cameras:
            return None
        return tuple(int(v) for v in self.images[self.cameras[0]][0].shape[1:])


def padded_indices(t: int, length: int, deltas: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Step indices ``t + delta`` clipped into ``[0, length)`` and a mask of the clipped (padded) ones."""
    raw = t + np.asarray(deltas, dtype=np.int64)
    clipped = np.clip(raw, 0, length - 1)
    return clipped, raw != clipped


class FrameDataset:
    """One item per (demo, step) with temporal stacking by delta offsets (see the module docstring)."""

    def __init__(self, store: TrajectoryStore, *, obs_delta: Sequence[int] | None = (0,), action_delta: Sequence[int] = (0,),
                 drop_n_last_frames: int = 0, task: str | None = None):
        """Args:
            store: the loaded trajectories.
            obs_delta: offsets of the observation frames relative to the current step (LeRobot convention); None
                means a single frame without a time axis (policies whose config has no observation deltas, e.g. π0).
            action_delta: offsets of the action targets.
            drop_n_last_frames: do not start items on the last n steps of a demo (LeRobot's sampler option).
            task: language instruction attached to every item as ``task`` (π0); None adds no key.
        """
        self.store = store
        self.obs_delta = None if obs_delta is None else tuple(int(d) for d in obs_delta)
        self.task = task
        self.action_delta = tuple(int(d) for d in action_delta)
        starts = np.concatenate([[0], np.cumsum(store.lengths)[:-1]]) if store.num_demos else np.zeros(0, np.int64)
        self._demo_of = np.concatenate([np.full(max(n - drop_n_last_frames, 0), i, dtype=np.int64)
                                        for i, n in enumerate(store.lengths)]) if store.num_demos else np.zeros(0, np.int64)
        self._step_of = np.concatenate([np.arange(max(n - drop_n_last_frames, 0), dtype=np.int64)
                                        for n in store.lengths]) if store.num_demos else np.zeros(0, np.int64)
        self._global_index = starts[self._demo_of] + self._step_of if len(self._demo_of) else self._demo_of

    def __len__(self) -> int:
        return len(self._demo_of)

    def __getitem__(self, i: int) -> dict:
        """The item of flat index ``i`` as numpy arrays (keys in the module docstring)."""
        d, t = int(self._demo_of[i]), int(self._step_of[i])
        n = int(self.store.lengths[d])
        obs_idx, _ = padded_indices(t, n, self.obs_delta if self.obs_delta is not None else (0,))
        act_idx, act_pad = padded_indices(t, n, self.action_delta)
        item = {f"observation.images.{cam}": np.ascontiguousarray(self.store.images[cam][d][obs_idx].transpose(0, 3, 1, 2))
                for cam in self.store.cameras}
        item["observation.state"] = self.store.state[d][obs_idx]
        if self.store.env_state is not None:
            item["observation.environment_state"] = self.store.env_state[d][obs_idx]
        if self.obs_delta is None:                       # single frame: drop the time axis
            item = {k: v[0] for k, v in item.items()}
        if self.task is not None:
            item["task"] = self.task
        item["action"] = self.store.actions[d][act_idx]
        item["action_is_pad"] = act_pad
        item["index"], item["episode_index"], item["frame_index"] = int(self._global_index[i]), d, t
        return item

    def batches(self, batch_size: int, rng: np.random.Generator):
        """Endless generator of index arrays: uniform shuffles of the whole index, cut into batches."""
        n = len(self)
        while True:
            perm = rng.permutation(n)
            for s in range(0, n - batch_size + 1, batch_size):
                yield perm[s:s + batch_size]


def collate(items: Sequence[dict]):
    """Stack items into a batch of torch tensors (images uint8, floats float32, pads bool, ints int64)."""
    import torch

    out = {}
    for key in items[0]:
        vals = [it[key] for it in items]
        if isinstance(vals[0], str):
            out[key] = list(vals)
        elif isinstance(vals[0], (int, np.integer)):
            out[key] = torch.as_tensor(np.asarray(vals, dtype=np.int64))
        else:
            out[key] = torch.from_numpy(np.stack(vals))
    return out


def to_device(batch: dict, device, *, images_to_float: bool = True) -> dict:
    """Move a collated batch to ``device``; uint8 images become float32 in [0, 1] when requested."""
    import torch

    out = {}
    for key, v in batch.items():
        if not isinstance(v, torch.Tensor):
            out[key] = v
            continue
        v = v.to(device, non_blocking=True)
        if images_to_float and key.startswith("observation.images.") and v.dtype == torch.uint8:
            v = v.float() / 255.0
        out[key] = v
    return out


def compute_stats(store: TrajectoryStore, *, max_image_frames: int = 20000, seed: int = 0) -> dict:
    """Per-key statistics in the format LeRobot's normalizers read.

    Args:
        store: the loaded trajectories.
        max_image_frames: images are sampled (uniformly over all frames) up to this many per camera.
        seed: sampling seed.

    Returns:
        ``{key: {"min", "max", "mean", "std"}}`` as float32 arrays; vector keys have shape (D,), images (3, 1, 1)
        on the [0, 1] scale.
    """
    def vec_stats(arr: np.ndarray) -> dict:
        return {"min": arr.min(0).astype(np.float32), "max": arr.max(0).astype(np.float32),
                "mean": arr.mean(0).astype(np.float32), "std": arr.std(0).astype(np.float32)}

    stats = {"observation.state": vec_stats(np.concatenate(store.state)), "action": vec_stats(np.concatenate(store.actions))}
    if store.env_state is not None:
        stats["observation.environment_state"] = vec_stats(np.concatenate(store.env_state))
    rng = np.random.default_rng(seed)
    for cam in store.cameras:
        frames = np.concatenate(store.images[cam])
        if len(frames) > max_image_frames:
            frames = frames[np.sort(rng.choice(len(frames), size=max_image_frames, replace=False))]
        x = frames.astype(np.float32) / 255.0
        per_channel = {"min": x.min(axis=(0, 1, 2)), "max": x.max(axis=(0, 1, 2)), "mean": x.mean(axis=(0, 1, 2)),
                       "std": x.std(axis=(0, 1, 2))}
        stats[f"observation.images.{cam}"] = {k: v.astype(np.float32).reshape(3, 1, 1) for k, v in per_channel.items()}
    return stats
