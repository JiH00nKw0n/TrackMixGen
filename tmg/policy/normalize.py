"""Per-key normalization of batches, applied outside the policies (as LeRobot's processors do).

Modes per key, chosen by the policy config's ``normalization`` mapping (keys ``VISUAL``, ``STATE``, ``ENV`` for the
object state ``observation.environment_state`` (defaults to the ``STATE`` mode), ``ACTION``):

- ``identity``: pass through (BC-RNN as in robomimic: images in [0, 1], state raw, actions already in [-1, 1]; actions
  outside [-1, 1], such as the Dex tasks' absolute poses, switch to ``min_max`` with one global range, see
  ``global_min_max``);
- ``mean_std``: ``(x - mean) / (std + eps)``;
- ``min_max``: ``2 (x - min) / (max - min) - 1`` to [-1, 1] (Diffusion Policy's state and action mode).

Statistics come from ``tmg.data.loader.compute_stats`` and are saved as ``stats.json`` with every checkpoint,
so the server normalizes observations and unnormalizes actions exactly as the training did.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tmg.core.features import ACTION, ENV_STATE, OBSERVATION_PREFIX, is_image_key

EPS = 1e-8
MODES = ("identity", "mean_std", "min_max")


def feature_type(key: str) -> str:
    """``VISUAL`` for images, ``ACTION`` for ``action``, ``ENV`` for the object state, ``STATE`` otherwise."""
    if is_image_key(key):
        return "VISUAL"
    if key == ACTION:
        return "ACTION"
    if key == ENV_STATE:
        return "ENV"
    return "STATE"


def exceeds_unit_range(stats: dict, key: str) -> bool:
    """Whether ``stats[key]`` reaches outside [-1, 1], where tanh-bounded outputs (BC-RNN's GMM means) cannot go."""
    s = stats[key]
    return bool(np.min(s["min"]) < -1.0 or np.max(s["max"]) > 1.0)


def global_min_max(stats: dict, key: str) -> dict:
    """Make the ``min_max`` normalization of ``stats[key]`` use one minimum and one maximum over all dimensions, as
    Isaac Lab's ``scripts/imitation_learning/robomimic/train.py --normalize_training_actions`` scales the actions of its
    GR1 humanoid datasets to [-1, 1] (``normalize_hdf5_actions``: the min and max of all action values)."""
    s = stats[key]
    lo, hi = float(np.min(s["min"])), float(np.max(s["max"]))
    s["min"] = np.full_like(np.asarray(s["min"], dtype=np.float32), lo)
    s["max"] = np.full_like(np.asarray(s["max"], dtype=np.float32), hi)
    return stats


class Normalizer:
    """Normalize observation keys and the action with per-key modes; ``unnormalize_action`` inverts the action."""

    def __init__(self, stats: dict, modes: dict[str, str]):
        """Args:
            stats: ``{key: {"min", "max", "mean", "std"}}`` numpy arrays (images (3, 1, 1), vectors (D,)).
            modes: ``{"VISUAL": mode, "STATE": mode, "ACTION": mode}`` with modes from ``MODES``.
        """
        for t, m in modes.items():
            if m not in MODES:
                raise ValueError(f"normalization mode {m!r} for {t}; expected one of {MODES}")
        self.stats = {k: {s: np.asarray(v, dtype=np.float32) for s, v in d.items()} for k, d in stats.items()}
        self.modes = {"VISUAL": "identity", "STATE": "identity", "ACTION": "identity", **modes}
        self.modes.setdefault("ENV", self.modes["STATE"])

    def mode_of(self, key: str) -> str:
        return self.modes[feature_type(key)]

    def _apply(self, key: str, x, inverse: bool):
        mode = self.mode_of(key)
        if mode == "identity" or key not in self.stats:
            return x
        import torch

        s = {k: torch.as_tensor(v, device=x.device, dtype=x.dtype) for k, v in self.stats[key].items()}
        if mode == "mean_std":
            return x * (s["std"] + EPS) + s["mean"] if inverse else (x - s["mean"]) / (s["std"] + EPS)
        span = s["max"] - s["min"] + EPS
        return (x + 1) / 2 * span + s["min"] if inverse else 2 * (x - s["min"]) / span - 1

    def __call__(self, batch: dict) -> dict:
        """Normalized copy of a batch (observation keys and ``action`` when present)."""
        out = {}
        for key, v in batch.items():
            normalized = (key.startswith(OBSERVATION_PREFIX) or key == ACTION) and not key.endswith("_is_pad")
            out[key] = self._apply(key, v, inverse=False) if normalized else v
        return out

    def unnormalize_action(self, action):
        return self._apply(ACTION, action, inverse=True)

    def save(self, path: Path) -> Path:
        """Write ``{"modes": ..., "stats": {key: {name: list}}}`` as JSON."""
        data = {"modes": self.modes, "stats": {k: {s: v.tolist() for s, v in d.items()} for k, d in self.stats.items()}}
        Path(path).write_text(json.dumps(data, indent=1))
        return Path(path)

    @classmethod
    def load(cls, path: Path) -> Normalizer:
        data = json.loads(Path(path).read_text())
        return cls(data["stats"], data["modes"])
