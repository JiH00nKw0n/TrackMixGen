"""Rotation encoding of absolute end-effector pose actions (CP-Gen), as the authors' training does.

CP-Gen's actions are absolute poses of 7 numbers: position (3), rotation as an axis-angle vector (3, the rotation axis
scaled by the angle) and gripper (1). The axis-angle vector is not continuous: near an angle of pi the same orientation
has two vectors of opposite direction, and CP-Gen's own data cross pi (Three Piece Assembly: norms 2.39-3.57 rad, 69
steps where the vector jumps by more than 1 in one step). The authors' training (adaflow-cpgen, diffusion_policy's
``RobomimicReplayImageDataset`` with ``abs_action: True``) turns the rotation into the 6D representation, the first
two rows of the rotation matrix (pytorch3d ``matrix_to_rotation_6d``), and normalizes only the position (min-max to
[-1, 1]); the 6D rotation and the gripper pass unnormalized (``normalize_util``: identity for those dimensions).

- ``encode``: dataset actions (..., 7) to (..., 10) before the statistics are computed;
- ``identity_dims``: sets the statistics of the rotation and gripper dimensions so that our min-max and mean/std
  normalizers leave them unchanged (min -1, max 1, mean 0, std 1);
- ``decode``: a predicted (..., 10) action back to the environment's (..., 7) (Gram-Schmidt on the two rows, as
  pytorch3d ``rotation_6d_to_matrix``, then the axis-angle vector of that matrix).

The encoding is the generator config's ``action_rotation`` (``axis_angle``: unchanged, ``rotation_6d``); a checkpoint
trained with ``rotation_6d`` holds ``action_rotation.json`` and the policy server decodes its actions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

AXIS_ANGLE = "axis_angle"
ROTATION_6D = "rotation_6d"
ENCODINGS = (AXIS_ANGLE, ROTATION_6D)
SAVED = "action_rotation.json"


def check(encoding: str) -> str:
    if encoding not in ENCODINGS:
        raise ValueError(f"action_rotation {encoding!r}; expected one of {ENCODINGS}")
    return encoding


def encode(actions: np.ndarray) -> np.ndarray:
    """``(..., 7)`` position, axis-angle, gripper to ``(..., 10)`` position, 6D rotation (matrix rows 0 and 1), gripper."""
    from scipy.spatial.transform import Rotation

    a = np.asarray(actions, dtype=np.float64)
    if a.shape[-1] != 7:
        raise ValueError(f"rotation_6d expects 7-dim absolute pose actions, got {a.shape[-1]}")
    mat = Rotation.from_rotvec(a[..., 3:6].reshape(-1, 3)).as_matrix()
    r6 = mat[:, :2, :].reshape(*a.shape[:-1], 6)
    return np.concatenate([a[..., :3], r6, a[..., 6:]], axis=-1).astype(np.float32)


def decode(actions: np.ndarray) -> np.ndarray:
    """``(..., 10)`` back to ``(..., 7)``: the two rows are orthonormalized (Gram-Schmidt) into a rotation matrix."""
    from scipy.spatial.transform import Rotation

    a = np.asarray(actions, dtype=np.float64)
    if a.shape[-1] != 10:
        raise ValueError(f"rotation_6d decoding expects 10-dim actions, got {a.shape[-1]}")
    a1, a2 = a[..., 3:6], a[..., 6:9]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    mat = np.stack([b1, b2, np.cross(b1, b2)], axis=-2)
    rotvec = Rotation.from_matrix(mat.reshape(-1, 3, 3)).as_rotvec().reshape(*a.shape[:-1], 3)
    return np.concatenate([a[..., :3], rotvec, a[..., 9:]], axis=-1).astype(np.float32)


def identity_dims(stats: dict, key: str) -> dict:
    """Make the normalization of ``stats[key]`` an identity on the rotation and gripper dimensions (3 onwards)."""
    s = stats[key]
    for name, value in (("min", -1.0), ("max", 1.0), ("mean", 0.0), ("std", 1.0)):
        if name in s:
            arr = np.array(s[name], dtype=np.float32, copy=True)
            arr[3:] = value
            s[name] = arr
    return stats


def save(folder: Path, encoding: str) -> None:
    """Record a non-default encoding in a checkpoint folder."""
    if check(encoding) != AXIS_ANGLE:
        (Path(folder) / SAVED).write_text(json.dumps({"action_rotation": encoding}))


def load(folder: Path) -> str:
    """The encoding a checkpoint was trained with (``axis_angle`` when the file is absent)."""
    path = Path(folder) / SAVED
    return check(json.loads(path.read_text())["action_rotation"]) if path.exists() else AXIS_ANGLE
