"""Pose arithmetic on 4x4 homogeneous matrices (numpy only)."""
from __future__ import annotations

import numpy as np


def pose_gap(a, b) -> tuple[float, float]:
    """Translation distance (m) and rotation angle (rad) between two 4x4 poses."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    dpos = float(np.linalg.norm(b[:3, 3] - a[:3, 3]))
    rel = a[:3, :3].T @ b[:3, :3]
    cos = (np.trace(rel) - 1.0) / 2.0
    return dpos, float(np.arccos(np.clip(cos, -1.0, 1.0)))


def rotation_angle(ra, rb) -> float:
    """Geodesic angle (rad) between two rotation matrices."""
    cos = (np.trace(np.asarray(ra)[:3, :3].T @ np.asarray(rb)[:3, :3]) - 1.0) / 2.0
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def pose_to_list(pose) -> list[float]:
    """4x4 pose -> [px, py, pz, r00 .. r22] (12 floats) for json records."""
    pose = np.asarray(pose, dtype=float)
    return [float(v) for v in np.concatenate([pose[:3, 3], pose[:3, :3].reshape(-1)])]


def list_to_pose(values) -> np.ndarray:
    """Inverse of :func:`pose_to_list`: 12 floats back to a 4x4 pose."""
    values = np.asarray(values, dtype=float).reshape(-1)
    out = np.eye(4)
    out[:3, 3] = values[:3]
    out[:3, :3] = values[3:12].reshape(3, 3)
    return out


def invert(pose) -> np.ndarray:
    """Inverse of a rigid 4x4 pose (transpose of the rotation, negated rotated translation)."""
    pose = np.asarray(pose, dtype=float)
    out = np.eye(4)
    rt = pose[:3, :3].T
    out[:3, :3] = rt
    out[:3, 3] = -rt @ pose[:3, 3]
    return out


def yaw_of(pose) -> float:
    """Yaw angle (rad) of a pose about the world z axis, ``atan2(r10, r00)``."""
    pose = np.asarray(pose, dtype=float)
    return float(np.arctan2(pose[1, 0], pose[0, 0]))
