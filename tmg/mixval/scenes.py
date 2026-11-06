"""Stored scenes, their fixed perturbations and the per-instance seed, shared by both generators' runtimes.

A scene instance is (scene, perturbation). Perturbation 0 is the stored state; perturbation p >= 1 moves every free
object (the task objects) by dx, dy uniform in [-1 cm, 1 cm] and a rotation dyaw uniform in [-5 deg, 5 deg] about the
world z axis, drawn from ``SeedSequence([PERT_SEED, scene, p])``, so the offsets are the same in every condition and
every process. ``instance_seed`` seeds Python's ``random`` and numpy (and torch when present) right after the scene is
set, so every condition of one instance sees the same random draws until the trajectories differ.
"""
from __future__ import annotations

import random

import h5py
import numpy as np

PERT_SEED = 20261106
PERT_POS_M = 0.01
PERT_YAW_DEG = 5.0
MJ_FREE = 0                # mjJNT_FREE


def instance_seed(scene: int, pert: int) -> int:
    return int(np.random.SeedSequence([PERT_SEED, int(scene), int(pert), 1]).generate_state(1)[0])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def perturbation(scene: int, pert: int, n_objects: int) -> np.ndarray:
    """``(n_objects, 3)`` offsets (dx m, dy m, dyaw deg) of perturbation ``pert`` of ``scene``; zeros for ``pert = 0``."""
    if pert == 0:
        return np.zeros((n_objects, 3))
    rng = np.random.default_rng(np.random.SeedSequence([PERT_SEED, int(scene), int(pert)]))
    out = np.empty((n_objects, 3))
    out[:, :2] = rng.uniform(-PERT_POS_M, PERT_POS_M, size=(n_objects, 2))
    out[:, 2] = rng.uniform(-PERT_YAW_DEG, PERT_YAW_DEG, size=n_objects)
    return out


def free_joints(sim) -> list[tuple[str, int]]:
    """(name, qpos address) of every free joint in model order."""
    model = sim.model
    return [(model.joint_id2name(j), int(model.jnt_qposadr[j])) for j in range(model.njnt) if int(model.jnt_type[j]) == MJ_FREE]


def _quat_mul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def apply_perturbation(sim, offsets: np.ndarray) -> list[dict]:
    """Move each free object by its (dx, dy, dyaw) offset (rotation about the world z axis through the object's
    origin) and run ``forward``; returns what was applied."""
    applied = []
    qpos = sim.data.qpos
    for (name, adr), (dx, dy, dyaw) in zip(free_joints(sim), offsets):
        qpos[adr] += dx
        qpos[adr + 1] += dy
        half = np.radians(dyaw) / 2
        q = _quat_mul(np.array([np.cos(half), 0.0, 0.0, np.sin(half)]), np.asarray(qpos[adr + 3:adr + 7], dtype=float))
        qpos[adr + 3:adr + 7] = q / np.linalg.norm(q)
        applied.append({"joint": name, "dx": round(float(dx), 5), "dy": round(float(dy), 5), "dyaw_deg": round(float(dyaw), 3)})
    sim.forward()
    return applied


def load_scenes(path: str, indices) -> dict[int, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {int(i): np.asarray(f[f"reset_{int(i)}"]["states"], dtype=float) for i in indices}


def check_state_size(state: np.ndarray, sim) -> None:
    n = sim.get_state().flatten().shape[0]
    if state.shape[0] != n:
        raise RuntimeError(f"stored scene has {state.shape[0]} state values, the simulator {n}")
