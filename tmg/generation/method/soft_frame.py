"""The object frame of a deformable object (SoftMimicGen's rope and towel) from its simulation nodes, for SkillMix's scene
and in-hand features.

A deformable object has no rigid pose, only node positions (``datagen_info.object_nodal_positions``: the same material nodes
in the same order in every source demonstration and in the current scene). Its frame (``nodal_frame``) is

- origin: the centroid of the nodes;
- z axis: world up;
- x and y axes: the planar rotation (yaw about world up) that best maps the reference cloud onto the nodes, in the least
  squares sense over corresponding nodes (centred clouds, table-plane coordinates). The reference is source demonstration
  0's node cloud at its first step (``reference_nodes``), so every frame of a task is measured against one cloud.

Fitting the yaw on node correspondences instead of taking the principal axis of the cloud needs no sign convention: a
straight rope's principal axis has two equally good signs, and a U-shaped rope's flips with small changes of its bend. The
frame is deterministic and equivariant: moving the nodes by a yaw R and a shift t moves the frame by the same R and t. A
cloud with no identifiable planar rotation (all nodes on one vertical line, or a reference that does not correlate with
it) raises instead of inventing an angle.

The frame is a feature, not a registration: the segments themselves are still moved with SoftMimicGen's own nodal
registration, and the frame changes nothing in a generation without SkillMix (method none). The same definition was used
by the Rope calibration code of 2026-11-07 on the Isaac server (``nodal_yaw_v1``, ``docs/isaaclab_mixing_provenance.md``).
"""
from __future__ import annotations

import hashlib

import numpy as np

FRAME_VERSION = "nodal_yaw_v1"           # recorded with every frame, so a fit refuses records of another definition


def reference_nodes(infos, object_name: str) -> np.ndarray:
    """The reference cloud of ``object_name``: source demonstration 0's nodes at its first step (N, 3)."""
    return np.asarray(_np(infos[0].object_nodal_positions[object_name][0]), dtype=float)


def reference_hash(reference) -> str:
    """sha256 of the reference cloud (float64 little endian), recorded next to the frames it defines."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(reference, dtype="<f8")).tobytes()).hexdigest()


def nodal_frame(nodes, reference) -> np.ndarray:
    """The 4x4 frame of the module docstring of a node cloud ``nodes`` (N, 3) against ``reference`` (N, 3)."""
    nodes, reference = np.asarray(nodes, dtype=float), np.asarray(reference, dtype=float)
    if nodes.ndim != 2 or nodes.shape[1] != 3 or nodes.shape != reference.shape or len(nodes) < 2:
        raise ValueError(f"a nodal frame needs two (N, 3) clouds of the same nodes, got {nodes.shape} and {reference.shape}")
    if not (np.isfinite(nodes).all() and np.isfinite(reference).all()):
        raise ValueError("a nodal frame needs finite node positions")
    center = nodes.mean(axis=0)
    a = reference[:, :2] - reference[:, :2].mean(axis=0)
    b = nodes[:, :2] - center[:2]
    dot = float((a * b).sum())
    cross = float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum())
    scale = float(np.linalg.norm(a) * np.linalg.norm(b))
    if scale < 1e-12 or np.hypot(dot, cross) <= 1e-8 * scale:
        raise ValueError("the planar rotation of this node cloud is undefined (no spread in the table plane, or no "
                         "correlation with the reference cloud)")
    yaw = np.arctan2(cross, dot)
    c, s = np.cos(yaw), np.sin(yaw)
    frame = np.eye(4)
    frame[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    frame[:3, 3] = center
    return frame


def _np(x) -> np.ndarray:
    """A torch tensor (or anything array-like) as a float numpy array."""
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=float)
