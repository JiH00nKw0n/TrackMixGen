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
frame is deterministic and equivariant: moving the nodes by a yaw R and a shift t moves the frame by the same R and t.

When the yaw means nothing. With the centred table-plane coordinates as complex numbers a_i (reference) and b_i (nodes),
the yaw is the angle of z = sum_i conj(a_i) b_i and the rotation-fit correlation r = |z| / (|a| |b|) (1 for a rigid
motion, 0 when no rotation relates the clouds). A cloud with r < ``MIN_ROTATION_FIT`` = 0.5 raises instead of getting an
angle. The value comes from the geometry: if the nodes fall into two equal groups turned by +phi and -phi from the
reference, r = cos(phi) and the yaw is their average; r = 0.5 is phi = 60 degrees, groups that disagree by 120 degrees,
past which the "yaw" averages motions that point in opposite directions and a few nodes decide its sign. The nearly
isotropic clouds a principal axis cannot orient are not a problem here (the yaw comes from corresponding nodes, so a flat
square towel turned by any angle has r = 1). Measured r of synthetic shapes against a straight or flat reference
(2026-11-27): rope bent into a half circle 0.91, a nearly closed loop 0.64; towel with one corner pulled in 1.00, folded
in half 0.63 (the unfolded half fixes the yaw); a rope bent so far that its sagitta is twice its length 0.43 (refused).
The frames are taken at subtask starts, before the rope is shaped or the towel folded. All nodes on one vertical line
(no spread in the table plane) also raise.

In-hand pose of a held rope or towel: the frame in the end-effector frame, so its position is the offset from the hand to
the node centroid, about 0.3 m for the rope held at one end (Rope pilot, 2026-11-27), not the distance to the grasped
node. That is the intended definition and the one of the 2026-11-07 code (``soft_pose.py``: ``_hand_in_object`` of the
same centroid frame): the in-hand features compare this offset between the current scene and a source, and both see the
same centroid. Whether the object is held at all is decided separately, from the nearest node
(``isaaclab.runtime.held_distance``).

The frame is a feature, not a registration: the segments themselves are still moved with SoftMimicGen's own nodal
registration, and the frame changes nothing in a generation without SkillMix (method none). The same definition was used
by the Rope calibration code of 2026-11-07 on the Isaac server (``nodal_yaw_v1``, ``docs/isaaclab_mixing_provenance.md``).
"""
from __future__ import annotations

import hashlib

import numpy as np

FRAME_VERSION = "nodal_yaw_v1"           # recorded with every frame, so a fit refuses records of another definition
MIN_ROTATION_FIT = 0.5                   # least rotation-fit correlation r of a cloud whose yaw is used (module docstring)


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
    if scale < 1e-12:
        raise ValueError("the planar rotation of this node cloud is undefined (no spread in the table plane)")
    fit = float(np.hypot(dot, cross)) / scale
    if fit < MIN_ROTATION_FIT:
        raise ValueError(f"the planar rotation of this node cloud is undefined: rotation-fit correlation {fit:.2f} < "
                         f"{MIN_ROTATION_FIT} with the reference cloud (soft_frame module docstring)")
    yaw = np.arctan2(cross, dot)
    c, s = np.cos(yaw), np.sin(yaw)
    frame = np.eye(4)
    frame[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    frame[:3, 3] = center
    return frame


def _np(x) -> np.ndarray:
    """A torch tensor (or anything array-like) as a float numpy array."""
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=float)
