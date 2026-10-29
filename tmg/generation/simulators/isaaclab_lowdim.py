"""Low-dimensional object observations for the Isaac Lab tasks (decision 2026-10-29).

MimicGen and CP-Gen train their low-dim policies on robosuite's ``object`` observation: for every task object its
position (3) and orientation quaternion (4), the joint angles of articulated objects, and the vector from each
end-effector to the object. The Isaac Lab tasks record no such term (Pick Place excepted: its ``object`` term is exactly
pos, quat, left eef to object, right eef to object), so this module provides observation-term functions of the same
level and :func:`add_lowdim_terms`, which writes them into an environment config's ``policy`` observation group before
the environment is built. The generation entrypoint and the evaluator both call it, so the generated files record the
term (Isaac Lab's recorder stores every term of the ``policy`` group) and the evaluation computes the same values from
the live environment. ``task.object_state`` names the terms ``tmg dataset`` concatenates into
``observation.environment_state``; ``task.lowdim_terms`` says how to build them::

    lowdim_terms:
      rope_state:
        func: tmg.generation.simulators.isaaclab_lowdim:deformable_nodes
        params: {object: object, layout: line, counts: [16], relative: ends, eef: {sensor: ee_frame}}

Terms:

- :func:`deformable_nodes`: positions of a fixed subset of a deformable object's mesh nodes (a rope has 549 nodes, a
  towel 5,989; a policy input needs a small fixed set) plus the vectors from the end-effector to the two ends (rope) or to
  the centroid (towel). The subset is chosen once per environment from the object's rest shape (``default_nodal_state_w``,
  the mesh at its spawn pose): the principal axes of the rest shape define a line (rope) or a plane (towel), ``counts``
  evenly spaced target coordinates are laid along each axis between the shape's extremes, and the node nearest each
  target is taken. The axis signs are fixed by making each axis's largest component positive, so the order of the
  selected nodes is the same in every process. Rope: 16 nodes (48) + eef to both ends (6) = 54. Towel: 6x6 grid
  (108) + eef to centroid (3) = 111.
- :func:`scene_object_state`: for each listed rigid or articulated object its position (3, environment frame), quaternion
  (4), the positions of the listed joints and the vector from each end-effector to the object. Microwave: pose (7) +
  door joint ``microjoint`` (1) + two hands (6) = 14. Bottle door: bottle pose (7) + hands (6), refrigerator pose (7) +
  ``fridge_door_joint`` (1) + hands (6) = 27.

Isaac Lab 3.0 wraps asset data fields (``root_pos_w.torch``); :func:`_t` reads both forms. Positions are expressed in the
environment frame (world position minus ``env.scene.env_origins``), as robosuite's are for a single environment.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

_NODE_CACHE: dict[int, object] = {}       # id(asset) -> LongTensor of selected node indices


def _t(x):
    """The tensor of an Isaac Lab data field (3.0 wraps them in an object with a ``torch`` attribute)."""
    return x.torch if hasattr(x, "torch") else x


def _origins(env):
    return _t(env.scene.env_origins)


def eef_positions(env, eef: Mapping | None) -> list:
    """End-effector positions in the environment frame, one (N, 3) tensor per end-effector.

    ``eef`` is ``{"sensor": name}`` (a FrameTransformer sensor, target 0; the SoftMimicGen tasks' ``ee_frame``) or
    ``{"links": [body names], "robot": name}`` (robot bodies; the GR1 tasks' ``*_hand_roll_link``). None gives no vectors.
    """
    if not eef:
        return []
    origins = _origins(env)
    if "sensor" in eef:
        sensor = env.scene[str(eef["sensor"])]
        return [_t(sensor.data.target_pos_w)[:, 0, :] - origins]
    robot = env.scene[str(eef.get("robot", "robot"))]
    names = list(robot.data.body_names)
    body_pos = _t(robot.data.body_pos_w)
    return [body_pos[:, names.index(str(link))] - origins for link in eef["links"]]


def select_nodes(rest_positions, layout: str, counts: Sequence[int]) -> np.ndarray:
    """Indices of the nodes nearest to an evenly spaced line (``layout='line'``, one count) or grid (``'grid'``, two
    counts) laid along the principal axes of the rest shape ``rest_positions`` (V, 3). The axes' signs are fixed by
    making each axis's largest component positive; nodes are returned in target order (along the first axis, then the
    second)."""
    counts = [int(c) for c in counts]
    k = {"line": 1, "grid": 2}.get(str(layout))
    if k is None or len(counts) != k:
        raise ValueError(f"layout {layout!r} needs {'one count' if layout == 'line' else 'two counts'}, got {counts}")
    pts = np.asarray(rest_positions, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < max(counts):
        raise ValueError(f"rest shape must be (V, 3) with V >= max(counts), got {pts.shape} for counts {counts}")
    centered = pts - pts.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    axes = vh[:k]
    signs = np.sign(axes[np.arange(k), np.abs(axes).argmax(axis=1)])
    axes = axes * signs[:, None]
    proj = centered @ axes.T                                                     # (V, k)
    grids = [np.linspace(proj[:, i].min(), proj[:, i].max(), counts[i]) for i in range(k)]
    targets = np.stack(np.meshgrid(*grids, indexing="ij"), axis=-1).reshape(-1, k)   # first axis outermost
    dist = ((targets[:, None, :] - proj[None, :, :]) ** 2).sum(axis=-1)          # (T, V)
    return dist.argmin(axis=1)


def deformable_nodes(env, object: str = "object", layout: str = "line", counts: Sequence[int] = (16,),
                     relative: str | None = "ends", eef: Mapping | None = None):
    """Positions of a fixed node subset of the deformable object ``object`` (see the module docstring), flattened, followed
    by the vectors from every end-effector of ``eef`` to the subset's two end nodes (``relative='ends'``) or to the
    object's centroid (``'centroid'``); ``relative=None`` adds none."""
    import torch

    asset = env.scene[str(object)]
    nodal = _t(asset.data.nodal_pos_w)                                           # (N, V, 3) world frame
    indices = _NODE_CACHE.get(id(asset))
    if indices is None:
        rest = _t(asset.data.default_nodal_state_w)[0, :, :3]
        chosen = select_nodes(np.asarray(rest.detach().cpu().numpy() if hasattr(rest, "detach") else rest), layout, counts)
        indices = _NODE_CACHE[id(asset)] = torch.as_tensor(chosen, dtype=torch.long, device=nodal.device)
    origins = _origins(env)
    picked = nodal[:, indices, :] - origins[:, None, :]                         # (N, k, 3)
    parts = [picked.reshape(picked.shape[0], -1)]
    if relative == "ends":
        anchors = [picked[:, 0, :], picked[:, -1, :]]
    elif relative == "centroid":
        anchors = [nodal.mean(dim=1) - origins]
    elif relative in (None, "none"):
        anchors = []
    else:
        raise ValueError(f"relative must be 'ends', 'centroid' or None, got {relative!r}")
    for eef_pos in eef_positions(env, eef):
        parts.extend(anchor - eef_pos for anchor in anchors)
    return torch.cat(parts, dim=1).to(torch.float32)


def scene_object_state(env, objects: Sequence[Mapping], eef: Mapping | None = None):
    """For every entry of ``objects`` (``{"name": scene entity, "joints": [joint names] or absent}``): position (3,
    environment frame), quaternion (4), the listed joint positions, then the vector from each end-effector of ``eef`` to
    the object (the layout of robosuite's object-state and of Isaac Lab's Pick Place ``object`` term)."""
    import torch

    eefs = eef_positions(env, eef)
    origins = _origins(env)
    parts = []
    for entry in objects:
        asset = env.scene[str(entry["name"])]
        pos = _t(asset.data.root_pos_w) - origins
        parts.extend([pos, _t(asset.data.root_quat_w)])
        joints = list(entry.get("joints") or [])
        if joints:
            names = list(asset.data.joint_names)
            parts.append(_t(asset.data.joint_pos)[:, [names.index(str(j)) for j in joints]])
        parts.extend(pos - eef_pos for eef_pos in eefs)
    return torch.cat(parts, dim=1).to(torch.float32)


def add_lowdim_terms(env_cfg, task: Mapping) -> dict:
    """Write the task's ``lowdim_terms`` into ``env_cfg.observations.policy`` (new terms only; an existing term of the
    same name is an error). Returns ``{term name: function path}``; empty when the task lists none."""
    terms = dict(task.get("lowdim_terms") or {})
    if not terms:
        return {}
    from isaaclab.managers import ObservationTermCfg

    from tmg.generation.simulators.isaaclab import _import_attr

    group = env_cfg.observations.policy
    added = {}
    for name, spec in terms.items():
        if getattr(group, name, None) is not None:
            raise KeyError(f"the environment config already has an observation term {name!r}")
        params = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(spec.get("params") or {}).items()}
        setattr(group, str(name), ObservationTermCfg(func=_import_attr(str(spec["func"])), params=params))
        added[str(name)] = str(spec["func"])
    return added
