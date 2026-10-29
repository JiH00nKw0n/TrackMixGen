"""robosuite side of the generator adapters: scene ranges applied to MimicGen's environment classes.

A scene is pure data: ``configs/experiment/generator/<generator>/scene/<name>.yaml`` holds the generator's placement
ranges for the condition ``name`` (``official_d1`` or ``ours``; every generator defines both). ``register_env_class`` turns it into a robosuite
environment class at runtime: a subclass of MimicGen's public ``<env_prefix>_D1`` class whose
``_get_initial_placement_bounds`` returns the configured ranges and, with a reach constraint, whose
``reset`` redraws the scene until every task object lies within ``radius_m`` of the robot base. The class is
named ``<env_prefix>_<scene.env_suffix>``; robosuite registers classes by name on creation, so this must run
in every process that builds the environment. The ``official_d1`` scene registers nothing.

Scene configs may name a ``parent`` scene; ranges are inherited leaf-first. The parent chain is resolved
from the generator's ``scene`` folder by ``load_scene_chain`` (the file reading is shared with the Isaac Lab
layer in ``tmg.generation.simulators.scenes``; this module adds the robosuite validation).
"""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np

from tmg.generation.simulators import scenes
from tmg.generation.simulators.scenes import (  # noqa: F401  re-exported: the scene file layout is shared by every generator
    load_scene_file,
    scene_chain,
    scene_details_path,
)

TABLE_OFFSET = "table_offset"
BASE_SUFFIX = "D1"
SCENE_META_ATTR = "_tmg_scene_meta"
INTERFACE_TYPE = "robosuite"
INTERFACE_PREFIX = "MG_"                   # MimicGen names its environment interfaces MG_<env_prefix>
_BUILT: dict[str, tuple[type, str]] = {}


# scene data ---------------------------------------------------------------------------------------
def load_scene_chain(scene: Mapping, generator: str | None = None, root: Path | None = None) -> dict:
    """The scene as a plain dict with the generator's details and ``parent`` inlined (recursively), validated as a
    robosuite scene (``validate_scene``). See ``tmg.generation.simulators.scenes.load_scene_chain`` for the arguments."""
    return scenes.load_scene_chain(scene, generator, root, validate=validate_scene)


def interface_name(task: Mapping) -> str:
    """The MimicGen environment interface of a task: ``task.datagen_interface`` or ``MG_<task.env_prefix>``."""
    return str(task.get("datagen_interface") or f"{INTERFACE_PREFIX}{task['env_prefix']}")


_default_interface_name = interface_name    # ``register_env_class`` has a parameter of the same name


def normalize_bounds(entry: Mapping, where: str) -> dict:
    """One object's placement bounds as ``{x, y, z_rot: (lo, hi), reference, rotation_axis}``; ``reference``
    is the string ``table_offset`` or an ``(x, y, z)`` tuple, ``rotation_axis`` one of x/y/z (default z; the
    Hammer Cleanup hammer rotates about y).
    """
    out = {}
    axis = str(entry.get("rotation_axis", "z"))
    if axis not in ("x", "y", "z"):
        raise ValueError(f"bounds {where}: rotation_axis must be x, y or z")
    out["rotation_axis"] = axis
    for key in ("x", "y", "z_rot"):
        if key not in entry:
            raise KeyError(f"bounds {where} lack '{key}'")
        lo, hi = (float(v) for v in entry[key])
        if lo > hi:
            raise ValueError(f"bounds {where}: {key} has lo > hi")
        out[key] = (lo, hi)
    ref = entry.get("reference")
    if ref is None:
        raise KeyError(f"bounds {where} lack 'reference'")
    if isinstance(ref, str):
        if ref != TABLE_OFFSET:
            raise ValueError(f"bounds {where}: reference must be '{TABLE_OFFSET}' or [x, y, z]")
        out["reference"] = TABLE_OFFSET
    else:
        out["reference"] = tuple(float(v) for v in ref)
        if len(out["reference"]) != 3:
            raise ValueError(f"bounds {where}: reference needs 3 numbers")
    return out


def validate_scene(scene: Mapping) -> None:
    """Check every scene of the chain: required keys, bound ranges, and the ``table_clip_m`` limit."""
    for node in scene_chain(scene):
        if "name" not in node or "env_suffix" not in node:
            raise KeyError("every scene needs 'name' and 'env_suffix'")
        clip = node.get("table_clip_m")
        for task_name, objects in (node.get("bounds") or {}).items():
            for obj, entry in objects.items():
                norm = normalize_bounds(entry, f"{node['name']}/{task_name}/{obj}")
                if clip is not None:
                    for key in ("x", "y"):
                        lo, hi = norm[key]
                        if lo < -float(clip) - 1e-9 or hi > float(clip) + 1e-9:
                            raise ValueError(f"scene {node['name']}: {task_name}/{obj} {key} exceeds table_clip_m")


def effective_bounds(scene: Mapping, task_name: str) -> dict | None:
    """Bounds of ``task_name`` from the nearest scene in the chain that defines them, or None."""
    for node in reversed(scene_chain(scene)):
        bounds = node.get("bounds") or {}
        if task_name in bounds:
            return {obj: normalize_bounds(e, f"{node['name']}/{task_name}/{obj}") for obj, e in bounds[task_name].items()}
    return None


def reach_constraint(node: Mapping) -> dict | None:
    """The enabled reach constraint of one scene node (``radius_m``, ``base_xy``, ``max_rejects``) or None."""
    rc = node.get("reach_constraint") or {}
    if not rc.get("enabled", False):
        return None
    return {"radius_m": float(rc["radius_m"]), "base_xy": tuple(float(v) for v in rc.get("base_xy", (-0.56, 0.0))),
            "max_rejects": int(rc.get("max_rejects", 500))}


def effective_reach_constraint(scene: Mapping) -> dict | None:
    """The reach constraint of the nearest scene in the chain that enables one, or None (as ``effective_bounds``)."""
    return next((reach_constraint(n) for n in reversed(scene_chain(scene)) if reach_constraint(n)), None)


def env_class_name(task: Mapping, scene: Mapping) -> str:
    """``<task.env_prefix>_<scene.env_suffix>``, or the scene's ``env_class`` pattern formatted with the prefix.

    ``env_class`` names classes whose name has no underscore, such as CP-Gen's ``CoffeeWide`` (``"{env_prefix}Wide"``).
    """
    if scene.get("env_class"):
        return str(scene["env_class"]).format(env_prefix=task["env_prefix"])
    return f"{task['env_prefix']}_{scene['env_suffix']}"


def base_env_class_name(task: Mapping, root: Mapping) -> str:
    """The registered class the scene chain's root derives from: ``base_env_class`` pattern (``"{env_prefix}Wide"``) or
    ``<env_prefix>_<base_env_suffix>`` (default suffix D1)."""
    if root.get("base_env_class"):
        return str(root["base_env_class"]).format(env_prefix=task["env_prefix"])
    return f"{task['env_prefix']}_{root.get('base_env_suffix', BASE_SUFFIX)}"


# class factories ----------------------------------------------------------------------------------
def make_bounds_method(entries: Mapping) -> Callable:
    """Build a ``_get_initial_placement_bounds`` method returning the configured ranges (``table_offset``
    resolved at call time).
    """
    entries = {k: dict(v) for k, v in entries.items()}

    def _get_initial_placement_bounds(self):
        out = {}
        for key, e in entries.items():
            ref = self.table_offset if e["reference"] == TABLE_OFFSET else np.array(e["reference"], dtype=float)
            out[key] = dict(x=tuple(e["x"]), y=tuple(e["y"]), z_rot=tuple(e["z_rot"]), reference=ref,
                            rotation_axis=e.get("rotation_axis", "z"))
        return out

    return _get_initial_placement_bounds


def robot_base_xy(env) -> tuple[float, float]:
    """World x, y of the first robot's base body (what the reach constraint's ``base_xy`` must match)."""
    body = env.robots[0].robot_model.root_body
    pos = env.sim.data.body_xpos[env.sim.model.body_name2id(body)]
    return float(pos[0]), float(pos[1])


def make_object_xy(interface_name: str) -> Callable:
    """Build ``object_xy(env) -> {object: (x, y)}`` reading positions through the MimicGen interface class."""
    cache: dict = {}

    def object_xy(env) -> dict:
        cls = cache.get("cls")
        if cls is None:
            from mimicgen.env_interfaces.base import REGISTERED_ENV_INTERFACES

            cls = REGISTERED_ENV_INTERFACES[INTERFACE_TYPE][interface_name]
            cache["cls"] = cls
        poses = cls(env).get_object_poses()
        return {k: np.asarray(p)[:2, 3].copy() for k, p in poses.items()}

    return object_xy


def make_reach_reset(parent: type, constraint: Mapping, object_xy: Callable, label: str) -> Callable:
    """Build a ``reset`` that redraws the scene until every object lies within the radius of the base and
    stores the reject count and distances on the env (``SCENE_META_ATTR``).
    """
    base = np.asarray(constraint["base_xy"], dtype=float)
    radius = float(constraint["radius_m"])
    max_rejects = int(constraint["max_rejects"])

    def reset(self):
        n_reject = 0
        for _ in range(max_rejects):
            out = parent.reset(self)
            xy = object_xy(self)
            if all(np.linalg.norm(p - base) <= radius for p in xy.values()):
                setattr(self, SCENE_META_ATTR, {"scene": label, "n_reject": n_reject,
                                                "base_dist": {k: round(float(np.linalg.norm(p - base)), 4)
                                                              for k, p in xy.items()}})
                return out
            n_reject += 1
        from robosuite.utils import RandomizationError

        raise RandomizationError(f"{type(self).__name__}: no reset with all objects within {radius} m of the base "
                                 f"in {max_rejects} draws")

    return reset


def _ensure_base_envs_registered() -> dict:
    """Import the packages whose environment classes the scene classes derive from; returns robosuite's registry."""
    import mimicgen  # noqa: F401  registers the public environments and interfaces

    # CP-Gen's image runs robosuite 1.5, where MimicGen's own environment modules fail to import (no
    # ``single_arm_env``); there the base classes (Coffee_D0, ...) come from CP-Gen's ``cpgen_envs`` package.
    try:
        import cpgen_envs  # noqa: F401
    except ImportError:
        pass
    from robosuite.environments.base import REGISTERED_ENVS

    return REGISTERED_ENVS


def register_env_class(task: Mapping, scene: Mapping, *, interface_name: str | None = None) -> str:
    """Register the environment class of (task, scene) and return its name. Idempotent."""
    registered = _ensure_base_envs_registered()
    scene = load_scene_chain(scene)
    name = env_class_name(task, scene)
    if name in _BUILT:
        return name
    chain = scene_chain(scene)
    base_name = base_env_class_name(task, chain[0])
    if base_name not in registered:
        raise KeyError(f"no registered robosuite environment '{base_name}' (MimicGen's or CP-Gen's environment packages)")
    interface = interface_name or _default_interface_name(task)
    object_xy = make_object_xy(str(interface))
    parent, parent_name = registered[base_name], base_name
    task_name = str(task["name"])
    for node in chain:
        cls_name = env_class_name(task, node)
        attrs: dict = {}
        if node.get("override_bounds", False):
            entries = effective_bounds(node, task_name)
            if entries is None:
                raise KeyError(f"scene {node['name']} overrides bounds but defines none for task {task_name}")
            attrs["_get_initial_placement_bounds"] = make_bounds_method(entries)
        constraint = reach_constraint(node)
        if constraint is not None:
            attrs["reset"] = make_reach_reset(parent, constraint, object_xy, str(node["name"]))
        if cls_name == parent_name:
            if attrs:
                raise ValueError(f"scene {node['name']} modifies the environment but keeps the name {cls_name}")
            continue
        fingerprint = json.dumps({"task": task_name, "parent": parent_name,
                                  "node": {k: v for k, v in node.items() if k != "parent"}}, sort_keys=True, default=str)
        cached = _BUILT.get(cls_name)
        if cached is not None:
            if cached[1] != fingerprint:
                raise ValueError(f"environment class {cls_name} already built from another scene definition")
            parent, parent_name = cached[0], cls_name
            continue
        if not attrs and node is not chain[-1]:
            continue                      # pass-through intermediate node; an empty leaf still gets its own name
        cls = type(cls_name, (parent,), attrs)
        _BUILT[cls_name] = (cls, fingerprint)
        parent, parent_name = cls, cls_name
    if name not in registered and name != base_name:
        raise RuntimeError(f"environment class {name} was not registered with robosuite")
    return name
