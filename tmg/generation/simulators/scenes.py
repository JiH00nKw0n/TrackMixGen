"""Scene files: where a generator's scene data lives and how a scene's ``parent`` chain is read. Simulator-independent.

A scene is pure data in ``configs/experiment/generator/<generator>/scene/<name>.yaml``. Every generator uses the same
file layout and the same ``parent`` inheritance (ranges are inherited leaf-first); what the ranges mean is the
simulator layer's business (``robosuite.py`` validates placement bounds, ``isaaclab.py`` reset-event ranges), so
``load_scene_chain`` takes the validation as a callable.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

import yaml

from tmg import CONFIGS_ROOT

GENERATOR_DIR = "experiment/generator"      # configs/experiment/generator/<generator>/ holds task, scene and ranking data
SCENE_DIR = "scene"
MAX_PARENT_DEPTH = 8


def generator_config_dir(generator: str, root: Path | None = None) -> Path:
    """``configs/experiment/generator/<generator>``: the folder of one generator's task, scene and ranking files."""
    return Path(root or CONFIGS_ROOT) / GENERATOR_DIR / generator


def scene_details_path(name: str, generator: str, root: Path | None = None) -> Path:
    """``configs/experiment/generator/<generator>/scene/<name>.yaml``: the generator's scene ``name``."""
    return generator_config_dir(generator, root) / SCENE_DIR / f"{name}.yaml"


def load_scene_file(name: str, generator: str, root: Path | None = None) -> dict:
    """The scene ``name`` of ``generator`` as a dict (``configs/experiment/generator/<generator>/scene/<name>.yaml``)."""
    details = scene_details_path(name, generator, root)
    if not details.exists():
        raise FileNotFoundError(f"scene {name!r} of generator {generator!r} not found: {details}")
    data = dict(yaml.safe_load(details.read_text()) or {})
    data.setdefault("name", name)
    data["generator"] = generator
    return data


def load_scene_chain(scene: Mapping, generator: str | None = None, root: Path | None = None, *,
                     validate: Callable[[Mapping], None] | None = None) -> dict:
    """The scene as a plain dict with the generator's details and ``parent`` inlined (recursively).

    Args:
        scene: the composed ``scene`` config (selector only, or details inline as the tests pass them).
        generator: generator name; required when ``scene`` carries no ``env_suffix`` (selector only) and
            taken from ``scene['generator']`` otherwise.
        root: configs root (default ``CONFIGS_ROOT``).
        validate: called with the inlined chain before it is returned (the simulator layer's checks).
    """
    data = dict(scene)
    generator = generator or data.get("generator")
    if "env_suffix" not in data:
        if generator is None:
            raise KeyError(f"scene {data.get('name')!r}: generator name needed to load its details")
        data = {**data, **load_scene_file(str(data["name"]), str(generator), root)}
    parent = data.get("parent")
    depth = 0
    node = data
    while parent is not None:
        depth += 1
        if depth > MAX_PARENT_DEPTH:
            raise RecursionError("scene parent chain nests too deeply")
        parent_data = load_scene_file(parent, str(generator), root) if isinstance(parent, str) else dict(parent)
        node["parent"] = parent_data
        node = parent_data
        parent = node.get("parent")
    if validate is not None:
        validate(data)
    return data


def scene_chain(scene: Mapping) -> list[dict]:
    """Root-first list of a scene and its ancestors (``parent`` already inlined)."""
    chain = []
    node = scene
    while node is not None:
        chain.append(dict(node))
        node = node.get("parent")
    chain.reverse()
    return chain
