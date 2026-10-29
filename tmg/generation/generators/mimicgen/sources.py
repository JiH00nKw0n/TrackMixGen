"""MimicGen source demonstrations: download from the Hugging Face release and add ``datagen_info``.

The released source files do not carry ``datagen_info``; MimicGen's ``prepare_src_dataset.py`` replays the
demos in the simulator to add it, so this runs inside the robosuite image. ``download`` and ``file_md5`` live in
``tmg.generation.generators._sources`` and are re-exported here because the other generators import them from this module.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import h5py

from tmg.core.layout import indexed_keys
from tmg.generation.generators._sources import (
    count_demos,
    download,
    ensure_downloaded,
    file_facts,
    file_md5,
    write_manifest,
)
from tmg.generation.simulators.robosuite import interface_name

__all__ = ["MANIFEST", "PREPARE_MODULE", "demos_without_datagen_info", "download", "fetch", "file_md5", "prepare"]

PREPARE_MODULE = "mimicgen.scripts.prepare_src_dataset"
MANIFEST = "manifest.json"


def demos_without_datagen_info(path: Path) -> list[str]:
    """Demo keys of a source hdf5 that lack ``datagen_info`` (empty list means ready for MimicGen)."""
    with h5py.File(str(path), "r") as f:
        demos = indexed_keys(f["data"].keys())
        return [d for d in demos if f"data/{d}/datagen_info" not in f]


def prepare(path: Path, interface: str) -> None:
    """Run MimicGen's ``prepare_src_dataset`` in place for ``interface`` (replays the demos in the simulator)."""
    cmd = [sys.executable, "-m", PREPARE_MODULE, "--dataset", str(path), "--env_interface", interface,
           "--env_interface_type", "robosuite"]
    print(f"[tmg] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def fetch(task: dict, generator: dict) -> dict:
    """Make the task's source file ready and return its manifest.

    Args:
        task: task config (``source``, ``source_registry_name``, ``datagen_interface``, ``name``).
        generator: generator config (``source_repo``).

    Returns:
        The manifest written next to the file (path, bytes, md5, demo count, whether it was prepared here).
    """
    import mimicgen

    dest = Path(str(task["source"])).expanduser()
    registry = mimicgen.DATASET_REGISTRY["source"]
    name = str(task.get("source_registry_name", task["name"]))
    if name not in registry:
        raise KeyError(f"mimicgen has no source dataset {name!r}; known: {sorted(registry)}")
    ensure_downloaded(str(generator.get("source_repo", mimicgen.HF_REPO_ID)), registry[name]["url"], dest)
    missing = demos_without_datagen_info(dest)
    prepared = bool(missing)
    if missing:
        prepare(dest, interface_name(task))
        if demos_without_datagen_info(dest):
            raise RuntimeError(f"{dest} still lacks datagen_info after prepare_src_dataset")
    n = count_demos(dest)
    manifest = {"task": task["name"], "file": str(dest), **file_facts(dest), "demos": n, "prepared_here": prepared,
                "registry_url": registry[name]["url"]}
    write_manifest(dest.parent / MANIFEST, manifest, indent=2, trailing_newline=True)
    print(f"[tmg] {task['name']}: {n} source demos ready (md5 {manifest['md5']})", flush=True)
    return manifest
