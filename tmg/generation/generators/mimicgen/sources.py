"""MimicGen source demonstrations: download from the Hugging Face release and add ``datagen_info``.

The released source files do not carry ``datagen_info``; MimicGen's ``prepare_src_dataset.py`` replays the
demos in the simulator to add it, so this runs inside the robosuite image.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PREPARE_MODULE = "mimicgen.scripts.prepare_src_dataset"
MANIFEST = "manifest.json"


def file_md5(path: Path) -> str:
    """Hex md5 of a file read in 1 MiB blocks."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(repo_id: str, filename: str, dest: Path, revision: str | None = None) -> Path:
    """Download one file of a Hugging Face dataset repo (at ``revision`` when given) to ``dest`` (written atomically)."""
    from huggingface_hub import hf_hub_download

    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(dest.parent)) as tmp:
        got = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", cache_dir=tmp, revision=revision)
        partial = dest.with_suffix(dest.suffix + ".partial")
        shutil.move(str(Path(got).resolve()), str(partial))
        partial.replace(dest)
    return dest


def demos_without_datagen_info(path: Path) -> list[str]:
    """Demo keys of a source hdf5 that lack ``datagen_info`` (empty list means ready for MimicGen)."""
    import h5py

    with h5py.File(str(path), "r") as f:
        demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
        return [d for d in demos if f"data/{d}/datagen_info" not in f]


def prepare(path: Path, interface: str) -> None:
    """Run MimicGen's ``prepare_src_dataset`` in place for ``interface`` (replays the demos in the simulator)."""
    cmd = [sys.executable, "-m", PREPARE_MODULE, "--dataset", str(path), "--env_interface", interface,
           "--env_interface_type", "robosuite"]
    print(f"[tmg] {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, stdin=subprocess.DEVNULL)


def fetch(task: dict, generator: dict, *, force: bool = False) -> dict:
    """Make the task's source file ready and return its manifest.

    Args:
        task: task config (``source``, ``source_registry_name``, ``datagen_interface``, ``name``).
        generator: generator config (``source_repo``).
        force: re-download even if the file exists.

    Returns:
        The manifest written next to the file (path, bytes, md5, demo count, whether it was prepared here).
    """
    import mimicgen

    dest = Path(str(task["source"])).expanduser()
    registry = mimicgen.DATASET_REGISTRY["source"]
    name = str(task.get("source_registry_name", task["name"]))
    if name not in registry:
        raise KeyError(f"mimicgen has no source dataset {name!r}; known: {sorted(registry)}")
    if force or not dest.exists():
        print(f"[tmg] downloading {registry[name]['url']} -> {dest}", flush=True)
        download(str(generator.get("source_repo", mimicgen.HF_REPO_ID)), registry[name]["url"], dest)
    else:
        print(f"[tmg] using existing {dest}", flush=True)
    missing = demos_without_datagen_info(dest)
    prepared = bool(missing)
    if missing:
        prepare(dest, str(task.get("datagen_interface") or f"MG_{task['env_prefix']}"))
        if demos_without_datagen_info(dest):
            raise RuntimeError(f"{dest} still lacks datagen_info after prepare_src_dataset")
    import h5py

    with h5py.File(str(dest), "r") as f:
        n = len(f["data"].keys())
    manifest = {"task": task["name"], "file": str(dest), "bytes": dest.stat().st_size, "md5": file_md5(dest),
                "demos": n, "prepared_here": prepared, "registry_url": registry[name]["url"]}
    (dest.parent / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[tmg] {task['name']}: {n} source demos ready (md5 {manifest['md5']})", flush=True)
    return manifest
