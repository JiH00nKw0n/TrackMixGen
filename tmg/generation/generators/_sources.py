"""Source-demonstration plumbing shared by the generators' ``sources.py`` modules: the Hugging Face download, file
facts for the manifests (size, md5, demo count) and the manifest writer.

Each generator keeps its own manifest location and keys (servers already hold the written files), so this module
never decides the layout of a manifest; it only supplies the pieces. ``huggingface_hub`` is imported inside
``download`` because the runner image does not ship it.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import h5py


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
        partial.chmod(0o644)      # the Hugging Face cache keeps its files read-only; the CP-Gen step appends constraint windows
        partial.replace(dest)
    return dest


def ensure_downloaded(repo_id: str, filename: str, dest: Path, *, revision: str | None = None) -> bool:
    """Download ``filename`` of ``repo_id`` to ``dest`` unless it already exists; True when it was downloaded now."""
    if dest.exists():
        print(f"[tmg] using existing {dest}", flush=True)
        return False
    print(f"[tmg] downloading {repo_id}:{filename} -> {dest}", flush=True)
    download(repo_id, filename, dest, revision=revision)
    return True


def count_demos(path: Path) -> int:
    """Number of demonstrations (groups under ``data``) of a source hdf5. A file without ``data`` is a broken source, so
    the ``KeyError`` is left to propagate."""
    with h5py.File(str(path), "r") as f:
        return len(f["data"].keys())


def file_facts(path: Path) -> dict:
    """``{"bytes": size, "md5": hex}`` of a file, the two facts every manifest records (in this key order)."""
    return {"bytes": path.stat().st_size, "md5": file_md5(path)}


def check_num_sources(path: Path, demos: int, task: dict) -> None:
    """Raise ``RuntimeError`` when ``task.num_sources`` is set and differs from the demonstrations found in ``path``."""
    if task.get("num_sources") and demos != int(task["num_sources"]):
        raise RuntimeError(f"{path}: {demos} demos, task config expects {task['num_sources']}")


def write_manifest(path: Path, manifest: dict, *, indent: int, trailing_newline: bool) -> Path:
    """Write a manifest as JSON. ``indent`` and ``trailing_newline`` are per generator so the files stay as they were."""
    path = Path(path)
    path.write_text(json.dumps(manifest, indent=indent) + ("\n" if trailing_newline else ""))
    return path
