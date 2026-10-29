"""Source demonstrations of the Isaac Lab generators: the released annotated datasets, downloaded once.

- SoftMimicGen: Hugging Face dataset repo ``generator.source_repo`` (``masoudmoghani/SoftMimicGen``), file
  ``task.source_file`` (``annotated_dataset_franka_rope.hdf5``).
- DexMimicGen: a direct URL (``task.source_url``, the Isaac assets server file ``dataset_annotated_gr1.hdf5``).

Both are already annotated with subtask boundaries, so nothing is prepared; the manifest records size and md5.
"""
from __future__ import annotations

import json
import hashlib
import shutil
import urllib.request
from pathlib import Path

from tmg.generation.generators.mimicgen.sources import file_md5


def compatible_source(task: dict) -> Path:
    """Add explicit aliases for renamed upstream objects in a separate source copy.

    HDF5 hard links expose the same pose arrays under the current asset name. The
    released source, actions, images and all numeric arrays remain unchanged.
    """
    original=Path(task['source'])
    aliases=dict(task.get('source_object_aliases') or {})
    if not aliases or not original.exists(): return original
    import h5py
    mapping=json.dumps(aliases,sort_keys=True)
    digest=hashlib.sha256(mapping.encode()).hexdigest()[:12]
    target=original.with_name(original.stem+f'.object_aliases_{digest}.hdf5')
    manifest=target.with_suffix('.manifest.json')
    original_md5=file_md5(original)
    if target.exists() and manifest.exists():
        prior=json.loads(manifest.read_text())
        if prior['original_md5']!=original_md5 or prior['aliases']!=aliases:
            raise RuntimeError('source alias copy does not match the original')
        return target
    if target.exists(): raise RuntimeError('source alias copy exists without its manifest')
    tmp=target.with_suffix('.part')
    if tmp.exists(): raise RuntimeError('unfinished source alias copy preserved for review')
    shutil.copyfile(original,tmp)
    with h5py.File(tmp,'r+') as f:
        for demo in f['data'].values():
            poses=demo['obs/datagen_info/object_pose']
            for old,new in aliases.items():
                if old not in poses or new in poses:
                    raise RuntimeError(f'unexpected source object names: {old} -> {new}')
                poses[new]=poses[old]
                assert poses[new].id == poses[old].id
    tmp.replace(target)
    manifest.write_text(json.dumps(dict(original=str(original),original_md5=original_md5,
        prepared=str(target),prepared_md5=file_md5(target),aliases=aliases,
        numeric_arrays_unchanged=True),indent=2))
    return target


def download_url(url: str, dest: Path) -> Path:
    """Stream ``url`` to ``dest`` (written atomically)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)
    return dest


def fetch(task: dict, generator: dict, *, force: bool = False) -> dict:
    """Make the task's source file ready and write its manifest next to it."""
    dest = Path(task["source"])
    manifest = dest.with_suffix(".manifest.json")
    if dest.exists() and manifest.exists() and not force:
        return json.loads(manifest.read_text())
    if task.get("source_url"):
        print(f"[tmg] downloading {task['source_url']} -> {dest}", flush=True)
        download_url(str(task["source_url"]), dest)
    elif task.get("source_file") and generator.get("source_repo"):
        from tmg.generation.generators.mimicgen.sources import download

        download(str(generator["source_repo"]), str(task["source_file"]), dest)
    else:
        raise KeyError(f"task {task['name']!r} needs source_url, or source_file with generator.source_repo")
    with __import__("h5py").File(dest, "r") as f:
        demos = len(f["data"].keys())
    out = {"path": str(dest), "bytes": dest.stat().st_size, "md5": file_md5(dest), "demos": demos,
           "expected_sources": task.get("num_sources")}
    if task.get("num_sources") and demos != int(task["num_sources"]):
        raise RuntimeError(f"{dest}: {demos} demos, task config expects {task['num_sources']}")
    manifest.write_text(json.dumps(out, indent=1))
    print(f"[tmg] {task['name']}: {demos} source demos ready (md5 {out['md5']})", flush=True)
    return out
