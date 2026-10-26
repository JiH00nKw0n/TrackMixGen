"""Source demonstrations of CP-Gen: the released source file and its constraint annotation.

Both come from the Hugging Face dataset repo ``generator.source_repo`` (``cpgen/datasets-src``):
``task.source_file`` (``datasets/source/<task>.hdf5``, 10 human demonstrations in robomimic layout) and
``task.annotation_file`` (``datasets/source/<task>/constraints.json``, the keypoint constraints of the first
demonstration, which CP-Gen's default configuration uses). Nothing is prepared; the manifest records sizes,
md5 sums, the demonstration count and the constraint count.
"""
from __future__ import annotations

import json
from pathlib import Path

from tmg.generation.generators.mimicgen.sources import download, file_md5


def fetch(task: dict, generator: dict, *, force: bool = False) -> dict:
    """Make the task's source file and annotation ready and write the manifest next to the source."""
    source, annotation = Path(task["source"]), Path(task["annotation"])
    manifest = source.with_suffix(".manifest.json")
    if source.exists() and annotation.exists() and manifest.exists() and not force:
        return json.loads(manifest.read_text())
    repo = str(generator["source_repo"])
    if not source.exists() or force:
        print(f"[tmg] downloading {repo}:{task['source_file']} -> {source}", flush=True)
        download(repo, str(task["source_file"]), source)
    if not annotation.exists() or force:
        print(f"[tmg] downloading {repo}:{task['annotation_file']} -> {annotation}", flush=True)
        download(repo, str(task["annotation_file"]), annotation)
    with __import__("h5py").File(source, "r") as f:
        demos = len(f["data"].keys())
    constraints = json.loads(annotation.read_text())
    out = {"path": str(source), "bytes": source.stat().st_size, "md5": file_md5(source), "demos": demos,
           "annotation": str(annotation), "annotation_md5": file_md5(annotation), "constraints": len(constraints),
           "expected_sources": task.get("num_sources")}
    if task.get("num_sources") and demos != int(task["num_sources"]):
        raise RuntimeError(f"{source}: {demos} demos, task config expects {task['num_sources']}")
    manifest.write_text(json.dumps(out, indent=1))
    print(f"[tmg] {task['name']}: {demos} source demos and {len(constraints)} constraints ready (md5 {out['md5']})", flush=True)
    return out
