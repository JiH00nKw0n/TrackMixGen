"""Source demonstrations of CP-Gen: the released source file, its constraint annotation, and the per-demonstration
constraint windows that let every demonstration be a source (``windows.py``, decision 2026-10-29).

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
    from tmg.generation.generators.cpgen import windows as win

    source, annotation = Path(task["source"]), Path(task["annotation"])
    manifest = source.with_suffix(".manifest.json")
    want_windows = bool(task.get("per_demo_windows", True))
    if source.exists() and annotation.exists() and manifest.exists() and not force:
        if not want_windows or win.has_constraint_data(str(source)):
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
    if want_windows and (force or not win.has_constraint_data(str(source))):
        # every demonstration gets its own constraint windows (demo 0: the released ones), so that selection=ours can use all
        # of them; the thresholds are calibrated on demonstration 0 against the released windows (windows.py)
        print(f"[tmg] deriving per-demonstration constraint windows of {task['name']} (CP-Gen segmentation, calibrated on demo 0)", flush=True)
        report = win.annotate_source(str(source), str(annotation), source_root=str(generator["source_root"]),
                                     report_path=str(source.with_suffix(".windows.json")))
        out["windows"] = {"thresholds_m": report["thresholds"], "calibration_demo0": report["calibration"], "written": report["written"],
                          "report": str(source.with_suffix(".windows.json"))}
        out["md5"] = file_md5(source)
        print(f"[tmg] windows written for demos {sorted(k for k, n in report['written'].items() if n)}; thresholds {report['thresholds']}", flush=True)
    manifest.write_text(json.dumps(out, indent=1))
    print(f"[tmg] {task['name']}: {demos} source demos and {len(constraints)} constraints ready (md5 {out['md5']})", flush=True)
    return out
