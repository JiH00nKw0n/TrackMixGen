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

from tmg.generation.generators._sources import (
    check_num_sources,
    count_demos,
    ensure_downloaded,
    file_facts,
    file_md5,
    write_manifest,
)

MANIFEST_SUFFIX = ".manifest.json"      # next to the source file: <task>/source.manifest.json
WINDOWS_SUFFIX = ".windows.json"        # the derivation report of the per-demonstration windows


def manifest_path(source: Path) -> Path:
    """``<source>.manifest.json`` beside the source file (the suffix replaces ``.hdf5``)."""
    return source.with_suffix(MANIFEST_SUFFIX)


def fetch(task: dict, generator: dict) -> dict:
    """Make the task's source file and annotation ready and write the manifest next to the source.

    When the source, the annotation and the manifest exist and the per-demonstration windows are present (or not
    wanted, ``task.per_demo_windows: false``), the stored manifest is returned as is.
    """
    from tmg.generation.generators.cpgen import windows as win

    source, annotation = Path(str(task["source"])).expanduser(), Path(str(task["annotation"])).expanduser()
    manifest = manifest_path(source)
    want_windows = bool(task.get("per_demo_windows", True))
    if source.exists() and annotation.exists() and manifest.exists() and (not want_windows or win.has_constraint_data(str(source))):
        return json.loads(manifest.read_text())
    repo = str(generator["source_repo"])
    ensure_downloaded(repo, str(task["source_file"]), source)
    ensure_downloaded(repo, str(task["annotation_file"]), annotation)
    demos = count_demos(source)
    constraints = json.loads(annotation.read_text())
    out = {"path": str(source), **file_facts(source), "demos": demos,
           "annotation": str(annotation), "annotation_md5": file_md5(annotation), "constraints": len(constraints),
           "expected_sources": task.get("num_sources")}
    check_num_sources(source, demos, task)
    if want_windows and not win.has_constraint_data(str(source)):
        # every demonstration gets its own constraint windows (demo 0: the released ones), so that selection=ours can use all
        # of them; the thresholds are calibrated on demonstration 0 against the released windows (windows.py)
        print(f"[tmg] deriving per-demonstration constraint windows of {task['name']} (CP-Gen segmentation, calibrated on demo 0)", flush=True)
        report_path = source.with_suffix(WINDOWS_SUFFIX)
        report = win.annotate_source(str(source), str(annotation), source_root=str(generator["source_root"]),
                                     report_path=str(report_path))
        out["windows"] = {"thresholds_m": report["thresholds"], "calibration_demo0": report["calibration"], "written": report["written"],
                          "report": str(report_path)}
        out["md5"] = file_md5(source)       # the windows were appended to the file
        print(f"[tmg] windows written for demos {sorted(k for k, n in report['written'].items() if n)}; thresholds {report['thresholds']}", flush=True)
    write_manifest(manifest, out, indent=1, trailing_newline=False)
    print(f"[tmg] {task['name']}: {demos} source demos and {len(constraints)} constraints ready (md5 {out['md5']})", flush=True)
    return out
