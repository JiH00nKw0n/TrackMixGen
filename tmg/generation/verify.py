"""Baseline identity check: the adapter with ``method=none`` must reproduce the official generator exactly."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import h5py
import numpy as np

from tmg.config import resolved
from tmg.core.chunks import CHUNK_SPEC, Chunk, write_json
from tmg.generation.generators import make_generator


def _run(generator, chunk: Chunk, folder: Path, *, disable_graft: bool, log: Path) -> dict:
    """Run one chunk in a fresh folder, officially (``disable_graft``) or through the adapter, and locate its results."""
    if folder.exists():
        shutil.rmtree(folder)
    spec = generator.prepare_chunk(chunk, folder)
    spec.extra["disable_graft"] = bool(disable_graft)
    spec_path = write_json(folder / CHUNK_SPEC, asdict(spec))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    with open(log, "w") as lf:
        code = subprocess.run(generator.chunk_command(spec, spec_path), stdout=lf, stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL, env=env, cwd=str(folder)).returncode
    stats = Path(spec.folder) / spec.extra["experiment"] / "important_stats.json"
    if not stats.exists():
        raise RuntimeError(f"chunk did not finish (exit {code}); see {log}")
    return {"spec": spec, "stats": json.loads(stats.read_text()), "experiment": stats.parent}


def _walk(h5, prefix=""):
    """Flatten an hdf5 group into ``{path: array}``."""
    out = {}
    for key, item in h5.items():
        path = f"{prefix}/{key}"
        if isinstance(item, h5py.Dataset):
            out[path] = item[()]
        else:
            out.update(_walk(item, path))
    return out


def compare_hdf5(a: Path, b: Path) -> list[str]:
    """Differences between two hdf5 files: missing datasets, shape or value differences."""
    diffs = []
    with h5py.File(a, "r") as fa, h5py.File(b, "r") as fb:
        da, db = _walk(fa), _walk(fb)
        for key in sorted(set(da) | set(db)):
            if key not in da or key not in db:
                diffs.append(f"{key}: only in {'official' if key in da else 'adapter'}")
                continue
            va, vb = da[key], db[key]
            if np.shape(va) != np.shape(vb):
                diffs.append(f"{key}: shape {np.shape(va)} vs {np.shape(vb)}")
            elif not np.array_equal(np.asarray(va), np.asarray(vb)):
                diffs.append(f"{key}: values differ (max abs {np.max(np.abs(np.asarray(va, dtype=float) - np.asarray(vb, dtype=float))) if np.asarray(va).dtype.kind in 'fiub' else 'n/a'})")
    return diffs


def verify_baseline(cfg) -> dict:
    """Run the same chunk twice (official generator, adapter with ``method=none``) and compare the outputs.

    Returns:
        The report written to ``VERIFY.json``: both stats, per-file differences and ``identical``.
    """
    plain = resolved(cfg)
    generator = make_generator(plain)
    problems = generator.check_sources()
    if problems:
        raise RuntimeError("; ".join(problems))
    output = Path(str(plain["output"])).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    chunk = Chunk(index=0, seed=int(plain["generation"]["seed_offset"]) + 1, attempts=int(plain["generation"]["attempts"]))
    official = _run(generator, chunk, output / "official", disable_graft=True, log=output / "official.log")
    adapter = _run(generator, chunk, output / "adapter", disable_graft=False, log=output / "adapter.log")
    report = {"chunk": asdict(chunk), "official_stats": official["stats"], "adapter_stats": adapter["stats"], "diffs": {}}
    for name in ("demo.hdf5", "demo_failed.hdf5"):
        pa, pb = official["experiment"] / name, adapter["experiment"] / name
        if pa.exists() or pb.exists():
            report["diffs"][name] = compare_hdf5(pa, pb) if pa.exists() and pb.exists() else [f"{name} missing on one side"]
    for key in ("num_success", "num_failures", "num_attempts", "num_problematic"):
        if official["stats"].get(key) != adapter["stats"].get(key):
            report["diffs"].setdefault("important_stats", []).append(f"{key}: {official['stats'].get(key)} vs {adapter['stats'].get(key)}")
    report["identical"] = not any(report["diffs"].values())
    write_json(output / "VERIFY.json", report)
    status = "IDENTICAL" if report["identical"] else "DIFFERENT"
    print(f"[tmg] baseline identity: {status} ({official['stats']['num_success']}/{official['stats']['num_attempts']} "
          f"successes official, {adapter['stats']['num_success']}/{adapter['stats']['num_attempts']} adapter) -> {output / 'VERIFY.json'}",
          flush=True)
    if not report["identical"]:
        for name, diffs in report["diffs"].items():
            for d in diffs[:20]:
                print(f"  {name}: {d}", flush=True)
    return report
