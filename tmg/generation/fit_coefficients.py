"""``tmg fit_coefficients``: fit the per-subtask success model of the logistic mixer on a uniform-source baseline pool
and write the task's coefficient file.

Input: the finished ``rank_human_demos`` pool of (generator, task, scene, selection) with ``method=none`` (the same pool
``tmg rank_sources`` reads; with ``selection=ours`` every subtask's source is drawn uniformly and independently of the
scene) and the task's source demonstrations (their ``datagen_info`` gives each source's in-hand pose). The model and
the labels are described in ``tmg.generation.method.success_model``. Output:

- ``<output>/coefficients.yaml`` and ``<output>/FIT_COEFFICIENTS.json`` (summary);
- ``configs/experiment/generator/<generator>/coefficients/<task>.yaml`` when ``write_config`` is true, the file
  ``method.mixing.coefficients`` names. The ranking file (``.../ranking/<task>.yaml``) is a separate file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from tmg.config import dump_yaml, resolved
from tmg.core.chunks import CHUNKS_DIR, POOL_DONE, RESOLVED_CONFIG, chunk_is_done, read_json, write_json
from tmg.core.provenance import write_provenance
from tmg.core.records import RECORDS_DIR, read_attempts
from tmg.generation.method.success_model import (
    INHAND_FEATURES, SCENE_FEATURES, STABILIZER, UNITS, coefficients_path, design_rows, fit_subtask, subtask_outcomes)

COEFFICIENT_FILE = "coefficients.yaml"
SUMMARY = "FIT_COEFFICIENTS.json"


def read_source_infos(path) -> list[dict]:
    """Per source demonstration (``demo_0``, ``demo_1``, ... in index order): ``eef_pose`` (T,4,4) and
    ``object_poses`` {name: (T,4,4)} of MimicGen's ``datagen_info``."""
    import h5py
    out = []
    with h5py.File(path, "r") as f:
        for name in sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1])):
            info = f["data"][name]["datagen_info"]
            out.append({"eef_pose": np.asarray(info["eef_pose"], dtype=float),
                        "object_poses": {k: np.asarray(v, dtype=float) for k, v in info["object_poses"].items()}})
    return out


def write_coefficients(path: Path, *, generator: str, task: str, basis: str, num_sources: int, signals, blocks: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    head = (f"# Success-model coefficients of {generator} {task} (tmg fit_coefficients). Per subtask k:\n"
            f"#   P(subtask k succeeds | reached, source j) = sigmoid(alpha[j] + dx*|dx| + dy*|dy| + rotation*rot\n"
            f"#                                            + inhand_position*dpos + inhand_rotation*drot)  (in-hand terms k >= 1)\n"
            f"# Units: {UNITS}. The source ranking is a separate file (../ranking/{task}.yaml).\n")
    data = {"generator": generator, "task": task, "basis": basis, "num_sources": int(num_sources),
            "features": {"scene": list(SCENE_FEATURES), "inhand": list(INHAND_FEATURES)}, "units": UNITS,
            "stabilizer": STABILIZER, "subtask_term_signals": list(signals),
            "subtasks": {int(k): v for k, v in sorted(blocks.items())}}
    path.write_text(head + yaml.safe_dump(data, sort_keys=False, width=120))
    return path


def fit_coefficients(cfg) -> dict:
    """Run the stage described in the module docstring and return the summary written to ``FIT_COEFFICIENTS.json``."""
    plain = resolved(cfg)
    pool, output = Path(plain["pool"]), Path(plain["output"])
    output.mkdir(parents=True, exist_ok=True)
    dump_yaml(plain, output / RESOLVED_CONFIG)
    write_provenance(output, extra={"command": "fit_coefficients", "output": str(output), "pool": str(pool)})
    task, generator = plain["task"], plain["generator"]
    signals = task.get("subtask_term_signals")
    if not signals:
        raise KeyError(f"task {task['name']} has no subtask_term_signals (one per subtask, null for the last)")
    chunks = sorted(p for p in (pool / CHUNKS_DIR).glob("seed*") if chunk_is_done(p))
    attempts, outcomes = [], []
    for chunk in chunks:
        folder = chunk / RECORDS_DIR
        for a in read_attempts(folder):
            with np.load(folder / a["npz"]) as z:
                steps = {k: z[k] for k in z.files if k in ("seg_sub", "seg_of_step") or k.startswith("term__")}
            attempts.append(a)
            outcomes.append(subtask_outcomes(a, steps, signals))
    if not attempts:
        raise FileNotFoundError(f"no finished chunk with attempt records under {pool}")
    if len(attempts) < int(plain["min_attempts"]):
        raise ValueError(f"{len(attempts)} attempts in {pool}, fewer than min_attempts={plain['min_attempts']}")
    num_sources = int(task.get("num_sources") or 1 + max(int(s["src_demo_ind"]) for a in attempts for s in a["subtasks"]))
    infos = read_source_infos(plain["source"]) if len(signals) > 1 else None
    blocks = {}
    for k, signal in enumerate(signals):
        X, y, carried = design_rows(attempts, outcomes, k, num_sources, infos)
        block = {"term_signal": signal if k < len(signals) - 1 else "task success", **fit_subtask(X, y, num_sources, k)}
        if carried is not None:
            block["carried_object"] = carried
        blocks[k] = block
    basis = (f"Maximum-likelihood fit (Newton, stabilizer {STABILIZER}) on the uniform-source baseline pool {pool} "
             f"({len(attempts)} attempts, {len(chunks)} chunks, scene {plain['scene']['name']}, selection "
             f"{plain['selection']['name']}, generation {plain['generation']['name']}); one fit per subtask on the attempts "
             f"that reached it, label = that subtask's success.")
    written = [str(write_coefficients(output / COEFFICIENT_FILE, generator=generator["name"], task=task["name"], basis=basis,
                                      num_sources=num_sources, signals=signals, blocks=blocks))]
    if plain.get("write_config", True):
        written.append(str(write_coefficients(coefficients_path(f"{generator['name']}/{task['name']}"), generator=generator["name"],
                                              task=task["name"], basis=basis, num_sources=num_sources, signals=signals,
                                              blocks=blocks)))
    summary = {"pool": str(pool), "pool_done": (pool / POOL_DONE).exists(), "chunks": [c.name for c in chunks],
               "attempts": len(attempts), "successes": int(sum(bool(a["success"]) for a in attempts)),
               "subtasks": {k: {key: b[key] for key in ("reached", "successes", "log_loss", "slopes")} for k, b in blocks.items()},
               "written": written}
    if (pool / POOL_DONE).exists():
        summary["pool_done_info"] = read_json(pool / POOL_DONE)
    write_json(output / SUMMARY, summary)
    print(f"[tmg] fit_coefficients {generator['name']}/{task['name']}: {len(attempts)} attempts from {len(chunks)} chunks")
    for k, b in blocks.items():
        print(f"[tmg]   subtask {k}: reached {b['reached']}, succeeded {b['successes']}, log loss {b['log_loss']}, slopes {b['slopes']}")
    print(f"[tmg] written: {', '.join(written)}")
    return summary
