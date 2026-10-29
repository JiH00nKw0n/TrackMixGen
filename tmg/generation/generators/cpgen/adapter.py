"""``SyntheticDataGenerator`` for CP-Gen (``demo_aug.generate.main`` of kevin-thankyou-lin/cpgen).

A chunk is one call of CP-Gen's ``main(Config(...))`` with ``n_demos = attempts`` and ``require_n_demos =
False`` (a fixed attempt count, like every other generator here), ``seed = chunk seed``, the task's source file
and constraint annotation, the official source selection (``task.source_indices``, by default the first
demonstration) and our scene's environment class as ``env_name``. The chunk process (``entrypoint.py``)
registers that class and installs the grafts before calling ``main``. Outputs of a chunk::

    <chunk>/generated/successes/*.hdf5, failures/*.hdf5   CP-Gen's per-attempt files (and failure videos)
    <chunk>/generated.hdf5, generated_failures.hdf5       merged successes / failures (states and actions)
    <chunk>/generated_obs.hdf5                            successes with observations rendered from the states
                                                          (robomimic dataset_states_to_obs: both cameras 84x84,
                                                          robot and object state), what ``tmg dataset`` reads
    <chunk>/records/attempts.jsonl, attempt_%06d.npz      our records (every attempt)

The k-th demonstration of ``generated_obs.hdf5`` is the k-th successful attempt (CP-Gen merges the success
files in attempt order). CP-Gen's own hdf5 actions are the noise-free labels; the executed actions are in the
records.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py

from tmg.core.chunks import Chunk
from tmg.core.records import ATTEMPTS_JSONL, RECORDS_DIR, read_attempts, validate_records
from tmg.generation.base import ChunkSpec
from tmg.generation.generators.mimicgen.adapter import RobosuiteGenerator

ENTRYPOINT = "tmg.generation.generators.cpgen.entrypoint"
GENERATED_OBS = "generated_obs.hdf5"
GENERATED = "generated.hdf5"


class CPGenGenerator(RobosuiteGenerator):
    """The ``SyntheticDataGenerator`` for CP-Gen (see the module docstring)."""

    name = "cpgen"

    def fetch_sources(self) -> dict:
        from tmg.generation.generators.cpgen.sources import fetch

        return fetch(self.task, self.generator_cfg)

    def all_sources(self) -> bool:
        """selection.all_sources: generate from every source demonstration (one chosen per attempt) instead of demo 0 only."""
        return bool((self.cfg.get("selection") or {}).get("all_sources"))

    def check_sources(self) -> list[str]:
        """Both the source file and its annotation must exist; with every source in use, the per-demonstration windows too."""
        missing = [str(p) for p in (Path(self.task["source"]), Path(self.task["annotation"])) if not p.exists()]
        if not missing and self.all_sources():
            from tmg.generation.generators.cpgen.windows import has_constraint_data

            if not has_constraint_data(str(self.task["source"])):
                missing.append(f"{self.task['source']}: no per-demonstration constraint_data (run `tmg sources` for this task)")
        return missing

    def cpgen_config(self, chunk: Chunk, folder: Path) -> dict:
        """Fields of ``demo_aug.generate.Config`` for the chunk (serialized into the chunk spec)."""
        task = self.task
        start, end = (int(v) for v in task.get("source_indices", [0, 1]))
        constraints_path = str(task["annotation"])
        if self.all_sources():
            # selection=ours: every source demonstration is loaded and the runtime picks one per attempt. The demonstrations'
            # own constraint windows (constraint_data groups written by `tmg sources`, windows.py) are read by CP-Gen's
            # load_demos, so the released file (demo 0's windows) is not passed.
            start, end = 0, int(task.get("num_sources") or 0) or None
            constraints_path = None
        cfg = {"seed": int(chunk.seed), "demo_path": str(self.source_path()), "load_demos_start_idx": start,
               "load_demos_end_idx": end, "custom_constraints_path": constraints_path, "n_demos": int(chunk.attempts),
               "require_n_demos": False, "env_name": self.env_name(), "save_dir": str(folder / "generated"),
               "merge_demo_save_path": str(folder / GENERATED), "use_wandb": False}
        cfg.update(dict(task.get("cpgen_config") or {}))
        cfg.update(dict(self.generator_cfg.get("config") or {}))
        return cfg

    def prepare_chunk(self, chunk: Chunk, folder: Path) -> ChunkSpec:
        self.require_generation_config()
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        if self.protocol.get("camera_names") not in (None, [], list(self.task.get("cameras", []))):
            raise ValueError("CP-Gen renders its own observations (both task cameras, 84x84) after generation; "
                             "generation.protocol.camera_names must be [] or the task's cameras")
        config = self.cpgen_config(chunk, folder)
        (folder / "cpgen_config.json").write_text(json.dumps(config, indent=1))
        return ChunkSpec(
            chunk={"index": chunk.index, "seed": chunk.seed, "attempts": chunk.attempts}, generator=self.generator_cfg["name"],
            task=self.task, scene=self.scene_chain(), method=self.method, protocol=self.protocol, generator_config="",
            folder=str(folder), records_dir=str(folder / RECORDS_DIR),
            expected_outputs=[str(folder / RECORDS_DIR / ATTEMPTS_JSONL)],
            extra={"cpgen_config": config, "source_root": str(self.generator_cfg["source_root"]),
                   "graft_hashes": dict(self.generator_cfg.get("graft_hashes") or {}), "all_sources": self.all_sources()})

    def chunk_command(self, spec: ChunkSpec, spec_path: Path) -> list[str]:
        return [sys.executable, "-m", ENTRYPOINT, "--spec", str(spec_path)]

    def chunk_result(self, spec: ChunkSpec) -> dict | None:
        """Validate the records and count the rendered successes; None when the chunk did not finish."""
        folder = Path(spec.folder)
        records = read_attempts(folder / RECORDS_DIR)
        attempts = int(spec.chunk["attempts"])
        if len(records) < attempts:
            return None
        validate_records(folder / RECORDS_DIR, attempts)
        successes = sum(1 for r in records if r.get("success"))
        rendered = folder / GENERATED_OBS
        exported = 0
        if rendered.exists():
            with h5py.File(rendered, "r") as f:
                exported = len(f["data"].keys()) if "data" in f else 0
        if successes and exported != successes:
            return None                      # the post-generation rendering has not finished (or failed)
        return {"attempts": len(records), "successes": successes, "exported_episodes": exported,
                "generated_hdf5": str(rendered) if rendered.exists() else None}

    def demo_layout(self) -> dict:
        return {"demo_glob": GENERATED_OBS, "image_key": "{cam}_image", "actions_key": "actions", "states_key": "states",
                "labels_from_records": False, "success_only_file": True}
