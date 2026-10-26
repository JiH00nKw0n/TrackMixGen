"""MimicGen adapter: prepares chunks for ``mimicgen.scripts.generate_dataset`` and reads their results."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tmg.core.chunks import Chunk
from tmg.core.records import ATTEMPTS_JSONL, RECORDS_DIR, validate_records
from tmg.generation.base import ChunkSpec, SyntheticDataGenerator
from tmg.generation.generators.mimicgen import config_builder
from tmg.generation.simulators.robosuite import env_class_name, load_scene_chain

IMPORTANT_STATS = "important_stats.json"
MG_CONFIG = "mg_config.json"
ENTRYPOINT = "tmg.generation.generators.mimicgen.entrypoint"


class RobosuiteGenerator(SyntheticDataGenerator):
    """Shared part of the adapters that run on robosuite: scene chain resolution and environment class name."""
    simulator = "robosuite"

    def scene_chain(self) -> dict:
        """The scene config with its parent chain inlined (``simulators.robosuite.load_scene_chain``)."""
        return load_scene_chain(self.scene, self.generator_cfg["name"])

    def env_name(self) -> str:
        """Name of the robosuite environment class of (task, scene), e.g. ``Square_D1`` or ``Square_OURS``."""
        return env_class_name(self.task, self.scene_chain())


class MimicGenGenerator(RobosuiteGenerator):
    """The ``SyntheticDataGenerator`` for NVlabs/mimicgen.

    A chunk is one run of ``mimicgen.scripts.generate_dataset`` with a config built from the task's template and
    our protocol; the chunk process (``entrypoint.py``) installs the grafts before running it.
    """
    name = "mimicgen"

    def template_path(self) -> Path:
        """Path of the MimicGen config template of the task under ``generator.source_root``.

        Raises:
            FileNotFoundError: when the pinned checkout (``TMG_MIMICGEN_SRC``) is missing.
        """
        root = Path(str(self.generator_cfg["source_root"])).expanduser()
        path = root / str(self.generator_cfg["templates"]) / str(self.task["template"])
        if not path.exists():
            raise FileNotFoundError(f"MimicGen template {path} not found; generator.source_root must point at a "
                                    f"checkout of mimicgen@{self.generator_cfg.get('commit')} (TMG_MIMICGEN_SRC)")
        return path

    def template(self) -> dict:
        """The task's MimicGen config template as a dict."""
        return json.loads(self.template_path().read_text())

    def fetch_sources(self) -> dict:
        """Download and prepare the task's human source demonstrations (``sources.fetch``)."""
        from tmg.generation.generators.mimicgen.sources import fetch

        return fetch(self.task, self.generator_cfg)

    def prepare_chunk(self, chunk: Chunk, folder: Path) -> ChunkSpec:
        """Write ``mg_config.json`` for the chunk and describe it.

        Args:
            chunk: index, seed and attempt count.
            folder: the chunk folder (created if needed).

        Returns:
            The ``ChunkSpec`` with the expected outputs (``important_stats.json``, ``attempts.jsonl``) and the
            graft hashes the chunk process must verify.
        """
        self.require_generation_config()
        folder = Path(folder)
        env_name = self.env_name()
        protocol = dict(self.protocol)
        if protocol.get("camera_size") is None:
            protocol["camera_size"] = list(self.task["image_size"])
        config = config_builder.build_generation_config(
            self.template(), protocol, env_name=env_name, source_dataset=str(self.source_path()),
            output_folder=str(folder), attempts=chunk.attempts, seed=chunk.seed)
        config_path = config_builder.save_config(config, folder / MG_CONFIG)
        experiment = config_builder.experiment_name(env_name, chunk.attempts, chunk.seed)
        return ChunkSpec(
            generator=self.name, chunk={"index": chunk.index, "seed": chunk.seed, "attempts": chunk.attempts},
            folder=str(folder), records_dir=str(folder / RECORDS_DIR), task=self.task, scene=self.scene_chain(),
            method=self.method, protocol=protocol, generator_config=str(config_path),
            expected_outputs=[str(folder / experiment / IMPORTANT_STATS), str(folder / RECORDS_DIR / ATTEMPTS_JSONL)],
            extra={"env_name": env_name, "experiment": experiment,
                   "graft_hashes": dict(self.generator_cfg.get("graft_hashes") or {}),
                   "commit": self.generator_cfg.get("commit")})

    def demo_layout(self) -> dict:
        """How ``tmg dataset`` reads MimicGen chunk outputs (``<experiment>/demo.hdf5``, robomimic layout)."""
        return {"demo_glob": "*/demo.hdf5", "image_key": "{cam}_image", "actions_key": "actions", "states_key": "states",
                "labels_from_records": False, "success_only_file": True}

    def chunk_command(self, spec: ChunkSpec, spec_path: Path) -> list[str]:
        """Command line of the chunk process: the entrypoint module with the spec path."""
        return [sys.executable, "-m", ENTRYPOINT, "--spec", str(spec_path)]

    def chunk_result(self, spec: ChunkSpec) -> dict | None:
        """Summary of a finished chunk, an ``error`` entry when MimicGen's outputs and our records disagree,
        or None when the chunk did not finish. Generic record checks live in ``tmg.core.records``."""
        import h5py

        stats_path = Path(spec.folder) / spec.extra["experiment"] / IMPORTANT_STATS
        if not stats_path.exists() or not (Path(spec.records_dir) / ATTEMPTS_JSONL).exists():
            return None
        checked = validate_records(spec.records_dir, spec.chunk["attempts"])
        if not checked["ok"]:
            return {"error": checked["error"]}
        stats = json.loads(stats_path.read_text())
        successes, failures = checked["successes"], checked["failures"]
        if (len(checked["lines"]), successes, failures) != tuple(int(stats.get(k, -1)) for k in
                                                                 ("num_attempts", "num_success", "num_failures")):
            return {"error": f"records ({len(checked['lines'])} attempts, {successes} successes, {failures} failures) "
                             f"differ from important_stats.json ({stats.get('num_attempts')}, {stats.get('num_success')}, "
                             f"{stats.get('num_failures')})"}
        cap = spec.protocol.get("max_num_failures")
        expected = {"demo.hdf5": successes,
                    "demo_failed.hdf5": (failures if cap is None else min(failures, int(cap))) if spec.protocol.get("keep_failed") else 0}
        counts = {}
        for name, want in expected.items():
            path = stats_path.parent / name
            if want == 0:
                continue
            if not path.exists():
                return {"error": f"{path} missing although {want} demonstrations were expected"}
            try:
                with h5py.File(path, "r") as f:
                    counts[name] = len(f["data"].keys()) if "data" in f else 0
            except Exception as error:  # noqa: BLE001
                return {"error": f"unreadable {path}: {error!r}"}
            if counts[name] != want:
                return {"error": f"{path} holds {counts[name]} demonstrations, expected {want}"}
        return {"attempts": len(checked["lines"]), "successes": successes, "failures": failures,
                "num_problematic": stats.get("num_problematic"), "time_hours": stats.get("time spent (hrs)"),
                "demo_hdf5": str(stats_path.parent / "demo.hdf5") if successes else None,
                "failed_hdf5": str(stats_path.parent / "demo_failed.hdf5") if expected["demo_failed.hdf5"] else None,
                "hdf5_counts": counts}
