"""``SyntheticDataGenerator`` for the Isaac Lab mimic generator SoftMimicGen (and the base of the AutoData adapter).

A chunk is one run of the official ``generate_dataset.py`` script of the generator's Isaac Lab checkout
(``generator.script`` under ``generator.source_root``) with ``--generation_num_trials <attempts>`` and one
environment per process. The chunk process (``entrypoint.py``) starts Isaac Sim through the script's own
``AppLauncher``, applies the scene ranges to the environment config, installs the grafts (``runtime.py``) and
lets the official script run. Outputs of a chunk::

    <chunk>/generated.hdf5          the official recorder's episodes (successes; Soft also writes *_failed.hdf5)
    <chunk>/records/attempts.jsonl, attempt_%06d.npz, labels_%06d.npz   our records (every attempt)
    <chunk>/run.log

The source demonstrations are the released annotated datasets (``sources.py``). ``chunk_result`` validates the
records against the attempt count and counts the exported episodes.

``prepare_chunk`` is a template: the official script's arguments come from ``chunk_inputs`` (files written and choices made
first), ``launcher_args`` (the script's own options), ``generator_args`` (``extra_args`` and ``kit_args`` of the generator
config) and ``provider_args`` (options after ``--``); ``task_extra`` and ``generator_config_text`` fill the spec. The AutoData
adapter overrides the hooks only.

Success is never computed by tmg: the grafts read the official ``success`` value inside ``MultiWaypoint.execute``.
SoftMimicGen's rope term fixes the rope end nodes at its first call of an episode, so calling it at reset (before the
first action) would change the verdict; see ``docs/GENERATORS.md`` and ``task.success_term``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py

from tmg.core.chunks import Chunk
from tmg.core.records import ATTEMPTS_JSONL, RECORDS_DIR, read_attempts, validate_records
from tmg.generation.base import ChunkSpec, SyntheticDataGenerator
from tmg.generation.simulators.isaaclab import validate_isaaclab_scene
from tmg.generation.simulators.scenes import load_scene_file

ENTRYPOINT = "tmg.generation.generators.isaaclab.entrypoint"
GENERATED = "generated.hdf5"


class IsaacLabGenerator(SyntheticDataGenerator):
    """Common part of the Isaac Lab adapters (SoftMimicGen here, the AutoData stack in ``..autodata.adapter``)."""

    simulator = "isaaclab"
    package = ""                       # official datagen package, e.g. "isaaclab_mimic.datagen"
    soft_objects = False               # trajectory builder is Soft's ``generate_trajectory``
    runtime_module: str | None = None  # module whose ``install_from_spec`` grafts the package (None: the Isaac Lab runtime)

    def scene_dict(self) -> dict:
        """The scene config of this generator (ranges per reset event), validated."""
        scene = dict(self.scene)
        if "override_bounds" not in scene:
            scene = load_scene_file(str(scene["name"]), self.generator_cfg["name"])
        validate_isaaclab_scene(scene)
        return scene

    def fetch_sources(self) -> dict:
        from tmg.generation.generators.isaaclab.sources import fetch

        return fetch(self.task, self.generator_cfg)

    def layout_dict(self) -> dict:
        """The ``IsaacLabLayout`` fields from the generator config (serialized into the chunk spec)."""
        g = self.generator_cfg
        return {"eefs": list(g["eefs"]), "arm_action_dims": {k: list(v) for k, v in g["arm_action_dims"].items()},
                "hand_prefixes": g.get("hand_prefixes"), "hand_term": g.get("hand_term"),
                "measure_links": g.get("measure_links"), "noise_draw_shapes": list(g.get("noise_draw_shapes") or [])}

    def script_path(self) -> str:
        """The official script: ``generator.script`` under ``generator.source_root``."""
        return str(Path(str(self.generator_cfg["source_root"])) / str(self.generator_cfg["script"]))

    # prepare_chunk and its hooks ----------------------------------------------------------------------
    def prepare_chunk(self, chunk: Chunk, folder: Path) -> ChunkSpec:
        """Describe the chunk: the official script's arguments and everything the entrypoint needs (hooks in the module
        docstring)."""
        self.require_generation_config()
        folder.mkdir(parents=True, exist_ok=True)
        inputs = self.chunk_inputs(chunk, folder)
        args = self.launcher_args(chunk, folder, inputs) + self.generator_args() + self.provider_args(chunk, inputs)
        extra = {"package": self.package, "soft": self.soft_objects,
                 **({"runtime_module": self.runtime_module} if self.runtime_module else {}),
                 "script": self.script_path(), "script_args": args, "layout": self.layout_dict(),
                 "graft_hashes": dict(self.generator_cfg.get("graft_hashes") or {}), **self.task_extra(inputs)}
        spec = ChunkSpec(
            chunk={"index": chunk.index, "seed": chunk.seed, "attempts": chunk.attempts}, generator=self.generator_cfg["name"],
            task=self.task, scene=self.scene_dict(), method=self.method,
            protocol=self.protocol, generator_config=self.generator_config_text(inputs), folder=str(folder),
            records_dir=str(folder / RECORDS_DIR), expected_outputs=[str(folder / RECORDS_DIR / ATTEMPTS_JSONL)], extra=extra)
        (folder / "chunk_args.json").write_text(json.dumps(args, indent=1))
        return spec

    def chunk_inputs(self, chunk: Chunk, folder: Path) -> dict:
        """Files written into ``folder`` and choices made before the arguments are built (nothing for SoftMimicGen)."""
        return {}

    def launcher_args(self, chunk: Chunk, folder: Path, inputs: dict) -> list[str]:
        """The official script's own options."""
        task = self.task
        args = ["--headless", "--enable_cameras", "--device", str(self.cluster_device()), "--num_envs", "1",
                "--generation_num_trials", str(chunk.attempts), "--input_file", str(self.source_path()),
                "--output_file", str(folder / GENERATED)]
        # the official scripts have no --seed: the chunk seed is written into env_cfg.datagen_config.seed by the
        # entrypoint's setup_env_config wrapper (the value the script seeds random, numpy and torch with)
        if task.get("camera_env"):
            args += ["--task", str(task["camera_env"])]
        return args

    def generator_args(self) -> list[str]:
        """``generator.extra_args`` and ``generator.kit_args`` (Isaac Sim kit settings) of the generator config."""
        args = [str(a) for a in (self.generator_cfg.get("extra_args") or [])]
        kit = self.generator_cfg.get("kit_args")
        if kit:
            args += ["--kit_args", str(kit)]
        return args

    def provider_args(self, chunk: Chunk, inputs: dict) -> list[str]:
        """Options after ``--`` for an environment provider (none for SoftMimicGen)."""
        return []

    def task_extra(self, inputs: dict) -> dict:
        """Task fields of ``spec.extra`` the entrypoint reads (camera environment module, environment name)."""
        return {"camera_module": self.task.get("camera_module"), "env_name": self.task.get("env_name")}

    def generator_config_text(self, inputs: dict) -> str:
        """``ChunkSpec.generator_config``: the official generator's own config file (none for the Isaac Lab scripts)."""
        return ""

    def cluster_device(self) -> str:
        """``cluster.device`` (the GPU Isaac Sim renders and simulates on), default ``cuda:0``."""
        return str((self.cfg.get("cluster") or {}).get("device", "cuda:0"))

    # chunk process and results --------------------------------------------------------------------------
    def chunk_command(self, spec: ChunkSpec, spec_path: Path) -> list[str]:
        return [sys.executable, "-m", ENTRYPOINT, "--spec", str(spec_path)]

    def chunk_result(self, spec: ChunkSpec) -> dict | None:
        """Validate the records and count the exported episodes; None when the chunk did not finish."""
        folder = Path(spec.folder)
        records = read_attempts(folder / RECORDS_DIR)
        attempts = int(spec.chunk["attempts"])
        if len(records) < attempts:
            return None
        validate_records(folder / RECORDS_DIR, attempts)
        exported = 0
        generated = folder / GENERATED
        if generated.exists():
            with h5py.File(generated, "r") as f:
                exported = len(f["data"].keys()) if "data" in f else 0
        successes = sum(1 for r in records if r.get("success"))
        return {"attempts": len(records), "successes": successes, "exported_episodes": exported,
                "generated_hdf5": str(generated) if generated.exists() else None}

    def demo_layout(self) -> dict:
        """How ``tmg dataset`` reads this generator's chunk outputs (see ``tmg.data.dataset.DemoLayout``)."""
        return {"demo_glob": GENERATED, "image_key": "{cam}", "actions_key": "actions", "states_key": None,
                "labels_from_records": True, "success_only_file": True}


class SoftMimicGenGenerator(IsaacLabGenerator):
    """SoftMimicGen (NVlabs/SoftMimicGen on its Isaac Lab fork): single Franka arm, soft objects."""

    name = "softmimicgen"
    package = "softmimicgen.datagen"
    soft_objects = True
