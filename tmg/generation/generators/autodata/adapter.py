"""``SyntheticDataGenerator`` for NVIDIA AutoData (isaac-sim/AutoData): DexMimicGen on Isaac Lab-Arena tasks.

AutoData is a rewrite of Isaac Lab Mimic's generator (same algorithm, its own ``autodata_core`` package, no dependency
on ``isaaclab_mimic``). A chunk runs the official ``scripts/generate_dataset.py`` of the AutoData checkout with the
Arena environment registered through ``--external_callback``, exactly as AutoData's own Arena test does, and one
environment per process. What differs from the Isaac Lab adapter:

- The generation settings live in a task descriptor yaml (``autodata_examples/tasks/<task>.yaml``), not in the
  environment config. The chunk gets a copy of the official descriptor with three fields changed:
  ``generation_policy.seed`` (the chunk seed), ``num_trials`` (the chunk's count) and ``guarantee_success: true``
  (so ``num_trials`` counts successes, as the Isaac Lab tasks do). Every other value (source selection per subtask,
  nearest-neighbour selection, action noise, interpolation steps) is the official one.
- The grafts target AutoData's anchors (``runtime.py`` of this package); the chunk process is the Isaac Lab
  entrypoint with ``runtime_module`` set to that module.
- Cameras: the Arena GR1 head camera (``robot_pov_cam_rgb``, 512x512) lives in a ``camera_obs`` group that Isaac Lab's
  recorder does not store, so the runtime captures it per step from ``env.obs_buf``, resizes it to ``image_size`` and
  writes ``records/images_%06d.npz`` for successful attempts; ``tmg dataset`` reads the images from there.

Outputs of a chunk are the same as the Isaac Lab adapter's (``generated.hdf5``, ``records/``, ``result.json`` from
AutoData itself).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

from tmg.core.chunks import Chunk
from tmg.core.records import ATTEMPTS_JSONL, RECORDS_DIR
from tmg.generation.base import ChunkSpec
from tmg.generation.generators.isaaclab.adapter import ENTRYPOINT, GENERATED, IsaacLabGenerator

RUNTIME_MODULE = "tmg.generation.generators.autodata.runtime"
DESCRIPTOR = "task_descriptor.yaml"


def chunk_descriptor(official: dict, *, seed: int, num_trials: int) -> dict:
    """The official task descriptor with the chunk's seed and trial count, successes guaranteed (see the module docstring)."""
    out = json.loads(json.dumps(official))          # yaml anchors already resolved by the loader; deep copy
    policy = dict(out.get("generation_policy") or {})
    policy["seed"] = int(seed)
    policy["num_trials"] = int(num_trials)
    policy["guarantee_success"] = True
    out["generation_policy"] = policy
    return out


class AutoDataGenerator(IsaacLabGenerator):
    """AutoData's DexMimicGen on the Arena GR1 tasks (microwave opening, bottle placement and door closing)."""

    name = "autodata"
    package = "autodata_core"
    soft_objects = False

    def prepare_chunk(self, chunk: Chunk, folder: Path) -> ChunkSpec:
        self.require_generation_config()
        folder.mkdir(parents=True, exist_ok=True)
        task, g = self.task, self.generator_cfg
        root = Path(str(g["source_root"]))
        official = yaml.safe_load((root / str(task["task_descriptor"])).read_text())
        descriptor = chunk_descriptor(official, seed=chunk.seed, num_trials=chunk.attempts)
        (folder / DESCRIPTOR).write_text(yaml.safe_dump(descriptor, sort_keys=False))
        args = ["--external_callback", str(g["external_callback"]), "--env_name", str(task["env_name"]),
                "--alg", str(descriptor.get("algo", "dexmimicgen")), "--task_descriptor", str(folder / DESCRIPTOR),
                "--embodiment", str(root / str(g["embodiment"])), "--input_file", str(self.source_path()),
                "--output_file", str(folder / GENERATED), "--result_file", str(folder / "result.json"),
                "--generation_num_trials", str(chunk.attempts), "--num_envs", "1", "--device", str(self.cluster_device()),
                "--viz", "none", "--enable_cameras"]       # Isaac Lab 3.0's AppLauncher has --viz none instead of --headless
        args += [str(a) for a in (g.get("extra_args") or [])]
        if g.get("kit_args"):
            args += ["--kit_args", str(g["kit_args"])]
        # provider options after "--" reach the Arena registration callback (its own seed and embodiment)
        args += ["--", "--seed", str(chunk.seed), *[str(a) for a in (task.get("env_args") or [])]]
        spec = ChunkSpec(
            chunk={"index": chunk.index, "seed": chunk.seed, "attempts": chunk.attempts}, generator=g["name"],
            task=task, scene=self.scene_dict(), method=self.method,
            protocol=self.protocol, generator_config=yaml.safe_dump(descriptor, sort_keys=False), folder=str(folder),
            records_dir=str(folder / RECORDS_DIR), expected_outputs=[str(folder / RECORDS_DIR / ATTEMPTS_JSONL)],
            extra={"package": self.package, "soft": False, "runtime_module": RUNTIME_MODULE,
                   "script": str(root / str(g["script"])), "script_args": args, "layout": self.layout_dict(),
                   "graft_hashes": dict(g.get("graft_hashes") or {}), "camera_module": None, "env_name": task["env_name"],
                   "cameras": list(task.get("cameras") or []), "image_size": list(task.get("image_size") or [])})
        (folder / "chunk_args.json").write_text(json.dumps(args, indent=1))
        return spec

    def chunk_command(self, spec: ChunkSpec, spec_path: Path) -> list[str]:
        return [sys.executable, "-m", ENTRYPOINT, "--spec", str(spec_path)]

    def demo_layout(self) -> dict:
        """As the Isaac Lab adapter, but the images come from our records (Isaac Lab's recorder keeps no camera group)."""
        return {**super().demo_layout(), "images_from_records": True}
