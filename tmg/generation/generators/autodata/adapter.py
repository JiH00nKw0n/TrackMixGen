"""``SyntheticDataGenerator`` of the DexMimicGen generator: NVIDIA AutoData (isaac-sim/AutoData) on Isaac Lab-Arena.

AutoData is a rewrite of Isaac Lab Mimic's generator (same algorithm, its own ``autodata_core`` package, no dependency
on ``isaaclab_mimic``). Since 2026-10-29 it runs every GR1 task of the paper plan: Pick Place (Isaac Lab Mimic's own
environment, registered by the nested Isaac Lab 3.0's ``isaaclab_mimic``) and the two Arena tasks (Microwave Opening,
Bottle Placement and Door Closing, registered through Arena's callback). A chunk runs the official
``scripts/generate_dataset.py`` of the AutoData checkout with one environment per process. What differs from the
SoftMimicGen (Isaac Lab fork) adapter, each a hook of ``IsaacLabGenerator.prepare_chunk``:

- The generation settings live in a task descriptor yaml (``autodata_examples/tasks/<task>.yaml``), not in the
  environment config. The chunk gets a copy of the official descriptor with three fields changed (``chunk_inputs``):
  ``generation_policy.seed`` (the chunk seed), ``num_trials`` (the chunk's count) and ``guarantee_success`` (from
  ``generation.protocol.guarantee``: true in the full pools, so ``num_trials`` counts successes; false in the test pilots
  and the ranking pools, so it counts attempts). Every other value (source selection per subtask,
  nearest-neighbour selection, action noise, interpolation steps) is the official one.
- Environment registration: a task with ``external_callback`` (the Arena tasks) passes it to the script with the provider
  options after ``--`` (``provider_args``); a task without it (Pick Place) lists the modules to import before the
  environment is built (``register_modules``, plus ``camera_module`` for our camera environment ``camera_env``).
- The grafts target AutoData's anchors (``runtime.py`` of this package); the chunk process is the Isaac Lab
  entrypoint with ``runtime_module`` set to that module. Source mixing and path tracking are wired there too.
- Cameras: Isaac Lab's recorder stores the ``policy`` observation group only. Arena keeps its head camera
  (``robot_pov_cam_rgb``, 512x512) in a ``camera_obs`` group and our Pick Place camera environment puts its two 128x128
  cameras in a ``camera_obs`` group as well, so no generated hdf5 carries images; the runtime captures the task's cameras
  per step from ``env.obs_buf``, resizes them to ``image_size`` and writes ``records/images_%06d.npz`` for successful
  attempts; ``tmg dataset`` reads the images from there.

Outputs of a chunk are the same as the Isaac Lab adapter's (``generated.hdf5``, ``records/``, ``result.json`` from
AutoData itself).
"""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from tmg.core.chunks import Chunk
from tmg.generation.generators.isaaclab.adapter import GENERATED, IsaacLabGenerator

RUNTIME_MODULE = "tmg.generation.generators.autodata.runtime"
DESCRIPTOR = "task_descriptor.yaml"


def chunk_descriptor(official: dict, *, seed: int, num_trials: int, guarantee: bool = True) -> dict:
    """The official task descriptor with the chunk's seed and trial count; ``guarantee`` (generation.protocol.guarantee)
    decides whether ``num_trials`` counts successes (true, the full pools) or attempts (false, test pilots and ranking pools)."""
    out = json.loads(json.dumps(official))          # yaml anchors already resolved by the loader; deep copy
    policy = dict(out.get("generation_policy") or {})
    policy["seed"] = int(seed)
    policy["num_trials"] = int(num_trials)
    policy["guarantee_success"] = bool(guarantee)
    out["generation_policy"] = policy
    return out


class DexMimicGenGenerator(IsaacLabGenerator):
    """DexMimicGen through AutoData: GR1 Pick Place (Isaac Lab Mimic environment) and the Arena GR1 tasks."""

    name = "dexmimicgen"
    package = "autodata_core"
    soft_objects = False
    runtime_module = RUNTIME_MODULE

    def chunk_inputs(self, chunk: Chunk, folder: Path) -> dict:
        """Write the chunk's task descriptor and choose the cameras and the environment.

        Returns ``descriptor`` (the written dict), ``cameras`` (the names captured per step) and ``env_name`` (the camera
        environment when cameras are wanted, else the official one).
        """
        task, g = self.task, self.generator_cfg
        root = Path(str(g["source_root"]))
        official = yaml.safe_load((root / str(task["task_descriptor"])).read_text())
        descriptor = chunk_descriptor(official, seed=chunk.seed, num_trials=chunk.attempts,
                                      guarantee=bool(self.protocol.get("guarantee", True)))
        (folder / DESCRIPTOR).write_text(yaml.safe_dump(descriptor, sort_keys=False))
        # generation.protocol.camera_names: null = the task's cameras, a list = those cameras ([] = none, the ranking pools).
        # Without cameras the official environment is used instead of our camera environment (no per-step rendering).
        names = self.protocol.get("camera_names")
        cameras = list(task.get("cameras") or []) if names is None else [str(c) for c in names]
        unknown = [c for c in cameras if c not in list(task.get("cameras") or [])]
        if unknown:
            raise ValueError(f"generation.protocol.camera_names {unknown} are not cameras of task {task['name']} ({task.get('cameras')})")
        env_name = str(task.get("camera_env") or task["env_name"]) if cameras else str(task["env_name"])
        return {"descriptor": descriptor, "cameras": cameras, "env_name": env_name}

    def launcher_args(self, chunk: Chunk, folder: Path, inputs: dict) -> list[str]:
        g = self.generator_cfg
        root = Path(str(g["source_root"]))
        args = []
        if self.task.get("external_callback"):
            args += ["--external_callback", str(self.task["external_callback"])]
        args += ["--env_name", inputs["env_name"],
                 "--alg", str(inputs["descriptor"].get("algo", "dexmimicgen")), "--task_descriptor", str(folder / DESCRIPTOR),
                 "--embodiment", str(root / str(g["embodiment"])), "--input_file", str(self.source_path()),
                 "--output_file", str(folder / GENERATED), "--result_file", str(folder / "result.json"),
                 "--generation_num_trials", str(chunk.attempts), "--num_envs", "1", "--device", str(self.cluster_device()),
                 "--viz", "none", "--enable_cameras"]       # Isaac Lab 3.0's AppLauncher has --viz none instead of --headless
        return args

    def provider_args(self, chunk: Chunk, inputs: dict) -> list[str]:
        """Provider options after ``--`` reach the Arena registration callback (its own seed and embodiment)."""
        if not self.task.get("external_callback"):
            return []
        return ["--", "--seed", str(chunk.seed), *[str(a) for a in (self.task.get("env_args") or [])]]

    def task_extra(self, inputs: dict) -> dict:
        task, cameras = self.task, inputs["cameras"]
        return {"camera_module": task.get("camera_module") if cameras else None,
                "register_modules": list(task.get("register_modules") or []), "env_name": inputs["env_name"],
                "cameras": cameras, "image_size": list(task.get("image_size") or [])}

    def generator_config_text(self, inputs: dict) -> str:
        return yaml.safe_dump(inputs["descriptor"], sort_keys=False)

    def demo_layout(self) -> dict:
        """As the Isaac Lab adapter, but the images come from our records (no generated hdf5 of this stack carries them)."""
        return {**super().demo_layout(), "images_from_records": True}
