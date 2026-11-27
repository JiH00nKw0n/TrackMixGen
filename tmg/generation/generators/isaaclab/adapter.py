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

``chunk_result`` also checks what the SkillMix fit and the mixing need of the records, so that a problem shows at the first
chunk and not days later at the fit (``records_check``). Every chunk result carries the chunk's sanity counts
(``fit_records.sanity_report``: attempts whose subtask scene could not be read, successful attempts that count as not
having reached a subtask, how often the carried object is held, how often a mix record left the segment unmixed) and
the settings the donors of a deformable subtask were registered with. The verdict is never one chunk's share against a
limit, which would stop a healthy pool on a few unlucky attempts: the counts of the pool's finished chunks are added up
(``merge_sanity``) and judged with the limits and confidence of the generator's ``record_checks``
(``fit_records.judge_sanity``). A failure of every attempt is beyond any limit in the first chunk, so that still stops
at once. In a pool that needs the scenes (``needs_subtask_scenes``: ``generation.purpose: calibration`` or a mixing
method) an error fails the chunk and stops the pool; any other pool keeps its chunk and warns. The share of successful
attempts that count as not reached is an error only for a calibration pool (the fit reads those verdicts); elsewhere
it is a warning.

Success is never computed by tmg: the grafts read the official ``success`` value inside ``MultiWaypoint.execute``.
SoftMimicGen's rope term fixes the rope end nodes at its first call of an episode, so calling it at reset (before the
first action) would change the verdict; see ``docs/GENERATORS.md`` and ``task.success_term``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py

from tmg.core.chunks import CHUNK_DONE, Chunk, read_json
from tmg.core.records import ATTEMPTS_JSONL, RECORDS_DIR, read_attempts, validate_records
from tmg.generation.base import CALIBRATION, ChunkSpec, SyntheticDataGenerator
from tmg.generation.method.fit_layout import FitLayout, SanityLimits
from tmg.generation.method.fit_records import Attempts, judge_sanity, merge_sanity, print_sanity, sanity_report, scene_errors
from tmg.generation.method.interfaces import Arms, HeldRule
from tmg.generation.method.success_model import FEATURES, HELD_FEATURES
from tmg.generation.simulators.isaaclab import validate_isaaclab_scene
from tmg.generation.simulators.scenes import load_scene_file

ENTRYPOINT = "tmg.generation.generators.isaaclab.entrypoint"
GENERATED = "generated.hdf5"


class IsaacLabGenerator(SyntheticDataGenerator):
    """Common part of the Isaac Lab adapters (SoftMimicGen here, the AutoData stack in ``..autodata.adapter``)."""

    package = ""                       # official datagen package, e.g. "isaaclab_mimic.datagen"
    soft_objects = False               # trajectory builder is Soft's ``generate_trajectory``
    runtime_module: str | None = None  # module whose ``install_from_spec`` grafts the package (None: the Isaac Lab runtime)
    default_reached = "term_window"    # reached rule of the fit layout unless the task declares another (task.fit.reached)

    # scene reading and the fit layout -------------------------------------------------------------------
    def scene_reading(self) -> dict:
        """``generator.scene_reading``: ``held_rule`` (the hold test of the runtime), and what records without a record
        version were made with: ``unversioned_held_rule`` and ``unversioned_donor_registration``."""
        return dict(self.generator_cfg.get("scene_reading") or {})

    def held_rule(self) -> HeldRule | None:
        return HeldRule.from_mapping(self.scene_reading().get("held_rule"))

    def fit_layout(self) -> FitLayout:
        """The arms of the generator config (``eefs``, the ones the runtime builds the mixer with); the selected source's
        poses are in the records; the reached rule is the task's ``fit.reached`` (declared for a task whose environment
        publishes no per-step subtask signals) or ``default_reached``; with a hold rule the features include the hold
        mismatch."""
        reading = self.scene_reading()
        fit = dict(self.task.get("fit") or {})
        return FitLayout(arms=Arms.of(self.generator_cfg["eefs"]), reached=str(fit.get("reached") or self.default_reached),
                         source_poses="records", features=HELD_FEATURES if self.held_rule() is not None else FEATURES,
                         unversioned_held_rule=HeldRule.from_mapping(reading.get("unversioned_held_rule")),
                         unversioned_donor_registration=reading.get("unversioned_donor_registration"),
                         sanity=SanityLimits.from_mapping(self.generator_cfg.get("record_checks")))

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
            records_dir=str(folder / RECORDS_DIR), expected_outputs=[str(folder / RECORDS_DIR / ATTEMPTS_JSONL)], extra=extra,
            scene_reading={"held_rule": None if self.held_rule() is None else self.held_rule().as_dict()})
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
        result = {"attempts": len(records), "successes": successes, "exported_episodes": exported,
                  "generated_hdf5": str(generated) if generated.exists() else None}
        result.update(self.records_check(records, folder))
        return result

    def records_check(self, records: list, folder: Path) -> dict:
        """What the fit and the mixing need of a chunk's records (module docstring): ``sanity`` (this chunk's counts),
        ``sanity_verdict`` (errors and warnings of the pool's finished chunks so far, this one included),
        ``donor_registration`` (the settings the donors of deformable subtasks were registered with) and, when the pool
        needs the scenes and the verdict has errors, ``error`` and ``stop_pool``."""
        layout = self.fit_layout()
        records_dir = folder / RECORDS_DIR
        sanity = sanity_report(Attempts(list(records), [records_dir] * len(records), [0] * len(records)), layout, self.task)
        earlier = finished_sanity(folder)
        pool = merge_sanity([*earlier, sanity])
        errors, warnings = judge_sanity(pool, layout.sanity, unreached_is_error=self.generation.get("purpose") == CALIBRATION)
        label = f"pool up to chunk {folder.name} ({len(earlier) + 1} chunks)"
        print_sanity(pool, label, errors, warnings)
        out = {"sanity": sanity, "sanity_verdict": {"chunks": len(earlier) + 1, "errors": errors, "warnings": warnings}}
        if sanity["without_scene"]:
            examples = scene_errors(records)
            out.update(attempts_without_scene=sanity["without_scene"], invalid_attempts=sum(1 for r in records if r.get("invalid")),
                       scene_error_examples=examples[:3])
        registrations = donor_registrations(records)
        if registrations:
            out["donor_registration"] = registrations
        if errors and self.needs_subtask_scenes():
            # the pool's later chunks would add to it: fail the chunk and stop the pool now instead of days later at the fit
            out.update(error=f"{label}: " + "; ".join(errors) + ": the SkillMix fit and mixing cannot use this pool", stop_pool=True)
        return out

    def demo_layout(self) -> dict:
        """How ``tmg dataset`` reads this generator's chunk outputs (see ``tmg.data.dataset.DemoLayout``)."""
        # The recorder keeps no flat state vector; its scene state is the nested initial_state/ and states/ groups
        # (articulation, rigid_object, deformable_object or deformable / <asset> / <field>), copied whole for replay.
        return {"demo_glob": GENERATED, "image_key": "{cam}", "actions_key": "actions", "states_key": None,
                "labels_from_records": True, "success_only_file": True, "replay_groups": ["initial_state", "states"],
                "per_step_groups": ["states"]}


def finished_sanity(folder: Path) -> list[dict]:
    """The sanity counts of the pool's other finished chunks (the ``sanity`` of their ``CHUNK_DONE.json``; a chunk
    finished before the counts were recorded has none)."""
    out = []
    for done in sorted(Path(folder).parent.glob(f"*/{CHUNK_DONE}")):
        if done.parent.resolve() == Path(folder).resolve():
            continue
        sanity = read_json(done).get("sanity")
        if sanity:
            out.append(sanity)
    return out


def donor_registrations(records) -> list[dict]:
    """The distinct settings the donors of deformable subtasks were registered with (mix records' ``registration``)."""
    seen = []
    for r in records:
        for s in r.get("subtasks") or []:
            settings = (s.get("mix") or {}).get("registration")
            if settings is not None and settings not in seen:
                seen.append(settings)
    return seen


class SoftMimicGenGenerator(IsaacLabGenerator):
    """SoftMimicGen (NVlabs/SoftMimicGen on its Isaac Lab fork): single Franka arm, soft objects."""

    name = "softmimicgen"
    package = "softmimicgen.datagen"
    soft_objects = True
