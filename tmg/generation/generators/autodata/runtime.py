"""The ``_tmg`` callbacks for AutoData's generator (the DexMimicGen stack): the Isaac Lab runtime re-targeted at AutoData's
anchors, plus source mixing and the shared-path tracking loop on AutoData's classes.

AutoData (``autodata_core``) keeps the algorithm of Isaac Lab Mimic but not its code, so the grafts land elsewhere:

``DataGenerator.generate`` (async)

| anchor                                                            | grafted statement                                        |
|-------------------------------------------------------------------|----------------------------------------------------------|
| after ``env_id_tensor, initial_state = await self._reset_environment_for_generation(...)`` | ``_tmg.begin_attempt(self, env_id, success_term)`` (also resets the class-based success term's state for this env) |
| before return #0 (planning failure)                               | ``_tmg.end_attempt(self, env_id, {"success": False})``    |
| before return #1 (the final ``GenerationResult``)                 | ``_tmg.end_attempt(self, env_id, {"success": buffers.success})`` |

``DataGenerator.generate_eef_subtask_trajectory`` (the Isaac Lab Mimic builder name and ``src_subtask_object_pose`` anchor the
base runtime already names; only the record anchor and the hand variable differ)

| after ``src_subtask_object_pose = ...`` | ``src_subtask_eef_poses, src_subtask_target_poses, src_subtask_passthrough_actions = _tmg.mix_segment(self, locals())`` |
| after ``seq = WaypointSequence.from_poses(...)`` | ``_tmg.record_subtask(locals())`` |

``MultiWaypoint.execute`` (async): after ``play_action = datastream.target_eef_pose_to_action(...)`` ->
``_tmg.record_command(self, datastream, env_id, play_action)``; its final ``return bool(success_term...)`` is bound to
``success`` and ``_tmg.record_step(self, env, env_id, None, success)`` is inserted before it.

Pre-noise labels come from wrapping ``datastream.target_eef_pose_to_action`` for the attempt (noise-free call with the torch
RNG state saved and restored, then the official call), as the base runtime does on the environment.

Source mixing (``method.mixing.enabled``): ``mix_segment`` hands the mixer every source's segment of the subtask (eef pose at
the segment start, target poses, the hand's passthrough commands), the reference object's pose at the segment start and the
scene of the subtask (current object and in-hand poses, read through the datastream), as the Isaac Lab runtime does; the
GR1's two arms number their subtasks from 0 each, so the mixer and the records key them by (end effector, subtask); the mixed segment replaces the selected source's before AutoData transforms it to the current
object pose. Non-eef passthrough channels (a base command) of the selected source are kept when their length matches.

Images: Isaac Lab's recorder stores the ``policy`` observation group only, and the tasks keep their cameras in a
``camera_obs`` group (Arena's head camera; our Pick Place camera environment), so the generated hdf5 has no images. The
runtime therefore reads the camera frames itself from ``env.obs_buf`` right before each step (the observation the policy
will see), resizes them to the task's image size and, for a successful attempt, writes ``records/images_%06d.npz`` (one
``(T, H, W, 3)`` uint8 array per camera). ``tmg dataset`` reads them from there (``demo_layout()['images_from_records']``).
"""
from __future__ import annotations

import functools
import importlib
import inspect
from pathlib import Path

import numpy as np

from tmg.core.records import IMAGES_FORMAT, record_file
from tmg.generation.generators.isaaclab.runtime import (
    IsaacLabLayout,
    IsaacLabRuntime,
    _np,
    graft_builder,
    held_rule_of,
    installed_about,
)
from tmg.generation.graft import Graft
from tmg.generation.simulators.isaaclab import measured_eef_poses


def initialize_success_term(success_term, env):
    """Instantiate a class-based success term (Arena's ``TaskSuccessTerm``) once, with Isaac Lab's own helper when present.

    AutoData's script passes ``env_cfg.terminations.success`` straight to the generator and calls ``func(env, **params)``;
    for a class-based term that constructs the class with the wrong arguments (observed 2026-10-28). Isaac Lab 3.0's
    ``isaaclab_mimic.datagen.success_term.initialize_success_term`` is what its own annotation script uses.
    """
    try:
        from isaaclab_mimic.datagen.success_term import initialize_success_term as official

        return official(success_term, env)
    except ImportError:
        if inspect.isclass(success_term.func):
            success_term.func = success_term.func(cfg=success_term, env=env)
        return success_term


def reset_success_term(success_term, env_ids) -> None:
    """Reset the per-environment state of a class-based success term (no-op for plain functions)."""
    fn = getattr(success_term, "func", None)
    if fn is not None and not isinstance(fn, type) and hasattr(fn, "reset") and not callable(getattr(fn, "__func__", None)):
        fn.reset(env_ids=env_ids)


class AutoDataRuntime(IsaacLabRuntime):
    """Callbacks for ``autodata_core`` (see the module docstring): the actions are built by the datastream, hand commands are
    the eef's channel of a passthrough dict, subtask settings come from the task descriptor's subtask list."""

    record_anchor = "seq"
    hand_var = "src_subtask_passthrough_actions"
    boundary_var = "selected_boundary"

    def __init__(self, method, layout: IsaacLabLayout, records_dir: Path | None, cameras=(), image_size=None, held_rule=None):
        super().__init__(method, layout, records_dir, held_rule=held_rule)
        self._said_no_terms = False
        self.cameras = [str(c) for c in (cameras or [])]
        self.image_size = tuple(int(v) for v in image_size) if image_size else None
        self._frames: dict[str, list[np.ndarray]] = {}

    # package hooks -------------------------------------------------------------------------------
    def action_owner(self, generator):
        return generator.datastream

    def environment(self, generator):
        return generator.datastream.get_env()

    def segment_hand(self, local_vars: dict, eef_name: str) -> tuple:
        passthrough = dict(local_vars[self.hand_var])
        return passthrough[eef_name], passthrough

    def source_hand(self, info, eef_name: str, start: int, end: int):
        return info.passthrough_action[eef_name][start:end]

    def pack_mixed(self, mixed, eef, targets, grippers, unchanged, eef_name: str):
        """The mixed hand next to the other passthrough channels of the selected source (which must have the mixed length)."""
        extra = {k: v for k, v in unchanged.items() if k != eef_name}
        if any(len(v) != len(mixed.targets) for v in extra.values()):
            raise NotImplementedError("source mixing with non-eef passthrough channels of another length is not supported")
        new_eef, new_targets, new_hand = super().pack_mixed(mixed, eef, targets, grippers, unchanged, eef_name)
        return new_eef, new_targets, {eef_name: new_hand, **extra}

    def subtask_settings(self, generator, local_vars: dict, eef_name: str, subtask_ind: int) -> tuple[int, int, float]:
        spec = local_vars["subtasks"][subtask_ind]
        return (int(getattr(spec, "num_interpolation_steps", 0)), int(getattr(spec, "num_fixed_steps", 0)),
                float(getattr(spec, "action_noise", 0.0)))

    def subtask_term_signal(self, generator, local_vars: dict, eef_name: str, subtask_ind: int) -> str | None:
        signal = getattr(local_vars["subtasks"][subtask_ind], "subtask_term_signal", None)
        return None if signal is None else str(signal)

    def current_eef_pose(self, generator, env_id: int, eef_name: str) -> np.ndarray:
        return _np(generator.datastream.get_robot_eef_pose(env_ids=[env_id], eef_name=eef_name)[0])

    def current_object_poses(self, generator, env_id: int) -> dict:
        return {k: _np(v[0]) for k, v in generator.datastream.get_object_poses(env_ids=[env_id]).items()}

    def measured_poses(self, datastream, env_id: int, eefs) -> dict:
        """Measured end-effector poses: the configured links (``measure_links``) or the embodiment's observed eef poses."""
        if self.layout.measure_links:
            return measured_eef_poses(datastream.get_env(), env_id, self.layout.measure_links)
        return {eef: _np(datastream.get_robot_eef_pose(env_ids=[env_id], eef_name=eef)[0]) for eef in eefs}

    def _label_wrapper(self, original, env_id: int):
        runtime = self

        def with_labels(target_eef_pose_dict, passthrough_action_dict, action_noise_dict=None, env_id=env_id, **kw):
            return runtime._labelled_action(original, target_eef_pose_dict, passthrough_action_dict, action_noise_dict,
                                            env_id, kw)

        return with_labels

    # attempt boundaries --------------------------------------------------------------------------
    def begin_attempt(self, generator, env_id: int, success_term=None) -> None:
        super().begin_attempt(generator, env_id)
        self._frames = {cam: [] for cam in self.cameras}
        if success_term is not None:
            reset_success_term(success_term, [env_id])      # Arena's class-based term keeps progress state per env

    def end_attempt(self, generator, env_id: int, results: dict) -> None:
        super().end_attempt(generator, env_id, results)
        if self.records_dir is not None and results.get("success") and self._frames and all(self._frames.values()):
            np.savez_compressed(record_file(Path(self.records_dir), IMAGES_FORMAT, self.rec.attempt),
                                **{cam: np.stack(frames) for cam, frames in self._frames.items()})
        self._frames = {}

    def capture_frames(self, env, env_id: int) -> None:
        """Read the task's cameras from ``env.obs_buf`` (any observation group) for the current step."""
        if not self.cameras:
            return
        from tmg.data.images import resize_frames

        groups = getattr(env, "obs_buf", None) or {}
        for cam in self.cameras:
            group = next((g for g in groups.values() if isinstance(g, dict) and cam in g), None)
            if group is None:
                raise KeyError(f"camera {cam!r} is in no observation group ({list(groups)}); was the environment built with cameras?")
            frame = _np(group[cam][env_id])
            if frame.dtype != np.uint8:
                frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)
            self._frames[cam].append(resize_frames(frame, self.image_size) if self.image_size else frame)

    # per-step records ----------------------------------------------------------------------------
    def record_step(self, multi_waypoint, env, env_id: int, obs, success: bool) -> None:
        """AutoData's ``MultiWaypoint.execute`` hands no observation to the graft (``obs`` None), so the subtask
        termination signals of the new state are read from the environment: its ``subtask_terms`` observation group
        (``env.obs_buf``, what Isaac Lab Mimic's ``get_subtask_term_signals`` reads), else that method."""
        if obs is None:
            terms = subtask_terms(env, env_id)
            obs = {"subtask_terms": terms} if terms else None
            if not terms and not self._said_no_terms:
                self._said_no_terms = True
                print("[tmg] the environment publishes no per-step subtask signals (no subtask_terms observation group, "
                      "and get_subtask_term_signals is absent or not implemented): the records carry none, so the task's "
                      "fit must be declared with fit.reached: task_success", flush=True)
        super().record_step(multi_waypoint, env, env_id, obs, success)

    def record_command(self, multi_waypoint, datastream, env_id: int, play_action) -> None:
        self.capture_frames(datastream.get_env(), env_id)       # pre-step observation, paired with this step's action
        self._pend_command(multi_waypoint, self.measured_poses(datastream, env_id, list(multi_waypoint.waypoints)), play_action)


def subtask_terms(env, env_id: int) -> dict:
    """The subtask termination signals of the environment's current state, ``{name: value}``; empty when it has none: no
    ``subtask_terms`` observation group and a ``get_subtask_term_signals`` that is absent or raises NotImplementedError."""
    groups = getattr(env, "obs_buf", None)
    if isinstance(groups, dict) and isinstance(groups.get("subtask_terms"), dict) and groups["subtask_terms"]:
        return dict(groups["subtask_terms"])
    getter = getattr(env, "get_subtask_term_signals", None)
    if callable(getter):
        try:
            signals = getter(env_ids=[env_id])
        except NotImplementedError:       # the Mimic base class declares the method; an env without signals leaves it so
            return {}
        return {k: (v[0] if hasattr(v, "__getitem__") and getattr(v, "ndim", 0) > 0 else v) for k, v in dict(signals or {}).items()}
    return {}


def install(runtime: AutoDataRuntime, graft_hashes: dict) -> dict:
    """Graft the recording and mixing callbacks into ``autodata_core`` (anchors in the module docstring)."""
    dg = importlib.import_module("autodata_core.data_generator")
    wp = importlib.import_module("autodata_core.waypoint")
    DataGenerator, MultiWaypoint = dg.DataGenerator, wp.MultiWaypoint
    if runtime.method.mixer is not None:
        runtime.bind_official(DataGenerator)
    gen = Graft(DataGenerator.generate, expected_hash=graft_hashes.get("generate"), label="DataGenerator.generate")
    gen.insert_after_assign("initial_state", "_tmg.begin_attempt(self, env_id, success_term)")
    gen.insert_before_return('_tmg.end_attempt(self, env_id, {"success": False})', occurrence=0, expected_returns=2)
    gen.insert_before_return('_tmg.end_attempt(self, env_id, {"success": buffers.success})', occurrence=1, expected_returns=2)
    builder = graft_builder(DataGenerator, runtime, graft_hashes.get("trajectory"))
    exe = Graft(MultiWaypoint.execute, expected_hash=graft_hashes.get("execute"), label="MultiWaypoint.execute")
    exe.insert_after_assign("play_action", "_tmg.record_command(self, datastream, env_id, play_action)", occurrence=0)
    exe.name_final_return("success")
    exe.insert_before_final_return("_tmg.record_step(self, env, env_id, None, success)")
    DataGenerator.generate = gen.compile({"_tmg": runtime})
    setattr(DataGenerator, runtime.builder_name, builder.compile({"_tmg": runtime}))
    MultiWaypoint.execute = exe.compile({"_tmg": runtime})
    return {"generate": gen.edits, runtime.builder_name: builder.edits, "execute": exe.edits,
            **installed_about(runtime, {"generate": gen.hash, "trajectory": builder.hash, "execute": exe.hash}, graft_hashes)}


def install_from_spec(spec: dict, namespace: dict | None = None) -> dict:
    """Entry used by the chunk entrypoint: build the runtime from the chunk spec, install the grafts, and wrap the script's ``setup_async_generation`` so the success term is instantiated with the
    environment before generation starts."""
    from tmg.generation.method import build_method

    if namespace is not None and callable(namespace.get("setup_async_generation")):
        original = namespace["setup_async_generation"]

        @functools.wraps(original)
        def setup_async_generation(*args, **kwargs):
            datastream = kwargs.get("datastream", args[0] if args else None)
            term = kwargs.get("success_term", args[2] if len(args) > 2 else None)
            if datastream is not None and term is not None:
                initialize_success_term(term, datastream.get_env())
                print(f"[tmg] success term ready: {type(term.func).__name__}", flush=True)
            return original(*args, **kwargs)

        namespace["setup_async_generation"] = setup_async_generation

    layout = IsaacLabLayout.from_spec(spec["extra"]["layout"])
    method = build_method(spec["method"], chunk_seed=int(spec["chunk"]["seed"]), arms=layout.eefs)
    runtime = AutoDataRuntime(method, layout, records_dir=Path(spec["records_dir"]), held_rule=held_rule_of(spec),
                              cameras=spec["extra"]["cameras"] if "cameras" in spec["extra"] else (spec["task"].get("cameras") or []),
                              image_size=spec["extra"].get("image_size") or spec["task"].get("image_size"))
    info = install(runtime, dict(spec["extra"].get("graft_hashes") or {}))
    return info
