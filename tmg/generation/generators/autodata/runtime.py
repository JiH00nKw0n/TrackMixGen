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
the segment start, target poses, the hand's passthrough commands) and the reference object's pose at the segment start, as
the Isaac Lab runtime does; the mixed segment replaces the selected source's before AutoData transforms it to the current
object pose. Non-eef passthrough channels (a base command) of the selected source are kept when their length matches.

Tracking (``method.tracking.enabled``): ``make_tracking_generate`` replaces ``DataGenerator.generate`` with the shared-path
loop of the Isaac Lab runtime (``isaaclab.tracking_loop.TrackingLoop``: one ``ErrorAdaptiveFollower`` per source-time
interval over both hands, errors measured on the configured links, official noise draws kept per base waypoint by a
``NoiseLedger``), with ``AutoDataTrackingLoop`` filling the hooks: AutoData's ``generate_eef_subtask_trajectory`` /
``merge_eef_subtask_trajectory`` / ``MultiWaypoint.execute``, waypoints whose hand command is the eef's entry of the
``passthrough_action`` dict, and AutoData's recorder lifecycle (``recorder_manager.set_success_to_episodes`` and
``export_episodes``). It needs one shared source demonstration per attempt (``generation_policy.select_src_per_subtask`` and
``select_src_per_arm`` false) and a task without coordination constraints or navigation; otherwise it raises
``NotImplementedError`` before the first attempt.

Images: Isaac Lab's recorder stores the ``policy`` observation group only, and the tasks keep their cameras in a
``camera_obs`` group (Arena's head camera; our Pick Place camera environment), so the generated hdf5 has no images. The
runtime therefore reads the camera frames itself from ``env.obs_buf`` right before each step (the observation the policy
will see), resizes them to the task's image size and, for a successful attempt, writes ``records/images_%06d.npz`` (one
``(T, H, W, 3)`` uint8 array per camera). ``tmg dataset`` reads them from there (``demo_layout()['images_from_records']``).
"""
from __future__ import annotations

import copy
import functools
import importlib
import inspect
from collections.abc import Callable
from pathlib import Path

import numpy as np

from tmg.core.records import IMAGES_FORMAT, record_file
from tmg.generation.generators.isaaclab.runtime import (
    Attempt,
    Interval,
    IsaacLabLayout,
    IsaacLabRuntime,
    TrackingLoop,
    _np,
    cut_interval,
    graft_builder,
)
from tmg.generation.graft import Graft
from tmg.generation.seeds import private_noise_seed
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

    def __init__(self, method, layout: IsaacLabLayout, records_dir: Path | None, cameras=(), image_size=None):
        super().__init__(method, layout, records_dir)
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
    def record_command(self, multi_waypoint, datastream, env_id: int, play_action) -> None:
        self.capture_frames(datastream.get_env(), env_id)       # pre-step observation, paired with this step's action
        self._pend_command(multi_waypoint, self.measured_poses(datastream, env_id, list(multi_waypoint.waypoints)), play_action)


def install(runtime: AutoDataRuntime, graft_hashes: dict) -> dict:
    """Graft the recording and mixing callbacks into ``autodata_core`` (anchors in the module docstring)."""
    dg = importlib.import_module("autodata_core.data_generator")
    wp = importlib.import_module("autodata_core.waypoint")
    DataGenerator, MultiWaypoint = dg.DataGenerator, wp.MultiWaypoint
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
            "hashes": {"generate": gen.hash, "trajectory": builder.hash, "execute": exe.hash}}


def install_from_spec(spec: dict, namespace: dict | None = None) -> dict:
    """Entry used by the chunk entrypoint: build the runtime from the chunk spec, install the grafts (and the tracking loop
    when the method tracks), and wrap the script's ``setup_async_generation`` so the success term is instantiated with the
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

    chunk_seed = int(spec["chunk"]["seed"])
    method = build_method(spec["method"], chunk_seed=chunk_seed)
    runtime = AutoDataRuntime(method, IsaacLabLayout.from_spec(spec["extra"]["layout"]), records_dir=Path(spec["records_dir"]),
                              cameras=spec["extra"]["cameras"] if "cameras" in spec["extra"] else (spec["task"].get("cameras") or []),
                              image_size=spec["extra"].get("image_size") or spec["task"].get("image_size"))
    info = install(runtime, dict(spec["extra"].get("graft_hashes") or {}))
    if method.follower_factory is not None:
        dg = importlib.import_module("autodata_core.data_generator")
        dg.DataGenerator.generate = make_tracking_generate(runtime, method.follower_factory,
                                                           private_seed=private_noise_seed(chunk_seed))
        info["tracking"] = "shared-path generate installed"
    return info


# tracking: shared-path loop on AutoData's classes --------------------------------------------------
def check_tracking_supported(datastream) -> None:
    """The shared-path loop needs one source per attempt and no coordination constraints or navigation (module docstring)."""
    policy = datastream.get_generation_policy()
    if getattr(policy, "select_src_per_subtask", False) or getattr(policy, "select_src_per_arm", False):
        raise NotImplementedError("tracking generate needs one shared source demonstration per attempt: the task descriptor's "
                                  "generation_policy must have select_src_per_subtask and select_src_per_arm false")
    if datastream.get_task_constraints():
        raise NotImplementedError("tracking generate does not apply the descriptor's coordination constraints")
    if getattr(policy, "use_navigation_controller", False):
        raise NotImplementedError("tracking generate does not drive a navigation controller")


class AutoDataTrackingLoop(TrackingLoop):
    """``TrackingLoop`` hooks for ``autodata_core``: the datastream resets and measures, a waypoint's hand command is its
    ``passthrough_action[eef]`` tensor (other channels are copied along), ``MultiWaypoint.execute`` returns the step's
    success and the attempt ends in a ``GenerationResult``."""

    def __init__(self, runtime: AutoDataRuntime, follower_factory: Callable, private_seed: int):
        super().__init__(runtime, follower_factory, private_seed)
        dg = importlib.import_module("autodata_core.data_generator")
        wp = importlib.import_module("autodata_core.waypoint")
        self.GenerationResult, self._optional_lock = dg.GenerationResult, dg._optional_lock
        self.Waypoint, self.MultiWaypoint = wp.Waypoint, wp.MultiWaypoint

    async def begin(self, generator, env_id: int, success_term, env_reset_queue, env_action_queue) -> Attempt:
        ds = generator.datastream
        check_tracking_supported(ds)
        eefs = sorted(ds.get_eef_names())
        env_ids, initial_state = await generator._reset_environment_for_generation(env_id=env_id, env_reset_queue=env_reset_queue)
        self.runtime.begin_attempt(generator, env_id, success_term)
        return Attempt(generator, env_id, success_term, env_action_queue, eefs, initial_state, env_ids)

    def lock(self, attempt: Attempt):
        return self._optional_lock(attempt.generator.src_demo_datagen_info_pool.asyncio_lock)

    def build(self, attempt: Attempt, eef: str, subtask_ind: int, boundaries: dict, selected: dict, previous) -> list:
        generator = attempt.generator
        traj = generator.generate_eef_subtask_trajectory(attempt.env_id, eef, subtask_ind, boundaries, {}, selected)
        return generator.merge_eef_subtask_trajectory(attempt.env_id, eef, subtask_ind, previous, traj)

    def subtask_steps(self, attempt: Attempt, eef: str) -> list[tuple[int, int]]:
        return [(int(c.num_interpolation_steps), int(c.num_fixed_steps)) for c in attempt.generator.datastream.get_subtasks(eef)]

    def num_subtasks(self, attempt: Attempt, eef: str) -> int:
        return attempt.generator.datastream.num_subtasks(eef)

    def waypoint_classes(self, attempt: Attempt) -> tuple:
        return self.Waypoint, self.MultiWaypoint

    def waypoint_gripper(self, waypoint, eef: str) -> np.ndarray:
        return _np(waypoint.passthrough_action[eef])

    def clone_hand(self, waypoint):
        return copy.deepcopy(waypoint.passthrough_action)

    def hand_command(self, base, eef: str, gripper: np.ndarray, device):
        import torch

        channels = dict(base.passthrough_action)
        channels[eef] = torch.as_tensor(gripper, device=device, dtype=base.passthrough_action[eef].dtype)
        return channels

    async def execute(self, attempt: Attempt, multi_waypoint) -> bool:
        return await multi_waypoint.execute(datastream=attempt.generator.datastream, success_term=attempt.success_term,
                                            env_id=attempt.env_id, env_action_queue=attempt.env_action_queue)

    def export(self, attempt: Attempt, success: bool, export_demo: bool) -> dict:
        import torch

        ds = attempt.generator.datastream
        env = ds.get_env()
        verdict = torch.tensor([[success]], dtype=torch.bool, device=ds.device)
        env.recorder_manager.set_success_to_episodes(attempt.env_ids, verdict)
        if export_demo:
            env.recorder_manager.export_episodes(attempt.env_ids)
        return {"success": success}

    def result(self, attempt: Attempt, results: dict):
        return self.GenerationResult(initial_state=attempt.initial_state, success=results["success"])


def make_tracking_generate(runtime: AutoDataRuntime, follower_factory: Callable, private_seed: int):
    """Build the replacement ``DataGenerator.generate`` (AutoData's signature) that drives paired trajectories with one
    follower."""
    loop = AutoDataTrackingLoop(runtime, follower_factory, private_seed)

    async def generate(self, env_id, success_term, *, env_reset_queue, env_action_queue, pause_subtask=False, export_demo=True):
        if pause_subtask:
            raise NotImplementedError("tracking generate does not support pause_subtask")
        attempt = await loop.begin(self, env_id, success_term, env_reset_queue, env_action_queue)
        return await loop.run(attempt, export_demo)

    return generate


def _clone_passthrough(waypoint):
    return copy.deepcopy(waypoint.passthrough_action)


def _cut_interval(active: dict, eefs: list[str], start: int, Waypoint, previous_last: dict) -> Interval:
    """``tracking_loop.cut_interval`` for AutoData waypoints (hand command: the ``passthrough_action`` dict)."""
    return cut_interval(active, eefs, start, Waypoint, previous_last, _clone_passthrough)
