"""The ``_tmg`` callbacks for AutoData's generator (the DexMimicGen stack): the Isaac Lab runtime re-targeted at AutoData's
anchors, plus source mixing and the shared-path tracking loop on AutoData's classes.

AutoData (``autodata_core``) keeps the algorithm of Isaac Lab Mimic but not its code, so the grafts land elsewhere:

``DataGenerator.generate`` (async)

| anchor                                                            | grafted statement                                        |
|-------------------------------------------------------------------|----------------------------------------------------------|
| after ``env_id_tensor, initial_state = await self._reset_environment_for_generation(...)`` | ``_tmg.begin_attempt(self, env_id, initial_state, success_term)`` (also resets the class-based success term's state for this env) |
| before return #0 (planning failure)                               | ``_tmg.end_attempt(self, env_id, {"success": False})``    |
| before return #1 (the final ``GenerationResult``)                 | ``_tmg.end_attempt(self, env_id, {"success": buffers.success})`` |

``DataGenerator.generate_eef_subtask_trajectory``

| after ``src_subtask_object_pose = ...`` | ``src_subtask_eef_poses, src_subtask_target_poses, src_subtask_passthrough_actions = _tmg.mix_segment(self, locals())`` |
| after ``seq = WaypointSequence.from_poses(...)`` | ``_tmg.record_subtask(locals())`` |

``MultiWaypoint.execute`` (async): after ``play_action = datastream.target_eef_pose_to_action(...)`` ->
``_tmg.record_command(self, datastream, env_id, play_action)``; its final ``return bool(success_term...)`` is bound to
``success`` and ``_tmg.record_step(self, env, env_id, None, success)`` is inserted before it.

Pre-noise labels come from wrapping ``datastream.target_eef_pose_to_action`` for the attempt (noise-free call with the torch
RNG state saved and restored, then the official call).

Source mixing (``method.mixing.enabled``): ``mix_segment`` hands the mixer every source's segment of the subtask (eef pose at
the segment start, target poses, the hand's passthrough commands) and the reference object's pose at the segment start, as
the Isaac Lab runtime does; the mixed segment replaces the selected source's before AutoData transforms it to the current
object pose. Non-eef passthrough channels (a base command) of the selected source are kept when their length matches.

Tracking (``method.tracking.enabled``): ``make_tracking_generate`` replaces ``DataGenerator.generate`` with the shared-path
loop of the Isaac Lab runtime (one ``ErrorAdaptiveFollower`` per source-time interval over both hands, errors measured on the
configured links, official noise draws kept per base waypoint by a ``NoiseLedger``), built on AutoData's
``generate_eef_subtask_trajectory`` / ``merge_eef_subtask_trajectory`` / ``MultiWaypoint.execute`` and AutoData's recorder
lifecycle (``recorder_manager.set_success_to_episodes`` and ``export_episodes``). It needs one shared source demonstration
per attempt (``generation_policy.select_src_per_subtask`` and ``select_src_per_arm`` false) and a task without coordination
constraints or navigation; otherwise it raises ``NotImplementedError`` before the first attempt.
"""
from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

import numpy as np

from tmg.generation.generators.isaaclab.runtime import (
    Interval,
    IsaacLabLayout,
    IsaacLabRuntime,
    NoiseLedger,
    _np,
    _wrap_randn_like,
    completed_last,
)
from tmg.generation.graft import Graft
from tmg.generation.method.interfaces import MixContext
from tmg.generation.simulators.isaaclab import measured_eef_poses


def initialize_success_term(success_term, env):
    """Instantiate a class-based success term (Arena's ``TaskSuccessTerm``) once, with Isaac Lab's own helper when present.

    AutoData's script passes ``env_cfg.terminations.success`` straight to the generator and calls ``func(env, **params)``;
    for a class-based term that constructs the class with the wrong arguments (observed 2026-10-28). Isaac Lab 3.0's
    ``isaaclab_mimic.datagen.success_term.initialize_success_term`` is what its own annotation script uses.
    """
    import inspect

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
    """Callbacks for ``autodata_core`` (see the module docstring).

    Images: Isaac Lab's recorder stores the ``policy`` observation group only, and the tasks keep their cameras in a
    ``camera_obs`` group (Arena's head camera; our Pick Place camera environment), so the generated hdf5 has no images. The
    runtime therefore reads the camera frames itself from ``env.obs_buf`` right before each step (the observation the policy
    will see), resizes them to the task's image size and, for a successful attempt, writes ``records/images_%06d.npz`` (one
    ``(T, H, W, 3)`` uint8 array per camera). ``tmg dataset`` reads them from there (``demo_layout()['images_from_records']``).
    """

    def __init__(self, method, layout: IsaacLabLayout, chunk_seed: int, records_dir: Path | None,
                 cameras=(), image_size=None):
        super().__init__(method, layout, chunk_seed, records_dir)
        self._datastream = None
        self.cameras = [str(c) for c in (cameras or [])]
        self.image_size = tuple(int(v) for v in image_size) if image_size else None
        self._frames: dict[str, list[np.ndarray]] = {}

    # attempt boundaries --------------------------------------------------------------------------
    def begin_attempt(self, generator, env_id: int, initial_state, success_term=None) -> None:
        import time

        self._attempt_t0 = time.time()
        self._labels, self._executed, self._pending, self._pending_track = [], [], None, None
        self.rec.begin_attempt()
        self._seg_id = -1
        self._datastream = generator.datastream
        self._env = generator.datastream.get_env()
        generator._tmg_selected = {}
        self._frames = {cam: [] for cam in self.cameras}
        if success_term is not None:
            reset_success_term(success_term, [env_id])      # Arena's class-based term keeps progress state per env
        self._install_label_wrapper(generator.datastream, env_id)

    def end_attempt(self, generator, env_id: int, results: dict) -> None:
        super().end_attempt(generator, env_id, results)
        if self.records_dir is not None and results.get("success") and self._frames and all(self._frames.values()):
            np.savez_compressed(Path(self.records_dir) / f"images_{self.rec.attempt:06d}.npz",
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

    def _install_label_wrapper(self, datastream, env_id: int) -> None:
        import torch

        original = datastream.target_eef_pose_to_action
        self._original_action_fn = original
        runtime = self

        def with_labels(target_eef_pose_dict, passthrough_action_dict, action_noise_dict=None, env_id=env_id, **kw):
            clone = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in target_eef_pose_dict.items()}
            cuda = torch.cuda.is_available()
            cpu_state = torch.random.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if cuda else None
            clean = original(clone, passthrough_action_dict, None, env_id, **kw)
            torch.random.set_rng_state(cpu_state)
            if cuda:
                torch.cuda.set_rng_state_all(cuda_state)
            runtime._labels.append(_np(clean).reshape(-1))
            noisy = original(target_eef_pose_dict, passthrough_action_dict, action_noise_dict, env_id, **kw)
            runtime._executed.append(_np(noisy).reshape(-1))
            return noisy

        datastream.target_eef_pose_to_action = with_labels

    def _restore_label_wrapper(self) -> None:
        if self._datastream is not None and self._original_action_fn is not None:
            self._datastream.target_eef_pose_to_action = self._original_action_fn
        self._original_action_fn = None

    # source mixing -------------------------------------------------------------------------------
    def mix_segment(self, generator, local_vars: dict):
        """Grafted after the source segment is sliced: the (possibly mixed) eef poses, targets and passthrough commands.

        Every source's segment is expressed in the primary's reference-object frame by the mixer (the object pose at the
        segment start is passed per source), as in the Isaac Lab runtime; AutoData's hand commands are the eef's own
        passthrough channel.
        """
        import torch

        eef = local_vars["src_subtask_eef_poses"]
        targets = local_vars["src_subtask_target_poses"]
        passthrough = dict(local_vars["src_subtask_passthrough_actions"])
        self._last_mix = None
        if self.method.mixer is None:
            return eef, targets, passthrough
        eef_name = str(local_vars["eef_name"])
        subtask_ind = int(local_vars["subtask_ind"])
        primary = int(local_vars["selected_src_demo_ind"])
        bounds = np.asarray(_np(local_vars["all_randomized_subtask_boundaries"][eef_name]), dtype=int)   # (sources, subtasks, 2)
        infos = generator.src_demo_datagen_info_pool.datagen_infos
        object_name = local_vars.get("subtask_object_name")
        grippers = passthrough[eef_name]
        segments, object_poses = [], []
        for i, info in enumerate(infos):
            start, end = (int(v) for v in bounds[i, subtask_ind])
            segments.append((_np(info.eef_pose[eef_name][start]), _np(info.target_eef_pose[eef_name][start:end]),
                             _np(info.passthrough_action[eef_name][start:end])))
            object_poses.append(None if object_name is None else _np(info.object_poses[object_name][start]))
        context = MixContext(subtask_ind=subtask_ind, primary=primary, primary_eef_start=_np(eef[0]),
                             primary_targets=_np(targets), primary_gripper=_np(grippers), src_object_poses=object_poses,
                             src_segments=segments, num_sources=len(infos), has_reference_object=object_name is not None,
                             rng=np.random.default_rng(0), extra={"eef": eef_name, "soft": False})
        mixed = self.method.mixer.mix(context)
        self._last_mix = mixed.record
        if not mixed.record.get("mixed"):
            return eef, targets, passthrough
        extra = {k: v for k, v in passthrough.items() if k != eef_name}
        if any(len(v) != len(mixed.targets) for v in extra.values()):
            raise NotImplementedError("source mixing with non-eef passthrough channels of another length is not supported")
        new_eef = torch.cat([torch.as_tensor(np.asarray(mixed.eef_start)[None], dtype=targets.dtype, device=targets.device), eef[1:]], dim=0)
        new_targets = torch.as_tensor(np.asarray(mixed.targets), dtype=targets.dtype, device=targets.device)
        new_pass = {eef_name: torch.as_tensor(np.asarray(mixed.gripper), dtype=grippers.dtype, device=grippers.device), **extra}
        return new_eef, new_targets, new_pass

    # subtasks and steps --------------------------------------------------------------------------
    def record_subtask(self, local_vars: dict) -> None:
        generator = local_vars["self"]
        eef_name = str(local_vars["eef_name"])
        subtask_ind = int(local_vars["subtask_ind"])
        selected = int(local_vars["selected_src_demo_ind"])
        chosen = getattr(generator, "_tmg_selected", None) or {}
        chosen[eef_name] = selected
        generator._tmg_selected = chosen
        boundary = [int(v) for v in _np(local_vars["selected_boundary"])]
        spec = local_vars["subtasks"][subtask_ind]
        object_pose = local_vars.get("subtask_object_pose")
        self.rec.add_subtask(subtask_ind=subtask_ind, eef=eef_name, object_name=local_vars.get("subtask_object_name"),
                             src_demo_ind=selected, src_subtask_start=boundary[0], src_subtask_end=boundary[1],
                             cur_obj_pose=_np(object_pose) if object_pose is not None else None,
                             num_interpolation_steps=int(getattr(spec, "num_interpolation_steps", 0)),
                             num_fixed_steps=int(getattr(spec, "num_fixed_steps", 0)),
                             action_noise=float(getattr(spec, "action_noise", 0.0)), mix=self._last_mix)

    def measured_poses(self, datastream, env_id: int, eefs) -> dict:
        """Measured end-effector poses: the configured links (``measure_links``) or the embodiment's observed eef poses."""
        if self.layout.measure_links:
            return measured_eef_poses(datastream.get_env(), env_id, self.layout.measure_links)
        return {eef: _np(datastream.get_robot_eef_pose(env_ids=[env_id], eef_name=eef)[0]) for eef in eefs}

    def record_command(self, multi_waypoint, datastream, env_id: int, play_action) -> None:
        self.capture_frames(datastream.get_env(), env_id)       # pre-step observation, paired with this step's action
        targets = {eef: _np(w.pose) for eef, w in multi_waypoint.waypoints.items()}
        grippers = {}
        for eef, w in multi_waypoint.waypoints.items():
            channels = getattr(w, "passthrough_action", None) or {}
            grippers[eef] = _np(channels[eef]).reshape(-1) if eef in channels else np.zeros(0)
        measured = self.measured_poses(datastream, env_id, list(multi_waypoint.waypoints))
        noise = max(float(w.noise) if w.noise is not None else 0.0 for w in multi_waypoint.waypoints.values())
        label = self._labels[-1] if self._labels else _np(play_action).reshape(-1)
        self._pending = {"targets": targets, "measured": measured, "grippers": grippers, "noise": noise,
                         "label": label, "executed": _np(play_action).reshape(-1)}


def install(runtime: AutoDataRuntime, graft_hashes: dict) -> dict:
    """Graft the recording and mixing callbacks into ``autodata_core`` (anchors in the module docstring)."""
    dg = importlib.import_module("autodata_core.data_generator")
    wp = importlib.import_module("autodata_core.waypoint")
    DataGenerator, MultiWaypoint = dg.DataGenerator, wp.MultiWaypoint
    gen = Graft(DataGenerator.generate, expected_hash=graft_hashes.get("generate"), label="DataGenerator.generate")
    gen.insert_after_assign("initial_state", "_tmg.begin_attempt(self, env_id, initial_state, success_term)")
    gen.insert_before_return('_tmg.end_attempt(self, env_id, {"success": False})', occurrence=0, expected_returns=2)
    gen.insert_before_return('_tmg.end_attempt(self, env_id, {"success": buffers.success})', occurrence=1, expected_returns=2)
    builder = Graft(DataGenerator.generate_eef_subtask_trajectory, expected_hash=graft_hashes.get("trajectory"),
                    label="DataGenerator.generate_eef_subtask_trajectory")
    builder.insert_after_assign("src_subtask_object_pose",
                                "src_subtask_eef_poses, src_subtask_target_poses, src_subtask_passthrough_actions = _tmg.mix_segment(self, locals())")
    builder.insert_after_assign("seq", "_tmg.record_subtask(locals())")
    exe = Graft(MultiWaypoint.execute, expected_hash=graft_hashes.get("execute"), label="MultiWaypoint.execute")
    exe.insert_after_assign("play_action", "_tmg.record_command(self, datastream, env_id, play_action)", occurrence=0)
    exe.name_final_return("success")
    exe.insert_before_final_return("_tmg.record_step(self, env, env_id, None, success)")
    DataGenerator.generate = gen.compile({"_tmg": runtime})
    DataGenerator.generate_eef_subtask_trajectory = builder.compile({"_tmg": runtime})
    MultiWaypoint.execute = exe.compile({"_tmg": runtime})
    return {"generate": gen.edits, "generate_eef_subtask_trajectory": builder.edits, "execute": exe.edits,
            "hashes": {"generate": gen.hash, "trajectory": builder.hash, "execute": exe.hash}}


def install_from_spec(spec: dict, namespace: dict | None = None) -> dict:
    """Entry used by the chunk entrypoint: build the runtime from the chunk spec, install the grafts (and the tracking loop
    when the method tracks), and wrap the script's ``setup_async_generation`` so the success term is instantiated with the
    environment before generation starts."""
    import functools

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

    layout = IsaacLabLayout(**{k: (tuple(v) if k == "noise_draw_shapes" else v) for k, v in spec["extra"]["layout"].items()},
                            soft_object_subtasks={})
    layout.arm_action_dims = {k: tuple(v) for k, v in layout.arm_action_dims.items()}
    method = build_method(spec["method"], chunk_seed=int(spec["chunk"]["seed"]))
    runtime = AutoDataRuntime(method, layout, chunk_seed=int(spec["chunk"]["seed"]), records_dir=Path(spec["records_dir"]),
                              cameras=spec["extra"].get("cameras") or spec["task"].get("cameras") or [],
                              image_size=spec["extra"].get("image_size") or spec["task"].get("image_size"))
    info = install(runtime, dict(spec["extra"].get("graft_hashes") or {}))
    if method.follower_factory is not None:
        dg = importlib.import_module("autodata_core.data_generator")
        dg.DataGenerator.generate = make_tracking_generate(runtime, method.follower_factory, private_seed=int(spec["chunk"]["seed"]) + 7919)
        info["tracking"] = "shared-path generate installed"
    return info


# tracking: shared-path loop on AutoData's classes --------------------------------------------------
def _cut_interval(active: dict, eefs: list[str], start: int, Waypoint, previous_last: dict) -> Interval:
    """Cut the active trajectories to the common source-time interval (as the Isaac Lab runtime's ``_cut_interval``, for
    AutoData waypoints whose hand command is the ``passthrough_action`` dict)."""
    import copy

    import torch

    end = min(active[e]["end"] for e in eefs)
    prefixes = {e: active[e]["prefix"] if active[e]["begin"] == start else 0 for e in eefs}
    prefix = max(prefixes.values())
    waypoints, kinds = {}, None
    for e in eefs:
        seg = active[e]
        first = seg["prefix"] + start - seg["begin"]
        body = seg["raw"][first:first + end - start]
        if len(body) != end - start:
            raise RuntimeError(f"{e}: trajectory shorter than its source interval")
        if prefixes[e]:
            if prefixes[e] != prefix:
                raise NotImplementedError("unequal simultaneous bridge lengths are not supported")
            bridge = seg["raw"][:prefix]
        elif prefix:
            prior = previous_last.get(e)
            if prior is None:
                raise RuntimeError(f"{e}: no previous waypoint to bridge from")
            a, b = _np(prior.pose), _np(body[0].pose)
            bridge = []
            for j in range(prefix):
                t = (j + 1) / (prefix + 1)
                pose = a.copy()
                pose[:3, 3] = (1 - t) * a[:3, 3] + t * b[:3, 3]
                pose[:3, :3] = b[:3, :3] if t >= 0.5 else a[:3, :3]
                bridge.append(Waypoint(torch.as_tensor(pose, device=body[0].pose.device, dtype=body[0].pose.dtype),
                                       copy.deepcopy(body[0].passthrough_action), body[0].noise))
        else:
            bridge = []
        waypoints[e] = bridge + body
        kinds = [0] * len(bridge) + [1] * len(body)
    return Interval(start=start, end=end, subtasks={e: active[e]["subtask"] for e in eefs}, waypoints=waypoints,
                    kinds=kinds or [], source_indices=[start] * prefix + list(range(start, end)))


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


def make_tracking_generate(runtime: AutoDataRuntime, follower_factory: Callable, private_seed: int):
    """Build the replacement ``DataGenerator.generate`` that drives paired trajectories with one follower (the loop of the
    Isaac Lab runtime's ``make_tracking_generate`` on AutoData's builders, executor and recorder lifecycle)."""
    import torch

    from autodata_core.data_generator import GenerationResult, _optional_lock
    from autodata_core.waypoint import MultiWaypoint, Waypoint

    from tmg.generation.method.tracking import EndEffectorPath

    async def generate(self, env_id, success_term, *, env_reset_queue, env_action_queue, pause_subtask=False, export_demo=True):
        if pause_subtask:
            raise NotImplementedError("tracking generate does not support pause_subtask")
        ds = self.datastream
        check_tracking_supported(ds)
        eefs = sorted(ds.get_eef_names())
        env_id_tensor, initial_state = await self._reset_environment_for_generation(env_id=env_id, env_reset_queue=env_reset_queue)
        runtime.begin_attempt(self, env_id, initial_state, success_term)
        lock = self.src_demo_datagen_info_pool.asyncio_lock
        async with _optional_lock(lock):
            boundaries = self.randomize_subtask_boundaries()
        selected = {eef: None for eef in eefs}
        active: dict[str, dict] = {}
        indices = {eef: 0 for eef in eefs}
        completed: dict[str, list] = {}
        success, offset, start = False, 0, 0
        previous_states: dict | None = None
        ledger: NoiseLedger | None = None
        try:
            while True:
                async with _optional_lock(lock):
                    for eef in eefs:
                        if eef in active:
                            continue
                        i = indices[eef]
                        traj = self.generate_eef_subtask_trajectory(env_id, eef, i, boundaries, {}, selected)
                        raw = self.merge_eef_subtask_trajectory(env_id, eef, i, completed.get(eef), traj)
                        source = int(selected[eef])
                        begin, end = (int(v) for v in boundaries[eef][source, i])
                        active[eef] = {"raw": list(raw), "begin": begin, "end": end, "prefix": len(raw) - (end - begin), "subtask": i}
                sources = {int(selected[e]) for e in eefs}
                if len(sources) != 1:
                    raise NotImplementedError("tracking generate needs one shared source demonstration per attempt")
                source = sources.pop()
                interval = _cut_interval(active, eefs, start, Waypoint, previous_last=completed_last(completed, eefs))
                first = interval.waypoints[eefs[0]][0].pose
                if ledger is None:
                    n_total = max(sum(int(b[1] - b[0]) + int(i == 0) + int(c.num_interpolation_steps) + int(c.num_fixed_steps)
                                      for i, (b, c) in enumerate(zip(boundaries[e][source], ds.get_subtasks(e)))) for e in eefs)
                    ledger = NoiseLedger(n_total, runtime.layout.noise_draw_shapes, first.device, first.dtype, private_seed)
                paths = [EndEffectorPath(e, np.stack([_np(w.pose) for w in interval.waypoints[e]]),
                                         np.stack([_np(w.passthrough_action[e]) for w in interval.waypoints[e]])) for e in eefs]
                noises = [float(interval.waypoints[eefs[0]][k].noise) if interval.waypoints[eefs[0]][k].noise is not None else 0.0
                          for k in range(len(interval.source_indices))]
                follower = follower_factory()
                follower.start_path(paths, noises, interval.kinds, previous_states)
                runtime.begin_segment(1)
                runtime.rec.add_extra(interval_start=start, interval_end=interval.end, interval_subtasks=interval.subtasks)
                while not follower.finished():
                    measured = runtime.measured_poses(ds, env_id, eefs)
                    targets, grippers, _noise, row = follower.next_target(measured)
                    k = int(row["k"])
                    slot = offset + row["new_slots"][0] if row["new_slots"] else None
                    bundle = ledger.bundle(slot)
                    waypoints = {}
                    for e in eefs:
                        base = interval.waypoints[e][k]
                        channels = dict(base.passthrough_action)
                        channels[e] = torch.as_tensor(grippers[e], device=first.device, dtype=base.passthrough_action[e].dtype)
                        waypoints[e] = Waypoint(torch.as_tensor(targets[e], device=first.device, dtype=first.dtype), channels, base.noise)
                    for extra_slot in row["new_slots"][1:]:
                        ledger.used.add(offset + extra_slot)       # waypoints skipped within one step consume their draw
                    runtime._pending_track = {**row, "slot": 1 if slot is not None else 0}
                    with _wrap_randn_like(torch, bundle):
                        ok = await MultiWaypoint(waypoints).execute(datastream=ds, success_term=success_term, env_id=env_id,
                                                                    env_action_queue=env_action_queue)
                    success = success or bool(ok)
                previous_states = follower.final_gripper_states()
                offset += len(interval.source_indices)
                for eef in eefs:
                    if active[eef]["end"] == interval.end:
                        completed[eef] = active[eef]["raw"]
                        del active[eef]
                        indices[eef] += 1
                if all(indices[e] == ds.num_subtasks(e) for e in eefs):
                    break
                start = interval.end
            env = ds.get_env()
            env.recorder_manager.set_success_to_episodes(env_id_tensor, torch.tensor([[success]], dtype=torch.bool, device=ds.device))
            if export_demo:
                env.recorder_manager.export_episodes(env_id_tensor)
            runtime.rec.add_extra(noise_slots_used=len(ledger.used) if ledger else 0, private_noise_steps=ledger.extra if ledger else 0)
            runtime.end_attempt(self, env_id, {"success": success})
            return GenerationResult(initial_state=initial_state, success=success)
        except BaseException:
            runtime._restore_label_wrapper()
            raise

    return generate
