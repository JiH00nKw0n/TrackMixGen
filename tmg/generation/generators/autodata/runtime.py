"""The ``_tmg`` callbacks for AutoData's generator: the Isaac Lab runtime re-targeted at AutoData's anchors.

AutoData (``autodata_core``) keeps the algorithm of Isaac Lab Mimic but not its code, so the grafts land elsewhere:

``DataGenerator.generate`` (async)

| anchor                                                            | grafted statement                                        |
|-------------------------------------------------------------------|----------------------------------------------------------|
| after ``env_id_tensor, initial_state = await self._reset_environment_for_generation(...)`` | ``_tmg.begin_attempt(self, env_id, initial_state, success_term)`` (also resets the class-based success term's state for this env) |
| before return #0 (planning failure)                               | ``_tmg.end_attempt(self, env_id, {"success": False})``    |
| before return #1 (the final ``GenerationResult``)                 | ``_tmg.end_attempt(self, env_id, {"success": buffers.success})`` |

``DataGenerator.generate_eef_subtask_trajectory``: after ``seq = WaypointSequence.from_poses(...)`` ->
``_tmg.record_subtask(locals())``. ``MultiWaypoint.execute`` (async): after ``play_action = datastream.target_eef_pose_to_action(...)``
-> ``_tmg.record_command(self, datastream, env_id, play_action)``; its final ``return bool(success_term...)`` is bound to
``success`` and ``_tmg.record_step(self, env, env_id, None, success)`` is inserted before it.

Pre-noise labels come from wrapping ``datastream.target_eef_pose_to_action`` for the attempt (noise-free call with the torch
RNG state saved and restored, then the official call). Source mixing and path tracking are not wired for AutoData yet:
``method=none`` only (the adapter refuses other methods).
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from tmg.generation.generators.isaaclab.runtime import IsaacLabLayout, IsaacLabRuntime, _np
from tmg.generation.graft import Graft


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

    Images: Isaac Lab's recorder stores the ``policy`` observation group only, and Arena keeps its head camera in a
    ``camera_obs`` group, so the generated hdf5 has no images. The runtime therefore reads the camera frames itself from
    ``env.obs_buf`` right before each step (the observation the policy will see), resizes them to the task's image size
    and, for a successful attempt, writes ``records/images_%06d.npz`` (one ``(T, H, W, 3)`` uint8 array per camera).
    ``tmg dataset`` reads them from there (``demo_layout()['images_from_records']``).
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
                             action_noise=float(getattr(spec, "action_noise", 0.0)), mix=None)

    def record_command(self, multi_waypoint, datastream, env_id: int, play_action) -> None:
        self.capture_frames(datastream.get_env(), env_id)       # pre-step observation, paired with this step's action
        targets = {eef: _np(w.pose) for eef, w in multi_waypoint.waypoints.items()}
        grippers = {}
        for eef, w in multi_waypoint.waypoints.items():
            channels = getattr(w, "passthrough_action", None) or {}
            grippers[eef] = _np(channels[eef]).reshape(-1) if eef in channels else np.zeros(0)
        measured = {eef: _np(datastream.get_robot_eef_pose(env_ids=[env_id], eef_name=eef)[0]) for eef in multi_waypoint.waypoints}
        noise = max(float(w.noise) if w.noise is not None else 0.0 for w in multi_waypoint.waypoints.values())
        label = self._labels[-1] if self._labels else _np(play_action).reshape(-1)
        self._pending = {"targets": targets, "measured": measured, "grippers": grippers, "noise": noise,
                         "label": label, "executed": _np(play_action).reshape(-1)}


def install(runtime: AutoDataRuntime, graft_hashes: dict) -> dict:
    """Graft the recording callbacks into ``autodata_core`` (anchors in the module docstring)."""
    dg = importlib.import_module("autodata_core.data_generator")
    wp = importlib.import_module("autodata_core.waypoint")
    DataGenerator, MultiWaypoint = dg.DataGenerator, wp.MultiWaypoint
    gen = Graft(DataGenerator.generate, expected_hash=graft_hashes.get("generate"), label="DataGenerator.generate")
    gen.insert_after_assign("initial_state", "_tmg.begin_attempt(self, env_id, initial_state, success_term)")
    gen.insert_before_return('_tmg.end_attempt(self, env_id, {"success": False})', occurrence=0, expected_returns=2)
    gen.insert_before_return('_tmg.end_attempt(self, env_id, {"success": buffers.success})', occurrence=1, expected_returns=2)
    builder = Graft(DataGenerator.generate_eef_subtask_trajectory, expected_hash=graft_hashes.get("trajectory"),
                    label="DataGenerator.generate_eef_subtask_trajectory")
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
    """Entry used by the chunk entrypoint: build the runtime from the chunk spec, install the grafts, and wrap the script's
    ``setup_async_generation`` so the success term is instantiated with the environment before generation starts."""
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

    if spec["method"].get("mixing", {}).get("enabled") or spec["method"].get("tracking", {}).get("enabled"):
        raise NotImplementedError("the AutoData adapter supports method=none only (mixing and tracking are not wired to autodata_core yet)")
    layout = IsaacLabLayout(**{k: (tuple(v) if k == "noise_draw_shapes" else v) for k, v in spec["extra"]["layout"].items()},
                            soft_object_subtasks={})
    layout.arm_action_dims = {k: tuple(v) for k, v in layout.arm_action_dims.items()}
    method = build_method(spec["method"], chunk_seed=int(spec["chunk"]["seed"]))
    runtime = AutoDataRuntime(method, layout, chunk_seed=int(spec["chunk"]["seed"]), records_dir=Path(spec["records_dir"]),
                              cameras=spec["extra"].get("cameras") or spec["task"].get("cameras") or [],
                              image_size=spec["extra"].get("image_size") or spec["task"].get("image_size"))
    return install(runtime, dict(spec["extra"].get("graft_hashes") or {}))
