"""The ``_tmg`` callbacks for the Isaac Lab mimic generators, and the shared-path tracking loop.

Official code is never edited on disk. ``install`` reads the pinned functions, checks their source hashes and
grafts one call per anchor (``tmg.generation.graft``):

``DataGenerator.generate`` (Soft: ``softmimicgen.datagen.data_generator``, Dex: ``isaaclab_mimic.datagen.data_generator``)

| anchor (assigned name)                            | grafted statement                                   |
|---------------------------------------------------|-----------------------------------------------------|
| after ``new_initial_state = ...``                 | ``_tmg.begin_attempt(self, env_id, new_initial_state)`` |
| before the final ``return results``               | ``_tmg.end_attempt(self, env_id, results)``          |

``DataGenerator.generate_eef_subtask_trajectory`` (Dex) / ``DataGenerator.generate_trajectory`` (Soft)

| after ``src_subtask_object_pose = ...`` (Dex) or ``src_subtask_nodal_positions = ...`` (Soft) | ``src_subtask_eef_poses, src_subtask_target_poses, src_subtask_gripper_actions = _tmg.mix_segment(self, locals())`` |
| after ``transformed_seq = WaypointSequence.from_poses(...)`` | ``_tmg.record_subtask(locals())`` |

``MultiWaypoint.execute``

| after the first ``play_action = env.target_eef_pose_to_action(...)`` | ``_tmg.record_command(self, env, env_id, state, play_action)`` |
| after ``success = ...``                           | ``_tmg.record_step(self, env, env_id, obs, success)``         |

Pre-noise labels: for the attempt's duration ``env.target_eef_pose_to_action`` is replaced by a wrapper that
first computes the noiseless action (with the torch RNG state saved and restored around that call, so the
official draw sequence is unchanged) and keeps it as the label, then calls the official method with the
noise. The official recorder writes the executed actions to the hdf5; the labels live in our records, and
``tmg dataset`` reads ``actions_pre_noise`` from there for these generators.

Tracking (``method.tracking.enabled``): the official loop advances one waypoint per step for each end
effector separately, so a follower that moves both hands along one path variable cannot be hosted by grafts.
``TrackingLoop`` replaces ``DataGenerator.generate`` for the attempt with a loop that still calls the
official builders (``generate_*trajectory``, ``merge_eef_subtask_trajectory``, ``MultiWaypoint.execute``) and
the official recorder, and drives the paired trajectories with ``tmg.generation.method.tracking``:
one ``ErrorAdaptiveFollower`` per source-time interval during which every end effector has a trajectory,
errors measured on the configured links (``simulators.isaaclab.measured_eef_poses``), gripper events taken
from the union of the hands' commanded events, and the official per-step noise draws preserved by a
``NoiseLedger`` (one original draw bundle per base waypoint, a private generator for extra steps). The
identity check (``method=none``) always runs the official ``generate``.
"""
from __future__ import annotations

import json
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tmg.core.records import AttemptRecorder
from tmg.generation.graft import Graft
from tmg.generation.method.interfaces import Method, MixContext
from tmg.generation.simulators.isaaclab import HandLayout, measured_eef_poses, observed_eef_poses

RECORD_ERROR = "RECORD_ERROR.json"


@dataclass
class IsaacLabLayout:
    """What differs between the Isaac Lab generators (filled from the generator and task configs)."""

    eefs: list[str]                                   # end-effector names in the official order (sorted)
    arm_action_dims: dict[str, tuple[int, int]]       # eef -> (start, end) of its arm pose block in the action
    hand_prefixes: dict[str, str] | None              # eef -> joint-name prefix (dexterous hands), or None (gripper)
    hand_term: str | None                             # action term holding the hand joint names (Dex: "upper_body_ik")
    measure_links: dict[str, str] | None              # eef -> body name to measure the follower error on (None: observed eef)
    soft_object_subtasks: dict[int, bool] = field(default_factory=dict)   # subtask index -> object is soft (Soft)
    noise_draw_shapes: tuple[int, ...] = ()           # per-step torch.randn_like shapes of target_eef_pose_to_action

    def hand_layout(self, env) -> HandLayout | None:
        """``HandLayout`` read from the environment's action term, or None for simple grippers."""
        if not self.hand_prefixes:
            return None
        term = env.action_manager.get_term(self.hand_term)
        arm_dim = max(end for _, end in self.arm_action_dims.values())
        layout = HandLayout(term._hand_joint_names, arm_dim=arm_dim, prefixes=self.hand_prefixes)
        configured = list(getattr(term.cfg, "hand_joint_names", layout.names))
        if layout.names != configured:
            raise RuntimeError("the controller's hand joint order differs from the configured hand_joint_names")
        return layout


def _np(x) -> np.ndarray:
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=float)


class IsaacLabRuntime:
    """Callbacks shared by both Isaac Lab generators (see the module docstring)."""

    def __init__(self, method: Method, layout: IsaacLabLayout, chunk_seed: int, records_dir: Path | None):
        self.method = method
        self.layout = layout
        self.chunk_seed = int(chunk_seed)
        self.rec = AttemptRecorder()
        self.rec.configure(records_dir)
        self.records_dir = records_dir
        self._attempt_t0 = 0.0
        self._last_mix: dict | None = None
        self._labels: list[np.ndarray] = []
        self._executed: list[np.ndarray] = []
        self._pending: dict | None = None
        self._pending_track: dict | None = None
        self._seg_id = -1
        self._env = None
        self._original_action_fn = None
        self._hand_layout: HandLayout | None = None

    # attempt boundaries --------------------------------------------------------------------------
    def begin_attempt(self, generator, env_id: int, initial_state) -> None:
        """Grafted after ``new_initial_state``: start the record and install the pre-noise label wrapper."""
        self._attempt_t0 = time.time()
        self._labels, self._executed, self._pending, self._pending_track = [], [], None, None
        self.rec.begin_attempt()
        self._seg_id = -1
        env = generator.env
        self._env = env
        if self._hand_layout is None:
            self._hand_layout = self.layout.hand_layout(env)
        self._install_label_wrapper(env, env_id)

    def end_attempt(self, generator, env_id: int, results: dict) -> None:
        """Grafted before ``return results``: restore the action method and flush the record."""
        self._restore_label_wrapper()
        try:
            self.rec.add_extra(elapsed_s=round(time.time() - self._attempt_t0, 3), labels=self.method.labels,
                               num_label_steps=len(self._labels), executed_steps=len(self._executed))
            src = {eef: int(v) for eef, v in (getattr(generator, "_tmg_selected", None) or {}).items()}
            self.rec.flush(success=bool(results.get("success", False)), src_demo_inds=[src[e] for e in self.layout.eefs if e in src])
            if self.records_dir is not None and self._labels:
                np.savez_compressed(Path(self.records_dir) / f"labels_{self.rec.attempt:06d}.npz",
                                    actions_pre_noise=np.stack(self._labels), actions_executed=np.stack(self._executed))
        except Exception:
            if self.records_dir is not None:
                (Path(self.records_dir) / RECORD_ERROR).write_text(json.dumps({"attempt": self.rec.attempt,
                                                                                "traceback": traceback.format_exc()}, indent=1))
            raise

    # pre-noise labels ----------------------------------------------------------------------------
    def _install_label_wrapper(self, env, env_id: int) -> None:
        import torch

        original = env.target_eef_pose_to_action
        self._original_action_fn = original
        runtime = self

        def with_labels(target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=env_id, **kw):
            clone = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in target_eef_pose_dict.items()}
            cuda = torch.cuda.is_available()
            cpu_state = torch.random.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if cuda else None
            clean = original(clone, gripper_action_dict, None, env_id, **kw)
            torch.random.set_rng_state(cpu_state)
            if cuda:
                torch.cuda.set_rng_state_all(cuda_state)
            runtime._labels.append(_np(clean).reshape(-1))
            noisy = original(target_eef_pose_dict, gripper_action_dict, action_noise_dict, env_id, **kw)
            runtime._executed.append(_np(noisy).reshape(-1))
            return noisy

        env.target_eef_pose_to_action = with_labels

    def _restore_label_wrapper(self) -> None:
        if self._env is not None and self._original_action_fn is not None:
            self._env.target_eef_pose_to_action = self._original_action_fn
        self._original_action_fn = None

    # source mixing -------------------------------------------------------------------------------
    def mix_segment(self, generator, local_vars: dict):
        """Grafted after the source segment is sliced: the (possibly mixed) eef poses, targets and gripper commands.

        The segments of every source are expressed in the primary's reference-object frame as the mixer expects.
        For rigid objects that frame is the object pose at the segment start; for a soft object (Soft's rope)
        every donor segment is first registered into the primary's rope geometry with the official nodal
        registration, after which the object frame is the world frame.
        """
        import torch

        eef = local_vars["src_subtask_eef_poses"]
        targets = local_vars["src_subtask_target_poses"]
        grippers = local_vars["src_subtask_gripper_actions"]
        self._last_mix = None
        if self.method.mixer is None:
            return eef, targets, grippers
        eef_name = local_vars["eef_name"]
        subtask_ind = int(local_vars["subtask_ind"])
        primary = int(local_vars["selected_src_demo_ind"])
        bounds = np.asarray(_np(local_vars["all_randomized_subtask_boundaries"][eef_name]), dtype=int)   # (sources, subtasks, 2)
        infos = generator.src_demo_datagen_info_pool.datagen_infos
        object_name = local_vars["subtask_object_name"]
        soft = bool(local_vars.get("subtask_object_soft", False))
        segments, object_poses = [], []
        nodal_primary = local_vars.get("src_subtask_nodal_positions") if soft else None
        for i, info in enumerate(infos):
            start, end = (int(v) for v in bounds[i, subtask_ind])
            targets_i = info.target_eef_pose[eef_name][start:end]
            eef_i = info.eef_pose[eef_name][start]
            if soft:
                from softmimicgen.datagen.data_generator import (
                    transform_source_data_segment_using_nodal_registration,
                )

                nodal_i = info.object_nodal_positions[object_name][start]
                if i != primary:
                    targets_i = transform_source_data_segment_using_nodal_registration(
                        tgt_obj_nodal_pos=nodal_primary, src_eef_poses=targets_i, src_obj_nodal_pos=nodal_i,
                        use_rotation_transform=True, bend_coef=0.1, rot_coef=1e-3)
                object_poses.append(np.eye(4))
            else:
                object_poses.append(None if object_name is None else _np(info.object_poses[object_name][start]))
            segments.append((_np(eef_i), _np(targets_i), _np(info.gripper_action[eef_name][start:end])))
        context = MixContext(subtask_ind=subtask_ind, primary=primary, primary_eef_start=_np(eef[0]),
                             primary_targets=_np(targets), primary_gripper=_np(grippers), src_object_poses=object_poses,
                             src_segments=segments, num_sources=len(infos), has_reference_object=object_name is not None,
                             rng=np.random.default_rng(0), extra={"eef": eef_name, "soft": soft})
        mixed = self.method.mixer.mix(context)
        self._last_mix = mixed.record
        if not mixed.record.get("mixed"):
            return eef, targets, grippers
        like = targets
        new_eef = torch.cat([torch.as_tensor(np.asarray(mixed.eef_start)[None], dtype=like.dtype, device=like.device), eef[1:]], dim=0)
        return (new_eef, torch.as_tensor(np.asarray(mixed.targets), dtype=like.dtype, device=like.device),
                torch.as_tensor(np.asarray(mixed.gripper), dtype=grippers.dtype, device=grippers.device))

    def record_subtask(self, local_vars: dict) -> None:
        """Grafted after ``transformed_seq``: the subtask's source, boundaries and object pose, plus the mix record."""
        generator = local_vars["self"]
        eef_name = local_vars["eef_name"]
        subtask_ind = int(local_vars["subtask_ind"])
        selected = int(local_vars["selected_src_demo_ind"])
        chosen = getattr(generator, "_tmg_selected", None) or {}
        chosen[eef_name] = selected
        generator._tmg_selected = chosen
        boundary = [int(v) for v in _np(local_vars["selected_src_subtask_boundary"])]
        object_pose = local_vars.get("subtask_object_pose")
        self.rec.add_subtask(subtask_ind=subtask_ind, eef=eef_name, object_name=local_vars.get("subtask_object_name"),
                             src_demo_ind=selected, src_subtask_start=boundary[0], src_subtask_end=boundary[1],
                             cur_obj_pose=_np(object_pose) if object_pose is not None else None,
                             num_interpolation_steps=int(generator.env_cfg.subtask_configs[eef_name][subtask_ind].num_interpolation_steps),
                             num_fixed_steps=int(generator.env_cfg.subtask_configs[eef_name][subtask_ind].num_fixed_steps),
                             action_noise=float(generator.env_cfg.subtask_configs[eef_name][subtask_ind].action_noise),
                             mix=self._last_mix)

    # per-step records ----------------------------------------------------------------------------
    def record_command(self, multi_waypoint, env, env_id: int, state, play_action) -> None:
        """Grafted after ``play_action``: remember the step's targets, measured poses and commands."""
        targets = {eef: _np(w.pose) for eef, w in multi_waypoint.waypoints.items()}
        grippers = {eef: _np(w.gripper_action) for eef, w in multi_waypoint.waypoints.items()}
        measured = (measured_eef_poses(env, env_id, self.layout.measure_links) if self.layout.measure_links
                    else observed_eef_poses(env, env_id, list(multi_waypoint.waypoints)))
        noise = max(float(w.noise) if w.noise is not None else 0.0 for w in multi_waypoint.waypoints.values())
        label = self._labels[-1] if self._labels else _np(play_action).reshape(-1)
        self._pending = {"targets": targets, "measured": measured, "grippers": grippers, "noise": noise,
                         "label": label, "executed": _np(play_action).reshape(-1)}

    def record_step(self, multi_waypoint, env, env_id: int, obs, success: bool) -> None:
        """Grafted after ``success``: write the step with the termination signals of the new state."""
        if self._pending is None:
            return
        p = self._pending
        self._pending = None
        signals = {}
        terms = obs.get("subtask_terms") if isinstance(obs, dict) else None
        if terms:
            for key, value in terms.items():
                v = value[env_id] if hasattr(value, "__getitem__") and getattr(value, "ndim", 0) > 0 else value
                signals[str(key)] = bool(_np(v).reshape(-1)[0])
        arm = np.concatenate([p["label"][s:e] for s, e in self.layout.arm_action_dims.values()])
        if self._seg_id < 0:
            self._seg_id = self.rec.begin_segment(1)
        self.rec.add_step(w_poses=p["targets"], e_poses=p["measured"], a_pre=arm, a_label=p["label"], a_exec=p["executed"],
                          noise=p["noise"], seg_id=self._seg_id, term_signals=signals, track=self._pending_track)
        self._pending_track = None
        self.rec.add_success(bool(success))

    def begin_segment(self, kind: int) -> int:
        """Open a record segment (the tracking loop opens one per source-time interval)."""
        self._seg_id = self.rec.begin_segment(kind)
        return self._seg_id


def install(package: str, runtime: IsaacLabRuntime, graft_hashes: dict, *, soft: bool) -> dict:
    """Install the grafts into the official package's classes; returns their descriptions for the chunk log.

    Args:
        package: ``softmimicgen.datagen`` or ``isaaclab_mimic.datagen``.
        runtime: the callbacks.
        graft_hashes: pinned source hashes (``generate``, ``trajectory``, ``execute``), None values skip the check.
        soft: Soft's trajectory builder is ``generate_trajectory`` with a nodal-position anchor.
    """
    import importlib

    dg = importlib.import_module(f"{package}.data_generator")
    wp = importlib.import_module(f"{package}.waypoint")
    DataGenerator, MultiWaypoint = dg.DataGenerator, wp.MultiWaypoint
    gen = Graft(DataGenerator.generate, expected_hash=graft_hashes.get("generate"), label="DataGenerator.generate")
    gen.insert_after_assign("new_initial_state", "_tmg.begin_attempt(self, env_id, new_initial_state)")
    gen.insert_before_final_return("_tmg.end_attempt(self, env_id, results)")
    builder_name = "generate_trajectory" if soft else "generate_eef_subtask_trajectory"
    builder = Graft(getattr(DataGenerator, builder_name), expected_hash=graft_hashes.get("trajectory"), label=f"DataGenerator.{builder_name}")
    builder.insert_after_assign("src_subtask_nodal_positions" if soft else "src_subtask_object_pose",
                                "src_subtask_eef_poses, src_subtask_target_poses, src_subtask_gripper_actions = _tmg.mix_segment(self, locals())")
    builder.insert_after_assign("transformed_seq", "_tmg.record_subtask(locals())")
    exe = Graft(MultiWaypoint.execute, expected_hash=graft_hashes.get("execute"), label="MultiWaypoint.execute")
    exe.insert_after_assign("play_action", "_tmg.record_command(self, env, env_id, state, play_action)", occurrence=0)
    exe.insert_after_assign("success", "_tmg.record_step(self, env, env_id, obs, success)")
    DataGenerator.generate = gen.compile({"_tmg": runtime})
    setattr(DataGenerator, builder_name, builder.compile({"_tmg": runtime}))
    MultiWaypoint.execute = exe.compile({"_tmg": runtime})
    return {"generate": gen.edits, builder_name: builder.edits, "execute": exe.edits}


# tracking: shared-path loop ------------------------------------------------------------------------
class NoiseLedger:
    """The official per-step noise draws, one bundle per base waypoint, plus a private stream for extra steps.

    ``target_eef_pose_to_action`` draws ``torch.randn_like`` once per shape in ``shapes`` at every step. To keep
    the official random stream, the ledger draws all ``n`` bundles up front from the global generator (the same
    values the official loop would have drawn in order) and hands bundle ``slot`` to the step that commands base
    waypoint ``slot``; steps the follower adds draw from a private generator seeded from the chunk seed.
    """

    def __init__(self, n: int, shapes: tuple[int, ...], device, dtype, private_seed: int):
        import torch

        self.torch, self.shapes, self.device, self.dtype = torch, tuple(shapes), device, dtype
        self.original = [[torch.randn(k, device=device, dtype=dtype) for k in self.shapes] for _ in range(int(n))]
        self.private = torch.Generator(device=device)
        self.private.manual_seed(int(private_seed))
        self.used: set[int] = set()
        self.extra = 0

    def bundle(self, slot: int | None) -> list:
        if slot is not None:
            if slot in self.used or not 0 <= slot < len(self.original):
                raise RuntimeError(f"noise slot {slot} reused or out of range")
            self.used.add(slot)
            return self.original[slot]
        self.extra += 1
        return [self.torch.randn(k, device=self.device, dtype=self.dtype, generator=self.private) for k in self.shapes]


def _wrap_randn_like(torch, values: list):
    """Context manager replacing ``torch.randn_like`` with the next value of ``values`` for each call."""
    class _Ctx:
        def __enter__(self):
            self.calls = 0
            self.old = torch.randn_like

            def draw(x, *a, **k):
                if self.calls >= len(values) or tuple(x.shape) != tuple(values[self.calls].shape):
                    raise RuntimeError("unexpected randn_like call while replaying a noise bundle")
                v = values[self.calls]
                self.calls += 1
                return v.clone()

            torch.randn_like = draw
            return self

        def __exit__(self, *exc):
            torch.randn_like = self.old

    return _Ctx()


@dataclass
class Interval:
    """A source-time interval [start, end) during which every end effector has an official trajectory."""

    start: int
    end: int
    subtasks: dict[str, int]
    waypoints: dict[str, list]           # eef -> list of official Waypoint objects (bridge + body)
    kinds: list[int]                     # 0 bridge, 1 skill, per waypoint
    source_indices: list[int]


def make_tracking_generate(runtime: IsaacLabRuntime, follower_factory: Callable, private_seed: int):
    """Build the replacement ``DataGenerator.generate`` that drives paired trajectories with one follower.

    The loop (a port of the validated 2026-10-26 bimanual implementation into this repository's follower):

    1. ``randomize_subtask_boundaries`` once; for each end effector whose trajectory is missing, call the official
       ``generate_*trajectory`` and ``merge_eef_subtask_trajectory`` for its current subtask (all end effectors
       share the selected source; checked).
    2. Cut every end effector's trajectory to the common source-time interval ``[start, min(end))``; an end
       effector whose official bridge belongs to another interval gets a linear bridge of the same length so
       that both timelines have one waypoint per base step.
    3. One ``ErrorAdaptiveFollower`` for the interval over all end effectors (shared path variable, gripper events
       from the union of the hands, errors on the configured links). Each step builds a ``MultiWaypoint`` from
       the follower's targets, replays the official noise bundle of the base waypoint the step covers (or a
       private draw for extra steps) and executes it through the official ``MultiWaypoint.execute``.
    4. When the interval ends, advance the end effectors whose subtask ended, and continue until every end
       effector finished its subtasks. Success, states, observations and actions are collected exactly as the
       official loop does, and the official recorder exports the episode.
    """
    import torch

    from tmg.generation.method.tracking import EndEffectorPath

    async def generate(self, env_id, success_term, env_reset_queue=None, env_action_queue=None, pause_subtask=False,
                       export_demo=True, motion_planner=None):
        if pause_subtask or motion_planner is not None:
            raise NotImplementedError("tracking generate supports neither pause_subtask nor motion planners")
        from importlib import import_module

        wp = import_module(type(self).__module__.rsplit(".", 1)[0] + ".waypoint")
        Waypoint, MultiWaypoint = wp.Waypoint, wp.MultiWaypoint
        cfg = self.env_cfg
        eefs = sorted(cfg.subtask_configs.keys())
        env_ids = torch.tensor([env_id], dtype=torch.int64, device=self.env.device)
        self.env.recorder_manager.reset(env_ids=env_ids)
        await env_reset_queue.put(env_id)
        await env_reset_queue.join()
        new_initial_state = self.env.scene.get_state(is_relative=True)
        runtime.begin_attempt(self, env_id, new_initial_state)
        async with self.src_demo_datagen_info_pool.asyncio_lock:
            boundaries = self.randomize_subtask_boundaries()
        selected = {eef: None for eef in eefs}
        active: dict[str, dict] = {}
        indices = {eef: 0 for eef in eefs}
        completed: dict[str, list] = {}
        generated_states, generated_obs, generated_actions = [], [], []
        success, offset, start = False, 0, 0
        previous_states: dict | None = None
        ledger: NoiseLedger | None = None
        builder = self.generate_trajectory if hasattr(self, "generate_trajectory") else self.generate_eef_subtask_trajectory
        try:
            while True:
                async with self.src_demo_datagen_info_pool.asyncio_lock:
                    for eef in eefs:
                        if eef in active:
                            continue
                        i = indices[eef]
                        if hasattr(self, "generate_trajectory"):
                            raw = builder(env_id, eef, i, boundaries, {}, selected, completed.get(eef))
                        else:
                            traj = builder(env_id, eef, i, boundaries, {}, selected)
                            raw = self.merge_eef_subtask_trajectory(env_id, eef, i, completed.get(eef), traj)
                        source = int(selected[eef])
                        begin, end = (int(v) for v in boundaries[eef][source, i])
                        active[eef] = {"raw": list(raw), "begin": begin, "end": end, "prefix": len(raw) - (end - begin), "subtask": i}
                sources = {int(selected[e]) for e in eefs}
                if len(sources) != 1:
                    raise NotImplementedError("tracking generate needs one shared source demonstration per attempt")
                source = sources.pop()
                interval = _cut_interval(active, eefs, start, Waypoint, previous_last=completed_last(completed, eefs))
                if ledger is None:
                    n_total = max(sum(int(b[1] - b[0]) + int(i == 0) + c.num_interpolation_steps + c.num_fixed_steps
                                      for i, (b, c) in enumerate(zip(boundaries[e][source], cfg.subtask_configs[e]))) for e in eefs)
                    first = interval.waypoints[eefs[0]][0].pose
                    ledger = NoiseLedger(n_total, runtime.layout.noise_draw_shapes, first.device, first.dtype, private_seed)
                paths = [EndEffectorPath(e, np.stack([_np(w.pose) for w in interval.waypoints[e]]),
                                         np.stack([_np(w.gripper_action) for w in interval.waypoints[e]])) for e in eefs]
                noises = [float(interval.waypoints[eefs[0]][k].noise or 0.0) for k in range(len(interval.source_indices))]
                follower = follower_factory()
                follower.start_path(paths, noises, interval.kinds, previous_states)
                runtime.begin_segment(1)
                runtime.rec.add_extra(interval_start=start, interval_end=interval.end, interval_subtasks=interval.subtasks)
                while not follower.finished():
                    measured = (measured_eef_poses(self.env, env_id, runtime.layout.measure_links) if runtime.layout.measure_links
                                else observed_eef_poses(self.env, env_id, eefs))
                    targets, grippers, noise, row = follower.next_target(measured)
                    k = int(row["k"])
                    slot = offset + row["new_slots"][0] if row["new_slots"] else None
                    bundle = ledger.bundle(slot)
                    waypoints = {e: Waypoint(torch.as_tensor(targets[e], device=first.device, dtype=first.dtype),
                                             torch.as_tensor(grippers[e], device=first.device, dtype=interval.waypoints[e][k].gripper_action.dtype),
                                             interval.waypoints[e][k].noise) for e in eefs}
                    for extra_slot in row["new_slots"][1:]:
                        ledger.used.add(offset + extra_slot)       # waypoints skipped within one step consume their draw
                    runtime._pending_track = {**row, "slot": 1 if slot is not None else 0}
                    with _wrap_randn_like(torch, bundle):
                        result = await MultiWaypoint(waypoints).execute(self.env, success_term, env_id, env_action_queue)
                    if result["states"]:
                        generated_states.extend(result["states"])
                        generated_obs.extend(result["observations"])
                        generated_actions.extend(result["actions"])
                        success = success or result["success"]
                previous_states = follower.final_gripper_states()
                offset += len(interval.source_indices)
                for eef in eefs:
                    if active[eef]["end"] == interval.end:
                        completed[eef] = active[eef]["raw"]
                        del active[eef]
                        indices[eef] += 1
                if all(indices[e] == len(cfg.subtask_configs[e]) for e in eefs):
                    break
                start = interval.end
            actions = torch.cat(generated_actions, dim=0)
            self.env.recorder_manager.set_success_to_episodes(env_ids, torch.tensor([[success]], dtype=torch.bool, device=self.env.device))
            if export_demo:
                self.env.recorder_manager.export_episodes(env_ids)
            results = {"initial_state": new_initial_state, "states": generated_states, "observations": generated_obs,
                       "actions": actions, "success": success}
            runtime.rec.add_extra(noise_slots_used=len(ledger.used) if ledger else 0, private_noise_steps=ledger.extra if ledger else 0)
            runtime.end_attempt(self, env_id, results)
            return results
        except BaseException:
            runtime._restore_label_wrapper()
            raise

    return generate


def completed_last(completed: dict, eefs: list[str]) -> dict:
    """Last executed waypoint per end effector (for bridges of the next interval)."""
    return {e: completed[e][-1] for e in eefs if e in completed}


def _cut_interval(active: dict, eefs: list[str], start: int, Waypoint, previous_last: dict) -> Interval:
    """Cut the active trajectories to the common source-time interval (step 2 of ``make_tracking_generate``)."""
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
            import torch

            bridge = []
            for j in range(prefix):
                t = (j + 1) / (prefix + 1)
                pose = a.copy()
                pose[:3, 3] = (1 - t) * a[:3, 3] + t * b[:3, 3]
                pose[:3, :3] = b[:3, :3] if t >= 0.5 else a[:3, :3]
                bridge.append(Waypoint(torch.as_tensor(pose, device=body[0].pose.device, dtype=body[0].pose.dtype),
                                       body[0].gripper_action.clone(), body[0].noise))
        else:
            bridge = []
        waypoints[e] = bridge + body
        kinds = [0] * len(bridge) + [1] * len(body)
    return Interval(start=start, end=end, subtasks={e: active[e]["subtask"] for e in eefs}, waypoints=waypoints,
                    kinds=kinds or [], source_indices=[start] * prefix + list(range(start, end)))
