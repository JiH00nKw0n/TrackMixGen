"""The ``_tmg`` callbacks for the Isaac Lab mimic generators, and the shared-path tracking loop on their classes.

Official code is never edited on disk. ``install`` reads the pinned functions, checks their source hashes and
grafts one call per anchor (``tmg.generation.graft``). The anchors of the trajectory builder are class attributes of the
runtime (``builder_name``, ``mix_anchor``, ``record_anchor``, ``hand_var``): ``IsaacLabRuntime`` has Isaac Lab Mimic's shape
(``isaaclab_mimic.datagen``, also the shape AutoData kept), ``SoftMimicGenRuntime`` SoftMimicGen's (``softmimicgen.datagen``).
``install_from_spec`` picks the class by ``spec["extra"]["soft"]``; every configured Isaac Lab generator of this repository
runs SoftMimicGen (DexMimicGen runs on AutoData, ``..autodata.runtime``).

``DataGenerator.generate``

| anchor (assigned name)                            | grafted statement                                   |
|---------------------------------------------------|-----------------------------------------------------|
| after ``new_initial_state = ...``                 | ``_tmg.begin_attempt(self, env_id)``                 |
| before the final ``return results``               | ``_tmg.end_attempt(self, env_id, results)``          |

``DataGenerator.generate_trajectory`` (Soft) / ``DataGenerator.generate_eef_subtask_trajectory`` (Isaac Lab Mimic)

| after ``src_subtask_nodal_positions = ...`` (Soft) or ``src_subtask_object_pose = ...`` (Mimic) | ``src_subtask_eef_poses, src_subtask_target_poses, src_subtask_gripper_actions = _tmg.mix_segment(self, locals())`` |
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
``TrackingLoop`` (``tracking_loop.py``; ``IsaacLabTrackingLoop`` below fills its hooks) replaces
``DataGenerator.generate`` for the attempt with a loop that still calls the official builders
(``generate_*trajectory``, ``merge_eef_subtask_trajectory``, ``MultiWaypoint.execute``) and the official recorder, and
drives the paired trajectories with ``tmg.generation.method.tracking``: one ``ErrorAdaptiveFollower`` per source-time
interval during which every end effector has a trajectory, errors measured on the configured links
(``simulators.isaaclab.measured_eef_poses``), gripper events taken from the union of the hands' commanded events, and the
official per-step noise draws preserved by a ``NoiseLedger`` (one original draw bundle per base waypoint, a private
generator for extra steps). The identity check (``method=none``) always runs the official ``generate``.
"""
from __future__ import annotations

import importlib
import json
import time
import traceback
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tmg.core.records import LABELS_FORMAT, AttemptRecorder, record_file
from tmg.generation.generators.isaaclab.tracking_loop import (  # noqa: F401  re-exported names
    Attempt,
    Interval,
    NoiseLedger,
    TrackingLoop,
    _np,
    _wrap_randn_like,
    completed_last,
    cut_interval,
    replay_randn_like,
)
from tmg.generation.graft import Graft
from tmg.generation.method.interfaces import Method, MixContext
from tmg.generation.seeds import private_noise_seed
from tmg.generation.simulators.isaaclab import HandLayout, measured_eef_poses, observed_eef_poses

RECORD_ERROR = "RECORD_ERROR.json"


@dataclass
class IsaacLabLayout:
    """What differs between the Isaac Lab generators (filled from the generator and task configs)."""

    eefs: list[str]                                   # end-effector names in the official order (sorted)
    arm_action_dims: dict[str, tuple[int, int]]       # eef -> (start, end) of its arm pose block in the action
    hand_prefixes: dict[str, str] | None              # eef -> joint-name prefix (dexterous hands), or None (gripper)
    hand_term: str | None                             # action term holding the hand joint names (GR1: "upper_body_ik")
    measure_links: dict[str, str] | None              # eef -> body name to measure the follower error on (None: observed eef)
    noise_draw_shapes: tuple[int, ...] = ()           # per-step torch.randn_like shapes of target_eef_pose_to_action

    @classmethod
    def from_spec(cls, layout: dict) -> IsaacLabLayout:
        """The layout serialised in ``spec["extra"]["layout"]`` (json turned its tuples into lists)."""
        out = cls(**{k: (tuple(v) if k == "noise_draw_shapes" else v) for k, v in layout.items()})
        out.arm_action_dims = {k: tuple(v) for k, v in out.arm_action_dims.items()}
        return out

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


@contextmanager
def torch_rng_preserved(torch):
    """Run the block and put the torch RNG state (CPU and every CUDA device) back where it was before it.

    The state is restored on the normal path only: an exception out of the block ends the attempt anyway, and the official
    code never catches it, so nothing downstream would observe the state.
    """
    cuda = torch.cuda.is_available()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if cuda else None
    yield
    torch.random.set_rng_state(cpu_state)
    if cuda:
        torch.cuda.set_rng_state_all(cuda_state)


class IsaacLabRuntime:
    """Callbacks shared by the Isaac Lab generators (see the module docstring).

    The package-specific steps are small hooks: where the action method lives (``action_owner``), how the segment's hand
    commands are read and returned (``segment_hand``, ``source_hand``, ``pack_mixed``), where a subtask's settings live
    (``subtask_settings``) and how the end effectors are measured (``measured_poses``).
    """

    # graft targets of the trajectory builder (Isaac Lab Mimic's shape, kept by AutoData; SoftMimicGen overrides)
    builder_name = "generate_eef_subtask_trajectory"     # DataGenerator method building one subtask's waypoints
    mix_anchor = "src_subtask_object_pose"               # assignment after which ``mix_segment`` runs
    record_anchor = "transformed_seq"                    # assignment after which ``record_subtask`` runs
    hand_var = "src_subtask_gripper_actions"             # the builder's local holding the segment's hand commands
    boundary_var = "selected_src_subtask_boundary"       # the builder's local holding the selected source boundary

    def __init__(self, method: Method, layout: IsaacLabLayout, records_dir: Path | None):
        self.method = method
        self.layout = layout
        self.rec = AttemptRecorder()
        self.rec.configure(records_dir)
        self.records_dir = records_dir
        self._attempt_t0 = 0.0
        self._last_mix: dict | None = None
        self._labels: list[np.ndarray] = []
        self._executed: list[np.ndarray] = []
        self._pending: dict | None = None
        self._pending_track: dict | None = None
        self._selected: dict[str, int] = {}           # eef -> source demonstration chosen in this attempt
        self._seg_id = -1
        self._action_owner = None                     # object whose target_eef_pose_to_action is wrapped
        self._original_action_fn = None
        self._hand_layout: HandLayout | None = None

    # package hooks -------------------------------------------------------------------------------
    def action_owner(self, generator):
        """The object whose ``target_eef_pose_to_action`` builds the actions (the environment here)."""
        return generator.env

    def environment(self, generator):
        return generator.env

    def segment_hand(self, local_vars: dict, eef_name: str) -> tuple:
        """``(hand commands of the eef's segment, the value to return when nothing is mixed)``."""
        hand = local_vars[self.hand_var]
        return hand, hand

    def source_hand(self, info, eef_name: str, start: int, end: int):
        """A source's hand commands over ``[start, end)``."""
        return info.gripper_action[eef_name][start:end]

    def source_segment(self, local_vars: dict, info, index: int, eef_name: str, object_name, start: int, end: int,
                       soft: bool) -> tuple:
        """One source's segment for the mixer: eef pose at the start, target poses, hand commands, reference-object pose."""
        targets = info.target_eef_pose[eef_name][start:end]
        object_pose = None if object_name is None else _np(info.object_poses[object_name][start])
        return _np(info.eef_pose[eef_name][start]), _np(targets), _np(self.source_hand(info, eef_name, start, end)), object_pose

    def pack_mixed(self, mixed, eef, targets, grippers, unchanged, eef_name: str):
        """The mixed segment in the builder's types: tensors like ``targets`` and ``grippers``."""
        import torch

        new_eef = torch.cat([torch.as_tensor(np.asarray(mixed.eef_start)[None], dtype=targets.dtype, device=targets.device), eef[1:]], dim=0)
        return (new_eef, torch.as_tensor(np.asarray(mixed.targets), dtype=targets.dtype, device=targets.device),
                torch.as_tensor(np.asarray(mixed.gripper), dtype=grippers.dtype, device=grippers.device))

    def subtask_settings(self, generator, local_vars: dict, eef_name: str, subtask_ind: int) -> tuple[int, int, float]:
        """``(num_interpolation_steps, num_fixed_steps, action_noise)`` of the subtask (the environment config here)."""
        cfg = generator.env_cfg.subtask_configs[eef_name][subtask_ind]
        return int(cfg.num_interpolation_steps), int(cfg.num_fixed_steps), float(cfg.action_noise)

    def measured_poses(self, owner, env_id: int, eefs) -> dict:
        """Measured end-effector poses: the configured links (``measure_links``) or the environment's observed eef poses."""
        if self.layout.measure_links:
            return measured_eef_poses(owner, env_id, self.layout.measure_links)
        return observed_eef_poses(owner, env_id, list(eefs))

    def _label_wrapper(self, original, env_id: int):
        """The replacement ``target_eef_pose_to_action`` (the official parameter names, so keyword calls bind)."""
        runtime = self

        def with_labels(target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=env_id, **kw):
            return runtime._labelled_action(original, target_eef_pose_dict, gripper_action_dict, action_noise_dict, env_id, kw)

        return with_labels

    # attempt boundaries --------------------------------------------------------------------------
    def begin_attempt(self, generator, env_id: int) -> None:
        """Grafted after ``new_initial_state``: start the record and install the pre-noise label wrapper."""
        self._attempt_t0 = time.time()
        self._labels, self._executed, self._pending, self._pending_track = [], [], None, None
        self._selected = {}
        self.rec.begin_attempt()
        self._seg_id = -1
        if self._hand_layout is None:
            self._hand_layout = self.layout.hand_layout(self.environment(generator))
        self._install_label_wrapper(self.action_owner(generator), env_id)

    def end_attempt(self, generator, env_id: int, results: dict) -> None:
        """Grafted before ``return results``: restore the action method and flush the record."""
        self._restore_label_wrapper()
        try:
            self.rec.add_extra(elapsed_s=round(time.time() - self._attempt_t0, 3), labels=self.method.labels,
                               num_label_steps=len(self._labels), executed_steps=len(self._executed))
            src = {eef: int(v) for eef, v in self._selected.items()}
            self.rec.flush(success=bool(results.get("success", False)), src_demo_inds=[src[e] for e in self.layout.eefs if e in src])
            if self.records_dir is not None and self._labels:
                np.savez_compressed(record_file(Path(self.records_dir), LABELS_FORMAT, self.rec.attempt),
                                    actions_pre_noise=np.stack(self._labels), actions_executed=np.stack(self._executed))
        except Exception:
            if self.records_dir is not None:
                (Path(self.records_dir) / RECORD_ERROR).write_text(json.dumps({"attempt": self.rec.attempt,
                                                                                "traceback": traceback.format_exc()}, indent=1))
            raise

    # pre-noise labels ----------------------------------------------------------------------------
    def _install_label_wrapper(self, owner, env_id: int) -> None:
        self._action_owner = owner
        self._original_action_fn = owner.target_eef_pose_to_action
        owner.target_eef_pose_to_action = self._label_wrapper(self._original_action_fn, env_id)

    def _restore_label_wrapper(self) -> None:
        if self._action_owner is not None and self._original_action_fn is not None:
            self._action_owner.target_eef_pose_to_action = self._original_action_fn
        self._original_action_fn = None

    def _labelled_action(self, original, target_eef_pose_dict, hand_dict, action_noise_dict, env_id, kw: dict):
        """The noiseless action (on a clone of the targets, RNG state preserved) as the label, then the official call."""
        import torch

        clone = {k: (v.clone() if hasattr(v, "clone") else v) for k, v in target_eef_pose_dict.items()}
        with torch_rng_preserved(torch):
            clean = original(clone, hand_dict, None, env_id, **kw)
        self._labels.append(_np(clean).reshape(-1))
        noisy = original(target_eef_pose_dict, hand_dict, action_noise_dict, env_id, **kw)
        self._executed.append(_np(noisy).reshape(-1))
        return noisy

    # source mixing -------------------------------------------------------------------------------
    def mix_segment(self, generator, local_vars: dict):
        """Grafted after the source segment is sliced: the (possibly mixed) eef poses, targets and hand commands.

        The segments of every source are expressed in the primary's reference-object frame as the mixer expects: for
        rigid objects that frame is the object pose at the segment start (``source_segment``; SoftMimicGen's rope is
        registered into the primary's geometry first, see ``SoftMimicGenRuntime``).
        """
        eef = local_vars["src_subtask_eef_poses"]
        targets = local_vars["src_subtask_target_poses"]
        eef_name = str(local_vars["eef_name"])
        grippers, unchanged = self.segment_hand(local_vars, eef_name)
        self._last_mix = None
        if self.method.mixer is None:
            return eef, targets, unchanged
        subtask_ind = int(local_vars["subtask_ind"])
        primary = int(local_vars["selected_src_demo_ind"])
        bounds = np.asarray(_np(local_vars["all_randomized_subtask_boundaries"][eef_name]), dtype=int)   # (sources, subtasks, 2)
        infos = generator.src_demo_datagen_info_pool.datagen_infos
        object_name = local_vars.get("subtask_object_name")
        soft = bool(local_vars.get("subtask_object_soft", False))
        segments, object_poses = [], []
        for i, info in enumerate(infos):
            start, end = (int(v) for v in bounds[i, subtask_ind])
            eef_i, targets_i, hand_i, object_pose = self.source_segment(local_vars, info, i, eef_name, object_name,
                                                                        start, end, soft)
            object_poses.append(object_pose)
            segments.append((eef_i, targets_i, hand_i))
        context = MixContext(subtask_ind=subtask_ind, primary=primary, primary_eef_start=_np(eef[0]),
                             primary_targets=_np(targets), primary_gripper=_np(grippers), src_object_poses=object_poses,
                             src_segments=segments, num_sources=len(infos), has_reference_object=object_name is not None,
                             rng=np.random.default_rng(0), extra={"eef": eef_name, "soft": soft})
        mixed = self.method.mixer.mix(context)
        self._last_mix = mixed.record
        if not mixed.record.get("mixed"):
            return eef, targets, unchanged
        return self.pack_mixed(mixed, eef, targets, grippers, unchanged, eef_name)

    def record_subtask(self, local_vars: dict) -> None:
        """Grafted after the transformed sequence: the subtask's source, boundaries and object pose, plus the mix record."""
        generator = local_vars["self"]
        eef_name = str(local_vars["eef_name"])
        subtask_ind = int(local_vars["subtask_ind"])
        selected = int(local_vars["selected_src_demo_ind"])
        self._selected[eef_name] = selected
        boundary = [int(v) for v in _np(local_vars[self.boundary_var])]
        object_pose = local_vars.get("subtask_object_pose")
        interpolation, fixed, noise = self.subtask_settings(generator, local_vars, eef_name, subtask_ind)
        self.rec.add_subtask(subtask_ind=subtask_ind, eef=eef_name, object_name=local_vars.get("subtask_object_name"),
                             src_demo_ind=selected, src_subtask_start=boundary[0], src_subtask_end=boundary[1],
                             cur_obj_pose=_np(object_pose) if object_pose is not None else None,
                             num_interpolation_steps=interpolation, num_fixed_steps=fixed, action_noise=noise, mix=self._last_mix)

    # per-step records ----------------------------------------------------------------------------
    def record_command(self, multi_waypoint, env, env_id: int, state, play_action) -> None:
        """Grafted after ``play_action``: remember the step's targets, measured poses and commands."""
        self._pend_command(multi_waypoint, self.measured_poses(env, env_id, list(multi_waypoint.waypoints)), play_action)

    def _pend_command(self, multi_waypoint, measured: dict, play_action) -> None:
        targets = {eef: _np(w.pose) for eef, w in multi_waypoint.waypoints.items()}
        noise = max(float(w.noise) if w.noise is not None else 0.0 for w in multi_waypoint.waypoints.values())
        label = self._labels[-1] if self._labels else _np(play_action).reshape(-1)
        self._pending = {"targets": targets, "measured": measured, "noise": noise, "label": label,
                         "executed": _np(play_action).reshape(-1)}

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

    def note_track(self, row: dict, *, slot_used: bool) -> None:
        """The follower's record of the step about to execute (``record_step`` attaches it to the step)."""
        self._pending_track = {**row, "slot": 1 if slot_used else 0}


class SoftMimicGenRuntime(IsaacLabRuntime):
    """SoftMimicGen's builder is ``generate_trajectory`` with a nodal-position anchor; its rope is a soft object.

    For a soft object every donor segment is first registered into the primary's rope geometry with the official nodal
    registration, after which the object frame is the world frame.
    """

    builder_name = "generate_trajectory"
    mix_anchor = "src_subtask_nodal_positions"

    def source_segment(self, local_vars: dict, info, index: int, eef_name: str, object_name, start: int, end: int,
                       soft: bool) -> tuple:
        if not soft:
            return super().source_segment(local_vars, info, index, eef_name, object_name, start, end, soft)
        from softmimicgen.datagen.data_generator import (
            transform_source_data_segment_using_nodal_registration,
        )

        targets = info.target_eef_pose[eef_name][start:end]
        nodal = info.object_nodal_positions[object_name][start]
        if index != int(local_vars["selected_src_demo_ind"]):
            targets = transform_source_data_segment_using_nodal_registration(
                tgt_obj_nodal_pos=local_vars.get("src_subtask_nodal_positions"), src_eef_poses=targets, src_obj_nodal_pos=nodal,
                use_rotation_transform=True, bend_coef=0.1, rot_coef=1e-3)
        return _np(info.eef_pose[eef_name][start]), _np(targets), _np(self.source_hand(info, eef_name, start, end)), np.eye(4)


def runtime_class(soft: bool) -> type[IsaacLabRuntime]:
    """The runtime of ``spec["extra"]["soft"]``: SoftMimicGen's or Isaac Lab Mimic's."""
    return SoftMimicGenRuntime if soft else IsaacLabRuntime


# grafts ---------------------------------------------------------------------------------------------
def graft_builder(DataGenerator, runtime: IsaacLabRuntime, expected_hash) -> Graft:
    """The trajectory builder's two grafts (mixing, subtask record) at the runtime's class-level anchors."""
    name = runtime.builder_name
    builder = Graft(getattr(DataGenerator, name), expected_hash=expected_hash, label=f"DataGenerator.{name}")
    mix = f"src_subtask_eef_poses, src_subtask_target_poses, {runtime.hand_var} = _tmg.mix_segment(self, locals())"
    builder.insert_after_assign(runtime.mix_anchor, mix)
    builder.insert_after_assign(runtime.record_anchor, "_tmg.record_subtask(locals())")
    return builder


def install(package: str, runtime: IsaacLabRuntime, graft_hashes: dict) -> dict:
    """Install the grafts into the official package's classes; returns their descriptions for the chunk log.

    Args:
        package: ``softmimicgen.datagen`` or ``isaaclab_mimic.datagen``.
        runtime: the callbacks (its class names the trajectory builder and its anchors).
        graft_hashes: pinned source hashes (``generate``, ``trajectory``, ``execute``), None values skip the check.
    """
    dg = importlib.import_module(f"{package}.data_generator")
    wp = importlib.import_module(f"{package}.waypoint")
    DataGenerator, MultiWaypoint = dg.DataGenerator, wp.MultiWaypoint
    gen = Graft(DataGenerator.generate, expected_hash=graft_hashes.get("generate"), label="DataGenerator.generate")
    gen.insert_after_assign("new_initial_state", "_tmg.begin_attempt(self, env_id)")
    gen.insert_before_final_return("_tmg.end_attempt(self, env_id, results)")
    builder = graft_builder(DataGenerator, runtime, graft_hashes.get("trajectory"))
    exe = Graft(MultiWaypoint.execute, expected_hash=graft_hashes.get("execute"), label="MultiWaypoint.execute")
    exe.insert_after_assign("play_action", "_tmg.record_command(self, env, env_id, state, play_action)", occurrence=0)
    exe.insert_after_assign("success", "_tmg.record_step(self, env, env_id, obs, success)")
    DataGenerator.generate = gen.compile({"_tmg": runtime})
    setattr(DataGenerator, runtime.builder_name, builder.compile({"_tmg": runtime}))
    MultiWaypoint.execute = exe.compile({"_tmg": runtime})
    return {"generate": gen.edits, runtime.builder_name: builder.edits, "execute": exe.edits}


def install_from_spec(spec: dict, namespace: dict | None = None) -> dict:
    """Entry used by the chunk entrypoint: build the runtime from the chunk spec, install the grafts and, when the method
    tracks, the shared-path ``generate``. ``namespace`` (the official script's globals) is not needed here."""
    from tmg.generation.method import build_method

    chunk_seed = int(spec["chunk"]["seed"])
    method = build_method(spec["method"], chunk_seed=chunk_seed)
    layout = IsaacLabLayout.from_spec(spec["extra"]["layout"])
    runtime = runtime_class(bool(spec["extra"]["soft"]))(method, layout, Path(spec["records_dir"]))
    info = install(spec["extra"]["package"], runtime, dict(spec["extra"].get("graft_hashes") or {}))
    if method.follower_factory is not None:
        dg = importlib.import_module(f"{spec['extra']['package']}.data_generator")
        dg.DataGenerator.generate = make_tracking_generate(runtime, method.follower_factory,
                                                           private_seed=private_noise_seed(chunk_seed))
        info["tracking"] = "shared-path generate installed"
    return info


# tracking: shared-path loop on the Isaac Lab classes ---------------------------------------------------
class IsaacLabTrackingLoop(TrackingLoop):
    """``TrackingLoop`` hooks for ``softmimicgen.datagen`` / ``isaaclab_mimic.datagen``: a waypoint's hand command is its
    ``gripper_action`` tensor, ``MultiWaypoint.execute`` returns the step's states, observations and actions, and the
    result dict of the official ``generate`` is rebuilt from them."""

    async def begin(self, generator, env_id: int, success_term, env_reset_queue, env_action_queue) -> Attempt:
        import torch

        env = generator.env
        eefs = sorted(generator.env_cfg.subtask_configs.keys())
        env_ids = torch.tensor([env_id], dtype=torch.int64, device=env.device)
        env.recorder_manager.reset(env_ids=env_ids)
        await env_reset_queue.put(env_id)
        await env_reset_queue.join()
        initial_state = env.scene.get_state(is_relative=True)
        self.runtime.begin_attempt(generator, env_id)
        return Attempt(generator, env_id, success_term, env_action_queue, eefs, initial_state, env_ids,
                       extra={"states": [], "observations": [], "actions": []})

    def lock(self, attempt: Attempt):
        return attempt.generator.src_demo_datagen_info_pool.asyncio_lock

    def build(self, attempt: Attempt, eef: str, subtask_ind: int, boundaries: dict, selected: dict, previous) -> list:
        generator = attempt.generator
        if hasattr(generator, "generate_trajectory"):      # Soft: one builder returns the merged trajectory
            return generator.generate_trajectory(attempt.env_id, eef, subtask_ind, boundaries, {}, selected, previous)
        traj = generator.generate_eef_subtask_trajectory(attempt.env_id, eef, subtask_ind, boundaries, {}, selected)
        return generator.merge_eef_subtask_trajectory(attempt.env_id, eef, subtask_ind, previous, traj)

    def subtask_steps(self, attempt: Attempt, eef: str) -> list[tuple[int, int]]:
        return [(c.num_interpolation_steps, c.num_fixed_steps) for c in attempt.generator.env_cfg.subtask_configs[eef]]

    def num_subtasks(self, attempt: Attempt, eef: str) -> int:
        return len(attempt.generator.env_cfg.subtask_configs[eef])

    def waypoint_classes(self, attempt: Attempt) -> tuple:
        wp = importlib.import_module(type(attempt.generator).__module__.rsplit(".", 1)[0] + ".waypoint")
        return wp.Waypoint, wp.MultiWaypoint

    def waypoint_gripper(self, waypoint, eef: str) -> np.ndarray:
        return _np(waypoint.gripper_action)

    def clone_hand(self, waypoint):
        return waypoint.gripper_action.clone()

    def hand_command(self, base, eef: str, gripper: np.ndarray, device):
        import torch

        return torch.as_tensor(gripper, device=device, dtype=base.gripper_action.dtype)

    async def execute(self, attempt: Attempt, multi_waypoint) -> bool:
        result = await multi_waypoint.execute(attempt.generator.env, attempt.success_term, attempt.env_id,
                                              attempt.env_action_queue)
        if not result["states"]:
            return False
        attempt.extra["states"].extend(result["states"])
        attempt.extra["observations"].extend(result["observations"])
        attempt.extra["actions"].extend(result["actions"])
        return result["success"]

    def export(self, attempt: Attempt, success: bool, export_demo: bool) -> dict:
        import torch

        env = attempt.generator.env
        actions = torch.cat(attempt.extra["actions"], dim=0)
        verdict = torch.tensor([[success]], dtype=torch.bool, device=env.device)
        env.recorder_manager.set_success_to_episodes(attempt.env_ids, verdict)
        if export_demo:
            env.recorder_manager.export_episodes(attempt.env_ids)
        return {"initial_state": attempt.initial_state, "states": attempt.extra["states"],
                "observations": attempt.extra["observations"], "actions": actions, "success": success}


def make_tracking_generate(runtime: IsaacLabRuntime, follower_factory: Callable, private_seed: int):
    """Build the replacement ``DataGenerator.generate`` (official signature) that drives paired trajectories with one follower."""
    loop = IsaacLabTrackingLoop(runtime, follower_factory, private_seed)

    async def generate(self, env_id, success_term, env_reset_queue=None, env_action_queue=None, pause_subtask=False,
                       export_demo=True, motion_planner=None):
        if pause_subtask or motion_planner is not None:
            raise NotImplementedError("tracking generate supports neither pause_subtask nor motion planners")
        attempt = await loop.begin(self, env_id, success_term, env_reset_queue, env_action_queue)
        return await loop.run(attempt, export_demo)

    return generate


def _clone_gripper(waypoint):
    return waypoint.gripper_action.clone()


def _cut_interval(active: dict, eefs: list[str], start: int, Waypoint, previous_last: dict) -> Interval:
    """``tracking_loop.cut_interval`` for Isaac Lab waypoints (hand command: the ``gripper_action`` tensor)."""
    return cut_interval(active, eefs, start, Waypoint, previous_last, _clone_gripper)
