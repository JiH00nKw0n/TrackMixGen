"""The ``_tmg`` callbacks for the Isaac Lab mimic generators.

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

Scene of a subtask (``subtask_scene``, read once per builder call into a ``SubtaskScene`` of
``tmg.generation.method.interfaces``): the reference object's pose now and in every source at the start of its segment, the
end-effector pose now and in every source, and the previous subtask's reference object of the same end effector (the
carried object) now and in every source, kept only where the hand holds it. Whether a hand holds it is the hold test
(``held_distance`` against the ``HeldRule`` of the chunk spec's ``scene_reading``, from the generator config
``scene_reading.held_rule``): a grasp leaves the object at the hand, while the previous subtask's reference object of a
hand that only reached, pushed or supported it is 0.4 to 1.3 m away (GR1 Bottle Placement pilot, 2026-11-27).
``mix_segment`` hands the scene to the mixer (``MixContext.scene``) and ``record_subtask`` writes the selected source's part
into the attempt record (``SubtaskScene.record``: ``cur_obj_pose``, ``src_obj_pose``, ``eef_pose``, ``src_eef_pose``,
``objects``, ``src_objects``, ``carried_object``, ``carried_candidate``, ``held_distance``, ``held_rule``,
``object_frame``) with ``term_signal`` and ``start_step``, the number of steps executed before the subtask was built (the
builders make a subtask when the end effector's previous one is exhausted), which is what ``tmg fit_coefficients`` reads.
The record is written for every method, the unmixed baseline included; the values are read from the simulation and the
source demonstrations and change nothing in the generation. A deformable reference object (SoftMimicGen's rope and
towel) has the nodal frame of ``tmg.generation.method.soft_frame`` (also recorded: ``object_frame``).

A scene that cannot be read (for example a towel folded so far that its nodal frame has no yaw) never stops the chunk:
the subtask record says why (``scene_error``), the attempt counts its errors (``scene_errors``) and the generation goes
on with the selected source's own segment. With a mixer that subtask was not mixed as the method says, so the mix record
says so (``mixed: false``, ``identity_reason: scene_error ...``) and the attempt is marked ``invalid: scene_error``
(a termination signal that cannot be read is a ``scene_error`` of the record too, but the subtask is still mixed):
``tmg dataset`` and ``tmg fit_coefficients`` leave such attempts out, and the adapter fails the chunk and stops the pool
when the pool's finished chunks show that more than ``record_checks.max_scene_error_rate`` of the attempts are affected
(``adapter.records_check``; never one chunk's share). The
official success verdict of the attempt is not changed: the official recorder has exported the episode by then, and the
k-th demonstration of its file must stay the k-th successful attempt of the records.

Pre-noise labels: for the attempt's duration ``env.target_eef_pose_to_action`` is replaced by a wrapper that
first computes the noiseless action (with the torch RNG state saved and restored around that call, so the
official draw sequence is unchanged) and keeps it as the label, then calls the official method with the
noise. The official recorder writes the executed actions to the hdf5; the labels live in our records, and
``tmg dataset`` reads ``actions_pre_noise`` from there for these generators.
"""
from __future__ import annotations

import importlib
import json
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tmg.core.poses import as_array as _np
from tmg.core.records import LABELS_FORMAT, AttemptRecorder, record_file
from tmg.generation.graft import Graft
from tmg.generation.method.interfaces import HeldRule, HeldTest, Method, MixContext, SubtaskScene
from tmg.generation.simulators.isaaclab import HandLayout, measured_eef_poses, observed_eef_poses

RECORD_ERROR = "RECORD_ERROR.json"
INVALID_SCENE = "scene_error"      # ``invalid`` of an attempt of a mixing method with a subtask whose scene could not be read


@dataclass
class IsaacLabLayout:
    """What differs between the Isaac Lab generators (filled from the generator and task configs)."""

    eefs: list[str]                                   # end-effector names in the official order (sorted)
    arm_action_dims: dict[str, tuple[int, int]]       # eef -> (start, end) of its arm pose block in the action
    hand_prefixes: dict[str, str] | None              # eef -> joint-name prefix (dexterous hands), or None (gripper)
    hand_term: str | None                             # action term holding the hand joint names (GR1: "upper_body_ik")
    measure_links: dict[str, str] | None              # eef -> body whose pose is recorded as the measured pose (None: observed eef)
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


def held_distance(eef_pose, object_pose, nodes=None) -> float:
    """How far the end effector is from an object (m): from its nearest node for a deformable object (``nodes`` (N, 3)),
    else from its origin. Compared with the ``HeldRule`` limits to decide whether the object is in the hand."""
    position = np.asarray(eef_pose, dtype=float)[:3, 3]
    if nodes is not None:
        return float(np.linalg.norm(np.asarray(nodes, dtype=float) - position, axis=1).min())
    return float(np.linalg.norm(np.asarray(object_pose, dtype=float)[:3, 3] - position))


@dataclass
class SubtaskRead:
    """What the runtime read for the subtask the builder is making: its scene and termination signal, and the first
    failure (``error``, "<field>: <exception>"). A scene that failed is None; the mixer needs the scene only."""
    eef: str
    subtask: int
    scene: SubtaskScene | None = None
    term_signal: str | None = None
    error: str | None = None


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

    def __init__(self, method: Method, layout: IsaacLabLayout, records_dir: Path | None, held_rule: HeldRule | None = None):
        """``method``: built with the layout's end effectors (``build_method(..., arms=layout.eefs)``). ``held_rule``: the
        hold test of the chunk spec's ``scene_reading``; None makes no test (a carried object counts as held)."""
        self.method = method
        self.layout = layout
        self.held_rule = held_rule
        self.rec = AttemptRecorder()
        self.rec.configure(records_dir)
        self.records_dir = records_dir
        self._attempt_t0 = 0.0
        # what was read and mixed for the subtask each end effector's builder call is making (mix_segment, then
        # record_subtask): keyed by end effector, so two arms whose builder calls interleave cannot swap them
        self._reads: dict[str, SubtaskRead] = {}
        self._mixes: dict[str, dict] = {}
        self._labels: list[np.ndarray] = []
        self._executed: list[np.ndarray] = []
        self._pending: dict | None = None
        self._selected: dict[str, int] = {}           # eef -> source demonstration chosen in this attempt
        self._carried: dict[str, tuple] = {}          # eef -> (reference object, soft) of its previous subtask in this attempt
        self._scene_errors = 0                        # subtask records of this attempt without their scene or term signal
        self._not_mixed = False                       # a mixing method left a subtask of this attempt unmixed for a scene error
        self._warned_scene = False
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

    def subtask_term_signal(self, generator, local_vars: dict, eef_name: str, subtask_ind: int) -> str | None:
        """The name of the subtask's termination signal (the environment config's ``subtask_term_signal``), or None."""
        cfg = generator.env_cfg.subtask_configs[eef_name][subtask_ind]
        signal = getattr(cfg, "subtask_term_signal", None)
        return None if signal is None else str(signal)

    def current_eef_pose(self, generator, env_id: int, eef_name: str) -> np.ndarray:
        """The end effector's pose now, in the frame of the sources' ``eef_pose`` (the controller's eef frame)."""
        return _np(generator.env.get_robot_eef_pose(eef_name=eef_name, env_ids=[env_id])[0])

    def current_object_poses(self, generator, env_id: int) -> dict:
        """Every rigid object's pose now (4x4), in the frame of the sources' ``object_poses``."""
        return {k: _np(v[0]) for k, v in generator.env.get_object_poses(env_ids=[env_id]).items()}

    def soft_object_poses(self, generator, local_vars: dict, infos, object_name: str, starts: list) -> tuple:
        """``(current pose, per-source poses, frame record)`` of a deformable reference object (SoftMimicGen only)."""
        raise NotImplementedError(f"{type(self).__name__}: no deformable objects")

    def bind_official(self, DataGenerator) -> None:
        """Read what the mixing takes from the official generator class itself (nothing here). ``install`` calls it only
        for a method with a mixer, so an unmixed run never depends on it."""

    def official_settings(self) -> dict:
        """What ``bind_official`` read, for the graft log (nothing here)."""
        return {}

    def mix_settings(self, soft: bool) -> dict:
        """Mix-record fields about what the runtime did to the sources before the mixer saw them (nothing here)."""
        return {}

    def measured_poses(self, owner, env_id: int, eefs) -> dict:
        """Measured end-effector poses: the configured links (``measure_links``) or the environment's observed eef poses."""
        if self.layout.measure_links:
            return measured_eef_poses(owner, env_id, self.layout.measure_links)
        return observed_eef_poses(owner, env_id, list(eefs))

    # scene of a subtask ---------------------------------------------------------------------------
    def read_subtask(self, generator, local_vars: dict) -> SubtaskRead:
        """The scene and termination signal of the subtask the builder is making, read once per builder call; a failure
        is kept as ``error`` (module docstring), never raised."""
        eef, k = str(local_vars["eef_name"]), int(local_vars["subtask_ind"])
        cached = self._reads.get(eef)
        if cached is not None and cached.subtask == k:
            return cached
        read = SubtaskRead(eef=eef, subtask=k)
        for field, reader in (("scene", lambda: self.subtask_scene(generator, local_vars)),
                              ("term_signal", lambda: self.subtask_term_signal(generator, local_vars, eef, k))):
            try:
                setattr(read, field, reader())
            except Exception as error:                 # noqa: BLE001  any failure to read is one scene error of the attempt
                read.error = read.error or f"{field}: {type(error).__name__}: {error}"
        if read.error is not None:
            self._scene_errors += 1
            if not self._warned_scene:
                self._warned_scene = True
                print(f"[tmg] WARNING subtask {eef}/{k}: not read ({read.error}); the attempt is left out of "
                      f"tmg fit_coefficients" + (", and of the datasets when the subtask cannot be mixed for it"
                                                 if self.method.mixer is not None else ""), flush=True)
        self._reads[eef] = read
        return read

    def last_mix(self, eef: str) -> dict | None:
        """The mix record of the subtask ``eef``'s builder call is making (None: not mixed, or no mixer)."""
        return self._mixes.get(str(eef))

    @staticmethod
    def source_bounds(local_vars: dict, eef: str) -> np.ndarray:
        """Every source's (start, end) of every subtask of ``eef``, ``(sources, subtasks, 2)``, as the builder randomized
        them for this attempt."""
        return np.asarray(_np(local_vars["all_randomized_subtask_boundaries"][eef]), dtype=int)

    def subtask_scene(self, generator, local_vars: dict) -> SubtaskScene:
        """The ``SubtaskScene`` of the subtask the builder is making (module docstring)."""
        eef = str(local_vars["eef_name"])
        k = int(local_vars["subtask_ind"])
        infos = generator.src_demo_datagen_info_pool.datagen_infos
        starts = [int(start) for start in self.source_bounds(local_vars, eef)[:, k, 0]]
        env_id = int(local_vars["env_id"])
        obj = local_vars.get("subtask_object_name")
        soft = bool(local_vars.get("subtask_object_soft", False))
        frame = None
        if obj is None:
            current, sources = None, [None] * len(infos)
        elif soft:
            current, sources, frame = self.soft_object_poses(generator, local_vars, infos, str(obj), starts)
        else:
            pose_now = local_vars.get("subtask_object_pose")
            current = _np(pose_now) if pose_now is not None else self.current_object_poses(generator, env_id)[obj]
            sources = [_np(info.object_poses[obj][t]) for info, t in zip(infos, starts)]
        scene = SubtaskScene(subtask=k, eef=eef, reference_object=None if obj is None else str(obj),
                             current_object_pose=current, source_object_poses=sources,
                             eef_pose=self.current_eef_pose(generator, env_id, eef),
                             source_eef_poses=[_np(info.eef_pose[eef][t]) for info, t in zip(infos, starts)],
                             object_frame=frame, held_rule=self.held_rule)
        carried, carried_soft = self._carried.get(eef, (None, False))
        if carried is not None:
            scene.carried_object = carried
            nodes_now = nodes_src = None
            if carried == scene.reference_object:              # the same object (Rope, Towel, Microwave): its poses above
                now_pose, src_poses = current, sources
                if soft:
                    nodes_now, nodes_src = self.held_nodes(local_vars, infos, str(carried), starts)
            elif not carried_soft:                             # a rigid object carried to another one (Bottle to the fridge)
                now_pose = self.current_object_poses(generator, env_id)[carried]
                src_poses = [_np(info.object_poses[carried][t]) for info, t in zip(infos, starts)]
            else:      # a deformable object carried to another object: its frame and hold test are not implemented
                raise NotImplementedError(f"subtask {eef}/{k}: the carried object {carried!r} is deformable and the reference "
                                          f"object is {scene.reference_object!r}; the in-hand pose of a deformable object "
                                          f"against another reference object is not implemented")
            self._hold(scene, now_pose, src_poses, nodes_now, nodes_src, soft and carried == scene.reference_object)
        return scene

    def held_nodes(self, local_vars: dict, infos, object_name: str, starts: list) -> tuple:
        """``(nodes now, per-source nodes)`` of a deformable object for the hold test (SoftMimicGen only)."""
        raise NotImplementedError(f"{type(self).__name__}: no deformable objects")

    def _hold(self, scene: SubtaskScene, now_pose, src_poses, nodes_now, nodes_src, deformable: bool) -> None:
        """Keep the carried object's pose only where it is held (the hold test of the module docstring): now, and in
        every source by itself, so a source that holds an object the hand does not hold now (or the reverse) is told
        apart from one in the same state (``inhand_mismatch``). Without a hold rule the pose is kept everywhere."""
        if self.held_rule is None:
            scene.carried_pose, scene.source_carried_poses = now_pose, list(src_poses)
            return
        now = held_distance(scene.eef_pose, now_pose, nodes_now)
        dist = [held_distance(e, p, None if nodes_src is None else nodes_src[j])
                for j, (e, p) in enumerate(zip(scene.source_eef_poses, src_poses))]
        scene.held = HeldTest(rule=self.held_rule, deformable=bool(deformable), now_m=now, sources_m=dist)
        scene.carried_pose = now_pose if scene.held.now else None
        scene.source_carried_poses = [p if held else None for p, held in zip(src_poses, scene.held.sources)]

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
        self._labels, self._executed, self._pending = [], [], None
        self._selected = {}
        self._carried, self._reads, self._mixes, self._scene_errors, self._not_mixed = {}, {}, {}, 0, False
        self.rec.begin_attempt()
        self._seg_id = -1
        if self.method.mixer is not None:
            self.method.mixer.begin_attempt()              # a new attempt draws new randmix weights
        if self._hand_layout is None:
            self._hand_layout = self.layout.hand_layout(self.environment(generator))
        self._install_label_wrapper(self.action_owner(generator), env_id)

    def end_attempt(self, generator, env_id: int, results: dict) -> None:
        """Grafted before ``return results``: restore the action method and flush the record."""
        self._restore_label_wrapper()
        try:
            self.rec.add_extra(elapsed_s=round(time.time() - self._attempt_t0, 3), labels=self.method.labels,
                               num_label_steps=len(self._labels), executed_steps=len(self._executed),
                               scene_errors=self._scene_errors)
            if self._not_mixed:
                self.rec.add_extra(invalid=INVALID_SCENE)      # a subtask was not mixed as the method says
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
        self._mixes.pop(eef_name, None)
        if self.method.mixer is None:
            return eef, targets, unchanged
        subtask_ind = int(local_vars["subtask_ind"])
        primary = int(local_vars["selected_src_demo_ind"])
        bounds = self.source_bounds(local_vars, eef_name)
        infos = generator.src_demo_datagen_info_pool.datagen_infos
        object_name = local_vars.get("subtask_object_name")
        soft = bool(local_vars.get("subtask_object_soft", False))
        read = self.read_subtask(generator, local_vars)
        if read.scene is None:                             # not mixed, and said so: the attempt is marked invalid
            self._not_mixed = True
            self._mixes[eef_name] = {"subtask": subtask_ind, "eef": eef_name, "source_a": primary, "mixed": False,
                                     "identity_reason": f"{INVALID_SCENE}: {read.error}"}
            return eef, targets, unchanged
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
                             scene=read.scene)
        mixed = self.method.mixer.mix(context)
        mixed.record.update(self.mix_settings(soft))       # what the runtime itself did to the sources, next to the mixer's record
        self._mixes[eef_name] = mixed.record
        if not mixed.record.get("mixed"):
            return eef, targets, unchanged
        return self.pack_mixed(mixed, eef, targets, grippers, unchanged, eef_name)

    def record_subtask(self, local_vars: dict) -> None:
        """Grafted after the transformed sequence: the subtask's source, boundaries and scene (module docstring), its
        termination signal, plus the mix record."""
        generator = local_vars["self"]
        eef_name = str(local_vars["eef_name"])
        subtask_ind = int(local_vars["subtask_ind"])
        selected = int(local_vars["selected_src_demo_ind"])
        object_name = local_vars.get("subtask_object_name")
        self._selected[eef_name] = selected
        boundary = [int(v) for v in _np(local_vars[self.boundary_var])]
        object_pose = local_vars.get("subtask_object_pose")
        interpolation, fixed, noise = self.subtask_settings(generator, local_vars, eef_name, subtask_ind)
        fields = {"cur_obj_pose": _np(object_pose) if object_pose is not None else None}
        read = self.read_subtask(generator, local_vars)
        if read.scene is not None:
            fields.update(read.scene.record(selected))
        fields["term_signal"] = read.term_signal
        if read.error is not None:
            fields["scene_error"] = read.error
        self._reads.pop(eef_name, None)
        self._carried[eef_name] = (object_name, bool(local_vars.get("subtask_object_soft", False)))
        self.rec.add_subtask(subtask_ind=subtask_ind, eef=eef_name, object_name=object_name,
                             start_step=len(self.rec.w_pos),          # steps executed before this subtask was built
                             src_demo_ind=selected, src_subtask_start=boundary[0], src_subtask_end=boundary[1],
                             num_interpolation_steps=interpolation, num_fixed_steps=fixed, action_noise=noise,
                             mix=self._mixes.pop(eef_name, None), **fields)

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
                          noise=p["noise"], seg_id=self._seg_id, term_signals=signals)
        self.rec.add_success(bool(success))



class SoftMimicGenRuntime(IsaacLabRuntime):
    """SoftMimicGen's builder is ``generate_trajectory`` with a nodal-position anchor; its rope is a soft object.

    For a soft object every donor segment (its start pose and its targets) is first registered into the primary's rope
    geometry with the official nodal registration (with the settings of the official builder's own call,
    ``registration_kwargs``), after which the blending frame is the world frame. The sources are
    scored with the nodal frame of ``tmg.generation.method.soft_frame`` (reference: source 0's first node cloud); the current
    nodes are the builder's ``subtask_nodal_positions``.
    """

    builder_name = "generate_trajectory"
    mix_anchor = "src_subtask_nodal_positions"
    current_nodes_var = "subtask_nodal_positions"        # the builder's local holding the reference object's nodes now
    registration_function = "transform_source_data_segment_using_nodal_registration"

    registration_data = ("tgt_obj_nodal_pos", "src_eef_poses", "src_obj_nodal_pos")   # the arguments source_segment passes itself

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the settings (bend_coef, rot_coef, use_rotation_transform) the official builder registers the selected source
        # with, read from its own call (``bind_official``); None until then: no donor is registered without them
        self.registration_kwargs: dict | None = None

    def bind_official(self, DataGenerator) -> None:
        self.registration_kwargs = literal_call_keywords(getattr(DataGenerator, self.builder_name), self.registration_function,
                                                         passed=self.registration_data)

    def official_settings(self) -> dict:
        return {} if self.registration_kwargs is None else {self.registration_function: dict(self.registration_kwargs)}

    def mix_settings(self, soft: bool) -> dict:
        """A deformable subtask's donors went through the official registration: with which settings."""
        return {"registration": dict(self.registration_kwargs)} if soft else {}

    def soft_object_poses(self, generator, local_vars: dict, infos, object_name: str, starts: list) -> tuple:
        from tmg.generation.method.soft_frame import (
            FRAME_VERSION,
            nodal_frame,
            reference_hash,
            reference_nodes,
        )

        if local_vars.get(self.current_nodes_var) is None:
            raise KeyError(f"SoftMimicGen builder has no {self.current_nodes_var!r} when the subtask is built; the nodal "
                           f"frame of {object_name!r} needs the current node positions")
        reference = reference_nodes(infos, object_name)
        current = nodal_frame(_np(local_vars[self.current_nodes_var]), reference)
        sources = [nodal_frame(_np(info.object_nodal_positions[object_name][t]), reference) for info, t in zip(infos, starts)]
        return current, sources, {"version": FRAME_VERSION, "reference_sha256": reference_hash(reference)}

    def held_nodes(self, local_vars: dict, infos, object_name: str, starts: list) -> tuple:
        return (_np(local_vars[self.current_nodes_var]),
                [_np(info.object_nodal_positions[object_name][t]) for info, t in zip(infos, starts)])

    def source_segment(self, local_vars: dict, info, index: int, eef_name: str, object_name, start: int, end: int,
                       soft: bool) -> tuple:
        if not soft:
            return super().source_segment(local_vars, info, index, eef_name, object_name, start, end, soft)
        from softmimicgen.datagen.data_generator import (
            transform_source_data_segment_using_nodal_registration,
        )

        eef_start = info.eef_pose[eef_name][start:start + 1]
        targets = info.target_eef_pose[eef_name][start:end]
        nodal = info.object_nodal_positions[object_name][start]
        if index != int(local_vars["selected_src_demo_ind"]):
            import torch

            if self.registration_kwargs is None:
                raise RuntimeError("the official builder's registration settings were not read (bind_official): a donor is "
                                   "never registered with settings other than the selected source's")

            # the start pose goes through the same registration as the targets: left in the donor's geometry it would
            # put the blended first waypoint off the primary's rope
            joined = transform_source_data_segment_using_nodal_registration(
                tgt_obj_nodal_pos=local_vars.get("src_subtask_nodal_positions"), src_eef_poses=torch.cat([eef_start, targets], dim=0),
                src_obj_nodal_pos=nodal, **self.registration_kwargs)
            eef_start, targets = joined[0:1], joined[1:]
        return _np(eef_start)[0], _np(targets), _np(self.source_hand(info, eef_name, start, end)), np.eye(4)


def literal_value(node, what: str):
    """The value of a literal expression of the official source: numbers, strings, booleans, None, signed numbers,
    ``not <literal>`` and tuples or lists of those. Anything else (a name, an attribute, a call) raises: its value is
    not in the source text."""
    import ast

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
        return not literal_value(node.operand, what)
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        raise RuntimeError(f"{what} is {ast.unparse(node)!r}, not a literal: its value cannot be read from the official source") from None


def literal_call_keywords(function, callee: str, passed: tuple = ()) -> dict:
    """The keyword arguments of the one call of ``callee`` in ``function``'s source, other than ``passed`` (the arguments
    the caller supplies itself), with their literal values. The official builder's registration call passes its
    coefficients as literals, not from a configuration, so this is where they are written. Strict, because a setting
    read wrongly or not at all would register the donors differently from the selected source without a trace: no such
    call or more than one, a positional argument, ``**kwargs``, or a value that is not a literal (``literal_value``)
    raises."""
    import ast
    import inspect
    import textwrap

    where = f"{getattr(function, '__qualname__', function)}: the call of {callee}"
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "id", getattr(n.func, "attr", None)) == callee]
    if len(calls) != 1:
        raise RuntimeError(f"{where}: expected one call, found {len(calls)}")
    call = calls[0]
    if call.args:
        raise RuntimeError(f"{where} has {len(call.args)} positional arguments; which settings they are cannot be read from "
                           f"the call (expected keyword arguments only)")
    out = {}
    for kw in call.keywords:
        if kw.arg is None:
            raise RuntimeError(f"{where} passes **{ast.unparse(kw.value)}: its settings cannot be read from the call")
        if kw.arg not in passed:
            out[kw.arg] = literal_value(kw.value, f"{where}: {kw.arg}")
    missing = [name for name in passed if name not in {kw.arg for kw in call.keywords}]
    if missing:
        raise RuntimeError(f"{where} does not pass {missing} by keyword: not the call this runtime mirrors")
    return out


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
    if runtime.method.mixer is not None:               # only mixing reads the official class; before the builder is grafted
        runtime.bind_official(DataGenerator)
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
    return {"generate": gen.edits, runtime.builder_name: builder.edits, "execute": exe.edits,
            **installed_about(runtime, {"generate": gen.hash, "trajectory": builder.hash, "execute": exe.hash}, graft_hashes)}


def installed_about(runtime: IsaacLabRuntime, hashes: dict, pinned: dict) -> dict:
    """For the graft log: the source hashes of the grafted official functions (the values to pin under the generator
    config's ``graft_hashes``) and what the runtime read from the official code (``official_settings``). A mixing run on
    a builder whose hash is not pinned says so loudly: its mixing mirrors that builder's source."""
    unpinned = sorted(name for name, value in hashes.items() if not pinned.get(name))
    if unpinned and runtime.method.mixer is not None:
        print(f"[tmg] WARNING mixing on official functions whose source hash is not pinned ({unpinned}); pin them in the "
              f"generator config: graft_hashes: {json.dumps(hashes)}", flush=True)
    return {"hashes": hashes, "official_settings": runtime.official_settings()}


def held_rule_of(spec: dict) -> HeldRule | None:
    """The hold rule of a chunk spec (``scene_reading.held_rule``, from the generator config), None when it has none."""
    return HeldRule.from_mapping((spec.get("scene_reading") or {}).get("held_rule"))


def install_from_spec(spec: dict, namespace: dict | None = None) -> dict:
    """Entry used by the chunk entrypoint: build the runtime from the chunk spec and install the grafts. ``namespace`` (the
    official script's globals) is not needed here."""
    from tmg.generation.method import build_method

    layout = IsaacLabLayout.from_spec(spec["extra"]["layout"])
    method = build_method(spec["method"], chunk_seed=int(spec["chunk"]["seed"]), arms=layout.eefs)
    runtime = runtime_class(bool(spec["extra"]["soft"]))(method, layout, Path(spec["records_dir"]), held_rule=held_rule_of(spec))
    info = install(spec["extra"]["package"], runtime, dict(spec["extra"].get("graft_hashes") or {}))
    return info
