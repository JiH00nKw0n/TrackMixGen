"""Callbacks grafted into CP-Gen's ``DemoGenerator.generate_demo`` (``demo_aug.generate``).

How CP-Gen produces one attempt: the environment is reset, then ``CPPolicy.get_action`` is called once per
control step. When its action queue is empty the policy solves the next constraint (an end-effector pose
sequence from the annotated source segment, re-solved for the current object poses) and motion-plans to its
start with cuRobo; the planned steps (``curr_motion_plan_steps``) followed by the constraint steps
(``curr_constraint_steps``) fill ``action_queue`` and one action is popped per step. ``generate_demo`` adds the
official uniform system noise to the popped action before stepping and stores the popped (noise-free) action
as the label, so CP-Gen's saved actions are already pre-noise labels.

Anchors in ``generate_demo`` (by the name the official statement assigns):

| anchor                                               | grafted code                                                            |
|------------------------------------------------------|-------------------------------------------------------------------------|
| after the first ``state = self.env.get_state()``     | ``_tmg.begin_attempt(self)``                                            |
| ``action_dct = self.policy.get_action(obs)``         | becomes ``action_dct = _tmg.get_action(self, obs)``                     |
| after the in-loop ``state = self.env.get_state()``   | ``_tmg.record_step(action, action_to_take, reward, in_constraint_segment, system_noise_cfg)`` |
| before the final ``return episode_data``             | ``_tmg.end_attempt(self, episode_data)``                                |

``get_action`` calls the official ``CPPolicy.get_action`` and, each time the policy has just planned a segment
(``new_constraint``), records the segment (bridge = motion-plan steps, skill = constraint steps) and the source
index. With tracking on, the planned actions become one ``EndEffectorPath`` (absolute poses from the action's
position and axis-angle, gripper from its last element) and the error-adaptive follower returns the next
target from the measured end-effector pose of the observation; the policy's queue is kept in step with the
follower's base index so the official noise rule (``in_constraint_segment``) still reads the right segment,
and when the path is finished the queue is emptied and ``check_constraint`` set, exactly the state the
official pop leaves behind. Source mixing has no counterpart in CP-Gen (it replays no source segment, it
re-solves constraints), so the generator config disables mixing. ``method.labels`` is likewise fixed to
``pre_noise`` because the official output already is; the executed (noisy) actions are kept in our records.

Source selection (``selection.all_sources``, ``extra["all_sources"]`` of the chunk spec): CP-Gen picks the constraint
sequence (source demonstration) only in its reset-near-constraint mode and otherwise always uses sequence 0, so with
every source in use :class:`SourcePicker` draws one per attempt from the chunk's private stream (``seeds.source_pick_rng``).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tmg.core.records import SEG_BRIDGE, SEG_SKILL, AttemptRecorder
from tmg.generation.execution import SegmentTable
from tmg.generation.generators._shared import record_error_guard
from tmg.generation.graft import Graft, Installed, install_grafts
from tmg.generation.method.interfaces import Method
from tmg.generation.method.tracking import EndEffectorPath
from tmg.generation.seeds import source_pick_rng
from tmg.generation.simulators.robosuite import SCENE_META_ATTR

EEF = "robot0"
_INSTALLED: Installed | None = None  # the graft of this process (apply installs once; restore undoes it)


# pose conversions (CP-Gen actions are absolute world-frame [pos 3, axis-angle 3, gripper 1]) --------------
def action_to_pose(action) -> np.ndarray:
    """4x4 pose of an absolute CP-Gen action (position and axis-angle rotation vector)."""
    a = np.asarray(action, dtype=float).reshape(-1)
    pose = np.eye(4)
    pose[:3, 3] = a[:3]
    pose[:3, :3] = Rotation.from_rotvec(a[3:6]).as_matrix()
    return pose


def pose_to_action(pose, gripper) -> np.ndarray:
    """Absolute CP-Gen action from a 4x4 pose and a gripper command."""
    pose = np.asarray(pose, dtype=float)
    return np.concatenate([pose[:3, 3], Rotation.from_matrix(pose[:3, :3]).as_rotvec(),
                           np.asarray(gripper, dtype=float).reshape(-1)])


def pose_from_obs(obs: dict) -> np.ndarray:
    """4x4 end-effector pose from robomimic's ``robot0_eef_pos`` and ``robot0_eef_quat`` (x, y, z, w)."""
    pose = np.eye(4)
    pose[:3, 3] = np.asarray(obs["robot0_eef_pos"], dtype=float).reshape(3)
    pose[:3, :3] = Rotation.from_quat(np.asarray(obs["robot0_eef_quat"], dtype=float).reshape(4)).as_matrix()
    return pose


class SourcePicker:
    """How the source demonstration (CP-Gen constraint sequence) of an attempt is chosen.

    ``all_sources=False`` is the official behaviour: the policy's current sequence is left alone (sequence 0 unless
    CP-Gen's own reset-near-constraint mode picked one). ``all_sources=True`` (selection=ours) draws a sequence index
    per attempt from the chunk's private stream and restarts that sequence at its first constraint.
    """

    def __init__(self, all_sources: bool, chunk_seed: int):
        self.all_sources = bool(all_sources)
        self._rng = source_pick_rng(chunk_seed)

    def pick(self, policy) -> None:
        """Set the policy's constraint sequence for the attempt that starts now (no-op in the official mode)."""
        sequences = getattr(policy, "constraint_sequences", None)
        if self.all_sources and sequences is not None and len(sequences) > 1:
            policy.current_constraint_sequence_idx = int(self._rng.integers(len(sequences)))
            policy.current_constraint_idx = 0


class CPGenRuntime:
    """The ``_tmg`` object the grafted ``generate_demo`` calls (anchors in the module docstring)."""

    def __init__(self, method: Method, chunk_seed: int, records_dir: Path | None, all_sources: bool = False):
        self.method = method
        self.picker = SourcePicker(all_sources, chunk_seed)
        self.rec = AttemptRecorder()
        self.rec.configure(records_dir)
        self.records_dir = records_dir
        self._attempt_t0 = 0.0
        self._reset_attempt_state()

    @property
    def all_sources(self) -> bool:
        """True when every source demonstration is in use (one picked per attempt)."""
        return self.picker.all_sources

    def _reset_attempt_state(self) -> None:
        self.follower = None
        self._path: list[np.ndarray] = []
        self._segments = SegmentTable()                   # record segments of the current planned path (bridge, skill)
        self._pos = 0                                    # executed steps of the current path (identity mode)
        self._subtasks = 0
        self._gripper_states: dict = {}
        self._measured_before: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._track: dict | None = None
        self._labels: list[np.ndarray] = []
        self._executed: list[np.ndarray] = []

    # attempt boundaries --------------------------------------------------------------------------
    def begin_attempt(self, generator) -> None:
        """Grafted after the reset: open the record, pick the attempt's source and read our scene metadata from the
        robosuite env."""
        self._attempt_t0 = time.time()
        self._reset_attempt_state()
        self.rec.begin_attempt()
        self.picker.pick(generator.policy)
        robosuite_env = getattr(getattr(generator.env, "env", None), "env", None)
        self.rec.set_scene_meta(getattr(robosuite_env, SCENE_META_ATTR, None))

    def end_attempt(self, generator, episode_data: dict) -> None:
        """Grafted before ``generate_demo`` returns: flush the attempt record (source index per segment)."""
        with record_error_guard(self.records_dir, self.rec):
            src = int(getattr(generator.policy, "current_constraint_sequence_idx", 0))
            self.rec.add_extra(elapsed_s=round(time.time() - self._attempt_t0, 3), labels="pre_noise",
                               failure_type=episode_data.get("failure_type"), executed_steps=len(self._executed))
            self.rec.flush(success=bool(episode_data.get("success", False)), src_demo_inds=[src] * self._subtasks)

    # per step ------------------------------------------------------------------------------------
    def get_action(self, generator, obs: dict) -> dict:
        """Replaces ``self.policy.get_action(obs)`` (see the module docstring)."""
        policy = generator.policy
        self._measured_before = pose_from_obs(obs)
        self._track = None
        if self.follower is not None:
            if not self.follower.finished():
                return self._follower_action(policy)
            self.follower = None
            policy.action_queue = []
            policy.check_constraint = True
        result = policy.get_action(obs)
        if result.get("action") is None:
            return result
        if getattr(policy, "new_constraint", False):
            self._begin_segment(policy, first_action=np.asarray(result["action"], dtype=float))
            if self.follower is not None:
                return self._follower_action(policy)
        self._target = action_to_pose(result["action"])
        return result

    def _begin_segment(self, policy, *, first_action: np.ndarray) -> None:
        """A segment was just planned: record it and, with tracking, start the follower on its path."""
        path = [first_action] + [np.asarray(a, dtype=float) for a in policy.action_queue]
        n_bridge = int(policy.curr_motion_plan_steps)
        n_skill = int(policy.curr_constraint_steps)
        if n_bridge + n_skill != len(path):
            raise RuntimeError(f"CP-Gen queue has {len(path)} actions, the policy reports {n_bridge} + {n_skill}")
        constraint = policy.current_constraint
        self.rec.add_subtask(subtask_ind=int(policy.current_constraint_idx),
                             src_demo_ind=int(policy.current_constraint_sequence_idx),
                             constraint_name=getattr(constraint, "name", None), obj_names=list(getattr(constraint, "obj_names", []) or []),
                             num_motion_plan_steps=n_bridge, num_constraint_steps=n_skill)
        self._subtasks += 1
        self._segments = SegmentTable()
        self._segments.add(n_bridge, self.rec.begin_segment(SEG_BRIDGE), SEG_BRIDGE)   # bridge opened first, as before
        self._segments.add(n_skill, self.rec.begin_segment(SEG_SKILL), SEG_SKILL)
        self._path, self._pos = path, 0
        if self.method.follower_factory is None:
            return
        kinds = self._segments.kinds_per_waypoint()
        eef = EndEffectorPath(EEF, np.stack([action_to_pose(a) for a in path]), np.stack([a[6:] for a in path]))
        self.follower = self.method.follower_factory()
        self.follower.start_path([eef], [None] * len(path), kinds, self._gripper_states or None)

    def _follower_action(self, policy) -> dict:
        """Next target of the follower as a CP-Gen action; keeps the policy's queue consistent with it."""
        poses, grippers, _noise, record = self.follower.next_target({EEF: self._measured_before})
        self._target, self._track = np.asarray(poses[EEF]), record
        self._pos = int(record["k"])
        policy.action_queue = [np.array(a) for a in self._path[self._pos + 1:]]
        policy.new_constraint = False
        policy.check_constraint = False
        if self.follower.finished():
            self._gripper_states.update(self.follower.final_gripper_states())
            policy.action_queue = []
            policy.check_constraint = True
        return {"action": pose_to_action(self._target, grippers[EEF]), "failure_type": None}

    def _seg_id_at(self, pos: int) -> int:
        return self._segments.seg_id_of(pos)

    def record_step(self, action, action_to_take, reward, in_constraint_segment, system_noise_cfg) -> None:
        """Grafted after the environment stepped: one record line with target, measured pose and both actions."""
        action = np.asarray(action, dtype=float).reshape(-1)
        executed = np.asarray(action_to_take, dtype=float).reshape(-1)
        magnitude = system_noise_cfg.constraint_segment_noise_magnitude if in_constraint_segment else system_noise_cfg.motion_segment_noise_magnitude
        seg_id = self._seg_id_at(self._pos)
        self.rec.add_step(w_poses={EEF: self._target if self._target is not None else action_to_pose(action)},
                          e_poses={EEF: self._measured_before if self._measured_before is not None else np.eye(4)},
                          a_pre=action[:6], a_label=action, a_exec=executed,
                          noise=float(np.max(np.abs(np.asarray(magnitude, dtype=float)))), seg_id=seg_id, term_signals=None,
                          track=self._track)
        self.rec.add_success(bool(np.asarray(reward).item() == 1))
        self._labels.append(action)
        self._executed.append(executed)
        if self.follower is None:
            self._pos += 1
            g = action[6:]
            if g.size and g[0] != 0:
                self._gripper_states[EEF] = bool(g[0] > 0)


def apply(method: Method, *, chunk_seed: int, records_dir: Path | None, graft_hashes: dict, all_sources: bool = False) -> dict:
    """Install the grafted ``generate_demo``; returns a description for the chunk log."""
    global _INSTALLED
    from demo_aug.generate import DemoGenerator

    if _INSTALLED is not None:
        raise RuntimeError("CP-Gen runtime already applied in this process")
    runtime = CPGenRuntime(method, chunk_seed, records_dir, all_sources=all_sources)
    graft = build_graft(DemoGenerator.generate_demo, graft_hashes.get("generate_demo"))
    _INSTALLED = install_grafts([(DemoGenerator, "generate_demo", graft)], {"_tmg": runtime})
    return {**_INSTALLED.info, "method": method.name, "labels": "pre_noise",
            "tracking": method.follower_factory is not None, "records_dir": None if records_dir is None else str(records_dir),
            "all_sources": runtime.all_sources}


def build_graft(generate_demo, expected_hash: str | None) -> Graft:
    """The four edits of ``DemoGenerator.generate_demo`` (module docstring), on any function with those anchors."""
    g = Graft(generate_demo, expected_hash=expected_hash, label="DemoGenerator.generate_demo")
    g.insert_after_assign("state", "_tmg.begin_attempt(self)", occurrence=0)
    g.replace_assign_value("action_dct", "_tmg.get_action(self, obs)")
    g.insert_after_assign("state", "_tmg.record_step(action, action_to_take, reward, in_constraint_segment, system_noise_cfg)",
                          occurrence=1)
    g.insert_before_final_return("_tmg.end_attempt(self, episode_data)")
    return g


def restore() -> None:
    """Put the original ``generate_demo`` back (tests and interactive use)."""
    global _INSTALLED
    if _INSTALLED is None:
        return
    _INSTALLED.restore()
    _INSTALLED = None
