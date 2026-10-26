"""Callbacks grafted into MimicGen's ``DataGenerator.generate`` and ``WaypointTrajectory.execute``.

This module only maps MimicGen's local variables onto the shared pieces: the attempt recorder
(``tmg.core.records``), the segment mixer (``tmg.generation.method``) and the execution bookkeeping
(``tmg.generation.execution``, which owns the path follower, its random-number accounting and the labels).

Anchors (by the name the official statement assigns):

generate():
  after  ``new_initial_state = env.get_state()``                 -> _tmg.begin_attempt(...)
  after  ``src_subtask_object_pose = ...``                       -> segment mixer (identity returns inputs)
  after  ``transformed_seq = WaypointSequence.from_poses(...)``   -> _tmg.record_subtask(locals())
  before the final ``return results``                            -> _tmg.end_attempt(results)
execute():
  first statement of ``for seq in self.waypoint_sequences``      -> _tmg.begin_segment(seq, self)
  ``for j in range(len(seq))``  becomes  ``for j in _tmg.step_indices(seq)``
  ``waypoint = seq[j]``         becomes  ``waypoint = _tmg.waypoint(seq, j, env, env_interface)``
  after  ``action_pose = env_interface.target_pose_to_action(...)`` -> _tmg.pre_noise(action_pose)
  after  ``datagen_info = env_interface.get_datagen_info(...)``   -> _tmg.record_step(waypoint, datagen_info, play_action)
  after  ``cur_success_metrics = env.is_success()``               -> _tmg.record_success(cur_success_metrics, env_interface)

With the identity method every callback only records: no random draw, no changed value, no changed file.
With the follower, grafted waypoints carry ``noise=None`` so the official ``np.random.randn`` line is skipped
and ``pre_noise`` applies the noise the execution bookkeeping chose (a global draw for an original waypoint,
a private draw for an extra step).
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

from tmg.core.records import AttemptRecorder
from tmg.generation.execution import TrajectoryExecution, WaypointView
from tmg.generation.graft import Graft
from tmg.generation.method.interfaces import Method, MixContext
from tmg.generation.simulators.robosuite import SCENE_META_ATTR

REC = AttemptRecorder()
RECORD_ERROR = "RECORD_ERROR.json"
EEF = "robot0"                      # MimicGen's robosuite tasks have one arm
_STATE: dict = {"applied": False, "originals": {}}


class MimicGenRuntime:
    """The ``_tmg`` object the grafted MimicGen functions call (see the module docstring for the anchors)."""
    def __init__(self, method: Method, chunk_seed: int, records_dir: Path | None):
        """Args:
            method: configured method.
            chunk_seed: seed of the private noise stream (``chunk_seed + 7919``).
            records_dir: records folder, or None to record nothing.
        """
        self.method = method
        self.rec = REC
        self.records_dir = records_dir
        private = np.random.RandomState(int(chunk_seed) + 7919)
        self.exec = TrajectoryExecution(method, REC, global_draw=lambda n: np.random.randn(n),
                                        private_draw=lambda n: private.randn(n))
        self._last_mix = None
        self._traj = None

    # generate() -----------------------------------------------------------------------------------
    def begin_attempt(self, env, env_interface, initial_state):
        """Grafted after ``env.reset()``: start the attempt record and read the scene metadata."""
        self.rec.begin_attempt()
        self.rec.set_scene_meta(getattr(getattr(env, "env", None), SCENE_META_ATTR, None))
        self.exec.begin_attempt()
        self._last_mix = None
        self._traj = None

    def mix_segment(self, generator, local_vars: dict):
        """Grafted after the source segment is extracted: return the (possibly mixed) first eef pose array,
        target poses and gripper commands of the selected source. Identity when no mixer is configured.
        """
        eef = local_vars["src_subtask_eef_poses"]
        targets = local_vars["src_subtask_target_poses"]
        grippers = local_vars["src_subtask_gripper_actions"]
        self._last_mix = None
        if self.method.mixer is None:
            return eef, targets, grippers
        subtask_ind = int(local_vars["subtask_ind"])
        object_name = local_vars["subtask_object_name"]
        all_inds = np.asarray(local_vars["all_subtask_inds"])
        infos = generator.src_dataset_infos
        segments, object_poses = [], []
        for i, info in enumerate(infos):
            start, end = (int(v) for v in all_inds[i, subtask_ind])
            segments.append((np.array(info.eef_pose[start]), np.array(info.target_pose[start:end]),
                             np.array(info.gripper_action[start:end])))
            object_poses.append(None if object_name is None else np.array(info.object_poses[object_name][start]))
        context = MixContext(
            subtask_ind=subtask_ind, primary=int(local_vars["selected_src_demo_ind"]),
            primary_eef_start=np.array(eef[0]), primary_targets=np.array(targets), primary_gripper=np.array(grippers),
            src_object_poses=object_poses, src_segments=segments, num_sources=len(infos),
            has_reference_object=object_name is not None, rng=np.random.default_rng(0))
        mixed = self.method.mixer.mix(context)
        self._last_mix = mixed.record
        if not mixed.record.get("mixed"):
            return eef, targets, grippers
        new_eef = np.concatenate([np.asarray(mixed.eef_start)[None], np.asarray(eef)[1:]], axis=0)
        return new_eef, np.asarray(mixed.targets), np.asarray(mixed.gripper)

    def record_subtask(self, local_vars: dict):
        """Grafted after the transformed segment is built: record the subtask's source, poses, junction and
        stitch settings (plus the mix record), and the initial object poses on subtask 0.
        """
        subtask_ind = int(local_vars["subtask_ind"])
        info = local_vars["cur_datagen_info"]
        if subtask_ind == 0:
            self.rec.set_objects_init(info.object_poses)
        spec = local_vars["self"].task_spec[subtask_ind]
        meta = dict(
            subtask_ind=subtask_ind,
            object_name=local_vars["subtask_object_name"],
            src_demo_ind=int(local_vars["selected_src_demo_ind"]),
            src_subtask_start=int(local_vars["selected_src_subtask_inds"][0]),
            src_subtask_end=int(local_vars["selected_src_subtask_inds"][1]),
            cur_obj_pose=local_vars["cur_object_pose"],
            src_obj_pose=local_vars["src_subtask_object_pose"],
            junction_start_pose=np.array(local_vars["init_sequence"][0].pose),
            junction_target_pose=np.array(local_vars["transformed_eef_poses"][0]),
            num_interpolation_steps=int(spec["num_interpolation_steps"]),
            num_fixed_steps=int(spec["num_fixed_steps"]),
            action_noise=float(spec["action_noise"]),
            objects=info.object_poses,
            eef_pose=np.array(info.eef_pose),
        )
        if self._last_mix is not None:
            meta["mix"] = self._last_mix
        self.rec.add_subtask(**meta)

    def end_attempt(self, results: dict):
        """Grafted before ``generate`` returns: swap in pre-noise labels when configured, then flush the attempt
        record. A record-writing failure leaves ``RECORD_ERROR.json`` and re-raises so the chunk fails visibly.
        """
        labels = self.exec.label_actions(results["actions"])
        if labels is not results["actions"]:
            results["actions_executed"] = np.array(results["actions"])
            results["actions"] = labels
        self.rec.add_extra(labels=self.method.labels)
        try:
            self.rec.flush(success=bool(results["success"]), src_demo_inds=results["src_demo_inds"])
        except Exception as error:
            if self.records_dir is not None:
                Path(self.records_dir).mkdir(parents=True, exist_ok=True)
                (Path(self.records_dir) / RECORD_ERROR).write_text(json.dumps(
                    {"attempt": self.rec.attempt, "error": repr(error), "traceback": traceback.format_exc(),
                     "time": time.time()}, indent=2))
            raise

    # execute() ------------------------------------------------------------------------------------
    @staticmethod
    def _views(seq):
        """A MimicGen ``WaypointSequence`` as a list of ``WaypointView`` (single end effector ``robot0``)."""
        return [WaypointView(poses={EEF: np.asarray(w.pose)}, grippers={EEF: np.asarray(w.gripper_action)}, noise=w.noise)
                for w in seq.sequence]

    def begin_segment(self, seq, traj):
        """Grafted at the start of each waypoint sequence: on the first sequence of a trajectory, hand the whole
        trajectory to the execution bookkeeping; then select the sequence's record segment.
        """
        if self._traj is not traj:
            self._traj = traj
            self.exec.begin_trajectory([self._views(s) for s in traj.waypoint_sequences])
        self.exec.enter_sequence(next(i for i, s in enumerate(traj.waypoint_sequences) if s is seq))

    def step_indices(self, seq):
        """Replaces ``range(len(seq))``: the official range, or the follower-driven iteration."""
        index = next(i for i, s in enumerate(self._traj.waypoint_sequences) if s is seq)
        return self.exec.steps_of_sequence(index, len(seq))

    def waypoint(self, seq, j, env, env_interface):
        """Replaces ``seq[j]``: the original waypoint (identity) or a ``Waypoint`` built from the follower's target
        with ``noise=None`` (the noise is applied by :meth:`pre_noise`).
        """
        if self.exec.follower is None:
            self.exec.target(self._views(seq)[j], None)
            return seq[j]
        from mimicgen.datagen.waypoint import Waypoint

        target = self.exec.target(None, {EEF: np.asarray(env_interface.get_robot_eef_pose())})
        return Waypoint(pose=target.poses[EEF], gripper_action=target.grippers[EEF], noise=None)

    def pre_noise(self, action_pose):
        """Grafted after the arm action is computed: keep the pre-noise copy and, on follower steps, add the
        chosen noise vector in place.
        """
        self.exec.apply_noise(action_pose)

    def record_step(self, waypoint, datagen_info, play_action):
        """Grafted after ``get_datagen_info``: hand the step's target, measured pose, executed action and gripper
        command to the execution bookkeeping.
        """
        self.exec.before_step(target_poses={EEF: np.asarray(waypoint.pose)},
                              measured_poses={EEF: np.asarray(datagen_info.eef_pose)},
                              executed_action=play_action, grippers={EEF: np.asarray(waypoint.gripper_action)},
                              noise_of_waypoint=waypoint.noise)

    def record_success(self, metrics, env_interface):
        """Grafted after ``env.is_success()``: complete the step record with the task predicate and the
        termination signals of the new state.
        """
        self.exec.after_step(task_success=bool(metrics["task"]), term_signals=env_interface.get_subtask_term_signals())


def apply(method: Method, *, chunk_seed: int, records_dir: Path | None, graft_hashes: dict) -> dict:
    """Install the grafted functions; returns a description for the chunk log."""
    from mimicgen.datagen.data_generator import DataGenerator
    from mimicgen.datagen.waypoint import WaypointTrajectory

    if _STATE["applied"]:
        raise RuntimeError("MimicGen runtime already applied in this process")
    runtime = MimicGenRuntime(method, chunk_seed, records_dir)
    REC.configure(records_dir)
    gen = Graft(DataGenerator.generate, expected_hash=graft_hashes.get("generate"), label="DataGenerator.generate")
    gen.insert_after_assign("new_initial_state", "_tmg.begin_attempt(env, env_interface, new_initial_state)")
    gen.insert_after_assign(
        "src_subtask_object_pose",
        "src_subtask_eef_poses, src_subtask_target_poses, src_subtask_gripper_actions = _tmg.mix_segment(self, locals())")
    gen.insert_after_assign("transformed_seq", "_tmg.record_subtask(locals())")
    gen.insert_before_final_return("_tmg.end_attempt(results)")
    exe = Graft(WaypointTrajectory.execute, expected_hash=graft_hashes.get("execute"), label="WaypointTrajectory.execute")
    exe.insert_at_for_start("seq", "_tmg.begin_segment(seq, self)")
    exe.replace_for_iter("j", "_tmg.step_indices(seq)")
    exe.replace_assign_value("waypoint", "_tmg.waypoint(seq, j, env, env_interface)")
    exe.insert_after_assign("action_pose", "_tmg.pre_noise(action_pose)", occurrence=0)
    exe.insert_after_assign("datagen_info", "_tmg.record_step(waypoint, datagen_info, play_action)")
    exe.insert_after_assign("cur_success_metrics", "_tmg.record_success(cur_success_metrics, env_interface)")
    _STATE["originals"] = {"generate": DataGenerator.generate, "execute": WaypointTrajectory.execute}
    DataGenerator.generate = gen.compile({"_tmg": runtime})
    WaypointTrajectory.execute = exe.compile({"_tmg": runtime})
    _STATE["applied"] = True
    _STATE["runtime"] = runtime
    return {"generate": DataGenerator.generate.__tmg_graft__, "execute": WaypointTrajectory.execute.__tmg_graft__,
            "method": method.name, "labels": method.labels, "records_dir": None if records_dir is None else str(records_dir)}


def restore() -> None:
    """Put the original MimicGen functions back (tests and interactive use)."""
    if not _STATE["applied"]:
        return
    from mimicgen.datagen.data_generator import DataGenerator
    from mimicgen.datagen.waypoint import WaypointTrajectory

    DataGenerator.generate = _STATE["originals"]["generate"]
    WaypointTrajectory.execute = _STATE["originals"]["execute"]
    _STATE["applied"] = False
