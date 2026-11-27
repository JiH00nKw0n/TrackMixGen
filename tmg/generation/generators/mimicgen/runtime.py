"""Callbacks grafted into MimicGen's ``DataGenerator.generate`` and ``WaypointTrajectory.execute``.

This module only maps MimicGen's local variables onto the shared pieces: the attempt recorder
(``tmg.core.records``), the segment mixer (``tmg.generation.method``) and the execution bookkeeping
(``tmg.generation.execution``, which owns the record segments and the labels).

Anchors (by the name the official statement assigns):

generate():
  after  ``new_initial_state = env.get_state()``                 -> _tmg.begin_attempt(env)
  after  ``src_subtask_object_pose = ...``                       -> segment mixer (identity returns inputs)
  after  ``transformed_seq = WaypointSequence.from_poses(...)``   -> _tmg.record_subtask(locals())
  before the final ``return results``                            -> _tmg.end_attempt(results)
execute():
  first statement of ``for seq in self.waypoint_sequences``      -> _tmg.begin_segment(seq, self)
  after  ``action_pose = env_interface.target_pose_to_action(...)`` -> _tmg.pre_noise(action_pose)
  after  ``datagen_info = env_interface.get_datagen_info(...)``   -> _tmg.record_step(waypoint, datagen_info, play_action)
  after  ``cur_success_metrics = env.is_success()``               -> _tmg.record_success(cur_success_metrics, env_interface)

With the identity method every callback only records: no random draw, no changed value, no changed file. A mixer
changes only the source segment the official code transforms; the official source selection and noise draws are kept.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tmg.core.records import AttemptRecorder
from tmg.generation.execution import TrajectoryExecution, WaypointView
from tmg.generation.generators._shared import record_error_guard
from tmg.generation.graft import Graft, Installed, install_grafts
from tmg.generation.method.interfaces import Method, MixContext, SubtaskScene
from tmg.generation.simulators.robosuite import SCENE_META_ATTR

EEF = "robot0"                      # MimicGen's robosuite tasks have one arm
_INSTALLED: Installed | None = None  # the grafts of this process (apply installs once; restore undoes it)


class MimicGenRuntime:
    """The ``_tmg`` object the grafted MimicGen functions call (see the module docstring for the anchors)."""
    def __init__(self, method: Method, chunk_seed: int, records_dir: Path | None):
        """Args:
            method: configured method.
            chunk_seed: seed of the chunk (kept for the entrypoint's call signature).
            records_dir: records folder, or None to record nothing.
        """
        self.method = method
        self.rec = AttemptRecorder()
        self.rec.configure(records_dir)
        self.records_dir = records_dir
        self.exec = TrajectoryExecution(method, self.rec)
        self._last_mix = None
        self._traj = None

    # generate() -----------------------------------------------------------------------------------
    def begin_attempt(self, env):
        """Grafted after ``env.reset()``: start the attempt record and read the scene metadata."""
        self.rec.begin_attempt()
        self.rec.set_scene_meta(getattr(getattr(env, "env", None), SCENE_META_ATTR, None))
        self.exec.begin_attempt()
        if self.method.mixer is not None:
            self.method.mixer.begin_attempt()              # a new attempt draws new randmix weights
        self._last_mix = None
        self._traj = None
        self._carried = None

    def subtask_scene(self, local_vars: dict, infos, all_inds, subtask_ind: int, object_name, object_poses: list) -> SubtaskScene:
        """What SkillMix scores of the current scene: the reference object's pose the segment is moved to and in every
        source and, from the second subtask on, the end-effector pose and the pose of the previous subtask's reference
        object (the in-hand pose) now and in every source at the start of its segment. The same quantities as the
        attempt record's ``cur_obj_pose``, ``eef_pose`` and ``objects`` (``tmg fit_coefficients`` reads those)."""
        current = local_vars.get("cur_object_pose")
        scene = SubtaskScene(subtask=subtask_ind, reference_object=object_name,
                             current_object_pose=None if current is None else np.array(current),
                             source_object_poses=object_poses)
        carried = self._carried
        if carried is not None:
            cur = local_vars["cur_datagen_info"]
            starts = [int(all_inds[i, subtask_ind][0]) for i in range(len(infos))]
            scene.carried_object = carried
            scene.eef_pose, scene.carried_pose = np.array(cur.eef_pose), np.array(cur.object_poses[carried])
            scene.source_eef_poses = [np.array(info.eef_pose[t]) for info, t in zip(infos, starts)]
            scene.source_carried_poses = [np.array(info.object_poses[carried][t]) for info, t in zip(infos, starts)]
        return scene

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
        scene = self.subtask_scene(local_vars, infos, all_inds, subtask_ind, object_name, object_poses)
        self._carried = object_name
        context = MixContext(
            subtask_ind=subtask_ind, primary=int(local_vars["selected_src_demo_ind"]),
            primary_eef_start=np.array(eef[0]), primary_targets=np.array(targets), primary_gripper=np.array(grippers),
            src_object_poses=object_poses, src_segments=segments, num_sources=len(infos),
            has_reference_object=object_name is not None, rng=np.random.default_rng(0), scene=scene)
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
        with record_error_guard(self.records_dir, self.rec):
            self.rec.flush(success=bool(results["success"]), src_demo_inds=results["src_demo_inds"])

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

    def pre_noise(self, action_pose):
        """Grafted after the arm action is computed, before the official noise: keep the pre-noise copy."""
        self.exec.keep_pre_noise(action_pose)

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


def build_grafts(generate, execute, graft_hashes: dict) -> tuple[Graft, Graft]:
    """The edits of ``DataGenerator.generate`` and ``WaypointTrajectory.execute`` (module docstring), on any pair of
    functions with those anchors; ``graft_hashes`` holds the pinned hashes under ``generate`` and ``execute``."""
    gen = Graft(generate, expected_hash=graft_hashes.get("generate"), label="DataGenerator.generate")
    gen.insert_after_assign("new_initial_state", "_tmg.begin_attempt(env)")
    gen.insert_after_assign(
        "src_subtask_object_pose",
        "src_subtask_eef_poses, src_subtask_target_poses, src_subtask_gripper_actions = _tmg.mix_segment(self, locals())")
    gen.insert_after_assign("transformed_seq", "_tmg.record_subtask(locals())")
    gen.insert_before_final_return("_tmg.end_attempt(results)")
    exe = Graft(execute, expected_hash=graft_hashes.get("execute"), label="WaypointTrajectory.execute")
    exe.insert_at_for_start("seq", "_tmg.begin_segment(seq, self)")
    exe.insert_after_assign("action_pose", "_tmg.pre_noise(action_pose)", occurrence=0)
    exe.insert_after_assign("datagen_info", "_tmg.record_step(waypoint, datagen_info, play_action)")
    exe.insert_after_assign("cur_success_metrics", "_tmg.record_success(cur_success_metrics, env_interface)")
    return gen, exe


def apply(method: Method, *, chunk_seed: int, records_dir: Path | None, graft_hashes: dict) -> dict:
    """Install the grafted functions; returns a description for the chunk log."""
    global _INSTALLED
    from mimicgen.datagen.data_generator import DataGenerator
    from mimicgen.datagen.waypoint import WaypointTrajectory

    if _INSTALLED is not None:
        raise RuntimeError("MimicGen runtime already applied in this process")
    runtime = MimicGenRuntime(method, chunk_seed, records_dir)
    gen, exe = build_grafts(DataGenerator.generate, WaypointTrajectory.execute, graft_hashes)
    _INSTALLED = install_grafts([(DataGenerator, "generate", gen), (WaypointTrajectory, "execute", exe)], {"_tmg": runtime})
    return {**_INSTALLED.info, "method": method.name, "labels": method.labels,
            "records_dir": None if records_dir is None else str(records_dir)}


def restore() -> None:
    """Put the original MimicGen functions back (tests and interactive use)."""
    global _INSTALLED
    if _INSTALLED is None:
        return
    _INSTALLED.restore()
    _INSTALLED = None
