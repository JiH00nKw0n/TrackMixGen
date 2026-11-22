"""Callbacks grafted into CP-Gen's ``DemoGenerator.generate_demo`` (``demo_aug.generate``).

How CP-Gen produces one attempt: the environment is reset, then ``CPPolicy.get_action`` is called once per
control step. When its action queue is empty the policy solves the next constraint (an end-effector pose
sequence from the annotated source segment, re-solved for the current object poses) and motion-plans to its
start with cuRobo; the planned steps (``curr_motion_plan_steps``) followed by the constraint steps
(``curr_constraint_steps``) fill ``action_queue`` and one action is popped per step. ``generate_demo`` adds uniform
noise scaled by ``SystemNoiseConfig`` (``motion_segment_noise_magnitude``, ``constraint_segment_noise_magnitude``) to
the popped action before stepping and stores the popped action as the label. Both magnitudes default to 0 in all 7
dimensions, and neither our config nor the authors' launcher sets them. So no noise is added during CP-Gen generation
(the uniform draw still happens and is multiplied by 0), the saved actions equal the executed ones, and they are
pre-noise labels.

Anchors in ``generate_demo`` (by the name the official statement assigns):

| anchor                                               | grafted code                                                            |
|------------------------------------------------------|-------------------------------------------------------------------------|
| after the first ``state = self.env.get_state()``     | ``_tmg.begin_attempt(self)``                                            |
| ``action_dct = self.policy.get_action(obs)``         | becomes ``action_dct = _tmg.get_action(self, obs)``                     |
| after the in-loop ``state = self.env.get_state()``   | ``_tmg.record_step(action, action_to_take, reward, in_constraint_segment, system_noise_cfg)`` |
| before the final ``return episode_data``             | ``_tmg.end_attempt(self, episode_data)``                                |

``get_action`` calls the official ``CPPolicy.get_action`` and, each time the policy has just planned a segment
(``new_constraint``), records the segment (bridge = motion-plan steps, skill = constraint steps) and the source
index.

Re-solve after the bridge (generator ``resolve_after_bridge``, on by default): CP-Gen solves a constraint before it plans
the bridge to the constraint's start. When the constraint carries an object (``attached_object_names``), the object can
sit differently in the hand once the bridge has been executed, so ``get_action`` solves the constraint again from that
observation and, when the new plan has the same number of steps, replaces the queued constraint steps. With a mixer the
re-solve is mixed again (the grafted ``solve_constraint`` calls ``prepare_mix``); the subtask record then keeps the mix
record of the solve that was executed (``mix``), the first one under ``mix_before_resolve``, and the outcome under
``resolve_after_bridge`` (``replaced`` or why the plan was kept). Constraints that carry nothing (grasps) keep their plan:
re-solving every constraint made attempts 1.4-1.7x slower without changing the success where nothing moves, and a rule
that also skipped carried objects that had not moved in the hand kept Threading at 7 % (2026-11-17). ``method.labels`` is fixed to ``pre_noise`` because the official output already is; the executed (noisy) actions
are kept in our records.

Source mixing (SkillMix, ``mixing.type: skillmix``, selection ours): CP-Gen re-solves constraints instead of
replaying segments, so the blended quantity is the desired end-effector pose of every solved step, after every retained
source is solved for the current scene (``tmg.generation.generators.cpgen.mixing``). It needs a second graft, of
``CPPolicy.solve_constraint``, installed only for a method with a mixer:

| anchor                                                          | grafted code                                      |
|-----------------------------------------------------------------|---------------------------------------------------|
| after ``X_obj0_transf, X_obj1_transf, X_eef_transf = find_best_transform(...)`` | ``_tmg.prepare_mix(self, constraint, obs, X_obj0_transf, X_obj1_transf, X_eef_transf)`` |
| ``X_W_eef_des = compute_X_W_eef_des(...)`` in the step loop     | becomes ``X_W_eef_des = _tmg.mixed_eef_des(t_idx, compute_X_W_eef_des(...))`` |

Source selection (``selection.all_sources``, ``extra["all_sources"]`` of the chunk spec): CP-Gen picks the constraint
sequence (source demonstration) only in its reset-near-constraint mode and otherwise always uses sequence 0, so with
every source in use :class:`SourcePicker` draws one per attempt from the chunk's private stream (``seeds.source_pick_rng``).
With ``selection.select_src_per_subtask`` too (selection=ours, ``extra["per_subtask_sources"]``) it draws again every time
the policy advances to the next constraint (subtask), so constraint k of the attempt is solved from a source drawn for k,
as MimicGen's selection=ours does. Without it every subtask of an attempt came from the same source, and the coefficient
pools never showed an in-hand pose that differs from the source's, so the in-hand terms of the mixing model could not
be fitted (CP-Gen Threading, 2026-11-14). The record keeps the source of every subtask (``subtasks[k].src_demo_ind`` and
``src_demo_inds``).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from tmg.core.records import SEG_BRIDGE, SEG_SKILL, AttemptRecorder
from tmg.generation.execution import SegmentTable
from tmg.generation.generators._shared import record_error_guard
from tmg.generation.generators.cpgen import mixing
from tmg.generation.graft import Graft, Installed, install_grafts
from tmg.generation.method.interfaces import Method
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


def pose_from_obs(obs: dict) -> np.ndarray:
    """4x4 end-effector pose from robomimic's ``robot0_eef_pos`` and ``robot0_eef_quat`` (x, y, z, w)."""
    pose = np.eye(4)
    pose[:3, 3] = np.asarray(obs["robot0_eef_pos"], dtype=float).reshape(3)
    pose[:3, :3] = Rotation.from_quat(np.asarray(obs["robot0_eef_quat"], dtype=float).reshape(4)).as_matrix()
    return pose


def _site_pose(pos, quat_xyzw) -> np.ndarray:
    """4x4 pose from a position and an (x, y, z, w) quaternion (robosuite's convention)."""
    pose = np.eye(4)
    pose[:3, 3] = np.asarray(pos, dtype=float).reshape(3)
    pose[:3, :3] = Rotation.from_quat(np.asarray(quat_xyzw, dtype=float).reshape(4)).as_matrix()
    return pose


class SourcePicker:
    """How the source demonstration (CP-Gen constraint sequence) of an attempt is chosen.

    ``all_sources=False`` is the official behaviour: the policy's current sequence is left alone (sequence 0 unless
    CP-Gen's own reset-near-constraint mode picked one). ``all_sources=True`` (selection=ours) draws a sequence index
    per attempt from the chunk's private stream and restarts that sequence at its first constraint. ``per_subtask=True``
    also draws a new sequence index each time the policy advances to the next constraint (``CPPolicy.advance_constraint``
    wrapped on the policy instance; CP-Gen's own code is not changed).
    """

    def __init__(self, all_sources: bool, chunk_seed: int, per_subtask: bool = False):
        self.all_sources = bool(all_sources)
        self.per_subtask = bool(per_subtask) and self.all_sources
        self._rng = source_pick_rng(chunk_seed)

    def _draw(self, policy, subtask_ind: int) -> int:
        """A source for constraint ``subtask_ind``: uniform over the sequences that have that constraint."""
        sequences = policy.constraint_sequences
        allowed = [j for j, seq in enumerate(sequences) if subtask_ind == 0 or len(seq) > subtask_ind]
        return allowed[int(self._rng.integers(len(allowed)))]

    def _wrap_advance(self, policy) -> None:
        """Make ``policy.advance_constraint`` draw the next constraint's source (installed once per policy)."""
        if getattr(policy, "_tmg_per_subtask", False):
            return
        advance = policy.advance_constraint                      # bound method of the official class

        def advance_and_draw():
            advance()
            k = int(policy.current_constraint_idx)
            if k < max(len(seq) for seq in policy.constraint_sequences):
                policy.current_constraint_sequence_idx = self._draw(policy, k)

        policy.advance_constraint = advance_and_draw
        policy._tmg_per_subtask = True

    def pick(self, policy) -> None:
        """Set the policy's constraint sequence for the attempt that starts now (no-op in the official mode)."""
        sequences = getattr(policy, "constraint_sequences", None)
        if self.all_sources and sequences is not None and len(sequences) > 1:
            policy.current_constraint_sequence_idx = self._draw(policy, 0)
            policy.current_constraint_idx = 0
            if self.per_subtask:
                self._wrap_advance(policy)


class CPGenRuntime:
    """The ``_tmg`` object the grafted ``generate_demo`` calls (anchors in the module docstring)."""

    def __init__(self, method: Method, chunk_seed: int, records_dir: Path | None, all_sources: bool = False,
                 per_subtask: bool = False, resolve_after_bridge: bool = False):
        if method.mixer is not None and not all_sources:
            raise ValueError(f"method {method.name!r}: source mixing needs every source loaded: use selection=ours")
        if method.mixer is not None and not hasattr(method.mixer, "weights"):
            raise ValueError(f"method {method.name!r}: CP-Gen mixing supports only SkillMix (mixing.type: skillmix)")
        self.method = method
        self.resolve_after_bridge = bool(resolve_after_bridge)
        self._mix_plan = None                              # MultiMixPlan of the constraint being solved (mixing only)
        self._last_mix: dict | None = None                 # mix record of that constraint, written with its subtask
        self.official = None                               # demo_aug.generate (tests put a stand-in here)
        self.picker = SourcePicker(all_sources, chunk_seed, per_subtask=per_subtask)
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
        self._segments = SegmentTable()                   # record segments of the current planned path (bridge, skill)
        self._pos = 0                                    # executed steps of the current path
        self._subtasks = 0
        self._sources: list[int] = []                      # source demonstration of every recorded subtask
        self._measured_before: np.ndarray | None = None
        self._target: np.ndarray | None = None
        self._labels: list[np.ndarray] = []
        self._executed: list[np.ndarray] = []
        self._prev_ref: str | None = None                  # reference object of the previous constraint (in-hand feature)
        self._last_mix = None
        self._mix_plan = None
        self._unplanned_mixes: list[dict] = []             # mix records of solves whose motion plan failed

    # attempt boundaries --------------------------------------------------------------------------
    def begin_attempt(self, generator) -> None:
        """Grafted after the reset: open the record, pick the attempt's source and read our scene metadata from the
        robosuite env."""
        self._attempt_t0 = time.time()
        self._reset_attempt_state()
        self.rec.begin_attempt()
        self.picker.pick(generator.policy)
        if self.method.mixer is not None:
            self.method.mixer.begin_attempt()                 # a new attempt draws new calibration weights
        robosuite_env = getattr(getattr(generator.env, "env", None), "env", None)
        self.rec.set_scene_meta(getattr(robosuite_env, SCENE_META_ATTR, None))

    def end_attempt(self, generator, episode_data: dict) -> None:
        """Grafted before ``generate_demo`` returns: flush the attempt record (source index per segment)."""
        with record_error_guard(self.records_dir, self.rec):
            src = int(getattr(generator.policy, "current_constraint_sequence_idx", 0))
            sources = self._sources if len(self._sources) == self._subtasks else [src] * self._subtasks
            self.rec.add_extra(elapsed_s=round(time.time() - self._attempt_t0, 3), labels="pre_noise",
                               failure_type=episode_data.get("failure_type"), executed_steps=len(self._executed))
            if self._last_mix is not None:
                self._unplanned_mixes.append(self._last_mix)
                self._last_mix = None
            if self._unplanned_mixes:                         # mixed constraints that never became a segment (plan failed)
                self.rec.add_extra(unplanned_mixes=self._unplanned_mixes)
            self.rec.flush(success=bool(episode_data.get("success", False)), src_demo_inds=sources)

    # per step ------------------------------------------------------------------------------------
    def get_action(self, generator, obs: dict) -> dict:
        """Replaces ``self.policy.get_action(obs)`` (see the module docstring)."""
        policy = generator.policy
        self._measured_before = pose_from_obs(obs)
        if self.resolve_after_bridge:
            self._resolve_after_bridge(policy, obs)
        result = policy.get_action(obs)
        if result.get("action") is None:
            return result
        if getattr(policy, "new_constraint", False):
            self._begin_segment(policy, first_action=np.asarray(result["action"], dtype=float), obs=obs)
        self._target = action_to_pose(result["action"])
        return result

    def _scene_record(self, constraint, obs: dict | None) -> dict:
        """What ``tmg fit_coefficients`` reads of a constraint (subtask): its reference object, that object's pose now and
        in the source at the window's first step, the end-effector site pose now and in the source, and every
        task-relevant object's pose now (``objects``) and in the source (``src_objects``, the constraint's objects only).
        The reference object is ``obj_names[0]`` for a robot-object constraint and ``obj_names[1]`` (the object the held
        object is placed on) for an object-object constraint. Empty when the observation lacks the poses."""
        names = list(getattr(constraint, "obj_names", None) or [])
        src_obj = getattr(constraint, "src_obj_pose", None) or {}
        src_obs = getattr(constraint, "src_obs", None) or {}
        if obs is None or not names:
            return {}
        ref = names[1] if getattr(constraint, "constraint_type", None) == "object-object" and len(names) > 1 else names[0]
        if f"{ref}_pose" not in obs or ref not in src_obj:
            return {}
        quat_key = "robot0_eef_quat_site" if "robot0_eef_quat_site" in obs else "robot0_eef_quat"
        out = {"object_name": ref, "constraint_type": getattr(constraint, "constraint_type", None),
               "cur_obj_pose": np.asarray(obs[f"{ref}_pose"], dtype=float),
               "src_obj_pose": np.asarray(src_obj[ref], dtype=float)[0],
               "eef_pose": _site_pose(obs["robot0_eef_pos"], obs[quat_key]),
               "objects": {k[:-5]: np.asarray(v, dtype=float) for k, v in obs.items()
                           if k.endswith("_pose") and np.asarray(v).shape == (4, 4)},
               "src_objects": {k: np.asarray(v, dtype=float)[0] for k, v in src_obj.items()},
               "src_subtask_start": 0}
        if "robot0_eef_pos" in src_obs and quat_key in src_obs:
            out["src_eef_pose"] = _site_pose(np.asarray(src_obs["robot0_eef_pos"])[0], np.asarray(src_obs[quat_key])[0])
        return out

    def _begin_segment(self, policy, *, first_action: np.ndarray, obs: dict | None = None) -> None:
        """A segment was just planned: record it and open its bridge and skill record segments."""
        path = [first_action] + [np.asarray(a, dtype=float) for a in policy.action_queue]
        n_bridge = int(policy.curr_motion_plan_steps)
        n_skill = int(policy.curr_constraint_steps)
        if n_bridge + n_skill != len(path):
            raise RuntimeError(f"CP-Gen queue has {len(path)} actions, the policy reports {n_bridge} + {n_skill}")
        constraint = policy.current_constraint
        scene = self._scene_record(constraint, obs)
        mix = {"mix": self._last_mix} if self._last_mix is not None else {}
        self._last_mix, self._mix_plan = None, None
        if scene.get("object_name"):
            self._prev_ref = scene["object_name"]
        self.rec.add_subtask(subtask_ind=int(policy.current_constraint_idx),
                             src_demo_ind=int(policy.current_constraint_sequence_idx),
                             constraint_name=getattr(constraint, "name", None), obj_names=list(getattr(constraint, "obj_names", []) or []),
                             num_motion_plan_steps=n_bridge, num_constraint_steps=n_skill,
                             **scene, **mix)
        self._subtasks += 1
        self._sources.append(int(policy.current_constraint_sequence_idx))
        self._segments = SegmentTable()
        self._segments.add(n_bridge, self.rec.begin_segment(SEG_BRIDGE), SEG_BRIDGE)   # bridge opened first, as before
        self._segments.add(n_skill, self.rec.begin_segment(SEG_SKILL), SEG_SKILL)
        self._pos = 0

    # re-solve after the bridge -----------------------------------------------------------------------------------------
    def _resolve_after_bridge(self, policy, obs: dict) -> str | None:
        """When the bridge of the current constraint has just been executed (the queue holds exactly the constraint steps)
        and the constraint carries an object, solve it again from ``obs`` and replace the queued steps if the new plan has
        the same length (module docstring). Returns the outcome written to the subtask record, or None when nothing was
        due (no bridge just finished, or already handled for this queue)."""
        queue = getattr(policy, "action_queue", None)
        n = int(getattr(policy, "curr_constraint_steps", 0) or 0)
        if not queue or int(getattr(policy, "curr_motion_plan_steps", 0) or 0) <= 0 or len(queue) != n:
            return None
        if getattr(policy, "_tmg_resolved_queue", None) is queue:
            return None
        policy._tmg_resolved_queue = queue
        if not carried_in_hand(policy, obs):
            return None                                     # a grasp: nothing in the hand can have moved
        executed_mix = self._last_mix                         # None here: the first solve's record went to the segment
        self._last_mix = None
        try:
            res = policy.solve_constraint(policy.current_constraint, obs)
            resolved_mix = self._last_mix                     # the re-solve's mix record (with a mixer)
        finally:
            self._last_mix, self._mix_plan = executed_mix, None
        if not res.success:
            outcome = "kept: the re-solve failed"
        else:
            actions = list(res.actions)
            if getattr(policy.current_constraint, "duplicate_actions", False):
                actions = [a for a in actions for _ in (0, 1)]
            if len(actions) != n:
                outcome = f"kept: the re-solve has {len(actions)} steps, the plan {n}"
            else:
                queue[:] = actions
                outcome = "replaced"
        meta = {"resolve_after_bridge": outcome}
        if outcome == "replaced" and resolved_mix is not None:
            first = self.rec.subtasks[-1].get("mix") if self.rec.subtasks else None
            meta.update(mix=resolved_mix, mix_before_resolve=first)
        self.rec.update_last_subtask(**meta)
        print(f"[resolve-after-bridge] {outcome}", flush=True)
        return outcome

    # source mixing (grafted into CPPolicy.solve_constraint) ----------------------------------------------------------
    def prepare_mix(self, policy, constraint, obs: dict, X_obj0_transf, X_obj1_transf, X_eef_transf) -> None:
        """Grafted after CP-Gen chose the primary's symmetry transforms: weigh every source, solve the retained ones for the
        current scene and prepare the blend (``mixing`` module docstring)."""
        if self._last_mix is not None:                        # an earlier solve of this attempt never reached a segment
            self._unplanned_mixes.append(self._last_mix)
        self._mix_plan, self._last_mix = None, None
        if self.method.mixer is None:
            return
        idx, primary = int(policy.current_constraint_idx), int(policy.current_constraint_sequence_idx)
        sequences = policy.constraint_sequences
        if constraint is not sequences[primary][idx]:
            self._last_mix = {"subtask": idx, "source_a": primary, "mixed": False,
                              "identity_reason": "solve_constraint called for a constraint other than the current one"}
            return
        if self.official is None:
            import demo_aug.generate

            self.official = demo_aug.generate
        carried = self._prev_ref if idx > 0 else None
        try:
            record, w = mixing.decide(self.method.mixer, sequences, idx, primary, obs, carried)
        except KeyError as err:                              # a pose the model needs is missing from the observation
            self._last_mix = {"subtask": idx, "source_a": primary, "mixed": False, "identity_reason": f"missing {err}"}
            return
        self._last_mix = record
        others = [j for j in range(len(w)) if w[j] > 0 and j != primary]
        if not others:
            record.update(mixed=False, identity_reason="only the primary is retained")
            return
        transforms = dict(X_obj0_transf=X_obj0_transf, X_obj1_transf=X_obj1_transf, X_eef_transf=X_eef_transf)
        chosen, symmetry = mixing.align_sources(self.official, policy, sequences, idx, primary, others, obs, transforms)
        record["symmetry"] = symmetry                       # kept even if solving or planning fails below
        solved = mixing.solve_sources(self.official, policy, sequences, idx, others, obs, chosen)
        weights = {"primary": float(w[primary]), **{j: float(w[j]) for j in others}}
        self._mix_plan = mixing.MultiMixPlan(mixing.gripper_series(constraint), solved, weights)
        record.update(mixed=True, alignment=self._mix_plan.meta)

    def mixed_eef_des(self, t_idx: int, primary_pose):
        """Grafted around the step loop's ``compute_X_W_eef_des``: the official pose of step ``t_idx``, blended with the
        retained sources' when this constraint is mixed."""
        if self._mix_plan is None:
            return primary_pose
        return self._mix_plan.pose(int(t_idx), primary_pose)

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
                          noise=float(np.max(np.abs(np.asarray(magnitude, dtype=float)))), seg_id=seg_id, term_signals=None)
        self.rec.add_success(bool(np.asarray(reward).item() == 1))
        self._labels.append(action)
        self._executed.append(executed)
        self._pos += 1


def apply(method: Method, *, chunk_seed: int, records_dir: Path | None, graft_hashes: dict, all_sources: bool = False,
          per_subtask: bool = False, resolve_after_bridge: bool = False) -> dict:
    """Install the grafted ``generate_demo``; returns a description for the chunk log."""
    global _INSTALLED
    from demo_aug.generate import CPPolicy, DemoGenerator

    if _INSTALLED is not None:
        raise RuntimeError("CP-Gen runtime already applied in this process")
    runtime = CPGenRuntime(method, chunk_seed, records_dir, all_sources=all_sources, per_subtask=per_subtask,
                           resolve_after_bridge=resolve_after_bridge)
    edits = [(DemoGenerator, "generate_demo", build_graft(DemoGenerator.generate_demo, graft_hashes.get("generate_demo")))]
    if method.mixer is not None:
        edits.append((CPPolicy, "solve_constraint", build_mix_graft(CPPolicy.solve_constraint, graft_hashes.get("solve_constraint"))))
    _INSTALLED = install_grafts(edits, {"_tmg": runtime})
    return {**_INSTALLED.info, "method": method.name, "labels": "pre_noise",
            "records_dir": None if records_dir is None else str(records_dir),
            "all_sources": runtime.all_sources, "per_subtask_sources": runtime.picker.per_subtask,
            "resolve_after_bridge": runtime.resolve_after_bridge,
            "mixing": method.mixer is not None}


def build_graft(generate_demo, expected_hash: str | None) -> Graft:
    """The four edits of ``DemoGenerator.generate_demo`` (module docstring), on any function with those anchors."""
    g = Graft(generate_demo, expected_hash=expected_hash, label="DemoGenerator.generate_demo")
    g.insert_after_assign("state", "_tmg.begin_attempt(self)", occurrence=0)
    g.replace_assign_value("action_dct", "_tmg.get_action(self, obs)")
    g.insert_after_assign("state", "_tmg.record_step(action, action_to_take, reward, in_constraint_segment, system_noise_cfg)",
                          occurrence=1)
    g.insert_before_final_return("_tmg.end_attempt(self, episode_data)")
    return g


def build_mix_graft(solve_constraint, expected_hash: str | None) -> Graft:
    """The two edits of ``CPPolicy.solve_constraint`` for source mixing (module docstring)."""
    g = Graft(solve_constraint, expected_hash=expected_hash, label="CPPolicy.solve_constraint")
    g.insert_after_assign("X_obj0_transf", "_tmg.prepare_mix(self, constraint, obs, X_obj0_transf, X_obj1_transf, X_eef_transf)",
                          occurrence=1)
    g.wrap_assign_value("X_W_eef_des", "_tmg.mixed_eef_des", extra_args=("t_idx",))
    return g


def restore() -> None:
    """Put the original ``generate_demo`` back (tests and interactive use)."""
    global _INSTALLED
    if _INSTALLED is None:
        return
    _INSTALLED.restore()
    _INSTALLED = None


def carried_in_hand(policy, obs: dict) -> dict:
    """Pose in the end-effector frame of every object the current constraint carries (``attached_object_names``), from
    ``obs``; empty when the constraint carries nothing or the observation lacks the poses."""
    constraint = getattr(policy, "current_constraint", None)
    names = list(getattr(constraint, "attached_object_names", None) or [])
    if not names or "robot0_eef_pos" not in obs or "robot0_eef_quat_site" not in obs:
        return {}
    eef = _site_pose(obs["robot0_eef_pos"], obs["robot0_eef_quat_site"])
    return {name: np.linalg.inv(eef) @ np.asarray(obs[f"{name}_pose"], dtype=float)
            for name in names if f"{name}_pose" in obs}
