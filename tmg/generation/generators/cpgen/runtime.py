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
that also skipped carried objects that had not moved in the hand kept Threading at 7 % (2026-11-17).

``method.labels`` is fixed to ``pre_noise`` because the official output already is; the executed (noisy) actions are kept
in our records.

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
from tmg.generation.method.skillmix import SkillMixer
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
    return mixing.site_pose(obs["robot0_eef_pos"], obs["robot0_eef_quat"])


def carried_objects(constraint, obs: dict) -> list[str]:
    """The objects the constraint carries in the hand (``attached_object_names``) whose pose ``obs`` has; empty for a
    constraint that carries nothing (a grasp)."""
    return [name for name in getattr(constraint, "attached_object_names", None) or [] if f"{name}_pose" in obs]


def bridge_just_finished(policy) -> bool:
    """True when the policy has executed the bridge of its current plan and nothing of the constraint yet: the plan had a
    bridge and the queue holds exactly the constraint's steps."""
    queue = getattr(policy, "action_queue", None)
    return bool(queue) and int(getattr(policy, "curr_motion_plan_steps", 0) or 0) > 0 \
        and len(queue) == int(getattr(policy, "curr_constraint_steps", 0) or 0)


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
        if method.mixer is not None and not isinstance(method.mixer, SkillMixer):
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
        self._references: dict[int, str] = {}              # reference object of every recorded constraint (by its index)
        self._resolved_queue: list | None = None           # the action queue whose constraint was already solved again
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
            self.method.mixer.begin_attempt()                 # a new attempt draws new randmix weights
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
        ref = mixing.reference_object(constraint)
        if f"{ref}_pose" not in obs or ref not in src_obj:
            return {}
        quat_key = mixing.quat_key(obs)
        out = {"object_name": ref, "constraint_type": getattr(constraint, "constraint_type", None),
               "cur_obj_pose": np.asarray(obs[f"{ref}_pose"], dtype=float),
               "src_obj_pose": np.asarray(src_obj[ref], dtype=float)[0],
               "eef_pose": mixing.site_pose(obs["robot0_eef_pos"], obs[quat_key]),
               "objects": {k[:-5]: np.asarray(v, dtype=float) for k, v in obs.items()
                           if k.endswith("_pose") and np.asarray(v).shape == (4, 4)},
               "src_objects": {k: np.asarray(v, dtype=float)[0] for k, v in src_obj.items()},
               "src_subtask_start": 0}
        if "robot0_eef_pos" in src_obs and quat_key in src_obs:
            out["src_eef_pose"] = mixing.site_pose(np.asarray(src_obs["robot0_eef_pos"])[0], np.asarray(src_obs[quat_key])[0])
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
            self._references[int(policy.current_constraint_idx)] = scene["object_name"]
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
    def _resolve_after_bridge(self, policy, obs: dict) -> None:
        """When the bridge of the current constraint has just been executed and the constraint carries an object, solve it
        again from ``obs`` and replace the queued steps if the new plan has the same length (module docstring). The
        outcome goes to the constraint's subtask record."""
        queue = policy.action_queue
        if not bridge_just_finished(policy) or queue is self._resolved_queue:
            return
        self._resolved_queue = queue                        # once per plan, whatever the outcome
        if not carried_objects(policy.current_constraint, obs):
            return                                          # a grasp: nothing in the hand can have moved
        result, mix = self._solve_again(policy, obs)
        actions = list(result.actions) if result.success else []
        if getattr(policy.current_constraint, "duplicate_actions", False):
            actions = [a for a in actions for _ in (0, 1)]    # as the official get_action queues them
        if not result.success:
            outcome = "kept: the re-solve failed"
        elif len(actions) != len(queue):
            outcome = f"kept: the re-solve has {len(actions)} steps, the plan {len(queue)}"
        else:
            queue[:] = actions
            outcome = "replaced"
        meta = {"resolve_after_bridge": outcome}
        if outcome == "replaced" and mix is not None:         # the record follows the solve that is executed
            meta.update(mix=mix, mix_before_resolve=self.rec.subtasks[-1].get("mix") if self.rec.subtasks else None)
        self.rec.update_last_subtask(**meta)
        print(f"[resolve-after-bridge] {outcome}", flush=True)

    def _solve_again(self, policy, obs: dict):
        """Solve the current constraint from ``obs``: ``(the official result, its mix record or None)``. The grafted solve
        leaves its mix record and plan on the runtime for the segment that would follow; this solve belongs to a
        constraint whose segment is already recorded, so what was pending before is put back."""
        pending, self._last_mix = self._last_mix, None
        try:
            result = policy.solve_constraint(policy.current_constraint, obs)
            return result, self._last_mix
        finally:
            self._last_mix, self._mix_plan = pending, None

    # source mixing (grafted into CPPolicy.solve_constraint) ----------------------------------------------------------
    def prepare_mix(self, policy, constraint, obs: dict, X_obj0_transf, X_obj1_transf, X_eef_transf):
        """Grafted after CP-Gen chose the drawn source's (primary's) symmetry transforms: weigh every source, align the
        retained ones, drop those beyond the method's distance cap (``SkillMixer.cap``; none without a cap), solve the rest
        for the current scene and prepare the blend (``mixing`` module docstring). Returns the
        constraint CP-Gen's step loop runs on and its transforms: unchanged when the primary is the only retained source;
        the retained source's own constraint (its own window and speed, aligned to the primary) when one other source is
        retained; a copy of the anchor's constraint on the blended timeline when several are retained. In the last two
        cases the policy's sequence index becomes the anchor (the retained source with the largest weight), so the record
        and a second solve name it."""
        unchanged = (constraint, X_obj0_transf, X_obj1_transf, X_eef_transf)
        if self._last_mix is not None:                        # an earlier solve of this attempt never reached a segment
            self._unplanned_mixes.append(self._last_mix)
        self._mix_plan, self._last_mix = None, None
        if self.method.mixer is None:
            return unchanged
        idx, primary = int(policy.current_constraint_idx), int(policy.current_constraint_sequence_idx)
        sequences = policy.constraint_sequences
        if constraint is not sequences[primary][idx]:
            self._last_mix = {"subtask": idx, "source_a": primary, "mixed": False,
                              "identity_reason": "solve_constraint called for a constraint other than the current one"}
            return unchanged
        if self.official is None:
            import demo_aug.generate

            self.official = demo_aug.generate
        carried = self._carried_into(idx)
        try:
            record, w = mixing.decide(self.method.mixer, sequences, idx, primary, obs, carried)
        except KeyError as err:                              # a pose the model needs is missing from the observation
            self._last_mix = {"subtask": idx, "source_a": primary, "mixed": False, "identity_reason": f"missing {err}"}
            return unchanged
        self._last_mix = record
        keep, anchor = mixing.retained(w)
        record["anchor"] = anchor
        if keep == [primary]:
            record.update(mixed=False, identity_reason="only the drawn source is retained")
            return unchanged
        transforms = dict(X_obj0_transf=X_obj0_transf, X_obj1_transf=X_obj1_transf, X_eef_transf=X_eef_transf)
        others = [j for j in keep if j != primary]
        chosen, symmetry = mixing.align_sources(self.official, policy, sequences, idx, primary, others, obs, transforms)
        chosen[primary] = transforms
        record["symmetry"] = symmetry                       # kept even if solving or planning fails below
        w = self.method.mixer.cap(w, symmetry, record)      # mixing.max_source_distance: drops the sources beyond the cap
        keep, anchor = mixing.retained(w)
        record["anchor"] = anchor
        if keep == [primary]:
            record.update(mixed=False, identity_reason="only the drawn source is within the distance cap")
            return unchanged
        policy.current_constraint_sequence_idx = anchor
        if len(keep) == 1:                                  # one source other than the drawn one: its own window and speed
            record.update(mixed=True)
            t = chosen[anchor]
            return sequences[anchor][idx], t["X_obj0_transf"], t["X_obj1_transf"], t["X_eef_transf"]
        solved = mixing.solve_sources(self.official, policy, sequences, idx, keep, obs, chosen)
        self._mix_plan = mixing.MixedWindow(solved, {j: float(w[j]) for j in keep}, anchor)
        record.update(mixed=True, alignment=self._mix_plan.meta)
        t = chosen[anchor]
        window = mixing.window_on(sequences[anchor][idx], self._mix_plan.anchor_index)
        return window, t["X_obj0_transf"], t["X_obj1_transf"], t["X_eef_transf"]

    def _carried_into(self, idx: int) -> str | None:
        """The object in the hand when constraint ``idx`` starts: the reference object of the latest recorded constraint
        before it (None for the first constraint). Asked by index, so a second solve of ``idx`` (after its bridge) gets
        the same object as the first."""
        earlier = [k for k in self._references if k < idx]
        return self._references[max(earlier)] if earlier else None

    def mixed_eef_des(self, t_idx: int, official_pose):
        """Grafted around the step loop's ``compute_X_W_eef_des``: the official pose of step ``t_idx``, or the blended pose
        of that step of the blended window when this constraint is mixed."""
        if self._mix_plan is None:
            return official_pose
        return self._mix_plan.pose(int(t_idx))

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
    g.insert_after_assign("X_obj0_transf", "constraint, X_obj0_transf, X_obj1_transf, X_eef_transf = "
                          "_tmg.prepare_mix(self, constraint, obs, X_obj0_transf, X_obj1_transf, X_eef_transf)", occurrence=1)
    g.wrap_assign_value("X_W_eef_des", "_tmg.mixed_eef_des", extra_args=("t_idx",))
    return g


def restore() -> None:
    """Put the original ``generate_demo`` back (tests and interactive use)."""
    global _INSTALLED
    if _INSTALLED is None:
        return
    _INSTALLED.restore()
    _INSTALLED = None
