"""MimicGen runtime with every attempt fixed in advance (scene, perturbation, primary, donor, primary share).

It extends ``tmg.generation.generators.mimicgen.runtime.MimicGenRuntime`` with one more edit of ``DataGenerator.generate``:

| anchor                                      | edit                                                       |
|---------------------------------------------|------------------------------------------------------------|
| after ``new_initial_state = env.get_state()`` | ``new_initial_state = _tmg.fixed_reset(env)`` (runs first) |

``fixed_reset`` overwrites the freshly reset simulator with the attempt's stored scene (``EnvRobosuite.reset_to`` with
the flat state), re-anchors the OSC controller to it (the stale-goal fix of ``tmg.evaluation.rollout.sync_controllers``),
applies the attempt's perturbation and seeds the random streams from the scene instance (``scenes``). The official
``randomize_subtask_boundaries``, source selection and action-noise draws that follow therefore repeat across conditions.

The source rule returns the attempt's primary for every subtask (both subtasks use the same source). ``FixedMixer`` is
``GuidedMixer`` with the decision replaced: the attempt's donor and primary share for every subtask, no ranking groups
and no rotation bound (the anchor rotations are recorded for the analysis). When the two sources' gripper events do not
match, the segment is left unmixed and the attempt is marked ``mix_error``. The blend itself is ``GuidedMixer.mix``
(constant weight: both sources in the primary's object frame, events moved to the weighted event steps), so donor
share 100 % is the donor's segment expressed relative to the primary's reference object.

One result line per attempt goes to ``results.jsonl``. An attempt that MimicGen aborted with a rollout exception is
written as a failure when the next attempt starts. After the last attempt the process exits at once
(``os._exit``), skipping MimicGen's dataset post-processing, which this check does not use.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from tmg.core.poses import invert, transform_all
from tmg.generation.generators.mimicgen import runtime as mgrt
from tmg.generation.graft import install_grafts
from tmg.generation.method.guided_mix import GuidedMixConfig, GuidedMixer
from tmg.generation.method.interfaces import Method
from tmg.mixval import scenes as sc

RECORD_KEYS = ("subtask", "mixed", "source_b", "anchor_rotation_degrees", "identity_reason", "blend_parts")


def sync_controllers(env) -> None:
    """``tmg.evaluation.rollout.sync_controllers``: initial joints, controller state and goal from the restored state."""
    robosuite_env = getattr(env, "env", env)
    for robot in getattr(robosuite_env, "robots", []):
        controller = getattr(robot, "controller", None)
        if controller is None or not hasattr(controller, "reset_goal"):
            continue
        controller.update_initial_joints(robot.sim.data.qpos[robot._ref_joint_pos_indexes])
        controller.update(force=True)
        controller.reset_goal()


class FixedSources:
    """``method.sources`` rule: the attempt's primary for every subtask."""

    def __init__(self, owner):
        self.owner = owner

    def pick(self, subtask_ind: int) -> int:
        return int(self.owner.current["primary"])

    def describe(self) -> dict:
        return {"type": "mixval_fixed"}


class FixedMixer(GuidedMixer):
    """``GuidedMixer`` whose decision is the attempt's donor and share (module docstring)."""

    def __init__(self, owner):
        super().__init__(ranking=None, chunk_seed=0, config=GuidedMixConfig(weight_a=0.5, max_anchor_rotation_deg=1e9))
        self.owner = owner

    def describe(self) -> dict:
        return {"type": "mixval_fixed"}

    def _decide(self, context):
        a = self.owner.current
        self.blend_parts = a.get("parts", "both")
        primary = int(context.primary)
        record = {"subtask": int(context.subtask_ind), "source_a": primary, "weight_a": a["weight_a"], "mixed": False}
        self.owner.mix_log.append(record)
        if primary != int(a["primary"]):
            record["identity_reason"] = f"primary {primary} is not the attempt's {a['primary']}"
            return record, None, None, None
        if a["donor"] is None:
            record["identity_reason"] = "original"
            return record, None, None, None
        if not context.has_reference_object:
            record["identity_reason"] = "subtask has no reference object"
            self.owner.mix_error = True
            return record, None, None, None
        obj_a = np.asarray(context.src_object_poses[primary], dtype=float)
        inv_a = invert(obj_a)
        pa = transform_all(inv_a, context.primary_targets)
        ga = np.asarray(context.primary_gripper, dtype=float)
        sa = self._event_series(ga)
        candidates = self._candidates(context, [int(a["donor"])], pa, sa)
        if not candidates:
            record["identity_reason"] = "gripper events of the pair differ"
            self.owner.mix_error = True
            return record, None, None, None
        return record, candidates[0], float(a["weight_a"]), (obj_a, inv_a, pa, ga, sa)


class FixedMimicGenRuntime(mgrt.MimicGenRuntime):
    """``MimicGenRuntime`` driven by a fixed attempt list (module docstring)."""

    def __init__(self, attempts: list[dict], scenes: dict[int, np.ndarray], results: Path, records_dir: Path):
        super().__init__(Method(name="mixval_fixed", mixer=None, follower_factory=None, labels="pre_noise"),
                         chunk_seed=0, records_dir=records_dir)
        self.method = Method(name="mixval_fixed", mixer=FixedMixer(self), follower_factory=None, labels="pre_noise",
                             sources=FixedSources(self))
        self.attempts, self.scenes, self.results = attempts, scenes, Path(results)
        self.k = -1
        self.current: dict | None = None
        self.open = False
        self.mix_log: list[dict] = []
        self.mix_error = False
        self.reset_info: dict = {}
        self.t0 = 0.0

    def _write(self, success: bool, failure: str | None) -> None:
        a = self.current
        line = {**a, "success": bool(success), "failure_type": failure, "elapsed_s": round(time.time() - self.t0, 2),
                "mix_error": self.mix_error, "mix": [{k: v for k, v in m.items() if k in RECORD_KEYS} for m in self.mix_log],
                **self.reset_info}
        with open(self.results, "a") as f:
            f.write(json.dumps(line, default=float) + "\n")
        self.open = False
        print(f"[mixval] {self.k + 1}/{len(self.attempts)} {a['id']} success={line['success']} {line['elapsed_s']}s", flush=True)

    def fixed_reset(self, env):
        """The attempt's stored scene and perturbation on the freshly reset env (module docstring)."""
        if self.open:
            self._write(False, "rollout_exception")
        self.k += 1
        self.current = a = self.attempts[self.k]
        self.mix_log, self.mix_error, self.t0 = [], False, time.time()
        sim = env.env.sim
        state = self.scenes[int(a["scene"])]
        sc.check_state_size(state, sim)
        env.reset_to({"states": state})
        sync_controllers(env)
        applied = sc.apply_perturbation(sim, sc.perturbation(a["scene"], a["pert"], len(sc.free_joints(sim))))
        sync_controllers(env)
        seed = sc.instance_seed(a["scene"], a["pert"], a.get("rep", 0))
        sc.seed_everything(seed)
        self.reset_info = {"perturbation": applied, "seed": seed}
        self.open = True
        return env.get_state()

    def end_attempt(self, results: dict):
        super().end_attempt(results)
        self._write(bool(results["success"]), None)
        if self.k + 1 >= len(self.attempts):
            print("[mixval] all attempts done", flush=True)
            os._exit(0)

    def finish(self) -> None:
        """After MimicGen returned: the last attempt, when a rollout exception ended it."""
        if self.open:
            self._write(False, "rollout_exception")


def apply(attempts, scenes, *, results: Path, records_dir: Path, graft_hashes: dict) -> FixedMimicGenRuntime:
    from mimicgen.datagen.data_generator import DataGenerator
    from mimicgen.datagen.waypoint import WaypointTrajectory

    rt = FixedMimicGenRuntime(attempts, scenes, results, records_dir)
    gen, exe = mgrt.build_grafts(DataGenerator.generate, WaypointTrajectory.execute, graft_hashes)
    gen.insert_after_assign("new_initial_state", "new_initial_state = _tmg.fixed_reset(env)")
    install_grafts([(DataGenerator, "generate", gen), (WaypointTrajectory, "execute", exe)], {"_tmg": rt})
    return rt
