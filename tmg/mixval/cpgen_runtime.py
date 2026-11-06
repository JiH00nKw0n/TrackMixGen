"""CP-Gen runtime with every attempt fixed in advance (scene, perturbation, primary, donor, primary share).

It extends ``tmg.generation.generators.cpgen.runtime.CPGenRuntime`` with one more edit of ``generate_demo``:

| anchor                                   | edit                                         |
|------------------------------------------|----------------------------------------------|
| first ``obs = self.env.reset()``         | value replaced by ``_tmg.fixed_reset(self)`` |

``fixed_reset`` keeps the official reset (fresh reset, collision retry, settle steps) and then overwrites the simulator
state with the attempt's stored scene, moves the free objects by the attempt's perturbation and lets the env settle
again (the same 6 steps the official reset uses). Python's ``random``, numpy and torch are seeded from (scene,
perturbation) right after, so every condition of one scene instance sees the same random draws.

The source picker sets the attempt's primary, and ``FixedMixer.decide`` returns the attempt's donor and primary share for
every constraint (no ranking, no compatibility rule); it records the anchor rotations of the pair so the analysis can
apply a compatibility bound afterwards. ``end_attempt`` appends one result line per attempt to ``results.jsonl`` and
raises ``AllAttemptsDone`` after the last one, which skips CP-Gen's post-loop merge and rendering.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from tmg.core.poses import invert, transform_all
from tmg.generation.generators.cpgen import runtime as cprt
from tmg.generation.graft import install_grafts
from tmg.generation.method.event_mix import anchor_rotation_degrees
from tmg.generation.method.interfaces import Method

PERT_SEED = 20261106
PERT_POS_M = 0.01          # uniform in [-1 cm, 1 cm] for x and y of every free object
PERT_YAW_DEG = 5.0         # uniform in [-5 deg, 5 deg] about the world z axis
MJ_FREE = 0                # mjJNT_FREE


class AllAttemptsDone(Exception):
    """Raised after the shard's last attempt so CP-Gen's ``main`` stops before merging and rendering."""


def instance_seed(scene: int, pert: int) -> int:
    return int(np.random.SeedSequence([PERT_SEED, int(scene), int(pert), 1]).generate_state(1)[0])


def perturbation(scene: int, pert: int, n_objects: int) -> np.ndarray:
    """``(n_objects, 3)`` offsets (dx m, dy m, dyaw deg) of perturbation ``pert`` of ``scene``; zeros for ``pert = 0``."""
    if pert == 0:
        return np.zeros((n_objects, 3))
    rng = np.random.default_rng(np.random.SeedSequence([PERT_SEED, int(scene), int(pert)]))
    out = np.empty((n_objects, 3))
    out[:, :2] = rng.uniform(-PERT_POS_M, PERT_POS_M, size=(n_objects, 2))
    out[:, 2] = rng.uniform(-PERT_YAW_DEG, PERT_YAW_DEG, size=n_objects)
    return out


def free_joints(sim) -> list[tuple[str, int]]:
    """(name, qpos address) of every free joint (the task objects) in model order."""
    model = sim.model
    out = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) == MJ_FREE:
            out.append((model.joint_id2name(j), int(model.jnt_qposadr[j])))
    return out


def apply_perturbation(sim, offsets: np.ndarray) -> list[dict]:
    """Move each free object by its (dx, dy, dyaw) offset; returns what was applied."""
    applied = []
    qpos = sim.data.qpos
    for (name, adr), (dx, dy, dyaw) in zip(free_joints(sim), offsets):
        qpos[adr] += dx
        qpos[adr + 1] += dy
        w, x, y, z = qpos[adr + 3:adr + 7]
        rot = Rotation.from_euler("z", dyaw, degrees=True) * Rotation.from_quat([x, y, z, w])
        qx, qy, qz, qw = rot.as_quat()
        qpos[adr + 3:adr + 7] = [qw, qx, qy, qz]
        applied.append({"joint": name, "dx": round(float(dx), 5), "dy": round(float(dy), 5), "dyaw_deg": round(float(dyaw), 3)})
    sim.forward()
    return applied


def load_scenes(path: str, indices) -> dict[int, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {int(i): np.asarray(f[f"reset_{int(i)}"]["states"], dtype=float) for i in indices}


class FixedPicker:
    """Sets the attempt's primary source (``SourcePicker.pick`` with the index given instead of drawn)."""

    def __init__(self, owner):
        self.owner = owner
        self.all_sources = True
        self.rule = None

    def pick(self, policy) -> None:
        policy.current_constraint_sequence_idx = int(self.owner.current["primary"])
        policy.current_constraint_idx = 0


class FixedMixer:
    """``decide`` of the attempt's pair and share for every constraint (module docstring)."""

    def __init__(self, owner):
        self.owner = owner

    def decide(self, context):
        a = self.owner.current
        primary, donor = int(context.primary), a["donor"]
        record = {"subtask": int(context.subtask_ind), "source_a": primary, "weight_a": a["weight_a"], "mixed": False}
        if primary != int(a["primary"]):
            record["identity_reason"] = f"primary {primary} is not the attempt's {a['primary']}"
            self.owner.mix_log.append(record)
            return record, None, None
        if donor is None:
            record["identity_reason"] = "original"
            self.owner.mix_log.append(record)
            return record, None, None
        donor = int(donor)
        pa = transform_all(invert(np.asarray(context.src_object_poses[primary], dtype=float)), context.primary_targets)
        _eb, tb, gb = context.src_segments[donor]
        pb = transform_all(invert(np.asarray(context.src_object_poses[donor], dtype=float)), np.asarray(tb, dtype=float))
        angles = anchor_rotation_degrees(pa, np.asarray(context.primary_gripper, dtype=float)[:, 0], pb, np.asarray(gb, dtype=float)[:, 0])
        record.update(source_b=donor, anchor_rotation_degrees=angles)
        self.owner.mix_log.append(record)
        return record, donor, float(a["weight_a"])


class FixedRuntime(cprt.CPGenRuntime):
    """``CPGenRuntime`` driven by a fixed attempt list (module docstring)."""

    def __init__(self, attempts: list[dict], scenes: dict[int, np.ndarray], results: Path, records_dir: Path):
        method = Method(name="mixval_fixed", mixer=None, follower_factory=None, labels="pre_noise")
        super().__init__(method, chunk_seed=0, records_dir=records_dir, all_sources=True)
        self.method = Method(name="mixval_fixed", mixer=FixedMixer(self), follower_factory=None, labels="pre_noise")
        self.picker = FixedPicker(self)
        self.attempts, self.scenes, self.results = attempts, scenes, Path(results)
        self.k = -1
        self.current: dict | None = None
        self.mix_log: list[dict] = []
        self.reset_info: dict = {}

    def fixed_reset(self, generator):
        """Official reset, then the attempt's stored scene and perturbation (module docstring)."""
        generator.env.reset()
        self.k += 1
        self.current = a = self.attempts[self.k]
        self.mix_log = []
        sim = generator.env.env.env.sim
        state = self.scenes[int(a["scene"])]
        if state.shape[0] != sim.get_state().flatten().shape[0]:
            raise RuntimeError(f"stored scene has {state.shape[0]} state values, the simulator {sim.get_state().flatten().shape[0]}")
        sim.set_state_from_flattened(state)
        sim.forward()
        applied = apply_perturbation(sim, perturbation(a["scene"], a["pert"], len(free_joints(sim))))
        collided = bool(type(generator.env).is_collision(sim.model._model, sim.data._data))
        seed = instance_seed(a["scene"], a["pert"])
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        try:
            import torch

            torch.manual_seed(seed)
        except ImportError:
            pass
        obs = generator.env.let_env_settle()
        self.reset_info = {"perturbation": applied, "collision_after_perturbation": collided, "seed": seed}
        return obs

    def prepare_mix(self, policy, constraint, obs, X_obj0_transf, X_obj1_transf, X_eef_transf) -> None:
        """The parent's donor solve; a failure of the solve or the alignment is recorded and the attempt marked."""
        try:
            super().prepare_mix(policy, constraint, obs, X_obj0_transf, X_obj1_transf, X_eef_transf)
        except Exception as exc:  # noqa: BLE001  recorded per attempt, the analysis drops these attempts
            self._mix_plan = None
            self.mix_log.append({"subtask": int(policy.current_constraint_idx), "mix_error": f"{type(exc).__name__}: {exc}"[:300]})

    def end_attempt(self, generator, episode_data: dict) -> None:
        super().end_attempt(generator, episode_data)
        a = self.current
        line = {**a, "success": bool(episode_data.get("success", False)), "failure_type": episode_data.get("failure_type"),
                "elapsed_s": round(time.time() - self._attempt_t0, 2), "subtasks_planned": self._subtasks,
                "mix_error": any("mix_error" in m for m in self.mix_log),
                "mix": [{k: v for k, v in m.items() if k in ("subtask", "mixed", "source_b", "anchor_rotation_degrees", "identity_reason", "mix_error")}
                        for m in self.mix_log], **self.reset_info}
        with open(self.results, "a") as f:
            f.write(json.dumps(line, default=float) + "\n")
        print(f"[mixval] {self.k + 1}/{len(self.attempts)} {a['id']} success={line['success']} {line['elapsed_s']}s", flush=True)
        if self.k + 1 >= len(self.attempts):
            raise AllAttemptsDone()


def apply(attempts, scenes, *, results: Path, records_dir: Path, graft_hashes: dict) -> FixedRuntime:
    """Install the grafted ``generate_demo`` (with ``fixed_reset``) and ``solve_constraint``."""
    from demo_aug.generate import CPPolicy, DemoGenerator

    rt = FixedRuntime(attempts, scenes, results, records_dir)
    gen = cprt.build_graft(DemoGenerator.generate_demo, graft_hashes.get("generate_demo"))
    gen.replace_assign_value("obs", "_tmg.fixed_reset(self)", occurrence=0)
    edits = [(DemoGenerator, "generate_demo", gen),
             (CPPolicy, "solve_constraint", cprt.build_mix_graft(CPPolicy.solve_constraint, graft_hashes.get("solve_constraint")))]
    install_grafts(edits, {"_tmg": rt})
    return rt
