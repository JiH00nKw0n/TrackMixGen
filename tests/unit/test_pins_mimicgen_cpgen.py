"""Pins of the MimicGen and CP-Gen behaviour SkillMix must keep: the mixer's output and record through each runtime's own
entry point (which also pins the random streams), and the fit's rule file from synthetic pools.

The expected values (``fixtures/pins_mimicgen_cpgen.json``) were written at commit 491e5e9, before the Isaac Lab refactor
of the scene, arm and fit interfaces, with ``TMG_WRITE_PINS=1 pytest tests/unit/test_pins_mimicgen_cpgen.py``. They must
never be regenerated to make a change pass: a difference is a change of MimicGen or CP-Gen behaviour.
"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from tmg.generation.method.interfaces import build_method

PINS = Path(__file__).parent / "fixtures" / "pins_mimicgen_cpgen.json"
WRITE = os.environ.get("TMG_WRITE_PINS") == "1"
_written: dict = {}


def plain(value):
    """Arrays and numpy scalars as json values (floats keep every digit through json)."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def same(a, b, path=""):
    """Equal structure, keys, strings and integers; floats to 1e-12 (absolute and relative)."""
    if isinstance(a, dict):
        assert isinstance(b, dict) and sorted(a) == sorted(b), f"{path}: keys {sorted(a)} vs {sorted(b) if isinstance(b, dict) else b}"
        for k in a:
            same(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, list):
        assert isinstance(b, list) and len(a) == len(b), f"{path}: lengths differ"
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
    elif isinstance(a, float) or isinstance(b, float):
        assert a == pytest.approx(b, rel=1e-12, abs=1e-12), f"{path}: {a} vs {b}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def check(name, value):
    value = json.loads(json.dumps(plain(value)))
    if WRITE:
        _written[name] = value
        PINS.write_text(json.dumps({**(json.loads(PINS.read_text()) if PINS.exists() else {}), **_written}, indent=1, sort_keys=True))
        return
    same(json.loads(PINS.read_text())[name], value, name)


def pose(x=0.0, y=0.0, z=0.8, yaw=0.0, roll=0.0):
    p = np.eye(4)
    p[:3, :3] = Rotation.from_euler("zx", [yaw, roll]).as_matrix()
    p[:3, 3] = [x, y, z]
    return p


# MimicGen: the runtime's mix_segment -------------------------------------------------------------------------------
def mimicgen_info(rng, steps=24):
    """A source demonstration's datagen_info: the gripper closes at step 5 of the first segment and opens at 19."""
    eef = np.stack([pose(*rng.normal(0, 0.05, 2), 1.0 + 0.01 * t, yaw=rng.normal(0, 0.2)) for t in range(steps)])
    grip = np.where((np.arange(steps) >= 5) & (np.arange(steps) < 19), 1.0, -1.0)[:, None]
    objects = {name: np.stack([pose(*rng.normal(0, 0.1, 2), yaw=rng.normal(0, 0.5))] * steps) for name in ("nut", "peg")}
    return SimpleNamespace(eef_pose=eef, target_pose=eef + 0.0, gripper_action=grip, object_poses=objects)


def mimicgen_calls(method_cfg, num_sources, seed):
    """Two attempts of two subtasks through ``MimicGenRuntime.mix_segment``: what it returns and the mix record."""
    from tmg.generation.generators.mimicgen.runtime import MimicGenRuntime

    rng = np.random.default_rng(seed)
    infos = [mimicgen_info(rng) for _ in range(num_sources)]
    bounds = np.array([[[0, 12], [12, 24]]] * num_sources)
    runtime = MimicGenRuntime(build_method(method_cfg, chunk_seed=seed), seed, None)
    generator = SimpleNamespace(src_dataset_infos=infos)
    out = []
    for attempt in range(2):
        runtime.begin_attempt(SimpleNamespace())
        for k, name in enumerate(("nut", "peg")):
            primary = int(rng.integers(num_sources))
            start, end = bounds[primary, k]
            now = SimpleNamespace(eef_pose=pose(*rng.normal(0, 0.05, 2), 1.0, yaw=rng.normal(0, 0.2)),
                                  object_poses={n: pose(*rng.normal(0, 0.1, 2), yaw=rng.normal(0, 0.5)) for n in ("nut", "peg")})
            local_vars = {"src_subtask_eef_poses": infos[primary].eef_pose[start:end],
                          "src_subtask_target_poses": infos[primary].target_pose[start:end],
                          "src_subtask_gripper_actions": infos[primary].gripper_action[start:end],
                          "subtask_ind": k, "subtask_object_name": name, "all_subtask_inds": bounds,
                          "selected_src_demo_ind": primary, "cur_object_pose": now.object_poses[name], "cur_datagen_info": now}
            eef, targets, grip = runtime.mix_segment(generator, local_vars)
            out.append({"attempt": attempt, "subtask": k, "primary": primary, "eef": np.asarray(eef), "targets": np.asarray(targets),
                        "gripper": np.asarray(grip), "record": runtime._last_mix})
    return out


RANDMIX = {"name": "skillmix_randmix", "mixing": {"enabled": True, "type": "skillmix", "rule": "randmix"}}


def softmax(rule, single_source="best"):
    return {"name": "skillmix", "mixing": {"enabled": True, "type": "skillmix", "rule": "softmax", "coefficients": rule,
                                           "single_source": single_source}}


def test_mimicgen_randmix_segments_and_records_are_unchanged():
    check("mimicgen_randmix", mimicgen_calls(RANDMIX, 4, 30001))


@pytest.mark.parametrize("single_source", ["best", "sample"])
def test_mimicgen_softmax_segments_and_records_with_the_committed_square_rule_are_unchanged(single_source):
    check(f"mimicgen_softmax_{single_source}", mimicgen_calls(softmax("mimicgen/square", single_source), 10, 30002))


# CP-Gen: mixing.decide ------------------------------------------------------------------------------------------------
def constraint(rng, n=6, kind="robot-object", obj="needle", carried=None, grip=None, with_carried_source=True):
    x = float(rng.normal(0, 0.1))
    src_obj = {obj: np.stack([pose(x, float(rng.normal(0, 0.1)), yaw=float(rng.normal(0, 0.4)))] * n)}
    if carried and with_carried_source:
        src_obj[carried] = np.stack([pose(x, 0.01, 1.0, roll=float(rng.normal(0, 0.2)))] * n)
    g = np.full(n, -1.0) if grip is None else np.asarray(grip, dtype=float)
    return SimpleNamespace(obj_names=[obj] if kind == "robot-object" else [carried, obj], constraint_type=kind,
                           src_obj_pose=src_obj, timesteps=list(range(n)), src_gripper_action=g,
                           src_obs={"robot0_eef_pos": np.tile([x, 0, 1.02], (n, 1)), "robot0_eef_quat_site": np.tile([0, 0, 0, 1.0], (n, 1))})


def cpgen_decisions(method_cfg, num_sources, seed):
    """Two attempts of two constraints through ``cpgen.mixing.decide`` (the second solved twice): records and weights."""
    from tmg.generation.generators.cpgen import mixing

    rng = np.random.default_rng(seed)
    mixer = build_method(method_cfg, chunk_seed=seed).mixer
    sequences = [[constraint(rng, grip=[-1, -1, -1, 1, 1, 1] if j != 1 else [-1, 1, 1, 1, 1, 1]),
                  constraint(rng, kind="object-object", obj="tripod", carried="needle", with_carried_source=j != 2)]
                 for j in range(num_sources)]
    out = []
    for attempt in range(2):
        mixer.begin_attempt()
        obs = {"needle_pose": pose(*rng.normal(0, 0.1, 2), yaw=0.3), "tripod_pose": pose(*rng.normal(0, 0.1, 2), yaw=-0.2),
               "robot0_eef_pos": [0.0, 0.0, 1.0], "robot0_eef_quat_site": [0, 0, 0, 1.0]}
        primary = int(rng.integers(num_sources))
        for idx, carried, seen in ((0, None, obs), (1, "needle", obs), (1, "needle", obs),
                                   (1, "needle", {k: v for k, v in obs.items() if k != "needle_pose"}), (1, None, obs)):
            record, w = mixing.decide(mixer, sequences, idx, primary, seen, carried)
            out.append({"attempt": attempt, "idx": idx, "record": record, "weights": w})
    return out


def test_cpgen_randmix_decisions_are_unchanged():
    check("cpgen_randmix", cpgen_decisions(RANDMIX, 4, 30003))


@pytest.mark.parametrize("single_source", ["best", "sample"])
def test_cpgen_softmax_decisions_with_the_committed_threading_rule_are_unchanged(single_source):
    num_sources = yaml.safe_load((Path(__file__).parents[2] / "configs/experiment/generator/cpgen/coefficients/threading.yaml").read_text())["num_sources"]
    check(f"cpgen_softmax_{single_source}", cpgen_decisions(softmax("cpgen/threading", single_source), int(num_sources), 30004))


# the fit ---------------------------------------------------------------------------------------------------------------
def flat(p):
    return [*p[:3, 3], *p[:3, :3].reshape(-1)]


def write_pool(folder, chunks, attempts):
    """A finished pool: ``attempts(chunk, records folder)`` gives the attempt records of every chunk."""
    for c in range(chunks):
        records = folder / "chunks" / f"seed{c:05d}" / "records"
        records.mkdir(parents=True)
        (records.parent / "CHUNK_DONE.json").write_text("{}")
        (records / "attempts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in attempts(c, records)))


def run_fit(tmp_path, monkeypatch, generator, task, source):
    from omegaconf import OmegaConf

    import tmg.generation.fit_coefficients as fc

    monkeypatch.chdir(tmp_path)                      # the rule file's basis names the pools: relative paths
    monkeypatch.setattr(fc, "write_provenance", lambda *a, **k: None)
    cfg = OmegaConf.create({"randmix_pools": ["randmix"], "unmix_pools": ["unmix"], "output": "out", "min_attempts": 1,
                            "task": task, "source": source, "write_config": False, "generator": generator})
    summary = fc.fit_coefficients(cfg)
    return {"rule_file": yaml.safe_load(Path("out/coefficients.yaml").read_text()),
            "rule_text_head": Path("out/coefficients.yaml").read_text().split("subtasks:")[0],
            "summary_subtasks": summary["subtasks"], "summary_keys": sorted(k for k in summary if k in (
                "randmix_pools", "unmix_pools", "randmix_attempts", "unmix_attempts", "subtasks", "written"))}


def test_mimicgen_fit_from_term_signals_and_datagen_info_is_unchanged(tmp_path, monkeypatch):
    """MimicGen: reached from the per-step ``term__<signal>`` arrays up to the end of the subtask, the sources' in-hand poses
    from the source file's ``datagen_info``."""
    import h5py

    rng = np.random.default_rng(11)
    J, T = 3, 20
    with h5py.File(tmp_path / "source.hdf5", "w") as f:
        for j in range(J):
            info = f.create_group(f"data/demo_{j}/datagen_info")
            info["eef_pose"] = np.stack([pose(0.01 * j, 0.0, 1.0, yaw=0.1 * j)] * T)
            for name in ("nut", "peg"):
                info[f"object_poses/{name}"] = np.stack([pose(0.02 * j, 0.01, 0.9, yaw=0.05 * j)] * T)

    def attempt(index, records, mixed):
        grasped = bool(rng.random() < 0.8)
        success = grasped and bool(rng.random() < 0.6)
        steps = 12
        np.savez(records / f"attempt_{index:06d}.npz", seg_sub=np.array([0, 1]), seg_of_step=np.array([0] * 6 + [1] * 6),
                 term__grasp=np.array([0] * 4 + [int(grasped)] * 8) if rng.random() < 0.9 or not grasped
                 else np.array([0] * 7 + [1] * 5))                 # a few signals fire only after the subtask ended
        subs = []
        for k, name in enumerate(("nut", "peg")):
            w = rng.dirichlet(np.full(J, 0.6)).round(4)
            w[-1] = round(1 - w[:-1].sum(), 4)
            s = {"subtask_ind": k, "object_name": name, "src_demo_ind": int(rng.integers(J)), "src_subtask_start": int(10 * k),
                 "cur_obj_pose": flat(pose(*rng.normal(0, 0.05, 2), yaw=rng.normal(0, 0.3))), "src_obj_pose": flat(pose()),
                 "eef_pose": flat(pose(0.0, 0.0, 1.0)), "objects": {n: flat(pose(*rng.normal(0, 0.02, 2), 0.9)) for n in ("nut", "peg")}}
            if mixed:
                s["mix"] = {"subtask": k, "weights": [float(v) for v in w], "scene_features": rng.random((J, 3)).round(4).tolist(),
                            **({"inhand_features": rng.random((J, 2)).round(4).tolist()} if k else {})}
            subs.append(s)
        return {"attempt": index, "success": success, "n_steps": steps, "npz": f"attempt_{index:06d}.npz", "subtasks": subs}

    write_pool(tmp_path / "randmix", 6, lambda c, records: [attempt(i, records, True) for i in range(40)])
    write_pool(tmp_path / "unmix", 6, lambda c, records: [attempt(i, records, False) for i in range(40)])
    check("fit_mimicgen", run_fit(tmp_path, monkeypatch, {"name": "mimicgen", "simulator": "robosuite"},
                                  {"name": "square", "num_sources": J, "subtask_term_signals": ["grasp", None]}, "source.hdf5"))


def test_cpgen_fit_of_a_generator_that_stops_at_the_first_failed_subtask_is_unchanged(tmp_path, monkeypatch):
    """CP-Gen: an unmix attempt reached k when it recorded k + 1 subtasks, a randmix attempt when it has a mix record of k
    (also from a failed plan, ``unplanned_mixes``); the source's in-hand pose is in the record, or unknown (features 0)."""
    rng = np.random.default_rng(12)
    J = 3

    def sub(k, mixed):
        s = {"subtask_ind": k, "object_name": ("needle", "tripod")[k], "src_demo_ind": int(rng.integers(J)), "src_subtask_start": 0,
             "cur_obj_pose": flat(pose(*rng.normal(0, 0.05, 2), yaw=rng.normal(0, 0.3))), "src_obj_pose": flat(pose()),
             "eef_pose": flat(pose(0.0, 0.0, 1.0)), "src_eef_pose": flat(pose(0.01, 0.0, 1.0)),
             "objects": {"needle": flat(pose(*rng.normal(0, 0.02, 2), 0.95))} if rng.random() < 0.9 else {},
             "src_objects": {"needle": flat(pose(0.0, 0.01, 0.95))} if rng.random() < 0.8 else {}}
        return dict(s, mix=mix(k)) if mixed else s

    def mix(k):
        w = rng.dirichlet(np.full(J, 0.6)).round(4)
        w[-1] = round(1 - w[:-1].sum(), 4)
        return {"subtask": k, "weights": [float(v) for v in w], "scene_features": rng.random((J, 3)).round(4).tolist(),
                **({"inhand_features": (rng.random((J, 2)) * (rng.random((J, 1)) < 0.8)).round(4).tolist()} if k else {})}

    def randmix(chunk, records):
        out = [{"success": bool(rng.random() < 0.5), "subtasks": [sub(0, True), sub(1, True)]} for _ in range(30)]
        out += [{"success": False, "subtasks": [sub(0, True)]} for _ in range(5)]
        return [*out, {"success": False, "subtasks": [sub(0, True)], "unplanned_mixes": [mix(1)]}]

    def unmix(chunk, records):
        return [{"success": bool(n == 2 and rng.random() < 0.5), "subtasks": [sub(k, False) for k in range(n)]}
                for n in (1, 2, 2, 2, 2) * 7]

    write_pool(tmp_path / "randmix", 6, randmix)
    write_pool(tmp_path / "unmix", 6, unmix)
    check("fit_cpgen", run_fit(tmp_path, monkeypatch, {"name": "cpgen", "simulator": "robosuite"},
                               {"name": "threading", "num_sources": J}, None))
