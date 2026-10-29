"""CP-Gen adapter units without CP-Gen installed: configs, scene agreement, pose conversions, and the grafted
``generate_demo`` (the official text, from the fixture) driven with a fake environment and a fake planner."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from tmg.config import compose, resolved
from tmg.core.chunks import plan_chunks
from tmg.core.records import ATTEMPTS_JSONL, SEG_BRIDGE, SEG_SKILL
from tmg.generation.generators import make_generator
from tmg.generation.generators.cpgen import runtime as rt
from tmg.generation.graft import source_hash
from tmg.generation.method import build_method

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "cpgen_generate_demo.py"
TASKS = ["coffee", "threading", "hammer_cleanup", "mug_cleanup", "three_piece_assembly"]


def load_fixture():
    spec = importlib.util.spec_from_file_location("cpgen_fixture", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cpgen_chunk_spec_and_method_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    cfg = resolved(compose("generate", ["generator=cpgen", "task=coffee", "scene=ours", "method=trackmix", "generation=test/pilot20"]))
    gen = make_generator(cfg)
    spec = gen.prepare_chunk(plan_chunks(20, 20, 9000)[0], tmp_path / "chunk")
    c = spec.extra["cpgen_config"]
    assert c["n_demos"] == 20 and c["require_n_demos"] is False and c["seed"] == 9001 and c["env_name"] == "Coffee_OURS"
    assert c["load_demos_start_idx"] == 0 and c["load_demos_end_idx"] == 1 and c["use_wandb"] is False
    assert c["custom_constraints_path"].endswith("cpgen/coffee/constraints.json")
    assert spec.method["mixing"]["enabled"] is False and spec.method["tracking"]["enabled"] is True and spec.method["labels"] == "pre_noise"
    assert spec.scene["base_env_suffix"] == "D0" and spec.scene["override_bounds"] is True
    assert gen.demo_layout()["demo_glob"] == "generated_obs.hdf5" and gen.demo_layout()["labels_from_records"] is False
    assert (tmp_path / "chunk" / "cpgen_config.json").exists()
    none = make_generator(resolved(compose("generate", ["generator=cpgen", "task=threading", "scene=official_d0", "method=none", "generation=test/pilot20"])))
    assert none.method["labels"] == "pre_noise" and none.scene_chain()["env_suffix"] == "D0" and none.env_name() == "Threading_D0"
    for t in TASKS:
        compose("generate", ["generator=cpgen", f"task={t}", "scene=ours", "method=none", "generation=test/pilot20"])


def test_cpgen_ours_scene_equals_mimicgen_ours_for_its_tasks():
    mg = yaml.safe_load((ROOT / "configs/experiment/generator/mimicgen/scene/ours.yaml").read_text())
    cp = yaml.safe_load((ROOT / "configs/experiment/generator/cpgen/scene/ours.yaml").read_text())
    assert set(cp["bounds"]) == set(TASKS)
    for t in TASKS:
        assert cp["bounds"][t] == mg["bounds"][t], t
    assert cp["reach_constraint"] == mg["reach_constraint"] and cp["table_clip_m"] == mg["table_clip_m"]


def test_pose_conversions_round_trip():
    action = np.array([0.1, -0.2, 0.9, 0.3, -0.1, 1.2, -1.0])
    pose = rt.action_to_pose(action)
    back = rt.pose_to_action(pose, action[6:])
    assert np.allclose(back, action, atol=1e-9)
    obs = {"robot0_eef_pos": action[:3], "robot0_eef_quat": Rotation.from_rotvec(action[3:6]).as_quat()}
    assert np.allclose(rt.pose_from_obs(obs), pose, atol=1e-9)


def test_fixture_hash_is_the_pinned_one():
    module = load_fixture()
    pinned = yaml.safe_load((ROOT / "configs/experiment/generator/cpgen.yaml").read_text())["graft_hashes"]["generate_demo"]
    assert source_hash(module.DemoGenerator.generate_demo) == pinned


class FakeEnv:
    """Steps instantly to the commanded absolute pose; reward 1 once ``success_at`` steps were taken."""

    def __init__(self, success_at):
        self.t, self.success_at = 0, success_at
        self.pose = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    def _obs(self):
        return {"robot0_eef_pos": self.pose[:3].copy(), "robot0_eef_quat": Rotation.from_rotvec(self.pose[3:6]).as_quat()}

    def reset(self):
        self.t = 0
        return self._obs()

    def get_state(self):
        return {"states": np.zeros(3), "model": "<xml/>"}

    def step(self, action):
        self.pose = np.asarray(action[:6], dtype=float)
        self.t += 1
        return self._obs(), (1.0 if self.t >= self.success_at else 0.0), False, {}


class FakePolicy:
    """Two segments of 2 motion-plan + 3 constraint actions, with CP-Gen's queue bookkeeping."""

    def __init__(self):
        fields = dict(obj_names=["pod"], obj_to_parent_attachment_frame=None, post_constraint_behavior=None, constraint_type="pose",
                      keypoints_obj_frame_annotation=None, keypoints_robot_link_frame_annotation=None,
                      reset_near_random_constraint_state=None, name="c")
        self.constraint_sequences = [[SimpleNamespace(**fields), SimpleNamespace(**fields)]]
        self.current_constraint_sequence_idx = 0
        self.current_constraint_idx = 0
        self.action_queue, self.check_constraint, self.verbose = [], False, False
        self.curr_motion_plan_steps = self.curr_constraint_steps = -1
        self.plans = 0

    @property
    def current_constraint_sequence(self):
        return self.constraint_sequences[self.current_constraint_sequence_idx]

    @property
    def current_constraint(self):
        return self.current_constraint_sequence[self.current_constraint_idx]

    @property
    def done(self):
        return (not self.action_queue) and self.current_constraint_idx >= len(self.current_constraint_sequence)

    def get_action(self, obs):
        if self.check_constraint:
            self.current_constraint_idx += 1
            self.check_constraint = False
        if self.done:
            return {"action": None, "failure_type": None}
        if not self.action_queue:
            base = 0.1 * (self.plans + 1)
            mp = [np.array([base + 0.01 * i, 0, 1, 0, 0, 0, -1.0]) for i in range(2)]
            cs = [np.array([base + 0.05 + 0.01 * i, 0, 1, 0, 0, 0, 1.0]) for i in range(3)]
            self.action_queue = mp + cs
            self.curr_motion_plan_steps, self.curr_constraint_steps = 2, 3
            self.new_constraint = True
            self.plans += 1
        else:
            self.new_constraint = False
        action = self.action_queue.pop(0)
        if not self.action_queue:
            self.check_constraint = True
        return {"action": action, "failure_type": None}

    def reset(self):
        self.current_constraint_idx, self.check_constraint, self.action_queue = 0, False, []


def run_fixture(tmp_path, method_cfg):
    module = load_fixture()
    method = build_method(method_cfg, chunk_seed=3)
    runtime = rt.CPGenRuntime(method, chunk_seed=3, records_dir=tmp_path / "records")
    grafted = rt.build_graft(module.DemoGenerator.generate_demo, source_hash(module.DemoGenerator.generate_demo)).compile({"_tmg": runtime})
    gen = module.DemoGenerator()
    gen.env, gen.policy = FakeEnv(success_at=10), FakePolicy()
    episode = grafted(gen, system_noise_cfg=module.SystemNoiseConfig())
    lines = [json.loads(l) for l in (tmp_path / "records" / ATTEMPTS_JSONL).read_text().splitlines()]
    return episode, lines, runtime


def test_grafted_generate_demo_records_the_official_run(tmp_path):
    episode, lines, _ = run_fixture(tmp_path, {"name": "none", "mixing": {"enabled": False}, "tracking": {"enabled": False}})
    assert episode["success"] is True and len(episode["actions"]) == 10
    assert len(lines) == 1 and lines[0]["success"] is True and lines[0]["n_steps"] == 10 and lines[0]["src_demo_inds"] == [0, 0]
    z = np.load(tmp_path / "records" / "attempt_000000.npz")
    assert z["seg_kind"].tolist() == [SEG_BRIDGE, SEG_SKILL, SEG_BRIDGE, SEG_SKILL] and z["seg_len"].tolist() == [2, 3, 2, 3]
    assert np.allclose(z["a_label"], episode["actions"]) and np.allclose(z["a_exec"], episode["actions"])   # zero system noise


def test_grafted_generate_demo_with_tracking_drives_the_planned_path(tmp_path):
    tracking = {"enabled": True, "ds_min": 0.25, "regions": {"default": {"alpha_p": 12.0, "alpha_r": 1.2}},
                "approach_window": 2, "extra_steps_cap": "segment", "gripper_events": {"type": "sign"}}
    episode, lines, runtime = run_fixture(tmp_path, {"name": "tracking_only", "mixing": {"enabled": False}, "tracking": tracking})
    assert lines[0]["n_steps"] == len(episode["actions"]) >= 10 and runtime.follower is None
    z = np.load(tmp_path / "records" / "attempt_000000.npz")
    assert (z["track_k"] >= 0).all() and z["seg_kind"].tolist() == [SEG_BRIDGE, SEG_SKILL, SEG_BRIDGE, SEG_SKILL]
    assert episode["success"] is True and lines[0]["src_demo_inds"] == [0, 0]
