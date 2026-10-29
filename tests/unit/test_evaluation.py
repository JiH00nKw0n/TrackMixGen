"""``tmg eval`` on the robosuite path without a simulator: a fake environment and a fake policy client stand in, so the
episode rows, the summary, the resume rule and the validation row can be checked exactly."""
import json
from pathlib import Path
from typing import ClassVar

import h5py
import numpy as np
import pytest

from tmg.config import compose, resolved
from tmg.evaluation import rollout, session
from tmg.evaluation.resets import resets_path

CKPT = "runs/r/checkpoints/0000500"


class FakeEnv:
    """Succeeds on even reset indices after three steps; the scene index is read from the stored state."""

    def __init__(self):
        self.env = self                       # ``evaluate`` closes ``env.env``
        self.closed = False
        self.calls = []

    def reset_to(self, scene):
        self.index = int(scene["states"][0])
        self.steps = 0
        self.calls.append(("reset_to", self.index, scene["model"]))
        return self.get_observation()

    def get_observation(self):
        return {"agentview_image": np.zeros((84, 84, 3), np.uint8), "robot0_eye_in_hand_image": np.zeros((84, 84, 3), np.uint8),
                "robot0_eef_pos": np.zeros(3), "robot0_eef_quat": np.zeros(4), "robot0_gripper_qpos": np.zeros(2)}

    def step(self, action):
        self.steps += 1
        self.calls.append(("step", self.index, [float(a) for a in action]))
        return self.get_observation(), 0.0, False, {}

    def is_success(self):
        return {"task": self.index % 2 == 0 and self.steps >= 3}

    def close(self):
        self.closed = True


class FakeClient:
    """Answers ``info`` with the served checkpoint and acts with a constant action; records the reset seeds."""

    resets: ClassVar[list] = []
    served = None

    def __init__(self, url, **kwargs):
        self.url = url

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def info(self):
        return {"policy": "diffusion", "checkpoint": FakeClient.served, "cameras": ["agentview", "robot0_eye_in_hand"],
                "state_dim": 9, "env_state_dim": 0, "action_dim": 7}

    def reset(self, seed=None, task=None):
        FakeClient.resets.append((seed, task))

    def act(self, obs):
        assert set(obs) == {"observation.images.agentview", "observation.images.robot0_eye_in_hand", "observation.state"}
        assert obs["observation.images.agentview"].shape == (3, 84, 84) and obs["observation.state"].shape == (9,)
        return np.full(7, 0.5, np.float32)


def write_resets(path: Path, n_val: int, n_test: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["n_val"], f.attrs["n_test"], f.attrs["env_name"] = n_val, n_test, "Square_OURS"
        for i in range(n_val + n_test):
            g = f.create_group(f"reset_{i}")
            g.create_dataset("states", data=np.full(4, float(i)))
            g.create_dataset("model", data=np.frombuffer(f"<mujoco>{i}</mujoco>".encode(), dtype=np.uint8))
            g.attrs["split"] = "val" if i < n_val else "test"


@pytest.fixture
def eval_setup(tmp_path, monkeypatch):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    env = FakeEnv()
    FakeClient.resets = []
    FakeClient.served = str(tmp_path / CKPT)
    monkeypatch.setattr(rollout, "register_env_class", lambda task, scene: "Square_OURS")
    monkeypatch.setattr(rollout, "make_env", lambda *a, **k: env)
    monkeypatch.setattr(session, "PolicyClient", FakeClient)    # play_scenes resolves the class at call time
    overrides = ["task=square", "scene=ours", f"checkpoint={tmp_path / CKPT}", "training_seed=5"]
    write_resets(resets_path(resolved(compose("eval", overrides))), 3, 2)
    return tmp_path, env, overrides


def test_evaluate_plays_the_split_and_writes_rows_summary_and_validation(eval_setup):
    tmp_path, env, overrides = eval_setup
    summary = rollout.evaluate(compose("eval", [*overrides, "split=val", f"append_validation={tmp_path / 'runs/r'}"]))
    out = tmp_path / "runs/r/eval/val_h600/0000500"
    assert out.is_dir() and (out / "resolved_config.yaml").exists() and (out / "provenance.json").exists()
    rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert [r["reset_index"] for r in rows] == [0, 1, 2] and [r["split"] for r in rows] == ["val"] * 3
    assert [r["success"] for r in rows] == [True, False, True] and [r["steps"] for r in rows] == [3, 600, 3]
    assert all(r["error"] is None and r["checkpoint"] == str(tmp_path / CKPT) and "seconds" in r for r in rows)
    # SeedSequence([202600, 5, reset_index]) for reset indices 0, 1, 2: values of the committed evaluator
    expected_seeds = [3655276531, 1001985337, 2855745476]
    assert [r["policy_seed"] for r in rows] == expected_seeds == [s for s, _ in FakeClient.resets]
    instruction = resolved(compose("eval", overrides))["task"]["instruction"]
    assert [t for _, t in FakeClient.resets] == [instruction] * 3
    assert set(rows[0]) == {"reset_index", "split", "success", "steps", "error", "policy_seed", "checkpoint", "seconds"}
    assert env.calls[0] == ("reset_to", 0, "<mujoco>0</mujoco>") and env.closed
    written = json.loads((out / "eval_summary.json").read_text())
    assert written["all"]["n"] == 3 and written["all"]["n_success"] == 2 and written["by_split"]["val"]["success_rate"] == 2 / 3
    assert written["split"] == "val" and written["horizon"] == 600 and written["checkpoint"] == str(tmp_path / CKPT)
    assert written["task"] == "square" and written["scene"] == "ours" and written == summary
    assert written["policy_seed_rule"] == "SeedSequence([evaluation.seed, training_seed, reset_index])"
    assert set(written) == {"checkpoint", "task", "scene", "split", "horizon", "all", "by_split", "policy_seed_rule"}
    block = written["all"]
    assert set(block) == {"n", "n_success", "success_rate", "ci_low", "ci_high", "mean_steps", "errors"}
    assert block["mean_steps"] == 202.0 and block["errors"] == 0 and list(written["by_split"]) == ["val"]
    assert block["ci_low"] == pytest.approx(0.20765, abs=1e-4) and block["ci_high"] == pytest.approx(0.93851, abs=1e-4)
    val_rows = [json.loads(line) for line in (tmp_path / "runs/r/validation.jsonl").read_text().splitlines()]
    assert val_rows == [{"checkpoint": str(tmp_path / CKPT), "success_rate": 2 / 3, "n": 3, "n_success": 2, "horizon": 600,
                         "eval_output": str(out)}]


def test_evaluate_resumes_and_refuses_another_checkpoint(eval_setup):
    tmp_path, env, overrides = eval_setup
    rollout.evaluate(compose("eval", [*overrides, "split=val", "max_scenes=2"]))
    assert len(env.calls) == 2 + 3 + 600        # two resets, 3 steps to success, 600 steps to the horizon
    rollout.evaluate(compose("eval", [*overrides, "split=val"]))
    assert len(env.calls) == 2 + 3 + 600 + 1 + 3   # only reset 2 was missing
    out = tmp_path / "runs/r/eval/val_h600/0000500"
    assert len((out / "episodes.jsonl").read_text().splitlines()) == 3
    FakeClient.served = "/elsewhere/checkpoints/0000500"
    with pytest.raises(RuntimeError, match="another server"):
        rollout.evaluate(compose("eval", [*overrides, "split=test"]))
    FakeClient.served = str(tmp_path / CKPT)
    with pytest.raises(RuntimeError, match="another checkpoint"):
        rollout.evaluate(compose("eval", [*overrides, "split=val", f"checkpoint={tmp_path}/runs/r/checkpoints/0001000",
                                          f"output={out}"]))


def test_evaluate_shards_append_and_the_merge_run_writes_the_summary(eval_setup):
    tmp_path, _, overrides = eval_setup
    for k in range(2):
        s = rollout.evaluate(compose("eval", [*overrides, "split=all", f"shard=[{k},2]"]))
        assert s["all"]["n"] == (3 if k == 0 else 2)
    out = tmp_path / "runs/r/eval/all_h600/0000500"
    assert not (out / "eval_summary.json").exists()
    merged = rollout.evaluate(compose("eval", [*overrides, "split=all", f"append_validation={tmp_path / 'runs/r'}"]))
    assert (out / "eval_summary.json").exists() and merged["all"]["n"] == 5 and merged["by_split"]["test"]["n"] == 2
    # the validation row carries the val block only (resets 0, 1, 2), not the whole summary
    val_rows = [json.loads(line) for line in (tmp_path / "runs/r/validation.jsonl").read_text().splitlines()]
    assert val_rows == [{"checkpoint": str(tmp_path / CKPT), "success_rate": 2 / 3, "n": 3, "n_success": 2, "horizon": 600,
                         "eval_output": str(out)}]
    rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert sorted(r["reset_index"] for r in rows) == [0, 1, 2, 3, 4]


def test_evaluate_fails_without_scenes_and_on_mismatched_cameras(eval_setup, monkeypatch):
    _, _, overrides = eval_setup
    with pytest.raises(FileNotFoundError):
        rollout.evaluate(compose("eval", [*overrides, "split=val", "evaluation.seed=1"]))     # another resets file
    monkeypatch.setattr(FakeClient, "info", lambda self: {"checkpoint": FakeClient.served, "cameras": ["agentview"],
                                                            "env_state_dim": 0})
    with pytest.raises(RuntimeError, match="cameras"):
        rollout.evaluate(compose("eval", [*overrides, "split=val"]))
    monkeypatch.setattr(FakeClient, "info", lambda self: {"checkpoint": FakeClient.served,
                                                            "cameras": ["agentview", "robot0_eye_in_hand"], "env_state_dim": 14})
    with pytest.raises(RuntimeError, match="env_state_dim"):
        rollout.evaluate(compose("eval", [*overrides, "split=val"]))
