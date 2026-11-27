"""The stage record of ``tmg eval`` (``tmg.evaluation.stages``): the first step at which each stage signal turned on, per
episode row and as reach rates in the summary. A fake MimicGen interface stands in for the simulator."""
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tmg.config import compose
from tmg.core.records import AttemptRecorder, SignalLog
from tmg.evaluation import rollout, session, stages
from tmg.generation.simulators.scenes import load_task_stages

from .test_evaluation import eval_setup  # noqa: F401  the fake environment, client and resets of the evaluation tests

GENERATORS = Path("configs/experiment/generator")


class FakeInterface:
    """A needle that is grasped after ``grasp_at`` steps and then rises 1.5 cm per step."""

    def __init__(self, env, grasp_at=2):
        self.env, self.grasp_at = env, grasp_at

    def get_subtask_term_signals(self):
        return {"grasp": self.env.steps >= self.grasp_at}

    def get_object_poses(self):
        pose = np.eye(4)
        pose[2, 3] = 0.8 + 0.015 * max(self.env.steps - self.grasp_at, 0)
        return {"needle": pose, "tripod": np.eye(4)}


class Steps:
    steps = 0


def test_signal_log_gives_the_first_step_each_signal_was_on():
    log = SignalLog()
    for signals in ({"grasp": False, "insert": False}, {"grasp": True, "insert": False}, None, {"grasp": False, "insert": False}):
        log.add(signals)
    assert log.first_on() == {"grasp": 1, "insert": -1} and log.series == {"grasp": [0, 1, 0], "insert": [0, 0, 0]}
    assert SignalLog().first_on() == {}
    assert isinstance(AttemptRecorder().term, SignalLog)          # generation records term_first with the same code


def test_stage_record_reads_term_signals_and_declared_checks(monkeypatch):
    env = Steps()
    monkeypatch.setattr(stages, "env_interface", lambda name, e: FakeInterface(e))
    task = {"name": "threading", "env_prefix": "Threading", "subtask_term_signals": ["grasp", None]}
    checks = [{"type": "object_lifted", "name": "needle_lifted", "object": "needle", "height": 0.04}]
    record = stages.build_stage_record(task, env, checks)
    for _episode in range(2):                                     # a second episode starts from a clean record
        env.steps = 0
        record.begin()
        for _ in range(9):
            env.steps += 1
            record.step()
        # grasped after the 2nd step (index 1); more than 4 cm above the start after the 5th (index 4: 4.5 cm)
        assert record.first_on() == {"grasp": 1, "needle_lifted": 4}
    env.steps = 0
    record.begin()
    env.steps = 1
    record.step()
    assert record.first_on() == {"grasp": -1, "needle_lifted": -1}
    # the sources follow the configuration: checks only (CP-Gen), term signals only (MimicGen), nothing
    assert [type(s) for s in stages.build_stage_record({"env_prefix": "Threading"}, env, checks).sources] == [stages.ObjectLifted]
    assert [type(s) for s in stages.build_stage_record(task, env).sources] == [stages.TermSignals]
    assert stages.build_stage_record({"env_prefix": "Threading", "subtask_term_signals": [None]}, env) is None
    with pytest.raises(ValueError, match="not an object of the task's interface"):
        stages.build_stage_record(task, env, [{"type": "object_lifted", "name": "x", "object": "pod", "height": 0.04}])
    with pytest.raises(ValueError, match="unknown type"):
        stages.build_stage_record(task, env, [{"type": "object_inserted", "name": "x"}])


def test_evaluate_writes_the_stage_record_and_the_reach_rates(eval_setup, monkeypatch):  # noqa: F811
    tmp_path, env, overrides = eval_setup
    # the grasp signal turns on after two steps on every scene; the fake environment succeeds after three on even scenes
    record = stages.StageRecord([stages.TermSignals(FakeInterface(env))])
    monkeypatch.setattr(rollout, "stage_record_of", lambda generator, task, e: record)
    summary = rollout.evaluate(compose("eval", [*overrides, "split=val", "task.horizon=5"]))
    out = tmp_path / "runs/r/eval/val_h5/0000500"
    rows = [json.loads(line) for line in (out / "episodes.jsonl").read_text().splitlines()]
    assert [r["term_first"] for r in rows] == [{"grasp": 1}] * 3 and [r["steps"] for r in rows] == [3, 5, 3]
    assert set(rows[0]) == {"reset_index", "split", "success", "steps", "error", "term_first", "policy_seed", "checkpoint", "seconds"}
    assert summary["all"]["stages"] == {"grasp": {"n": 3, "n_reached": 3, "reach_rate": 1.0}}
    assert summary["by_split"]["val"]["stages"] == summary["all"]["stages"] and summary["all"]["n_success"] == 2


def test_reach_rates_count_only_the_rows_with_a_stage_record():
    rows = [{"split": "test", "success": True, "steps": 9, "term_first": {"grasp": 3, "needle_lifted": 5}},
            {"split": "test", "success": False, "steps": 9, "term_first": {"grasp": 4, "needle_lifted": -1}},
            {"split": "test", "success": False, "steps": 9, "term_first": {"grasp": -1, "needle_lifted": -1}},
            {"split": "test", "success": True, "steps": 9}]                # a row written before the stage record existed
    block = session.summarize(rows)["all"]
    assert block["n"] == 4 and block["n_success"] == 2
    assert block["stages"] == {"grasp": {"n": 3, "n_reached": 2, "reach_rate": 2 / 3},
                               "needle_lifted": {"n": 3, "n_reached": 1, "reach_rate": 1 / 3}}
    old = session.summarize([{"split": "test", "success": True, "steps": 9}])
    assert "stages" not in old["all"] and "stages" not in old["by_split"]["test"]     # old files summarize as before


def test_run_episode_without_a_stage_record_returns_the_old_row():
    class Env:
        steps = 0

        def reset_to(self, scene):
            return {}

        def get_observation(self):
            return {"robot0_eef_pos": np.zeros(3)}

        def step(self, action):
            self.steps += 1
            return self.get_observation(), 0.0, False, {}

        def is_success(self):
            return {"task": self.steps >= 2}

    class Policy:
        def reset(self, seed, task):
            pass

        def act(self, obs):
            return np.zeros(7)

    scene, observation = {"states": np.zeros(1), "model": ""}, {"robot_state": ["robot0_eef_pos"]}
    row = rollout.run_episode(Env(), Policy(), scene, {}, observation, horizon=5, seed=1)
    assert row == {"success": True, "steps": 2, "error": None}
    env = Env()
    record = stages.StageRecord([stages.TermSignals(FakeInterface(env, grasp_at=1))])
    row = rollout.run_episode(env, Policy(), scene, {}, observation, horizon=5, seed=1, stages=record)
    assert row == {"success": True, "steps": 2, "error": None, "term_first": {"grasp": 0}}


def test_committed_stages_files_match_the_tasks_and_rule_files():
    for path in sorted(GENERATORS.glob("*/stages/*.yaml")):
        generator, task_name = path.parts[-3], path.stem
        declared = load_task_stages(generator, task_name)
        task = yaml.safe_load((GENERATORS / generator / "task" / f"{task_name}.yaml").read_text())
        rule = yaml.safe_load((GENERATORS / generator / "coefficients/softmax" / f"{task_name}.yaml").read_text())
        assert set(declared) <= {"subtask_kinds", "stage_checks"}
        if "subtask_kinds" in declared:                            # one kind per subtask of the rule file
            assert len(declared["subtask_kinds"]) == len(rule["subtasks"]) and "grasp" in declared["subtask_kinds"]
        for check in declared.get("stage_checks", []):
            assert check["type"] in stages.CHECKS and check["object"] in task["reference_objects"] and check["height"] > 0
        # a generator whose tasks have termination signals needs no checks; one without them declares its own
        assert bool(declared.get("stage_checks")) != bool(task.get("subtask_term_signals"))
    assert load_task_stages("mimicgen", "no_such_task") == {}
    cpgen = {p.stem for p in (GENERATORS / "cpgen/task").glob("*.yaml")}
    assert cpgen == {p.stem for p in (GENERATORS / "cpgen/stages").glob("*.yaml")}      # every CP-Gen task has a stage record
    # the task configs themselves are untouched: they are part of every finished chunk's identity
    assert all("stage_checks" not in yaml.safe_load(p.read_text()) for p in GENERATORS.glob("*/task/*.yaml"))
