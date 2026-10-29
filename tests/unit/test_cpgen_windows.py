"""Per-demonstration constraint windows of CP-Gen (tmg.generation.generators.cpgen.windows): the simulator-free parts."""
import json

import h5py
import numpy as np
import pytest

from tmg.generation.generators.cpgen import runtime as rt
from tmg.generation.generators.cpgen.windows import (
    best_threshold,
    has_constraint_data,
    interactions_of,
    released_windows,
    windowed_constraints,
    write_constraint_data,
)
from tmg.generation.method import build_method

CONSTRAINTS = [{"name": "gripper0_right_right_gripper-coffee_pod_main", "obj_names": ["coffee_pod_main"], "timesteps": list(range(47, 87)),
                "keypoints_obj_frame_annotation": {"coffee_pod_main": [[[0, 0, 0.01]]]}, "symmetries": [], "constraint_type": "robot-object"},
               {"name": "coffee_pod_main-coffee_machine_pod_holder_cup_body_root", "obj_names": ["coffee_pod_main", "coffee_machine_pod_holder_cup_body_root"],
                "timesteps": list(range(125, 159)), "constraint_type": "object-object", "obj_to_parent_attachment_frame": {"coffee_pod_main": "gripper0_right_eef"}}]


def test_interactions_and_released_windows_come_from_the_annotation():
    assert interactions_of(CONSTRAINTS) == [("gripper0_right_right_gripper", "coffee_pod_main"), ("coffee_pod_main", "coffee_machine_pod_holder_cup_body_root")]
    assert released_windows(CONSTRAINTS) == [(47, 87), (125, 159)]


def test_windowed_constraints_replace_only_the_timesteps():
    out = windowed_constraints(CONSTRAINTS, [(40, 80), (110, 160)])
    assert out[0]["timesteps"] == list(range(40, 80)) and out[1]["timesteps"] == list(range(110, 160))
    assert out[0]["keypoints_obj_frame_annotation"] == CONSTRAINTS[0]["keypoints_obj_frame_annotation"] and CONSTRAINTS[0]["timesteps"][0] == 47
    with pytest.raises(ValueError):
        windowed_constraints(CONSTRAINTS, [(40, 80)])


def test_best_threshold_picks_the_nearest_window_and_the_smaller_threshold_on_ties():
    thr, info = best_threshold({0.01: (44, 86), 0.02: (47, 88), 0.03: (43, 89), 0.05: None}, (47, 87))
    assert thr == 0.02 and info["error_steps"] == 1 and info["found"]
    thr, info = best_threshold({0.01: (40, 80), 0.02: (40, 80)}, (47, 87))
    assert thr == 0.01
    assert best_threshold({0.01: None}, (47, 87)) == (0.03, {"threshold": 0.03, "found": False, "released": [47, 87]})


def test_write_constraint_data_gives_every_demonstration_its_own_windows(tmp_path):
    path = tmp_path / "source.hdf5"
    with h5py.File(path, "w") as f:
        for i in range(3):
            g = f.create_group(f"data/demo_{i}")
            g.create_dataset("actions", data=np.zeros((5, 7)))
    assert not has_constraint_data(str(path))
    written = write_constraint_data(str(path), CONSTRAINTS, {1: [[40, 80], [110, 160]], 2: [[30, 70]]})   # demo 2: one window short
    assert written == {0: 2, 1: 2, 2: 0} and not has_constraint_data(str(path))
    with h5py.File(path, "r") as f:
        d0 = [json.loads(f["data/demo_0/constraint_data"][k][()]) for k in sorted(f["data/demo_0/constraint_data"].keys())]
        d1 = [json.loads(f["data/demo_1/constraint_data"][k][()]) for k in sorted(f["data/demo_1/constraint_data"].keys())]
        assert d0[0]["timesteps"] == list(range(47, 87)) and d1[0]["timesteps"] == list(range(40, 80)) and d1[1]["timesteps"][-1] == 159
        assert f["data/demo_1"].attrs["interactions"] == "gripper0_right_right_gripper:coffee_pod_main,coffee_pod_main:coffee_machine_pod_holder_cup_body_root"
        assert "constraint_data" not in f["data/demo_2"]
    write_constraint_data(str(path), CONSTRAINTS, {1: [[40, 80], [110, 160]], 2: [[30, 70], [100, 140]]})   # idempotent rewrite
    assert has_constraint_data(str(path))


def test_runtime_picks_one_source_per_attempt_when_all_sources_are_used():
    method = build_method({"name": "none", "mixing": {"enabled": False}, "tracking": {"enabled": False}}, chunk_seed=3)
    runtime = rt.CPGenRuntime(method, chunk_seed=3, records_dir=None, all_sources=True)

    class Policy:
        constraint_sequences = [object()] * 10
        current_constraint_sequence_idx = 0
        current_constraint_idx = 1

    class Gen:
        policy = Policy()
        env = None

    picks = set()
    for _ in range(40):
        runtime.begin_attempt(Gen)
        picks.add(Gen.policy.current_constraint_sequence_idx)
        assert Gen.policy.current_constraint_idx == 0
    assert picks <= set(range(10)) and len(picks) > 3
    official = rt.CPGenRuntime(method, chunk_seed=3, records_dir=None)        # selection=official: CP-Gen's own choice (sequence 0)
    Gen.policy.current_constraint_sequence_idx = 0
    official.begin_attempt(Gen)
    assert Gen.policy.current_constraint_sequence_idx == 0
