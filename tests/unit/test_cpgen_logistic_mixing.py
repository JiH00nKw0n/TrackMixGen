"""CP-Gen logistic mixing: per-source prediction and weights from the constraint data, and the multi-source blend on the
primary's timeline."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tmg.generation.generators.cpgen import mixing
from tmg.generation.method.success_model import SubtaskModel, SuccessModel


def pose(x=0.0, y=0.0, yaw=0.0):
    p = np.eye(4)
    p[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    p[:3, 3] = [x, y, 0.8]
    return p


def constraint(x, n=6, kind="robot-object", obj="pod", carried=None, grip=None):
    src_obj = {obj: np.stack([pose(x)] * n)}
    if carried:
        src_obj[carried] = np.stack([pose(x, 0, 0)] * n)
        src_obj[carried][:, 2, 3] = 1.0
    g = np.full(n, -1.0) if grip is None else np.asarray(grip, dtype=float)
    return SimpleNamespace(obj_names=[obj] if kind == "robot-object" else [carried, obj], constraint_type=kind,
                           src_obj_pose=src_obj, timesteps=list(range(n)), src_gripper_action=g,
                           src_obs={"robot0_eef_pos": np.tile([x, 0, 1.0], (n, 1)), "robot0_eef_quat_site": np.tile([0, 0, 0, 1.0], (n, 1))})


def model(n):
    return SuccessModel(name="cpgen/t", num_sources=n, carried={}, subtasks={
        0: SubtaskModel(np.zeros(n), {"dx": -20.0, "dy": 0.0, "rotation": 0.0}),
        1: SubtaskModel(np.zeros(n), {"dx": -20.0, "dy": 0.0, "rotation": 0.0, "inhand_position": -50.0, "inhand_rotation": 0.0})})


def test_decide_keeps_sources_at_or_above_the_mean_and_compatible():
    seqs = [[constraint(0.0)], [constraint(0.05)], [constraint(0.3)], [constraint(0.01, grip=[-1, -1, 1, 1, 1, 1])]]
    obs = {"pod_pose": pose(0.0), "robot0_eef_pos": [0, 0, 1.0], "robot0_eef_quat_site": [0, 0, 0, 1.0]}
    record, w = mixing.decide(model(4), seqs, 0, 1, obs, None)
    assert record["incompatible"] == [3]                 # gripper events differ from the primary's
    # predictions 0.5, 0.27, 0.0025, 0.45 (mean 0.31): only source 0 stays; the primary (1) gives the timeline only
    assert w[0] == pytest.approx(1.0) and w[1] == 0 and w[2] == 0 and w[3] == 0


def test_decide_uses_inhand_features_from_the_second_constraint():
    seqs = [[constraint(0.0), constraint(0.0, kind="object-object", obj="machine", carried="pod")] for _ in range(2)]
    obs = {"machine_pose": pose(0.0), "pod_pose": pose(0.0), "robot0_eef_pos": [0, 0, 1.0], "robot0_eef_quat_site": [0, 0, 0, 1.0]}
    obs["pod_pose"][2, 3] = 1.0
    record, w = mixing.decide(model(2), seqs, 1, 0, obs, "pod")
    assert "inhand_features" in record and np.allclose(record["inhand_features"], 0, atol=1e-6)


def test_multi_mix_plan_blends_on_the_primary_timeline():
    n = 5
    primary_grip = np.full((n, 1), -1.0)
    other = np.stack([pose(0.1 * t + 1.0) for t in range(n)])
    plan = mixing.MultiMixPlan(primary_grip, {2: (other, np.full((n, 1), -1.0))}, {"primary": 0.5, 2: 0.5})
    out = plan.pose(0, pose(0.0))
    assert out[0, 3] == pytest.approx(0.5) and np.allclose(out[:3, :3], np.eye(3))
    plan0 = mixing.MultiMixPlan(primary_grip, {2: (other, np.full((n, 1), -1.0))}, {"primary": 0.0, 2: 1.0})
    assert plan0.pose(n - 1, pose(0.0))[0, 3] == pytest.approx(other[-1, 0, 3])
    with pytest.raises(IndexError):
        plan.pose(n, pose(0.0))
