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


# symmetry alignment ----------------------------------------------------------------------------------------------------
def _se3(transforms):
    out = []
    for p in transforms:
        t = np.eye(4)
        t[:3, :3] = Rotation.from_euler(p["rot_axis"], p["rot_param"]).as_matrix()
        out.append(t)
    return out


class FakeOfficial:
    """CP-Gen's ``compute_X_W_eef_des`` without keypoint refinement: the source grasp in the (transformed) object frame
    put on the current object, then the end-effector transform."""
    generate_se3_transforms = staticmethod(_se3)

    @staticmethod
    def compute_X_W_eef_des(c, obs, rc, ec, X_obj0_transf=None, X_obj1_transf=None, X_eef_transf=None, src_t_idx=0,
                            ignore_scaling=False):
        obj = c.obj_names[0]
        src_obj = c.src_obj_pose[obj][src_t_idx] @ X_obj0_transf
        src_eef = mixing.site_pose(c.src_obs["robot0_eef_pos"][src_t_idx], c.src_obs["robot0_eef_quat_site"][src_t_idx])
        return obs[f"{obj}_pose"] @ np.linalg.inv(src_obj) @ src_eef @ X_eef_transf


def grasp(y, yaw, n=4, obj="hammer", x=0.0):
    """A source that grasps the hammer (at the origin, yaw 0) at (x, y) in the hammer frame with gripper yaw ``yaw``."""
    quat = Rotation.from_euler("xz", [np.pi, yaw]).as_quat()       # pointing down, turned about the approach axis
    sym = SimpleNamespace(type="obj0", transforms=[{"rot_axis": "z", "rot_param": np.pi, "translation": [0, 0, 0]}])
    return SimpleNamespace(obj_names=[obj], constraint_type="robot-object", reflect_eef=True, symmetries=[sym],
                           src_obj_pose={obj: np.stack([pose()] * n)}, timesteps=list(range(n)), src_gripper_action=np.full(n, -1.0),
                           src_obs={"robot0_eef_pos": np.tile([x, y, 0.85], (n, 1)), "robot0_eef_quat_site": np.tile(quat, (n, 1))})


POLICY = SimpleNamespace(robot_configuration=None, eef_configuration=None, ignore_scaling=True)
IDENTITY = dict(X_obj0_transf=np.eye(4), X_obj1_transf=np.eye(4), X_eef_transf=np.eye(4))


def test_symmetry_candidates_follow_find_best_transform():
    labels = [label for label, _ in mixing.symmetry_candidates(FakeOfficial, grasp(0.0, 0.0))]
    assert labels == ["identity", "obj0:z180", "eef:z180", "obj0:z180+eef:z180"]


def test_sources_held_from_the_other_side_are_aligned_to_the_primary_before_blending():
    # sources 0 and 2 hold the hammer at y=-0.06, source 1 (and 3, also turned about the approach axis) from the other
    # side, 180 degrees about the hammer z axis: the object symmetry maps them onto the primary's side
    seqs = [[grasp(-0.06, 0.0)], [grasp(0.06, np.pi)], [grasp(-0.05, 0.05)], [grasp(0.06, 0.0)]]
    obs = {"hammer_pose": pose(0.1, 0.2, 0.3)}
    chosen, info = mixing.align_sources(FakeOfficial, POLICY, seqs, 0, 0, [1, 2, 3], obs, IDENTITY)
    assert info["1"]["symmetry"] == "obj0:z180" and info["2"]["symmetry"] == "identity"
    assert info["3"]["symmetry"] == "obj0:z180+eef:z180"
    assert info["1"]["pos_cm"] < 0.1 and info["1"]["rot_deg"] < 0.1
    assert info["1"]["primary_symmetry_pos_cm"] == pytest.approx(12.0, abs=0.01)
    assert info["1"]["primary_symmetry_rot_deg"] == pytest.approx(180.0, abs=0.1)
    solved = mixing.solve_sources(FakeOfficial, POLICY, seqs, 0, [1, 2, 3], obs, chosen)
    plan = mixing.MultiMixPlan(mixing.gripper_series(seqs[0][0]), solved, {"primary": 0.25, 1: 0.25, 2: 0.25, 3: 0.25})
    primary = FakeOfficial.compute_X_W_eef_des(seqs[0][0], obs, None, None, src_t_idx=0, **IDENTITY)
    dp, dr = mixing.pose_distance(plan.pose(0, primary), primary)
    assert dp < 0.01 and np.degrees(dr) < 2.0             # without alignment: the hammer's middle, 180 degrees off


def test_alignment_records_the_distance_to_the_primary():
    seqs = [[grasp(-0.06, 0.0)], [grasp(-0.06, 0.0, x=0.08)]]
    obs = {"hammer_pose": pose()}
    chosen, info = mixing.align_sources(FakeOfficial, POLICY, seqs, 0, 0, [1], obs, IDENTITY)
    assert info["1"]["symmetry"] == "identity" and info["1"]["pos_cm"] == pytest.approx(8.0, abs=0.01)
    solved = mixing.solve_sources(FakeOfficial, POLICY, seqs, 0, [1], obs, chosen)
    plan = mixing.MultiMixPlan(mixing.gripper_series(seqs[0][0]), solved, {"primary": 0.5, 1: 0.5})
    assert plan.pose(0, FakeOfficial.compute_X_W_eef_des(seqs[0][0], obs, None, None, **IDENTITY))[0, 3] == pytest.approx(0.04)
