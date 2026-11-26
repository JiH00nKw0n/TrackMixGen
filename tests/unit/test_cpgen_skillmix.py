"""CP-Gen SkillMix: per-source features and weights from the constraint data, the multi-source blend on the weighted
timeline, and the step loop's constraint resampled onto it."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tmg.generation.generators.cpgen import mixing
from tmg.generation.method.skillmix import SkillMixer, SkillMixRule, SubtaskRule


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


def mixer(n, d=0.05, rule="softmax"):
    g = np.array([-20.0, 0.0, 0.0, -50.0, 0.0])
    fitted = SkillMixRule(name="cpgen/t", rule="softmax", num_sources=n, subtasks={k: SubtaskRule(a=np.zeros(n), g=g, d=d, c0=0.0) for k in (0, 1)})
    return SkillMixer(rule=rule, coefficients=None if rule == "randmix" else fitted, chunk_seed=3)


def test_decide_weighs_the_compatible_sources_by_the_softmax_of_their_scores():
    seqs = [[constraint(0.0)], [constraint(0.05)], [constraint(0.3)], [constraint(0.01, grip=[-1, -1, 1, 1, 1, 1])]]
    obs = {"pod_pose": pose(0.0), "robot0_eef_pos": [0, 0, 1.0], "robot0_eef_quat_site": [0, 0, 0, 1.0]}
    record, w = mixing.decide(mixer(4), seqs, 0, 1, obs, None)
    assert record["incompatible"] == [3] and w[3] == 0     # gripper events differ from the primary's
    # scores 0, -1, -6 (-20 per metre of dx): softmax over d = 0.05 leaves source 0 alone
    assert record["rule"] == "softmax" and record["scores"][:3] == [0.0, -1.0, -6.0]
    assert w[0] == pytest.approx(1.0) and w[1] == 0 and w[2] == 0
    record, w = mixing.decide(mixer(4, d=1.0), seqs, 0, 1, obs, None)
    e = np.exp([0.0, -1.0, -6.0])
    assert np.allclose(w[:3], e / e.sum(), atol=1e-6) and w[3] == 0
    record, w = mixing.decide(mixer(4, d=-0.3), seqs, 0, 1, obs, None)      # d <= 0: the best compatible source alone
    assert w.tolist() == [1.0, 0.0, 0.0, 0.0] and record["d"] == -0.3


def test_decide_uses_inhand_features_from_the_second_constraint():
    seqs = [[constraint(0.0), constraint(0.0, kind="object-object", obj="machine", carried="pod")] for _ in range(2)]
    obs = {"machine_pose": pose(0.0), "pod_pose": pose(0.0), "robot0_eef_pos": [0, 0, 1.0], "robot0_eef_quat_site": [0, 0, 0, 1.0]}
    obs["pod_pose"][2, 3] = 1.0
    record, _ = mixing.decide(mixer(2), seqs, 1, 0, obs, "pod")
    assert "inhand_features" in record and np.allclose(record["inhand_features"], 0, atol=1e-6)
    record, _ = mixing.decide(mixer(2), seqs, 1, 0, obs, None)               # nothing carried: in-hand features 0
    assert np.allclose(record["inhand_features"], 0)


def test_decide_randmix_draws_once_per_constraint_and_attempt():
    seqs = [[constraint(0.0)], [constraint(0.05)], [constraint(0.3)]]
    obs = {"pod_pose": pose(0.0), "robot0_eef_pos": [0, 0, 1.0], "robot0_eef_quat_site": [0, 0, 0, 1.0]}
    m = mixer(3, rule="randmix")
    first, w1 = mixing.decide(m, seqs, 0, 0, obs, None)
    _, w2 = mixing.decide(m, seqs, 0, 0, obs, None)                          # a second solve of the same constraint
    assert first["rule"] == "randmix" and np.array_equal(w1, w2) and w1.sum() == pytest.approx(1.0)
    m.begin_attempt()
    _, w3 = mixing.decide(m, seqs, 0, 0, obs, None)
    assert not np.array_equal(w1, w3)


def test_mixed_window_blends_on_the_weighted_timeline():
    a = np.stack([pose(0.0)] * 4)
    b = np.stack([pose(1.0)] * 8)
    win = mixing.MixedWindow({0: (a, np.full((4, 1), -1.0)), 2: (b, np.full((8, 1), -1.0))}, {0: 0.5, 2: 0.5}, anchor=2)
    assert win.n == 6 and win.pose(0)[0, 3] == pytest.approx(0.5)            # length 0.5 * 4 + 0.5 * 8
    assert win.anchor_index.tolist() == [0, 1, 2, 4, 5, 6]                  # the anchor's (2) steps on the blended window
    assert win.meta["sources"]["2"]["source_samples"] == 8
    lone = mixing.MixedWindow({2: (b, np.full((8, 1), -1.0))}, {2: 1.0}, anchor=2)
    assert lone.n == 8 and lone.anchor_index.tolist() == list(range(8))     # one source: its own window, its own speed
    assert np.allclose(lone.pose(7), b[7])
    with pytest.raises(IndexError):
        win.pose(6)


def test_window_on_resamples_the_per_step_data_and_keeps_the_rest():
    c = constraint(0.0, n=6, grip=[-1, -1, 1, 1, 1, 1])
    c.src_obj_transform = {"pod": len}                                      # a callable stays as it is
    c.symmetries, c.post_constraint_behavior = ["s"], ["lift"]
    w = mixing.window_on(c, np.array([0, 0, 1, 2, 3, 3, 4, 5, 5]))
    assert len(w.timesteps) == 9 and w.src_gripper_action.tolist() == [-1, -1, -1, 1, 1, 1, 1, 1, 1]
    assert w.src_obs["robot0_eef_pos"].shape == (9, 3) and w.src_obj_pose["pod"].shape == (9, 4, 4)
    assert w.src_obj_transform["pod"] is len and w.symmetries == ["s"] and w.post_constraint_behavior == ["lift"]
    assert len(c.timesteps) == 6 and c.src_gripper_action.shape == (6,)    # the source's own constraint is untouched
    w.src_gripper_action[0] = 5.0                                           # CP-Gen writes into it: a copy
    assert c.src_gripper_action[0] == -1
    with pytest.raises(ValueError):
        mixing.window_on(c, np.array([0, 6]))


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
    two = SimpleNamespace(type="obj0", transforms=[{"rot_axis": "z", "rot_param": np.pi}, {"rot_axis": "y", "rot_param": np.pi}])
    placement = SimpleNamespace(constraint_type="object-object", symmetries=[two])
    assert [label for label, _ in mixing.symmetry_candidates(FakeOfficial, placement)] == ["identity", "obj0:z180", "obj0:y180",
                                                                                          "obj0:z180*y180"]


def test_sources_held_from_the_other_side_are_aligned_to_the_primary_before_blending():
    # sources 0 and 2 hold the hammer at y=-0.06, source 1 (and 3, also turned about the approach axis) from the other
    # side, 180 degrees about the hammer z axis: the object symmetry maps them onto the primary's (0) side
    seqs = [[grasp(-0.06, 0.0)], [grasp(0.06, np.pi)], [grasp(-0.05, 0.05)], [grasp(0.06, 0.0)]]
    obs = {"hammer_pose": pose(0.1, 0.2, 0.3)}
    chosen, info = mixing.align_sources(FakeOfficial, POLICY, seqs, 0, 0, [1, 2, 3], obs, IDENTITY)
    assert info["1"]["symmetry"] == "obj0:z180" and info["2"]["symmetry"] == "identity"
    assert info["3"]["symmetry"] == "obj0:z180+eef:z180"
    assert info["1"]["pos_cm"] < 0.1 and info["1"]["rot_deg"] < 0.1
    assert info["1"]["primary_symmetry_pos_cm"] == pytest.approx(12.0, abs=0.01)
    assert info["1"]["primary_symmetry_rot_deg"] == pytest.approx(180.0, abs=0.1)
    chosen[0] = IDENTITY
    solved = mixing.solve_sources(FakeOfficial, POLICY, seqs, 0, [0, 1, 2, 3], obs, chosen)
    win = mixing.MixedWindow(solved, {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}, anchor=0)
    primary = FakeOfficial.compute_X_W_eef_des(seqs[0][0], obs, None, None, src_t_idx=0, **IDENTITY)
    dp, dr = mixing.pose_distance(win.pose(0), primary)
    assert dp < 0.01 and np.degrees(dr) < 2.0             # without alignment: the hammer's middle, 180 degrees off


def test_alignment_records_the_distance_to_the_primary():
    seqs = [[grasp(-0.06, 0.0)], [grasp(-0.06, 0.0, x=0.08)]]
    obs = {"hammer_pose": pose()}
    chosen, info = mixing.align_sources(FakeOfficial, POLICY, seqs, 0, 0, [1], obs, IDENTITY)
    assert info["1"]["symmetry"] == "identity" and info["1"]["pos_cm"] == pytest.approx(8.0, abs=0.01)
    chosen[0] = IDENTITY
    solved = mixing.solve_sources(FakeOfficial, POLICY, seqs, 0, [0, 1], obs, chosen)
    win = mixing.MixedWindow(solved, {0: 0.5, 1: 0.5}, anchor=0)
    assert win.pose(0)[0, 3] == pytest.approx(0.04)
