"""CP-Gen source mixing without CP-Gen installed: the donor time warp and blend (``MixPlan``), the mix context built
from constraints, and the grafted official ``CPPolicy.solve_constraint`` (the fixture text) run with fake kinematics,
checking that every solved step is the blend of the two sources' solutions for the current scene. Expected values
are worked out in the comments."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from tmg.generation.generators.cpgen import mixing
from tmg.generation.generators.cpgen import runtime as rt
from tmg.generation.graft import source_hash
from tmg.generation.method.geometric_mix import GeometricMixConfig, JunctionMixer, TransformMixer
from tmg.generation.method.interfaces import Method
from tmg.generation.method.ranking import RankedSources, SourceRanking

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = Path(__file__).parent / "fixtures" / "cpgen_solve_constraint.py"


def load_fixture():
    spec = importlib.util.spec_from_file_location("cpgen_solve_fixture", FIXTURE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    t = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    t[:2, :2] = [[c, -s], [s, c]]
    t[:3, 3] = (x, y, z)
    return t


def grip(n_open, n_closed):
    return np.r_[-np.ones(n_open), np.ones(n_closed)][:, None]


def test_fixture_hash_is_the_pinned_one():
    pinned = yaml.safe_load((ROOT / "configs/experiment/generator/cpgen.yaml").read_text())["graft_hashes"]["solve_constraint"]
    assert source_hash(load_fixture().CPPolicy.solve_constraint) == pinned


def test_mix_plan_warps_the_donor_onto_the_primary_events_and_blends():
    # primary: 4 open + 4 closed steps (event at 4); donor: 2 open + 6 closed (event at 2), donor x = 0.1 u.
    # On the primary's timeline the donor step is u = t/2 before the event and u = 2 + (t - 4) * 6/4 after it.
    donor = np.stack([pose(x=0.1 * u) for u in range(8)])
    plan = mixing.MixPlan(grip(4, 4), donor, grip(2, 6), 0.25)
    u = np.r_[np.arange(4) / 2, 2 + (np.arange(4, 8) - 4) * 1.5]
    assert np.allclose(plan.donor[:, 0, 3], 0.1 * u)
    assert plan.meta["source_a_event_steps"] == [4] and plan.meta["source_b_event_steps"] == [2] and len(plan) == 8
    primary = pose(x=1.0, yaw=0.4)
    mixed = plan.pose(6, primary)                              # w = 0.25: x = 0.25 * 1 + 0.75 * 0.5, yaw = 0.25 * 0.4
    assert mixed[0, 3] == pytest.approx(0.25 + 0.75 * 0.5) and np.allclose(mixed[:3, :3], pose(yaw=0.1)[:3, :3])
    with pytest.raises(IndexError):
        plan.pose(8, primary)


def test_mix_plan_takes_a_per_step_schedule_and_rejects_bad_weights():
    donor = np.stack([pose(x=1.0)] * 6)
    head = JunctionMixer(ranking=None, chunk_seed=0, config=GeometricMixConfig(head_steps=3))._schedule(0.4)
    plan = mixing.MixPlan(grip(3, 3), donor, grip(3, 3), head)
    assert np.allclose(plan.weights, [0.4, 0.6, 0.8, 1, 1, 1])          # 1 - 0.6 * max(0, 1 - t/3)
    assert np.allclose([plan.pose(t, pose())[0, 3] for t in range(6)], 1 - plan.weights)
    with pytest.raises(ValueError):
        mixing.MixPlan(grip(3, 3), donor, grip(3, 3), 1.5)


def constraint(index, n_open, n_closed, obj_x, kind="robot-object"):
    """Source ``index``'s constraint: the hand starts 0.1 above the object at its first step and moves 0.01 in y per
    step, with ``n_open`` open then ``n_closed`` closed commands. ``fake_des[t]`` is what the fake
    ``compute_X_W_eef_des`` returns for step t (x = 0.1 * index, y = 0.01 t, z = 0.5)."""
    n = n_open + n_closed
    obj = pose(x=obj_x)
    eef = np.stack([obj @ pose(y=0.01 * u, z=0.1) for u in range(n)])
    names = ["pod", "machine"]
    return SimpleNamespace(
        index=index, obj_names=names, constraint_type=kind, timesteps=list(range(100, 100 + n)), name=f"c{index}",
        src_obs={"robot0_eef_pos": eef[:, :3, 3], "robot0_eef_quat_site": np.tile([0, 0, 0, 1.0], (n, 1)),
                 "robot0_gripper_qpos": np.zeros((n, 2))},
        src_obj_pose={"pod": np.stack([obj] * n), "machine": np.stack([pose(x=obj_x + 1.0)] * n)},
        src_gripper_action=grip(n_open, n_closed), src_obj_geoms_size={"pod": [{}] * n}, src_action=np.zeros((n, 7)),
        keypoints_obj_frame={"pod": [np.zeros((2, 3))] * n}, keypoints_robot_link_frame={},
        post_constraint_behavior=[], duplicate_actions=False, attached_object_names=[],
        fake_des=np.stack([pose(x=0.1 * index, y=0.01 * t, z=0.5) for t in range(n)]))


def obs_now():
    return {"robot0_eef_pos": np.array([0.0, 0.0, 0.1]), "robot0_eef_quat_site": np.array([0, 0, 0, 1.0]),
            "pod_pose": pose(), "machine_pose": pose(x=1.0), "pod_geoms_size": {}, "robot_q": np.zeros(9)}


def test_source_context_reads_the_reference_object_window_and_current_scene():
    fixture = load_fixture()
    seqs = [[constraint(i, 3, 2, 0.1 * i)] for i in range(3)]
    ctx = mixing.source_context(seqs, 0, 2, obs_now(), fixture.quat2mat)
    assert ctx.primary == 2 and ctx.num_sources == 3 and ctx.has_reference_object
    assert np.allclose([o[0, 3] for o in ctx.src_object_poses], [0, 0.1, 0.2])           # the pod (robot-object)
    assert ctx.primary_targets.shape == (5, 4, 4) and np.allclose(ctx.primary_targets[4, :3, 3], [0.2, 0.04, 0.1])
    assert np.allclose(ctx.extra["current_object_pose"], pose()) and np.allclose(ctx.extra["junction_pose"], pose(z=0.1))
    placing = [[constraint(i, 3, 2, 0.1 * i, kind="object-object")] for i in range(2)]
    ctx = mixing.source_context(placing, 0, 0, obs_now(), fixture.quat2mat)
    assert np.allclose([o[0, 3] for o in ctx.src_object_poses], [1.0, 1.1])              # the machine (object-object)
    assert np.all(ctx.primary_gripper == 1) and np.allclose(ctx.extra["current_object_pose"], pose(x=1.0))


class FakeConfiguration:
    """Kinematics stand-in: ``update(q)`` with q = [x, y, z, qw, qx, qy, qz, g1, g2] puts the grip site there."""

    def __init__(self):
        self.q = np.zeros(9)
        self.robot_idxs = np.arange(9)

    def update(self, q=None, model=None):
        if q is not None:
            self.q = np.asarray(q, dtype=float).copy()

    def set_keypoints(self, *args, **kwargs):
        pass

    def get_transform_frame_to_world(self, name, kind):
        from scipy.spatial.transform import Rotation

        out = np.eye(4)
        out[:3, 3] = self.q[:3]
        out[:3, :3] = Rotation.from_quat(np.roll(self.q[3:7], -1)).as_matrix()
        return SimpleNamespace(as_matrix=lambda: out)


def ranking():
    return SourceRanking(name="cpgen/t", generator="cpgen", task="t", basis="", default=None,
                         subtasks={0: RankedSources(order=(0, 1, 2))}, path="")


def solve_with(mixer, primary=0):
    """Run the grafted official solve_constraint for constraint 0 of source ``primary`` with fake kinematics; sources 0,
    1, 2 have their pod at x = 0.3, 0.15, 0.2 while the current pod is at x = 0, so their transform sizes are 0.3, 0.15
    and 0.2. Returns (planning result, runtime)."""
    fixture = load_fixture()
    calls = []

    def fake_des(constraint, obs, robot_configuration, eef_configuration, X_obj0_transf=None, X_obj1_transf=None,
                 X_eef_transf=None, src_t_idx=0, **kwargs):
        calls.append((constraint.index, src_t_idx))
        return constraint.fake_des[src_t_idx]

    def fake_optimize(robot_configuration, eef_configuration, **kwargs):
        return eef_configuration.q.copy()           # the joint solution reaching the desired pose exactly

    fixture.compute_X_W_eef_des = fake_des
    fixture.find_best_transform = lambda c, o, r, e: (np.eye(4), np.eye(4), np.eye(4))
    fixture.optimize_robot_configuration = fake_optimize

    class Policy(fixture.CPPolicy):
        def _update_robot_configuration(self, *args, **kwargs):
            pass

    policy = Policy()
    policy.constraint_sequences = [[constraint(0, 4, 4, 0.3)], [constraint(1, 2, 4, 0.15)], [constraint(2, 3, 3, 0.2)]]
    policy.current_constraint_sequence_idx, policy.current_constraint_idx = primary, 0
    policy.robot_configuration, policy.eef_configuration = FakeConfiguration(), FakeConfiguration()
    policy.viz_robot_kpts, policy.verbose, policy.ignore_scaling, policy.max_iter, policy.env = False, False, False, 100, None
    policy.eef_interp_mink_motion_planer = None
    runtime = rt.CPGenRuntime(Method(name="mx", mixer=mixer, follower_factory=None), chunk_seed=1, records_dir=None,
                              all_sources=True)
    runtime.official = fixture
    graft = rt.build_mix_graft(fixture.CPPolicy.solve_constraint,
                               yaml.safe_load((ROOT / "configs/experiment/generator/cpgen.yaml").read_text())["graft_hashes"]["solve_constraint"])
    solve = graft.compile({"_tmg": runtime})
    result = solve(policy, policy.constraint_sequences[primary][0], obs_now())
    return result, runtime, calls


def test_grafted_solve_constraint_blends_both_sources_solved_for_the_current_scene():
    mixer = TransformMixer(ranking=ranking(), chunk_seed=1, config=GeometricMixConfig(max_anchor_rotation_deg=90.0))
    result, runtime, calls = solve_with(mixer)
    # trigger: the median of (0.3, 0.15, 0.2) is 0.2 < 0.3; donors with a smaller transform: 1 (0.15) and 2 (0.2), so 1;
    # the primary keeps w = 0.15 / (0.3 + 0.15) = 1/3
    record = runtime._last_mix
    assert record["mixed"] and record["source_b"] == 1 and record["transform"]["weight_a"] == pytest.approx(1 / 3, abs=1e-4)
    assert calls[:6] == [(1, u) for u in range(6)] and calls[6:] == [(0, t) for t in range(8)]   # donor first, then official
    # donor step on the primary's timeline: u = t/2 before the primary's event at 4, u = 2 + (t - 4) after it
    u = np.r_[np.arange(4) / 2, 2 + np.arange(4)]
    actions = np.stack(result.actions)
    w = 1 / 3
    assert result.success and actions.shape == (8, 7)
    assert np.allclose(actions[:, 0], (1 - w) * 0.1) and np.allclose(actions[:, 2], 0.5)
    assert np.allclose(actions[:, 1], w * 0.01 * np.arange(8) + (1 - w) * 0.01 * u)
    assert np.allclose(actions[:, 6], grip(4, 4)[:, 0])                     # the primary's gripper commands
    assert np.allclose(np.stack(result.robot_configurations)[0, :3], actions[0, :3])   # cuRobo's goal is the mixed start


def test_grafted_solve_constraint_is_official_when_the_mixer_keeps_the_primary():
    mixer = TransformMixer(ranking=ranking(), chunk_seed=1, config=GeometricMixConfig(max_anchor_rotation_deg=90.0))
    result, runtime, calls = solve_with(mixer, primary=1)                  # 0.15 is below the median: not mixed
    assert not runtime._last_mix["mixed"] and "not above the trigger quantile" in runtime._last_mix["identity_reason"]
    assert calls == [(1, t) for t in range(6)]
    assert np.allclose(np.stack(result.actions)[:, :3], load_fixture_des(1))


def load_fixture_des(index):
    n = {0: 8, 1: 6, 2: 6}[index]
    return np.stack([[0.1 * index, 0.01 * t, 0.5] for t in range(n)])


def test_mixing_needs_every_source():
    with pytest.raises(ValueError, match="selection=ours"):
        rt.CPGenRuntime(Method(name="mx", mixer=object(), follower_factory=None), chunk_seed=1, records_dir=None)
