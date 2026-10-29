import numpy as np
import pytest

from tmg.generation.method.grippers import ClosureGripper, SignGripper, event_indices
from tmg.generation.method.tracking import (
    EndEffectorPath,
    ErrorAdaptiveFollower,
    advance_inverse,
    clamp_advance,
    region_table,
)

CFG = {"ds_min": 0.25, "regions": {"default": {"alpha_p": 12.0, "alpha_r": 1.2}, "approach": {"alpha_p": 50.0, "alpha_r": 5.0}},
       "approach_window": 10, "extra_steps_cap": "segment"}


def _path(n=12, step=0.01, close_at=8):
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * step
    grips = -np.ones((n, 1))
    grips[close_at:] = 1.0
    return poses, grips


def _follower(cfg=CFG, n=12, close_at=8, kinds=None, previous=None):
    poses, grips = _path(n=n, close_at=close_at)
    f = ErrorAdaptiveFollower.factory(cfg)()
    f.start_path([EndEffectorPath("arm", poses, grips)], [0.05] * n, kinds or [1] * n, previous)
    return f, poses


def test_advance_rule():
    assert advance_inverse(0.0, 0.0, 12.0, 1.2, 0.25) == 1.0
    assert advance_inverse(10.0, 0.0, 12.0, 1.2, 0.25) == 0.25
    assert clamp_advance(5.0, 1.0, [7], 11.0) == 6.0
    assert clamp_advance(6.5, 1.0, [7], 11.0) == 7.0
    assert clamp_advance(10.5, 1.0, [], 11.0) == 11.0


def test_region_table_marks_approach_window():
    r = region_table(12, [8], 3)
    assert r[4] == "rest" and r[5:9] == ["approach"] * 4 and r[9] == "rest"


def test_gripper_detectors():
    _, grips = _path(10, close_at=6)
    assert event_indices(SignGripper(), grips, None) == [6]
    assert event_indices(SignGripper(), grips, True) == [0, 6]        # previous path ended closed
    hand = np.zeros((20, 4))
    hand[8:, :] = -1.0                                                  # flexion grows (negated closure)
    assert event_indices(ClosureGripper(columns=[0, 1, 2, 3]), hand, None) == [8]


def test_zero_error_is_identity():
    f, poses = _follower()
    steps = []
    while not f.finished():
        s_before = f.s
        measured = {"arm": poses[0] if s_before is None else f.targets(s_before)["arm"]}
        _, _, _, rec = f.next_target(measured)
        steps.append(rec)
    assert len(steps) == 12
    assert [r["k"] for r in steps] == list(range(12))
    assert [r["new_slots"] for r in steps] == [[k] for k in range(12)]
    assert all(r["ds_actual"] == (0.0 if i == 0 else 1.0) for i, r in enumerate(steps))


def test_lag_slows_and_cap_restores_base_speed():
    f, _ = _follower(n=6)
    far = np.eye(4)
    far[0, 3] = -1.0
    count, slots = 0, []
    while not f.finished():
        _, _, _, rec = f.next_target({"arm": far})
        slots += rec["new_slots"]
        count += 1
    assert 6 < count <= 12
    assert slots == list(range(6))     # every original slot exactly once


def test_clamp_records_actual_advance_and_bridge_region():
    cfg = dict(CFG, approach_window=2, regions=dict(CFG["regions"], bridge={"alpha_p": 0.0, "alpha_r": 0.0}))
    f, poses = _follower(cfg, kinds=[0] * 5 + [1] * 7, previous={"arm": False})
    assert f.region[:5] == ["bridge"] * 5 and f.region[6:9] == ["approach"] * 3 and f.region[5] == "rest"
    behind = np.eye(4)
    behind[0, 3] = -0.5
    f.next_target({"arm": poses[0]})
    for _ in range(4):
        f.next_target({"arm": behind})
    assert f.s == 4.0
    _, _, _, rec = f.next_target({"arm": behind})
    assert rec["ds_actual"] == 1.0 and rec["s_after"] == 5.0
    _, _, _, rec = f.next_target({"arm": behind})
    assert rec["ds_proposed"] == 0.25 and rec["s_after"] == 5.25 and rec["new_slots"] == []


def test_two_end_effectors_share_one_path_variable():
    """A bimanual path: the lagging hand governs the advance and both hands reach the event together."""
    n = 12
    left_poses, left_grips = _path(n=n, close_at=9)          # left closes at 9
    right_poses, right_grips = _path(n=n, close_at=5)        # right opens... (sign flips at 5)
    f = ErrorAdaptiveFollower.factory(CFG)()
    f.start_path([EndEffectorPath("left", left_poses, left_grips), EndEffectorPath("right", right_poses, right_grips)],
                 [0.05] * n, [1] * n, None)
    assert f.events == [5, 9]                                # union of both hands' events
    behind = np.eye(4)
    behind[0, 3] = -0.5
    f.next_target({"left": left_poses[0], "right": right_poses[0]})
    # right hand tracks perfectly, left hand lags: the shared s still slows down for the left hand
    _, _, _, rec = f.next_target({"left": behind, "right": f.targets(0.0)["right"]})
    assert rec["e_pos"] == pytest.approx(0.5) and rec["ds_proposed"] == 0.25
    targets, grippers, _, _ = f.next_target({"left": behind, "right": f.targets(f.s)["right"]})
    assert set(targets) == {"left", "right"} and set(grippers) == {"left", "right"}
    # both hands are commanded from the same waypoint index
    k = int(np.floor(f.s + 1e-9))
    assert grippers["left"][0] == left_grips[k, 0] and grippers["right"][0] == right_grips[k, 0]
    assert f.final_gripper_states() == {"left": True, "right": True}


def test_build_method_picks_implementations_by_type():
    import pytest

    from tmg.generation.method import build_method
    from tmg.generation.method.tracking import ErrorAdaptiveFollower

    tracking = {"enabled": True, "type": "error_adaptive", "regions": {"default": {"alpha_p": 12.0, "alpha_r": 1.2}}}
    m = build_method({"name": "t", "mixing": {"enabled": False}, "tracking": tracking}, chunk_seed=1)
    assert isinstance(m.follower_factory(), ErrorAdaptiveFollower) and m.mixer is None
    with pytest.raises(ValueError):
        build_method({"name": "t", "mixing": {"enabled": False}, "tracking": {**tracking, "type": "nope"}}, chunk_seed=1)
    with pytest.raises(ValueError):
        build_method({"name": "t", "mixing": {"enabled": True, "type": "nope", "ranking": "mimicgen/square"}, "tracking": {"enabled": False}}, chunk_seed=1)


def test_gripper_state_helpers_live_in_grippers_and_are_re_exported():
    from tmg.generation.method import event_mix
    from tmg.generation.method.grippers import (
        CLOSE_THRESHOLD,
        OPEN_THRESHOLD,
        IncompatibleGripperEvents,
        build_detector,
        change_indices,
        gripper_states,
    )

    states = gripper_states([-1.0, 0.0, 1.0, 1.0, 0.0, -1.0])
    assert states.tolist() == [False, False, True, True, True, False]            # zeros carry the state
    assert change_indices(states).tolist() == [2, 5]
    assert change_indices([True, True]).tolist() == []
    with pytest.raises(IncompatibleGripperEvents):
        gripper_states([0.0, 1.0])
    assert event_mix.gripper_states is gripper_states and event_mix.IncompatibleGripperEvents is IncompatibleGripperEvents
    closure = build_detector({"type": "closure", "columns": [0]})
    assert (closure.close_threshold, closure.open_threshold) == (CLOSE_THRESHOLD, OPEN_THRESHOLD) == (0.65, 0.35)


def test_guided_mix_config_matches_the_single_value_constructor():
    from tmg.generation.method.guided_mix import GuidedMixConfig, GuidedMixer
    from tmg.generation.method.ranking import RankedSources, SourceRanking

    ranking = SourceRanking(name="x/y", generator="x", task="y", basis="t", default=RankedSources(order=(0, 1, 2)), subtasks={}, path="")
    cfg = GuidedMixConfig.from_mapping({"weight_a": 0.7, "good_count": 1, "gripper_events": {"type": "sign"}})
    assert cfg == GuidedMixConfig(weight_a=0.7, max_anchor_rotation_deg=20.0, seed_offset=817263, good_fraction=0.5,
                                  good_count=1, gripper_events={"type": "sign"})
    by_config = GuidedMixer(ranking=ranking, chunk_seed=3, config=cfg)
    by_values = GuidedMixer(ranking=ranking, chunk_seed=3, weight_a=0.7, max_anchor_rotation_deg=20.0, seed_offset=817263,
                            good_count=1, gripper_events={"type": "sign"})
    assert by_config.describe() == by_values.describe() and by_config.groups(0) == ({0}, {1, 2})
    with pytest.raises(TypeError):
        GuidedMixer(ranking=ranking, chunk_seed=3, config=cfg, weight_a=0.5)
