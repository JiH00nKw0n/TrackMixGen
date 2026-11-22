import numpy as np
import pytest

from tmg.generation.method.grippers import ClosureGripper, SignGripper, event_indices


def test_gripper_detectors():
    grips = -np.ones((10, 1))
    grips[6:] = 1.0
    assert event_indices(SignGripper(), grips, None) == [6]
    assert event_indices(SignGripper(), grips, True) == [0, 6]        # previous path ended closed
    hand = np.zeros((20, 4))
    hand[8:, :] = -1.0                                                  # flexion grows (negated closure)
    assert event_indices(ClosureGripper(columns=[0, 1, 2, 3]), hand, None) == [8]


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


def test_build_method_refuses_removed_blocks_and_unknown_mixers():
    from tmg.generation.method import build_method

    m = build_method({"name": "none", "mixing": {"enabled": False}, "tracking": {"enabled": False}}, chunk_seed=1)
    assert m.mixer is None and m.labels == "executed" and m.is_identity
    with pytest.raises(ValueError, match="tracking"):
        build_method({"name": "t", "mixing": {"enabled": False}, "tracking": {"enabled": True}}, chunk_seed=1)
    with pytest.raises(ValueError, match="sources"):
        build_method({"name": "t", "sources": {"type": "rank_based", "top_k": 1}}, chunk_seed=1)
    with pytest.raises(ValueError, match="unknown"):
        build_method({"name": "t", "mixing": {"enabled": True, "type": "nope"}}, chunk_seed=1)
