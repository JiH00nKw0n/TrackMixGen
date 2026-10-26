import pytest

from tmg.core.actions import split_by_prefix

GR1_HAND_JOINTS = ["L_index_proximal_joint", "L_middle_proximal_joint", "L_pinky_proximal_joint", "L_ring_proximal_joint",
                   "L_thumb_proximal_yaw_joint", "R_index_proximal_joint", "R_middle_proximal_joint",
                   "R_pinky_proximal_joint", "R_ring_proximal_joint", "R_thumb_proximal_yaw_joint",
                   "L_index_intermediate_joint", "L_middle_intermediate_joint", "L_pinky_intermediate_joint",
                   "L_ring_intermediate_joint", "L_thumb_proximal_pitch_joint", "R_index_intermediate_joint",
                   "R_middle_intermediate_joint", "R_pinky_intermediate_joint", "R_ring_intermediate_joint",
                   "R_thumb_proximal_pitch_joint", "L_thumb_distal_joint", "R_thumb_distal_joint"]


def test_gr1_hands_are_interleaved_so_a_contiguous_split_is_wrong():
    groups = split_by_prefix(GR1_HAND_JOINTS, {"left": "L_", "right": "R_"})
    assert groups["left"] == [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 20]
    assert groups["right"] == [5, 6, 7, 8, 9, 15, 16, 17, 18, 19, 21]
    assert groups["left"] != list(range(11))      # the official mimic env's actions[:, 14:25] "left" split


def test_unmatched_joint_is_an_error():
    with pytest.raises(ValueError):
        split_by_prefix(["L_a", "X_b"], {"left": "L_", "right": "R_"})
