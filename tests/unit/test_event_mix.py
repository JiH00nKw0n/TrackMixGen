import numpy as np
import pytest

from tmg.generation.method.event_mix import (
    IncompatibleGripperEvents,
    anchor_rotation_degrees,
    blend_poses,
    events,
    mix_segments,
)


def _segment(n, close_at, dx=0.1):
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.linspace(0, dx, n)
    grip = -np.ones(n)
    grip[close_at:] = 1.0
    return poses, grip


def test_events_detect_close():
    _, grip = _segment(10, 6)
    states, idx, labels = events(grip)
    assert idx.tolist() == [6] and labels == ["close"] and not states[0]


def test_identity_weights_return_inputs():
    pa, ga = _segment(10, 6)
    pb, gb = _segment(14, 9)
    for w, src in [(1.0, pa), (0.0, pb)]:
        mixed, grip, meta = mix_segments(pa, ga, pb, gb, w)
        np.testing.assert_array_equal(mixed, src)
        assert meta["identity_source"] == ("A" if w == 1 else "B")


def test_mix_aligns_events_with_weighted_time():
    pa, ga = _segment(10, 6)
    pb, gb = _segment(14, 9)
    mixed, grip, meta = mix_segments(pa, ga, pb, gb, 0.8)
    assert meta["actual_event_steps"] == [round(0.8 * 6 + 0.2 * 9)]
    assert len(mixed) == round(0.8 * 10 + 0.2 * 14)
    # gripper never interpolates across the event
    assert set(np.round(grip, 6)) <= {-1.0, 1.0}
    # the last sample sits one warped step before each source's end (times run 0 .. n-1), so the mixed end
    # lies within one source step of the weighted blend of the two ends
    blend_end = 0.8 * pa[-1, 0, 3] + 0.2 * pb[-1, 0, 3]
    assert abs(mixed[-1, 0, 3] - blend_end) < max(np.diff(pa[:, 0, 3]).max(), np.diff(pb[:, 0, 3]).max())
    # the first sample is the blend of both starts
    assert mixed[0, 0, 3] == pytest.approx(0.8 * pa[0, 0, 3] + 0.2 * pb[0, 0, 3])


def test_incompatible_events_raise():
    pa, ga = _segment(10, 6)
    pb, gb = _segment(10, 6)
    gb[:] = 1.0  # starts closed
    with pytest.raises(IncompatibleGripperEvents):
        mix_segments(pa, ga, pb, gb, 0.8)
    assert anchor_rotation_degrees(pa, ga, pb, gb) is None


def test_blend_poses_rotation_fraction():
    a = np.tile(np.eye(4), (1, 1, 1))
    b = a.copy()
    theta = np.deg2rad(40)
    b[0, :3, :3] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
    out = blend_poses(a, b, 0.75)
    angle = np.degrees(np.arctan2(out[0, 1, 0], out[0, 0, 0]))
    assert angle == pytest.approx(10.0, abs=1e-6)


def _ranking(order):
    from tmg.generation.method.ranking import RankedSources, SourceRanking

    return SourceRanking(name="x/y", generator="x", task="y", basis="test", default=RankedSources(order=tuple(order)), subtasks={}, path="")


def _hand_segment(n, close_at, dx):
    """A GR1-like hand: 11 joint commands per step; the four proximal joints flex (negative) when the hand closes."""
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.linspace(0, dx, n)
    hand = np.zeros((n, 11))
    hand[close_at:, :4] = -1.2
    hand[:, 10] = 0.3                                   # an unrelated joint with a constant command
    return poses, hand


def test_guided_mixer_mixes_many_joint_hand_commands_with_the_closure_rule():
    """The mixer aligns Dex segments by hand-closure events and returns 11-D hand commands on the mixed timeline."""
    from tmg.generation.method.guided_mix import GuidedMixer
    from tmg.generation.method.interfaces import MixContext

    pa, ha = _hand_segment(10, 6, 0.1)
    pb, hb = _hand_segment(14, 9, 0.1)
    rng = np.random.default_rng(0)
    ctx = MixContext(subtask_ind=0, primary=1, primary_eef_start=np.eye(4), primary_targets=pa, primary_gripper=ha,
                     src_object_poses=[np.eye(4), np.eye(4)], src_segments=[(np.eye(4), pb, hb), (np.eye(4), pa, ha)],
                     num_sources=2, has_reference_object=True, rng=rng)
    mixer = GuidedMixer(ranking=_ranking([0, 1]), weight_a=0.8, max_anchor_rotation_deg=20.0, chunk_seed=1, seed_offset=7,
                        good_count=1, gripper_events={"type": "closure", "columns": [0, 1, 2, 3]})
    out = mixer.mix(ctx)
    assert out.record["mixed"] and out.record["source_b"] == 0 and mixer.describe()["gripper_events"] == "closure"
    n = round(0.8 * 10 + 0.2 * 14)
    assert out.targets.shape == (n, 4, 4) and out.gripper.shape == (n, 11)
    closed = out.gripper[:, :4].mean(axis=1) < -0.5
    assert not closed[0] and closed[-1] and int(np.flatnonzero(closed)[0]) == round(0.8 * 6 + 0.2 * 9)   # one close event at the blended step
    assert np.allclose(out.gripper[:, 10], 0.3) and set(np.round(out.gripper[:, 0], 6)) <= {0.0, -1.2}  # same phase in both sources: no half-closed hands
    # the scalar rule is unchanged: raw +-1 commands are blended as before
    ga = np.where(ha[:, 0] < 0, 1.0, -1.0); gb = np.where(hb[:, 0] < 0, 1.0, -1.0)
    ctx2 = MixContext(subtask_ind=0, primary=1, primary_eef_start=np.eye(4), primary_targets=pa, primary_gripper=ga,
                      src_object_poses=[np.eye(4), np.eye(4)], src_segments=[(np.eye(4), pb, gb), (np.eye(4), pa, ga)],
                      num_sources=2, has_reference_object=True, rng=rng)
    out2 = GuidedMixer(ranking=_ranking([0, 1]), weight_a=0.8, max_anchor_rotation_deg=20.0, chunk_seed=1, seed_offset=7, good_count=1).mix(ctx2)
    assert out2.record["mixed"] and out2.gripper.shape == (n,) and set(np.round(out2.gripper, 6)) <= {-1.0, 1.0}
