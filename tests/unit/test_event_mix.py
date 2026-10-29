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
    # no event on either side (both label lists empty): only the initial state tells them apart
    always_open, always_closed = -np.ones(10), np.ones(10)
    assert events(always_open)[2] == events(always_closed)[2] == []
    with pytest.raises(IncompatibleGripperEvents):
        mix_segments(pa, always_open, pb, always_closed, 0.8)
    assert anchor_rotation_degrees(pa, always_open, pb, always_closed) is None
    assert anchor_rotation_degrees(pa, always_open, pb, always_open) == [0.0, 0.0]        # start and end anchors only


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


def _hand_segment(n, close_at, dx, closed=-1.2, rest=0.3):
    """A GR1-like hand: 11 joint commands per step; the four proximal joints flex (negative, to ``closed``) when the
    hand closes and joint 10 holds the constant command ``rest``."""
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.linspace(0, dx, n)
    hand = np.zeros((n, 11))
    hand[close_at:, :4] = closed
    hand[:, 10] = rest
    return poses, hand


def test_guided_mixer_mixes_many_joint_hand_commands_with_the_closure_rule():
    """The mixer aligns Dex segments by hand-closure events and returns 11-D hand commands on the mixed timeline."""
    from tmg.generation.method.guided_mix import GuidedMixer
    from tmg.generation.method.interfaces import MixContext

    pa, ha = _hand_segment(10, 6, 0.1)
    pb, hb = _hand_segment(14, 9, 0.1, closed=-1.0, rest=0.5)       # the donor's joints differ, so the weights show
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
    # joint commands are the 0.8 / 0.2 blend of both sources at the same phase: 0.8 * -1.2 + 0.2 * -1.0 closed, 0 open;
    # the constant joint blends to 0.8 * 0.3 + 0.2 * 0.5. Nothing half-closed.
    assert np.allclose(out.gripper[:, 10], 0.34) and set(np.round(out.gripper[:, 0], 6)) <= {0.0, -1.16}
    assert np.allclose(out.gripper[closed, :4], -1.16) and np.allclose(out.gripper[~closed, :4], 0.0)
    # the scalar rule is unchanged: raw +-1 commands are blended as before
    ga = np.where(ha[:, 0] < 0, 1.0, -1.0); gb = np.where(hb[:, 0] < 0, 1.0, -1.0)
    ctx2 = MixContext(subtask_ind=0, primary=1, primary_eef_start=np.eye(4), primary_targets=pa, primary_gripper=ga,
                      src_object_poses=[np.eye(4), np.eye(4)], src_segments=[(np.eye(4), pb, gb), (np.eye(4), pa, ga)],
                      num_sources=2, has_reference_object=True, rng=rng)
    out2 = GuidedMixer(ranking=_ranking([0, 1]), weight_a=0.8, max_anchor_rotation_deg=20.0, chunk_seed=1, seed_offset=7, good_count=1).mix(ctx2)
    assert out2.record["mixed"] and out2.gripper.shape == (n,) and set(np.round(out2.gripper, 6)) <= {-1.0, 1.0}


def _rz(deg):
    t = np.deg2rad(deg)
    out = np.eye(4)
    out[:2, :2] = [[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]]
    return out


def _translate(x, y=0.0, z=0.0):
    out = np.eye(4)
    out[:3, 3] = [x, y, z]
    return out


def test_transform_all_applies_one_pose_on_the_left_of_every_pose():
    from tmg.core.poses import transform_all

    transform = _rz(90.0) @ _translate(0.0) ; transform[:3, 3] = [1.0, 2.0, 3.0]
    poses = np.stack([np.eye(4), _translate(1.0)])
    out = transform_all(transform, poses)
    assert out.shape == (2, 4, 4) and np.allclose(out[0], transform)
    assert np.allclose(out[1][:3, 3], [1.0, 3.0, 3.0])                 # +x of the pose turns into +y, then the offset
    assert np.allclose(out[1][:3, :3], transform[:3, :3])


def _mix_context(primary, obj_poses, segments, has_reference_object=True, subtask=0):
    from tmg.generation.method.interfaces import MixContext

    eef, targets, grip = segments[primary]
    return MixContext(subtask_ind=subtask, primary=primary, primary_eef_start=eef, primary_targets=targets, primary_gripper=grip,
                      src_object_poses=obj_poses, src_segments=segments, num_sources=len(segments), has_reference_object=has_reference_object)


def test_guided_mixer_identity_reasons_donor_filter_and_frames():
    """Sources 0 and 1 are the high-success half of the ranking (0, 1, 2, 3); 2 and 3 are the primaries that get mixed.

    Every source carries the same object-frame path (10 steps, close at 6); source 1's path is turned 10 degrees
    about z in its object frame and source 3's starts closed, so only sources 0 and 1 are compatible donors of
    source 2 and source 1 only within a 20 degree anchor bound. Object poses differ per source so the frame
    handling is visible: the mixed world targets must come back through the primary's object pose.
    """
    from tmg.generation.method.guided_mix import GuidedMixer

    path, grip = _segment(10, 6, dx=0.1)
    closed_grip = np.ones(10)
    obj = [_translate(0.5, 0.1, 0.8), _rz(45.0) @ _translate(0.2, -0.3, 0.8), _rz(90.0) @ _translate(0.0, 0.4, 0.8), _translate(1.0)]
    eef_b = _translate(1.0)                                           # donor 0 starts its segment 1 m along x in the object frame
    segments = [(obj[0] @ eef_b, np.einsum("ij,tjk->tik", obj[0], path), grip),
                (obj[1] @ np.eye(4), np.einsum("ij,tjk->tik", obj[1] @ _rz(10.0), path), grip),
                (obj[2] @ np.eye(4), np.einsum("ij,tjk->tik", obj[2], path), grip),
                (obj[3] @ np.eye(4), np.einsum("ij,tjk->tik", obj[3], path), closed_grip)]

    def make(**overrides):
        values = {"weight_a": 0.8, "max_anchor_rotation_deg": 20.0, "seed_offset": 7, "good_count": 2, **overrides}
        return GuidedMixer(ranking=_ranking([0, 1, 2, 3]), chunk_seed=11, **values)
    mixer = make()
    assert mixer.groups(0) == ({0, 1}, {2, 3})

    # identity reasons, in the order the pre-refactor code checked them
    out = mixer.mix(_mix_context(0, obj, segments))
    assert out.record["identity_reason"] == "primary is a high-success source" and out.record["mixed"] is False
    assert out.targets is segments[0][1] and out.gripper is segments[0][2] and out.eef_start is segments[0][0]
    assert out.record["call"] == 0 and out.record["source_a"] == 0 and out.record["good"] == [0, 1] and out.record["weight_a"] == 0.8
    assert mixer.mix(_mix_context(7, obj + [np.eye(4)] * 4, segments + [segments[2]] * 4)).record["identity_reason"] == "primary not ranked"
    assert mixer.mix(_mix_context(2, obj, segments, has_reference_object=False)).record["identity_reason"] == "subtask has no reference object"
    assert make(weight_a=1.0).mix(_mix_context(2, obj, segments)).record["identity_reason"] == "weight_a is 1"
    assert mixer.calls == 3

    # donor filter: source 1 exceeds a 5 degree anchor bound, source 3 is never a donor (not high-success)
    strict = make(max_anchor_rotation_deg=5.0).mix(_mix_context(2, obj, segments))
    assert strict.record["mixed"] is True and strict.record["eligible_donors"] == [0] and strict.record["source_b"] == 0
    assert np.allclose(strict.record["anchor_rotation_degrees"], [0.0, 0.0, 0.0])          # start, end, close event
    none_left = make(max_anchor_rotation_deg=5.0).mix(_mix_context(3, obj, segments))      # primary 3 starts closed
    assert none_left.record["identity_reason"] == "no compatible donor" and none_left.record["eligible_donors"] == []

    # with the 20 degree bound both donors qualify; the choice is the committed private RNG of (chunk seed, call, seed offset)
    out = mixer.mix(_mix_context(2, obj, segments))
    assert out.record["eligible_donors"] == [0, 1] and out.record["mixed"] is True and out.record["call"] == 3
    expected = int(np.random.default_rng(np.random.SeedSequence([11, 3, 7])).integers(2))
    assert out.record["source_b"] == expected
    angle = 0.0 if expected == 0 else 10.0
    assert np.allclose(out.record["anchor_rotation_degrees"], [angle] * 3, atol=1e-6)

    # frames: donor 0 has the primary's object-frame path, so the mixed world targets equal the primary's own
    out0 = make(max_anchor_rotation_deg=5.0).mix(_mix_context(2, obj, segments))
    assert np.allclose(out0.targets, segments[2][1]) and out0.gripper.shape == (10,) and set(out0.gripper) == {-1.0, 1.0}
    assert np.flatnonzero(out0.gripper > 0)[0] == 6 and out0.record["alignment"]["actual_event_steps"] == [6]
    # the first eef pose is the 0.8 / 0.2 blend of both object-frame starts (identity and 1 m along x), in the primary's frame
    assert np.allclose(out0.eef_start, obj[2] @ _translate(0.2))
