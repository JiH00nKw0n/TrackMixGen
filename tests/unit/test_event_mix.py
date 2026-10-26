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
