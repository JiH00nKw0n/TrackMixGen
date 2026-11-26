"""The nodal frame of a deformable object: deterministic, equivariant under table-plane motions, explicit failures."""
from types import SimpleNamespace

import numpy as np
import pytest

from tmg.generation.method.soft_frame import FRAME_VERSION, nodal_frame, reference_hash, reference_nodes


def yaw_matrix(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rope(n=40, bend=0.0, seed=0):
    """A rope-like cloud along x with a bend and a little noise (fixed seed)."""
    t = np.linspace(-0.2, 0.2, n)
    rng = np.random.default_rng(seed)
    return np.stack([t, bend * t ** 2, np.full(n, 0.02)], axis=1) + 1e-4 * rng.standard_normal((n, 3))


def test_the_reference_cloud_has_the_identity_rotation_at_its_centroid():
    ref = rope(bend=2.0)
    frame = nodal_frame(ref, ref)
    assert np.allclose(frame[:3, :3], np.eye(3)) and np.allclose(frame[:3, 3], ref.mean(axis=0))
    assert frame[3].tolist() == [0.0, 0.0, 0.0, 1.0] and FRAME_VERSION == "nodal_yaw_v1"


def test_the_frame_is_deterministic():
    ref, nodes = rope(bend=2.0), rope(bend=1.5, seed=3) @ yaw_matrix(0.7).T + [0.1, -0.05, 0.0]
    assert np.array_equal(nodal_frame(nodes, ref), nodal_frame(nodes.copy(), ref.copy()))
    assert reference_hash(ref) == reference_hash(ref.copy()) and reference_hash(ref) != reference_hash(ref + 1e-9)


@pytest.mark.parametrize("angle", [0.3, -1.2, 2.9, np.pi])
def test_the_frame_is_equivariant_under_yaw_and_shift(angle):
    """Moving the nodes by a yaw R and a shift t moves the frame by the same R and t (also a half turn, where a principal
    axis would have flipped its sign arbitrarily)."""
    ref, nodes = rope(bend=2.0), rope(bend=1.0, seed=1) @ yaw_matrix(0.4).T
    shift = np.array([0.3, -0.2, 0.01])
    moved = nodes @ yaw_matrix(angle).T + shift
    motion = np.eye(4)
    motion[:3, :3], motion[:3, 3] = yaw_matrix(angle), shift
    assert np.allclose(nodal_frame(moved, ref), motion @ nodal_frame(nodes, ref), atol=1e-9)


def test_a_rotated_copy_of_the_reference_recovers_the_rotation():
    ref = rope(bend=3.0)
    frame = nodal_frame(ref @ yaw_matrix(1.1).T, ref)
    assert np.allclose(frame[:3, :3], yaw_matrix(1.1), atol=1e-9)


def test_undefined_or_mismatched_clouds_raise():
    ref = rope()
    with pytest.raises(ValueError, match="same nodes"):
        nodal_frame(ref[:-1], ref)
    vertical = np.zeros((5, 3))
    vertical[:, 2] = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match="no spread"):
        nodal_frame(vertical, ref[:5])
    # two halves turned by +-70 degrees (r = cos 70 = 0.34 < 0.5): refused; by +-50 degrees (r = 0.64): a yaw of 0
    # (alternate nodes, so both groups have the same spread and the centroid stays put)
    centred = ref - ref.mean(axis=0)
    for angle, refused in ((np.radians(70), True), (np.radians(50), False)):
        split = centred.copy()
        split[0::2] = centred[0::2] @ yaw_matrix(angle).T
        split[1::2] = centred[1::2] @ yaw_matrix(-angle).T
        if refused:
            with pytest.raises(ValueError, match="rotation-fit correlation"):
                nodal_frame(split, centred)
        else:
            assert np.allclose(nodal_frame(split, centred)[:3, :3], np.eye(3), atol=1e-3)   # alternate groups nearly balance
    bad = ref.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        nodal_frame(bad, ref)


def test_the_reference_is_source_zero_at_its_first_step():
    infos = [SimpleNamespace(object_nodal_positions={"rope": np.stack([rope(seed=s), rope(seed=s + 1)])}) for s in (0, 5)]
    assert np.array_equal(reference_nodes(infos, "rope"), rope(seed=0))


def test_a_folded_towel_keeps_a_yaw_and_a_straight_rope_bent_into_a_u_too():
    g = np.linspace(-0.15, 0.15, 12)
    x, y = np.meshgrid(g, g)
    flat = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], axis=1)
    folded = flat.copy()
    folded[folded[:, 0] > 0, 0] *= -1                                  # folded in half onto the other side
    assert np.allclose(nodal_frame(folded @ yaw_matrix(0.5).T, flat)[:3, :3], yaw_matrix(0.5), atol=1e-6)
    t = np.linspace(-np.pi / 2, np.pi / 2, 60)
    straight = np.stack([np.linspace(-0.2, 0.2, 60), np.zeros(60), np.zeros(60)], axis=1)
    u = np.stack([0.4 / np.pi * np.cos(t), 0.4 / np.pi * np.sin(t), np.zeros(60)], axis=1)
    nodal_frame(u, straight)                                           # r = 0.91: accepted
