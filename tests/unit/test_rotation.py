"""6D rotation encoding of absolute pose actions (CP-Gen): round trip, continuity across pi, the normalization that
leaves the rotation and gripper unchanged, and (with torch and LeRobot) a short training run whose checkpoint the
policy server answers in the environment's 7 numbers."""
import json

import h5py
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tmg.policy import rotation


def actions(n: int, rng) -> np.ndarray:
    pos = rng.uniform([-0.25, -0.25, 0.83], [0.28, 0.28, 1.17], size=(n, 3))
    rotvec = Rotation.random(n, random_state=1).as_rotvec()
    grip = rng.choice([-1.0, 1.0], size=(n, 1))
    return np.concatenate([pos, rotvec, grip], axis=1).astype(np.float32)


def test_round_trip_keeps_position_gripper_and_orientation():
    a = actions(500, np.random.default_rng(0))
    a[0, 3:6] = [0, 0, np.pi - 1e-4]                              # angles at pi, where axis-angle is ambiguous
    a[1, 3:6] = [0, np.pi, 0]
    enc = rotation.encode(a)
    assert enc.shape == (500, 10)
    dec = rotation.decode(enc)
    assert np.allclose(dec[:, :3], a[:, :3], atol=1e-6) and np.array_equal(dec[:, 6:], a[:, 6:])
    same = Rotation.from_rotvec(dec[:, 3:6]).as_matrix(), Rotation.from_rotvec(a[:, 3:6]).as_matrix()
    assert np.allclose(*same, atol=1e-5)


def test_the_encoding_is_continuous_where_axis_angle_jumps():
    # the same orientation step written across pi: the axis-angle vector flips, the 6D rows barely move
    before = np.array([[0, 0, 0, 0, 0, np.pi - 0.01, 1.0]], np.float32)
    after = np.array([[0, 0, 0, 0, 0, -(np.pi - 0.01), 1.0]], np.float32)   # angle pi + 0.01 about +z
    assert np.abs(after[0, 3:6] - before[0, 3:6]).max() > 6
    assert np.abs(rotation.encode(after) - rotation.encode(before)).max() < 0.05


def test_decode_orthonormalizes_a_noisy_prediction():
    enc = rotation.encode(actions(50, np.random.default_rng(2)))
    noisy = enc + np.random.default_rng(3).normal(0, 0.02, enc.shape).astype(np.float32)
    dec = rotation.decode(noisy)
    mats = Rotation.from_rotvec(dec[:, 3:6]).as_matrix()
    assert np.allclose(np.einsum("nij,nkj->nik", mats, mats), np.eye(3), atol=1e-6)


def test_identity_dims_normalize_only_the_position():
    torch = pytest.importorskip("torch")
    from tmg.core.features import ACTION
    from tmg.policy.normalize import Normalizer

    enc = rotation.encode(actions(200, np.random.default_rng(4)))
    stats = {ACTION: {"min": enc.min(0), "max": enc.max(0), "mean": enc.mean(0), "std": enc.std(0)}}
    rotation.identity_dims(stats, ACTION)
    for mode in ("min_max", "mean_std"):
        out = Normalizer(stats, {"ACTION": mode})({ACTION: torch.as_tensor(enc)})[ACTION].numpy()
        assert np.allclose(out[:, 3:], enc[:, 3:], atol=1e-5)
    out = Normalizer(stats, {"ACTION": "min_max"})({ACTION: torch.as_tensor(enc)})[ACTION].numpy()
    assert out[:, :3].min() >= -1 - 1e-5 and out[:, :3].max() <= 1 + 1e-5 and out[:, 2].max() > 0.99


def test_save_and_load(tmp_path):
    assert rotation.load(tmp_path) == "axis_angle"
    rotation.save(tmp_path, "axis_angle")
    assert not (tmp_path / rotation.SAVED).exists()
    rotation.save(tmp_path, "rotation_6d")
    assert rotation.load(tmp_path) == "rotation_6d"
    with pytest.raises(ValueError):
        rotation.check("quaternion")


def write_cpgen_like_dataset(path, rng):
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["cameras"] = json.dumps(["agentview"])
        for j in range(3):
            g = data.create_group(f"demo_{j}")
            T = 12
            g.create_dataset("obs/agentview_image", data=rng.integers(0, 255, (T, 32, 32, 3), dtype=np.uint8))
            g.create_dataset("obs/robot_state", data=rng.normal(size=(T, 9)).astype(np.float32))
            g.create_dataset("actions", data=actions(T, rng))


def test_training_with_rotation_6d_saves_it_and_the_server_answers_7_numbers(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("lerobot")
    from tmg.config import compose
    from tmg.core.features import STATE, image_feature
    from tmg.policy.server import PolicyRunner
    from tmg.policy.trainer import list_checkpoints, train

    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    rng = np.random.default_rng(5)
    ds = tmp_path / "d.hdf5"
    write_cpgen_like_dataset(ds, rng)
    cfg = compose("train", ["generator=cpgen", "task=coffee", "scene=official_wide", "policy=bc_rnn", "observation=image",
                            f"dataset.path={ds}", f"output={tmp_path / 'run'}", "updates=2", "device=cpu",
                            "training.updates_per_epoch=1", "training.checkpoint_every_epochs=2",
                            "policy.params.crop_shape=null", "policy.params.rnn_hidden_dim=8", "policy.params.seq_length=3",
                            "training.by_policy.bc_rnn.image.batch_size=2"])
    train(cfg)
    ckpt = list_checkpoints(tmp_path / "run")[-1]
    assert rotation.load(ckpt) == "rotation_6d"
    saved = json.loads((ckpt / "stats.json").read_text())
    assert saved["modes"]["ACTION"] == "min_max"
    assert saved["stats"]["action"]["min"][3:] == [-1.0] * 7 and saved["stats"]["action"]["max"][3:] == [1.0] * 7
    runner = PolicyRunner(ckpt, "cpu")
    assert runner.info()["action_dim"] == 7
    runner.reset(seed=0)
    out = runner.act({image_feature("agentview"): np.zeros((3, 32, 32), np.uint8), STATE: np.zeros(9, np.float32)})
    assert out.shape == (7,) and np.isfinite(out).all()
