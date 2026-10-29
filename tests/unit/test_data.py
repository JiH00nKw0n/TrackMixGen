"""Dataset writer and frame loader on a synthetic two-chunk pool (no simulator)."""
import json

import h5py
import numpy as np
import pytest

from tmg.core.chunks import CHUNK_DONE, CHUNKS_DIR, POOL_DONE, write_json
from tmg.core.records import ATTEMPTS_JSONL, RECORDS_DIR
from tmg.data.dataset import DatasetError, candidates, select, write_dataset
from tmg.data.loader import FrameDataset, TrajectoryStore, compute_stats, padded_indices

IMAGE = {"name": "image", "cameras": ["agentview"], "robot_state": ["robot0_eef_pos", "robot0_gripper_qpos"], "object_state": []}
LOW_DIM = {"name": "low_dim", "cameras": [], "robot_state": ["robot0_eef_pos", "robot0_gripper_qpos"], "object_state": ["object"]}


def make_pool(root, *, chunks=2, attempts=4, h=8, w=8, seed=0):
    """A pool whose even attempts succeed; demo.hdf5 lists the successes in order, records list every attempt."""
    rng = np.random.default_rng(seed)
    pool = root / "pool"
    for c in range(chunks):
        chunk = pool / CHUNKS_DIR / f"seed{c + 1}"
        rec = chunk / RECORDS_DIR
        rec.mkdir(parents=True)
        exp = chunk / "Square_X_n4_seed1"
        exp.mkdir()
        lines = []
        with h5py.File(exp / "demo.hdf5", "w") as f:
            data = f.create_group("data")
            data.attrs["env_args"] = json.dumps({"env_name": "Square_X", "env_kwargs": {}})
            k = 0
            for a in range(attempts):
                n = 5 + a + 3 * c
                success = a % 2 == 0
                np.savez(rec / f"attempt_{a:06d}.npz", a_exec=rng.normal(size=(n, 7)).astype(np.float32),
                         a_label=rng.normal(size=(n, 7)).astype(np.float32), seg_of_step=np.repeat([0, 1], [n // 2, n - n // 2]),
                         seg_sub=np.array([0, 1]), seg_kind=np.array([0, 1]), seg_len=np.array([n // 2, n - n // 2]))
                lines.append(json.dumps({"attempt": a, "success": success, "n_steps": n, "src_demo_inds": [a % 3, (a + 1) % 3]}))
                if success:
                    g = data.create_group(f"demo_{k}")
                    k += 1
                    obs = g.create_group("obs")
                    obs.create_dataset("agentview_image", data=rng.integers(0, 255, size=(n, h, w, 3), dtype=np.uint8))
                    obs.create_dataset("robot0_eef_pos", data=rng.normal(size=(n, 3)))
                    obs.create_dataset("robot0_gripper_qpos", data=rng.normal(size=(n, 2)))
                    obs.create_dataset("object", data=rng.normal(size=(n, 14)))
                    g.create_dataset("actions", data=rng.normal(size=(n, 7)))
                    g.create_dataset("states", data=rng.normal(size=(n, 11)))
                    g.create_dataset("src_demo_inds", data=np.array([a % 3, (a + 1) % 3]))
        (rec / ATTEMPTS_JSONL).write_text("\n".join(lines) + "\n")
        write_json(chunk / CHUNK_DONE, {"ok": True})
    write_json(pool / POOL_DONE, {"attempts": chunks * attempts})
    return pool


def test_candidates_select_and_write(tmp_path):
    pool = make_pool(tmp_path)
    cands = candidates(pool)
    assert len(cands) == 4 and [c["attempt"] for c in cands] == [0, 2, 0, 2]
    chosen = select(cands, 3, seed=1)
    assert len(chosen) == 3 and len({(c["chunk"], c["demo_key"]) for c in chosen}) == 3
    with pytest.raises(DatasetError):
        select(cands, 5, seed=1)
    out = tmp_path / "ds.hdf5"
    stats = write_dataset(out, chosen, pool=pool, observation=IMAGE, meta={"label": "pre_noise", "task": "square"})
    assert stats["demos"] == 3 and stats["image_shape"] == [8, 8, 3] and out.exists() and not out.with_suffix(".hdf5.part").exists()
    with h5py.File(out, "r") as f:
        d = f["data"]["demo_0"]
        n = d.attrs["num_samples"]
        assert d["obs"]["agentview_image"].shape == (n, 8, 8, 3) and d["obs"]["robot_state"].shape == (n, 5)
        assert d["actions"].shape == (n, 7) and d["actions_pre_noise"].shape == (n, 7) and d["actions_executed"].shape == (n, 7)
        assert set(np.unique(d["subtask_of_step"][()])) == {0, 1}
        assert "object_state" not in d["obs"] and json.loads(f["data"].attrs["object_state_keys"]) == []
        assert json.loads(f["data"].attrs["cameras"]) == ["agentview"] and f["data"].attrs["label"] == "pre_noise"
        assert len(f["mask"]["train"]) == 3
        # the complete layout of the file (keys, dtypes, attributes), as the committed writer produced it
        assert sorted(f) == ["data", "mask"] and sorted(f["mask"]) == ["train"]
        assert f["mask"]["train"][()].tolist() == [b"demo_0", b"demo_1", b"demo_2"]
        assert sorted(f["data"]) == ["demo_0", "demo_1", "demo_2"]
        assert sorted(f["data"].attrs) == ["cameras", "env_args", "label", "object_state_keys", "robot_state_keys", "task", "total"]
        assert json.loads(f["data"].attrs["env_args"]) == {"env_name": "Square_X", "env_kwargs": {}}
        assert json.loads(f["data"].attrs["robot_state_keys"]) == ["robot0_eef_pos", "robot0_gripper_qpos"]
        assert f["data"].attrs["total"] == sum(f["data"][k].attrs["num_samples"] for k in f["data"]) == sum(c["n_steps"] for c in chosen)
        assert sorted(d) == ["actions", "actions_executed", "actions_pre_noise", "obs", "src_demo_inds", "states", "subtask_of_step"]
        assert sorted(d["obs"]) == ["agentview_image", "robot0_eef_pos", "robot0_gripper_qpos", "robot_state"]
        assert sorted(d.attrs) == ["attempt", "chunk", "num_samples", "pool"]
        assert d.attrs["chunk"] == chosen[0]["chunk"] and d.attrs["attempt"] == chosen[0]["attempt"] and d.attrs["pool"] == str(pool)
        dtypes = {k: str(d[k].dtype) for k in ("actions", "actions_executed", "actions_pre_noise", "states", "subtask_of_step", "src_demo_inds")}
        assert dtypes == {"actions": "float32", "actions_executed": "float32", "actions_pre_noise": "float32", "states": "float64",
                          "subtask_of_step": "int16", "src_demo_inds": "int64"}
        assert str(d["obs"]["agentview_image"].dtype) == "uint8" and d["obs"]["agentview_image"].compression == "gzip"
        assert d["obs"]["agentview_image"].chunks == (1, 8, 8, 3) and str(d["obs"]["robot_state"].dtype) == "float32"
        # the three action arrays come from the right sources: the demo file's actions, the record's a_exec and a_label
        z = np.load(pool / CHUNKS_DIR / chosen[0]["chunk"] / RECORDS_DIR / f"attempt_{chosen[0]['attempt']:06d}.npz")
        with h5py.File(chosen[0]["demo_file"], "r") as src:
            assert np.array_equal(d["actions"][()], np.asarray(src["data"][chosen[0]["demo_key"]]["actions"], np.float32))
            assert np.array_equal(d["states"][()], src["data"][chosen[0]["demo_key"]]["states"][()])
        assert np.array_equal(d["actions_executed"][()], z["a_exec"]) and np.array_equal(d["actions_pre_noise"][()], z["a_label"])
        assert d["src_demo_inds"][()].tolist() == chosen[0]["src_demo_inds"]
        assert d["subtask_of_step"][()].tolist() == [0] * (n // 2) + [1] * (n - n // 2)


def test_candidates_follow_the_numeric_demo_order(tmp_path):
    """Twelve successes in one chunk: demo_10 and demo_11 come after demo_9 (an alphabetical sort would pair demo_10
    with the third success and fail the step-count check)."""
    pool = make_pool(tmp_path, chunks=1, attempts=24)
    cands = candidates(pool)
    assert [c["demo_key"] for c in cands] == [f"demo_{k}" for k in range(12)]
    assert [c["attempt"] for c in cands] == list(range(0, 24, 2)) and [c["n_steps"] for c in cands] == [5 + a for a in range(0, 24, 2)]


def test_step_count_mismatch_is_an_error(tmp_path):
    pool = make_pool(tmp_path)
    rec = pool / CHUNKS_DIR / "seed1" / RECORDS_DIR / ATTEMPTS_JSONL
    lines = rec.read_text().splitlines()
    first = json.loads(lines[0])
    first["n_steps"] += 1
    rec.write_text("\n".join([json.dumps(first), *lines[1:]]) + "\n")
    with pytest.raises(DatasetError):
        candidates(pool)


def test_loader_stacks_with_padding_and_computes_stats(tmp_path):
    pool = make_pool(tmp_path)
    out = tmp_path / "ds.hdf5"
    write_dataset(out, select(candidates(pool), 4, seed=0), pool=pool, observation=IMAGE, meta={})
    store = TrajectoryStore.load(out)
    assert store.num_demos == 4 and store.state_dim == 5 and store.action_dim == 7 and store.image_shape == (8, 8, 3)
    assert store.env_state is None and store.env_state_dim == 0
    ds = FrameDataset(store, obs_delta=(-1, 0), action_delta=range(-1, 3))
    assert len(ds) == store.num_frames
    first = ds[0]
    assert first["observation.images.agentview"].shape == (2, 3, 8, 8) and first["observation.images.agentview"].dtype == np.uint8
    assert first["observation.state"].shape == (2, 5) and first["action"].shape == (4, 7)
    assert first["action_is_pad"].tolist() == [True, False, False, False]      # offset -1 falls before the demo
    assert np.array_equal(first["observation.state"][0], first["observation.state"][1])  # edge padding repeats step 0
    last = ds[int(store.lengths[0]) - 1]
    assert last["action_is_pad"].tolist() == [False, False, True, True]
    idx, pad = padded_indices(3, 5, (-4, 0, 1, 2))
    assert idx.tolist() == [0, 3, 4, 4] and pad.tolist() == [True, False, False, True]
    stats = compute_stats(store)
    assert stats["action"]["mean"].shape == (7,) and stats["observation.images.agentview"]["mean"].shape == (3, 1, 1)
    assert 0.0 <= float(stats["observation.images.agentview"]["mean"].max()) <= 1.0
    dropped = FrameDataset(store, drop_n_last_frames=2)
    assert len(dropped) == store.num_frames - 2 * store.num_demos
    batches = ds.batches(3, np.random.default_rng(0))
    b = next(batches)
    assert len(b) == 3 and len(set(b.tolist())) == 3


def test_loader_single_frame_items_and_task_strings(tmp_path):
    pool = make_pool(tmp_path)
    out = tmp_path / "ds.hdf5"
    write_dataset(out, select(candidates(pool), 2, seed=0), pool=pool, observation=IMAGE, meta={})
    store = TrajectoryStore.load(out)
    ds = FrameDataset(store, obs_delta=None, action_delta=range(50), task="insert the nut")
    item = ds[0]
    assert item["observation.images.agentview"].shape == (3, 8, 8) and item["observation.state"].shape == (5,)
    assert item["action"].shape == (50, 7) and item["action_is_pad"].sum() == 50 - int(store.lengths[0])
    assert item["task"] == "insert the nut"


def test_low_dim_dataset_has_object_state_and_no_images(tmp_path):
    pool = make_pool(tmp_path)
    out = tmp_path / "ds_lowdim.hdf5"
    stats = write_dataset(out, select(candidates(pool), 3, seed=0), pool=pool, observation=LOW_DIM, meta={})
    assert stats["image_shape"] is None
    with h5py.File(out, "r") as f:
        d = f["data"]["demo_0"]
        n = d.attrs["num_samples"]
        assert "agentview_image" not in d["obs"] and d["obs"]["object_state"].shape == (n, 14) and d["obs"]["robot_state"].shape == (n, 5)
        assert json.loads(f["data"].attrs["object_state_keys"]) == ["object"] and json.loads(f["data"].attrs["cameras"]) == []
    store = TrajectoryStore.load(out)
    assert store.cameras == [] and store.image_shape is None and store.env_state_dim == 14 and store.state_dim == 5
    item = FrameDataset(store, obs_delta=(-1, 0), action_delta=range(4))[0]
    assert item["observation.environment_state"].shape == (2, 14) and not any(k.startswith("observation.images.") for k in item)
    stats = compute_stats(store)
    assert stats["observation.environment_state"]["std"].shape == (14,) and "observation.images.agentview" not in stats
    # a dataset must carry something besides the robot state
    with pytest.raises(DatasetError):
        write_dataset(tmp_path / "bad.hdf5", select(candidates(pool), 1, seed=0), pool=pool,
                      observation={"name": "x", "cameras": [], "robot_state": ["robot0_eef_pos"], "object_state": []}, meta={})


def test_egl_device_appended_to_cuda_visible_devices():
    from tmg.core import egl

    env = {"MUJOCO_EGL_DEVICE_ID": "8", "CUDA_VISIBLE_DEVICES": "7"}
    assert egl.apply(env)["CUDA_VISIBLE_DEVICES"] == "7,8"
    assert egl.apply({"MUJOCO_EGL_DEVICE_ID": "8", "CUDA_VISIBLE_DEVICES": "7,8"})["CUDA_VISIBLE_DEVICES"] == "7,8"
    assert "CUDA_VISIBLE_DEVICES" not in egl.apply({"MUJOCO_EGL_DEVICE_ID": "8"})
    assert egl.apply({"CUDA_VISIBLE_DEVICES": "7"})["CUDA_VISIBLE_DEVICES"] == "7"


def test_demo_layout_warns_on_unknown_keys(capsys):
    from tmg.data.dataset import DemoLayout

    layout = DemoLayout.from_mapping({"demo_glob": "generated.hdf5", "image_key": "{cam}", "states_key": None,
                                      "labels_from_record": True})      # misspelt: the default stays and the user is told
    assert layout.labels_from_records is False and layout.states_key is None and layout.demo_glob == "generated.hdf5"
    assert "unknown keys ['labels_from_record']" in capsys.readouterr().out
    DemoLayout.from_mapping({"demo_glob": "*/demo.hdf5", "image_key": "{cam}_image"})
    assert capsys.readouterr().out == ""


def test_camera_obs_key_keeps_the_two_lookup_rules():
    from tmg.data.images import camera_obs_key, to_uint8_frame

    mimicgen = {"agentview_image": 1, "robot_state": 2}
    isaaclab = {"agentview_image": 1}
    arena = {"camera_obs/robot_pov_cam_rgb": 1}
    # a generator's file: the layout's pattern, else Arena's camera_obs group, else None
    assert camera_obs_key(mimicgen, "agentview", "{cam}_image") == "agentview_image"
    assert camera_obs_key(isaaclab, "agentview_image", "{cam}") == "agentview_image"
    assert camera_obs_key(arena, "robot_pov_cam_rgb", "{cam}") == "camera_obs/robot_pov_cam_rgb"
    assert camera_obs_key(mimicgen, "sideview", "{cam}_image") is None
    # the common hdf5: the camera's own name when present, else <cam>_image without checking
    assert camera_obs_key(isaaclab, "agentview_image") == "agentview_image"
    assert camera_obs_key(mimicgen, "agentview") == "agentview_image" and camera_obs_key({}, "x") == "x_image"
    u8 = np.array([[[0, 128, 255]]], np.uint8)
    assert to_uint8_frame(u8) is u8
    assert to_uint8_frame(u8.astype(np.float32) / 255.0).tolist() == [[[0, 128, 255]]]      # [0, 1] floats are scaled
    assert to_uint8_frame(u8.astype(np.float32)).tolist() == [[[0, 128, 255]]]              # already on the 0..255 scale


def test_write_dataset_reads_images_and_labels_from_the_records(tmp_path):
    """The Isaac Lab layouts: labels in labels_%06d.npz (pre-noise label when the pool says so), frames in images_%06d.npz."""
    from tmg.core.records import IMAGES_FORMAT, LABELS_FORMAT, record_file
    from tmg.data.dataset import DemoLayout

    pool = make_pool(tmp_path)
    layout = DemoLayout(demo_glob="*/demo.hdf5", image_key="{cam}", states_key=None, labels_from_records=True,
                        images_from_records=True)
    cands = candidates(pool, layout)
    chosen, without_records = cands[:2], cands[2:3]
    for c in chosen:
        records = pool / CHUNKS_DIR / c["chunk"] / RECORDS_DIR
        n = c["n_steps"]
        np.savez(record_file(records, LABELS_FORMAT, c["attempt"]), actions_executed=np.zeros((n, 7), np.float32),
                 actions_pre_noise=np.ones((n, 7), np.float32))
        np.savez(record_file(records, IMAGES_FORMAT, c["attempt"]), agentview=np.full((n, 8, 8, 3), 7, np.uint8))
    out = tmp_path / "ds_records.hdf5"
    write_dataset(out, chosen, pool=pool, observation=IMAGE, meta={"label": "pre_noise"}, layout=layout)
    with h5py.File(out, "r") as f:
        d = f["data"]["demo_0"]
        assert np.all(d["actions"][()] == 1.0) and np.all(d["actions_executed"][()] == 0.0) and "states" not in d
        assert d["obs"]["agentview"].shape[1:] == (8, 8, 3) and int(d["obs"]["agentview"][0, 0, 0, 0]) == 7
    with pytest.raises(DatasetError, match="no recorded images"):
        write_dataset(tmp_path / "bad.hdf5", without_records, pool=pool, observation=IMAGE, meta={}, layout=layout)
    # label "executed": ``actions`` stays the demonstration file's actions, the two record arrays are still stored
    write_dataset(tmp_path / "ds_exec.hdf5", chosen, pool=pool, observation=IMAGE, meta={"label": "executed"}, layout=layout)
    with h5py.File(tmp_path / "ds_exec.hdf5", "r") as f, h5py.File(chosen[0]["demo_file"], "r") as src:
        d = f["data"]["demo_0"]
        assert np.array_equal(d["actions"][()], np.asarray(src["data"][chosen[0]["demo_key"]]["actions"], np.float32))
        assert np.all(d["actions_executed"][()] == 0.0) and np.all(d["actions_pre_noise"][()] == 1.0)
