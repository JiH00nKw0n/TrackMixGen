"""Isaac Lab evaluation units that run without Isaac Sim: seed-fixed scenes, observation conversion, task configs."""
import json

import numpy as np

from tmg.config import compose, resolved
from tmg.evaluation.isaaclab_rollout import observation_input
from tmg.evaluation.resets import (
    is_isaaclab,
    isaaclab_scenes,
    make_resets,
    read_resets,
    resets_path,
    scene_seed,
)


def _plain(tmp_path, monkeypatch, *overrides):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    return resolved(compose("resets", list(overrides)))


def test_scene_seeds_are_fixed_by_the_evaluation_seed_and_index():
    assert scene_seed(7, 0) == scene_seed(7, 0) and scene_seed(7, 0) != scene_seed(7, 1) and scene_seed(7, 3) != scene_seed(8, 3)
    scenes = isaaclab_scenes(202600, 25, 300)
    assert len(scenes) == 325 and [s["split"] for s in scenes[:25]] == ["val"] * 25 and scenes[25]["split"] == "test"
    assert len({s["seed"] for s in scenes}) == 325
    # frozen values of evaluation.seed 202600: a change here would change every evaluation's scenes
    assert scenes[0]["seed"] == scene_seed(202600, 0) == 3775830342 and scenes[25]["seed"] == 2788069247


def test_isaaclab_resets_write_a_seed_file_that_every_run_reproduces(tmp_path, monkeypatch):
    plain = _plain(tmp_path, monkeypatch, "generator=softmimicgen", "task=rope", "scene=ours")
    assert is_isaaclab(plain) and resets_path(plain).suffix == ".json"
    out = make_resets(compose("resets", ["generator=softmimicgen", "task=rope", "scene=ours"]))
    path = resets_path(plain)
    assert out["path"] == str(path) and path.exists() and out["n_val"] == 25 and out["n_test"] == 300
    meta = json.loads(path.read_text())
    assert meta["env_name"] == "Isaac-Rope-Franka-IK-Rel-Mimic-v0" and meta["scenes"] == isaaclab_scenes(202600, 25, 300)
    assert meta["seed"] == 202600
    val = read_resets(path, "val")
    assert len(val) == 25 and val[0] == {"index": 0, "split": "val", "seed": scene_seed(202600, 0)}
    assert len(read_resets(path)) == 325
    again = make_resets(compose("resets", ["generator=softmimicgen", "task=rope", "scene=ours"]))   # kept, not redrawn
    assert again["path"] == out["path"] and json.loads(path.read_text()) == meta


def test_dex_resets_name_the_camera_environment(tmp_path, monkeypatch):
    plain = _plain(tmp_path, monkeypatch, "generator=dexmimicgen", "task=pick_place", "scene=ours")
    make_resets(compose("resets", ["generator=dexmimicgen", "task=pick_place", "scene=ours"]))
    assert json.loads(resets_path(plain).read_text())["env_name"] == "TMG-PickPlace-GR1T2-Images-Mimic-v0"


def test_robosuite_resets_path_keeps_the_hdf5_layout(tmp_path, monkeypatch):
    plain = _plain(tmp_path, monkeypatch, "generator=mimicgen", "task=square", "scene=ours")
    assert not is_isaaclab(plain) and resets_path(plain).suffix == ".hdf5"


def test_observation_input_converts_isaaclab_policy_terms():
    class T:   # stands in for a torch tensor with a leading batch of one
        def __init__(self, a):
            self.a = np.asarray(a)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    img = (np.random.default_rng(1).random((1, 128, 128, 3)) * 255).astype(np.uint8)
    obs = {"agentview_image": T(img), "robot0_eye_in_hand_image": T(img[:, ::-1]), "robot0_eef_pos": T([[1.0, 2.0, 3.0]]),
           "robot0_eef_quat": T([[1.0, 0.0, 0.0, 0.0]]), "robot0_gripper_qpos": T([[0.04, 0.04]]), "unused": T([[9.0]])}
    observation = {"cameras": ["agentview_image", "robot0_eye_in_hand_image"], "image_size": [128, 128],
                   "robot_state": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"], "object_state": []}
    out = observation_input(obs, observation)
    assert out["observation.images.agentview_image"].shape == (3, 128, 128) and out["observation.images.agentview_image"].dtype == np.uint8
    assert np.array_equal(out["observation.images.agentview_image"], img[0].transpose(2, 0, 1))
    assert np.allclose(out["observation.state"], [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0, 0.04, 0.04]) and out["observation.state"].dtype == np.float32
    assert "observation.environment_state" not in out


def test_isaaclab_task_configs_compose_with_the_pool_isaac_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    for generator, task, env in (("softmimicgen", "rope", "Isaac-Rope-Franka-IK-Rel-Mimic-v0"),
                                 ("softmimicgen", "towel", "Isaac-Fold-Towel-Franka-IK-Rel-Mimic-v0"),
                                 ("softmimicgen", "jenga", "Isaac-Jenga-Franka-IK-Rel-Mimic-v0"),
                                 ("dexmimicgen", "pick_place", "Isaac-PickPlace-GR1T2-Abs-Mimic-v0")):
        cfg = resolved(compose("generate", [f"generator={generator}", f"task={task}", "scene=ours", "method=none",
                                            "generation=pool_isaac", "cluster=ada"]))
        assert cfg["task"]["env_name"] == env and cfg["generation"]["name"] == "pool_isaac"
        assert cfg["generation"]["attempts"] == 1000 and cfg["generation"]["chunk"] == 250 and cfg["generation"]["seed_offset"] == 30000
        assert cfg["cluster"]["device"] == "cuda:0" and cfg["cluster"]["runtime"] == "docker"
        assert generator == "dexmimicgen" or cfg["task"]["success_term"]["first_call"] == "after_first_action"
        assert all(k.endswith("_image") or k in ("front_image", "side_image") for k in cfg["task"]["cameras"])
    soft = resolved(compose("dataset", ["generator=softmimicgen", "task=towel", "scene=ours", "method=none", "generation=pool_isaac", "observation=image"]))
    assert soft["observation"]["robot_state"] == ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"]
    assert soft["observation"]["cameras"] == ["agentview_image", "robot0_eye_in_hand_image"]


def test_trajectory_store_reads_isaaclab_camera_keys(tmp_path):
    """Isaac Lab datasets store a camera under its own observation name (agentview_image), MimicGen under <cam>_image."""
    import h5py

    from tmg.data.loader import TrajectoryStore

    path = tmp_path / "d.hdf5"
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["cameras"] = json.dumps(["agentview_image"])
        data.attrs["robot_state_keys"] = json.dumps(["robot0_eef_pos"])
        data.attrs["object_state_keys"] = json.dumps([])
        g = data.create_group("demo_0")
        g.create_dataset("obs/agentview_image", data=np.zeros((4, 8, 8, 3), np.uint8))
        g.create_dataset("obs/robot_state", data=np.zeros((4, 3), np.float32))
        g.create_dataset("actions", data=np.zeros((4, 7), np.float32))
    store = TrajectoryStore.load(path)
    assert store.cameras == ["agentview_image"] and store.images["agentview_image"][0].shape == (4, 8, 8, 3)


def test_task_policy_overrides_follow_the_official_configs(tmp_path, monkeypatch):
    def task_policy_overrides(task, name):   # mirrors tmg.policy.lerobot_policies.task_policy_overrides (that module needs lerobot)
        return dict(((task or {}).get("policy_overrides") or {}).get(name) or {})

    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    rope = resolved(compose("train", ["generator=softmimicgen", "task=rope", "scene=ours", "method=none", "generation=pool_isaac",
                                     "observation=image", "policy=diffusion_policy"]))
    assert rope["task"]["horizon"] == 400
    assert task_policy_overrides(rope["task"], "diffusion_policy") == {"crop_shape": None}
    assert task_policy_overrides(rope["task"], "bc_rnn") == {"crop_shape": None}
    dex = resolved(compose("train", ["generator=dexmimicgen", "task=pick_place", "scene=ours", "method=none", "generation=pool_isaac",
                                    "observation=image", "policy=diffusion_policy"]))
    assert dex["task"]["horizon"] == 600 and task_policy_overrides(dex["task"], "diffusion_policy") == {"crop_shape": None}
    square = resolved(compose("train", ["generator=mimicgen", "task=square", "scene=ours", "method=none", "generation=pool_robosuite_image",
                                       "observation=image", "policy=diffusion_policy"]))
    assert task_policy_overrides(square["task"], "diffusion_policy") == {} and square["policy"]["params"]["crop_shape"] == [76, 76]
