import types

import numpy as np
import pytest

from tmg.generation.simulators.isaaclab import (
    HandLayout,
    apply_scene,
    measured_eef_poses,
    pose_from_pos_quat,
    pose_range_of,
)

GR1_HAND_NAMES = [f"{h}_{j}_proximal" for j in ("pinky", "ring", "middle", "index", "thumb") for h in ("L", "R")] + \
                 [f"{h}_{j}_intermediate" for j in ("pinky", "ring", "middle", "index", "thumb") for h in ("L", "R")] + \
                 ["L_thumb_distal", "R_thumb_distal"]


def test_hand_layout_groups_interleaved_joints_by_prefix():
    layout = HandLayout(GR1_HAND_NAMES, arm_dim=14)
    assert layout.indices["left"] == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20] and len(layout.indices["right"]) == 11
    actions = np.arange(36.0)[None]
    split = layout.split(actions)
    assert split["left"][0, 0] == 14 and split["right"][0, 0] == 15      # the official 14:25 split would mix both hands
    assert np.array_equal(layout.assemble({"left": split["left"], "right": split["right"]}), actions[..., 14:])
    with pytest.raises(ValueError):
        HandLayout(["L_a", "X_b"], arm_dim=2)


def test_apply_scene_writes_pose_ranges():
    term = types.SimpleNamespace(params={"pose_range": {"x": (0.0, 0.1), "y": (0.0, 0.0)}})
    env_cfg = types.SimpleNamespace(events=types.SimpleNamespace(reset_object_position=term))
    scene = {"name": "ours", "override_bounds": True,
             "bounds": {"rope": {"reset_object_position": {"x": [-0.025, 0.125], "y": [-0.05, 0.05], "yaw": [-0.2618, 0.2618]}}}}
    applied = apply_scene(env_cfg, scene, "rope")
    assert term.params["pose_range"] == {"x": (-0.025, 0.125), "y": (-0.05, 0.05), "yaw": (-0.2618, 0.2618)}
    assert applied["reset_object_position"]["x"] == (-0.025, 0.125)
    assert apply_scene(env_cfg, {"name": "official", "override_bounds": False}, "rope") == {}
    with pytest.raises(KeyError):
        apply_scene(env_cfg, scene, "towel")
    with pytest.raises(ValueError):
        pose_range_of({"x": [1.0, 0.0]})


def test_measured_eef_poses_reads_the_configured_link():
    class Tensor(np.ndarray):
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.asarray(self)

    def t(a):
        return np.asarray(a, dtype=float).view(Tensor)

    data = types.SimpleNamespace(body_names=["base", "left_hand_roll_link", "left_hand_pitch_link"],
                                 body_pos_w=t([[[0, 0, 0], [1.0, 2.0, 3.0], [1.1, 2.0, 3.0]]]),
                                 body_quat_w=t([[[1, 0, 0, 0], [1, 0, 0, 0], [0.70710678, 0, 0, 0.70710678]]]))
    class Scene:
        env_origins = t([[1.0, 0.0, 0.0]])

        def __getitem__(self, key):
            return {"robot": types.SimpleNamespace(data=data)}[key]

    env = types.SimpleNamespace(scene=Scene())
    poses = measured_eef_poses(env, 0, {"left": "left_hand_pitch_link"})
    assert np.allclose(poses["left"][:3, 3], [0.1, 2.0, 3.0])
    assert np.allclose(poses["left"][:3, :3], pose_from_pos_quat([0, 0, 0], [0.70710678, 0, 0, 0.70710678])[:3, :3])
    with pytest.raises(KeyError):
        measured_eef_poses(env, 0, {"left": "missing_link"})
