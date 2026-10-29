import pathlib
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
    assert apply_scene(env_cfg, scene, "towel") == {}            # a task the scene lists no bounds for keeps its official ranges
    with pytest.raises(KeyError):
        apply_scene(env_cfg, {**scene, "bounds": {"rope": {"no_such_event": {"x": [0, 1]}}}}, "rope")
    with pytest.raises(ValueError):
        pose_range_of({"x": [1.0, 0.0]})


def test_apply_scene_nested_parameters_and_added_events(monkeypatch):
    import sys
    from tmg.generation.simulators.isaaclab import ranges_of, validate_isaaclab_scene

    term = types.SimpleNamespace(params={"pose_range": {"x": (0.0, 0.1)}, "partial_pose_range": {"x": (0.0, 0.1), "yaw": (-0.5, 0.5)}})
    env_cfg = types.SimpleNamespace(events=types.SimpleNamespace(reset_object_position=term))
    rope = {"reset_object_position": {"pose_range": {"x": [-0.05, 0.15], "y": [-0.1, 0.1]}, "partial_pose_range": {"yaw": [-1.0, 1.0]}}}
    applied = apply_scene(env_cfg, {"name": "ours", "override_bounds": True, "bounds": {"rope": rope}}, "rope")
    assert term.params["pose_range"] == {"x": (-0.05, 0.15), "y": (-0.1, 0.1)}
    assert term.params["partial_pose_range"] == {"x": (0.0, 0.1), "yaw": (-1.0, 1.0)}      # the other axes stay official
    assert set(applied["reset_object_position"]) == {"pose_range", "partial_pose_range"}
    assert ranges_of({"x": [0, 1]}) == {"pose_range": {"x": (0.0, 1.0)}}

    # added events: a fake isaaclab.managers and a target function resolved by "module:attribute"
    calls = []
    managers = types.ModuleType("isaaclab.managers")
    managers.EventTermCfg = lambda **kw: types.SimpleNamespace(**kw)
    managers.SceneEntityCfg = lambda name: ("entity", name)
    isaaclab = types.ModuleType("isaaclab"); isaaclab.managers = managers
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab); monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    target = types.ModuleType("tmg_fake_events"); target.reset_pose = lambda *a, **k: calls.append(a)
    monkeypatch.setitem(sys.modules, "tmg_fake_events", target)
    scene = {"name": "ours", "override_bounds": True, "bounds": {"microwave": {
        "tmg_microwave_pose": {"add": True, "func": "tmg_fake_events:reset_pose", "asset": "microwave",
                               "pose_range": {"x": [-0.05, 0.05]}, "params": {"velocity_range": {}}}}}}
    validate_isaaclab_scene(scene)
    env_cfg = types.SimpleNamespace(events=types.SimpleNamespace(existing=term))
    applied = apply_scene(env_cfg, scene, "microwave")
    added = env_cfg.events.tmg_microwave_pose
    assert added.func is target.reset_pose and added.mode == "reset"
    assert added.params == {"asset_cfg": ("entity", "microwave"), "velocity_range": {}, "pose_range": {"x": (-0.05, 0.05)}}
    assert applied["tmg_microwave_pose"]["added"] == "tmg_fake_events:reset_pose"
    assert list(vars(env_cfg.events)) == ["existing", "tmg_microwave_pose"]                 # appended after the official events
    with pytest.raises(KeyError):
        apply_scene(env_cfg, scene, "microwave")                                              # already added
    with pytest.raises(ValueError):
        validate_isaaclab_scene({"bounds": {"microwave": {"e": {"add": True, "func": "nomodule", "asset": "m"}}}})


def test_shipped_isaaclab_scene_files_validate():
    import yaml
    from tmg.generation.simulators.isaaclab import validate_isaaclab_scene

    root = pathlib.Path(__file__).resolve().parents[2] / "configs" / "experiment" / "generator"
    for gen in ("softmimicgen", "dexmimicgen"):
        for f in (root / gen / "scene").glob("*.yaml"):
            validate_isaaclab_scene(yaml.safe_load(f.read_text()))
    ours = yaml.safe_load((root / "softmimicgen" / "scene" / "ours.yaml").read_text())
    assert ours["override_bounds"] is True and set(ours["bounds"]) == {"rope", "towel"}
    assert set(ours["bounds"]["rope"]["reset_object_position"]) == {"pose_range", "partial_pose_range"}
    auto = yaml.safe_load((root / "dexmimicgen" / "scene" / "ours.yaml").read_text())
    assert all(e["add"] for e in auto["bounds"]["microwave"].values())


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


def test_pickplace_camera_quaternion_is_xyzw_and_looks_at_the_target():
    """Isaac Lab 3.0 wants (x, y, z, w); the camera's forward axis (ROS: +z) must point from the eye to the target."""
    from scipy.spatial.transform import Rotation

    from tmg.generation.simulators.isaaclab_tasks.gr1_pickplace_cameras import FRONT, SIDE, look_at_quat_xyzw

    for eye, target in (FRONT, SIDE):
        q = look_at_quat_xyzw(eye, target)
        forward = Rotation.from_quat(q).apply([0.0, 0.0, 1.0])          # scipy takes (x, y, z, w)
        want = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
        want /= np.linalg.norm(want)
        assert np.allclose(forward, want, atol=1e-6) and abs(np.linalg.norm(q) - 1.0) < 1e-9


def test_isolate_controller_scratch_gives_each_pink_controller_its_own_folder(tmp_path):
    from tmg.generation.simulators.isaaclab import isolate_controller_scratch

    pink = types.SimpleNamespace(controller=types.SimpleNamespace(urdf_output_dir="/tmp/urdf", usd_path="robot.usd"))
    plain = types.SimpleNamespace(controller=types.SimpleNamespace(gains=1.0))       # no URDF conversion
    env_cfg = types.SimpleNamespace(actions=types.SimpleNamespace(upper_body_ik=pink, gripper=plain, arm=types.SimpleNamespace()))
    moved = isolate_controller_scratch(env_cfg, tmp_path / "chunk")
    assert moved == {"upper_body_ik": str(tmp_path / "chunk" / "urdf" / "upper_body_ik")}
    assert pink.controller.urdf_output_dir == str(tmp_path / "chunk" / "urdf" / "upper_body_ik") and (tmp_path / "chunk" / "urdf" / "upper_body_ik").is_dir()
    assert plain.controller.gains == 1.0 and isolate_controller_scratch(types.SimpleNamespace(), tmp_path) == {}
