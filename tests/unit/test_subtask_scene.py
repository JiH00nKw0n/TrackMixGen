"""One typed scene (``SubtaskScene``) for every runtime, the mixer, the records and the fit: MimicGen's and CP-Gen's
runtimes fill it (the Isaac Lab runtimes: test_isaaclab_runtime.py, test_autodata_runtime.py), and what the record keeps
of it gives the fit the features the mixer scored."""
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tmg.generation.method.interfaces import HeldRule, HeldTest, MixContext, SubtaskScene
from tmg.generation.method.success_model import (
    FEATURES,
    HELD_FEATURES,
    current_in_hand,
    held_flags,
    in_hand,
    scene_of_record,
    source_features,
    source_in_hand,
)


def pose(x=0.0, y=0.0, z=0.8, yaw=0.0):
    p = np.eye(4)
    p[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    p[:3, 3] = [x, y, z]
    return p


def test_the_mix_context_carries_a_typed_scene_and_no_string_keyed_extras():
    names = set(MixContext.__dataclass_fields__)
    assert "scene" in names and "extra" not in names
    assert MixContext.__dataclass_fields__["scene"].type == "SubtaskScene | None"
    scene = SubtaskScene(subtask=0)
    assert scene.num_sources == 0 and scene.carried_object is None and scene.held is None and scene.held_rule is None


def test_the_mimicgen_runtime_fills_the_scene_from_the_builders_locals():
    from tmg.generation.generators.mimicgen.runtime import MimicGenRuntime
    from tmg.generation.method.interfaces import Method

    seen = []
    mixer = SimpleNamespace(begin_attempt=lambda: None,
                            mix=lambda ctx: seen.append(ctx) or SimpleNamespace(record={"mixed": False}))
    runtime = MimicGenRuntime(Method(name="m", mixer=mixer), 1, None)
    infos = [SimpleNamespace(eef_pose=np.stack([pose(0.1 * j, z=1.0 + 0.01 * t) for t in range(8)]),
                             target_pose=np.stack([pose(0.1 * j)] * 8), gripper_action=np.full((8, 1), -1.0),
                             object_poses={"nut": np.stack([pose(0.2 * j, yaw=0.1 * j)] * 8), "peg": np.stack([pose(0.5 + 0.1 * j)] * 8)})
             for j in range(3)]
    bounds = np.array([[[0, 4], [4, 8]]] * 3)
    now = SimpleNamespace(eef_pose=pose(0.03, z=1.0), object_poses={"nut": pose(0.04, yaw=0.2), "peg": pose(0.6)})
    runtime.begin_attempt(SimpleNamespace())
    for k, name in enumerate(("nut", "peg")):
        runtime.mix_segment(SimpleNamespace(src_dataset_infos=infos), {
            "src_subtask_eef_poses": infos[1].eef_pose[4 * k:4 * k + 4], "src_subtask_target_poses": infos[1].target_pose[4 * k:4 * k + 4],
            "src_subtask_gripper_actions": infos[1].gripper_action[4 * k:4 * k + 4], "subtask_ind": k, "subtask_object_name": name,
            "all_subtask_inds": bounds, "selected_src_demo_ind": 1, "cur_object_pose": now.object_poses[name], "cur_datagen_info": now})
    first, second = (ctx.scene for ctx in seen)
    assert isinstance(first, SubtaskScene) and (first.subtask, first.eef, first.reference_object) == (0, None, "nut")
    assert first.carried_object is None and first.held is None and first.held_rule is None and current_in_hand(first) is None
    np.testing.assert_array_equal(first.current_object_pose, now.object_poses["nut"])
    np.testing.assert_array_equal(first.source_object_poses[2], infos[2].object_poses["nut"][0])
    assert second.carried_object == "nut" and second.reference_object == "peg" and second.num_sources == 3
    np.testing.assert_array_equal(current_in_hand(second), in_hand(now.eef_pose, now.object_poses["nut"]))
    np.testing.assert_array_equal(source_in_hand(second, 2), in_hand(infos[2].eef_pose[4], infos[2].object_poses["nut"][4]))
    assert held_flags(second) == (True, [True, True, True])           # no hold test: the pose is known everywhere


def test_the_cpgen_runtime_fills_the_scene_from_the_constraints_and_leaves_unknown_poses_out():
    from tmg.generation.generators.cpgen import mixing

    def constraint(x, carried_known=True):
        src = {"tripod": np.stack([pose(x)] * 4)}
        if carried_known:
            src["needle"] = np.stack([pose(x, 0.01, 1.0)] * 4)
        return SimpleNamespace(obj_names=["needle", "tripod"], constraint_type="object-object", src_obj_pose=src,
                               timesteps=list(range(4)), src_gripper_action=np.ones(4),
                               src_obs={"robot0_eef_pos": np.tile([x, 0, 1.02], (4, 1)), "robot0_eef_quat_site": np.tile([0, 0, 0, 1.0], (4, 1))})

    constraints = [constraint(0.0), constraint(0.1, carried_known=False), constraint(0.2)]
    obs = {"tripod_pose": pose(0.05), "needle_pose": pose(0.0, 0.0, 1.0), "robot0_eef_pos": [0.0, 0.0, 1.03], "robot0_eef_quat_site": [0, 0, 0, 1.0]}
    scene = mixing.subtask_scene(constraints, 1, obs, "needle")
    assert isinstance(scene, SubtaskScene) and (scene.subtask, scene.eef, scene.reference_object, scene.carried_object) == (1, None, "tripod", "needle")
    assert held_flags(scene) == (True, [True, False, True]) and scene.held is None
    f = source_features(scene)
    assert f.shape == (3, 5) and np.allclose(f[:, 0], [0.05, 0.05, 0.15]) and f[1, 3:].tolist() == [0.0, 0.0] and f[0, 3] > 0
    unseen = mixing.subtask_scene(constraints, 1, {k: v for k, v in obs.items() if k != "needle_pose"}, "needle")
    assert held_flags(unseen) == (False, [False, False, False]) and np.all(source_features(unseen)[:, 3:] == 0)
    assert mixing.subtask_scene(constraints, 0, obs, None).carried_object is None
    with pytest.raises(KeyError, match="tripod_pose"):                # the runtime turns this into a recorded identity
        mixing.subtask_scene(constraints, 1, {}, "needle")


def test_what_a_scene_records_gives_the_fit_the_features_the_mixer_scored():
    """``SubtaskScene.record(selected)`` -> (json-like fields) -> ``scene_of_record`` -> the selected source's features,
    for the three hold situations: both hold, only the current hand, only the source."""
    rule = HeldRule(version=1, rigid_m=0.2, nodes_m=0.1)
    eef, src_eef = pose(0.0, z=1.0), [pose(0.1 * j, z=1.0) for j in range(3)]
    carried_now, carried_src = pose(0.02, z=0.95), [pose(0.1 * j + 0.03, z=0.95) for j in range(3)]
    for now_m, sources_m in ((0.05, [0.05, 0.05, 0.6]), (0.05, [0.6, 0.6, 0.6]), (0.7, [0.05, 0.6, 0.05])):
        test = HeldTest(rule=rule, deformable=False, now_m=now_m, sources_m=sources_m)
        scene = SubtaskScene(subtask=1, eef="right", reference_object="fridge", current_object_pose=pose(0.3, yaw=0.2),
                             source_object_poses=[pose(0.1 * j, yaw=0.1 * j) for j in range(3)], eef_pose=eef,
                             source_eef_poses=src_eef, carried_object="bottle", carried_pose=carried_now if test.now else None,
                             source_carried_poses=[p if h else None for p, h in zip(carried_src, test.sources)], held=test,
                             held_rule=rule)
        scored = source_features(scene, HELD_FEATURES)
        for j in range(3):
            record = {"subtask_ind": 1, "eef": "right", "object_name": "fridge", "src_demo_ind": j, **scene.record(j)}
            assert record["held"] == {"now": test.now, "source": test.sources[j], "deformable": False}
            assert record["held_rule"] == rule.as_dict() and record["carried_candidate"] == "bottle"
            assert record["carried_object"] == ("bottle" if test.now else None)
            assert sorted(record["objects"]) == (["bottle"] if test.now else []) and sorted(record["src_objects"]) == (["bottle"] if test.sources[j] else [])
            back = scene_of_record(record, "bottle")
            np.testing.assert_allclose(source_features(back, HELD_FEATURES)[0], scored[j], atol=1e-12)
            assert back.held.now == test.now and back.held.sources == [test.sources[j]] and back.held_rule == rule
        assert np.array_equal(source_features(scene, FEATURES), scored[:, :5])
