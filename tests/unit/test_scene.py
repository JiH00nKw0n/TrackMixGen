import pytest

from tmg.generation.simulators.robosuite import (
    effective_bounds,
    env_class_name,
    load_scene_chain,
    reach_constraint,
)


def test_ours_scene_has_every_task_and_the_reach_constraint():
    scene = load_scene_chain({"name": "ours"}, "mimicgen")
    official = load_scene_chain({"name": "official_d1"}, "mimicgen")
    for task in ("square", "coffee", "threading", "stack_three", "hammer_cleanup", "mug_cleanup", "three_piece_assembly"):
        ours, d1 = effective_bounds(scene, task), effective_bounds(official, task)
        assert ours.keys() == d1.keys()
        for obj in ours:
            for key in ("x", "y"):                     # same centre, wider, within the clip
                lo, hi = ours[obj][key]
                dlo, dhi = d1[obj][key]
                assert lo <= dlo and hi >= dhi and -0.35 <= lo <= hi <= 0.35
    assert effective_bounds(scene, "square")["nut"]["x"] == (-0.23, 0.23)
    assert effective_bounds(scene, "hammer_cleanup")["hammer"]["rotation_axis"] == "y"
    assert reach_constraint(scene)["radius_m"] == 0.85 and reach_constraint(official) is None
    assert env_class_name({"env_prefix": "Square"}, scene) == "Square_OURS"
    assert env_class_name({"env_prefix": "Square"}, official) == "Square_D1"


def test_parent_chain_inherits_bounds():
    scene = load_scene_chain({"name": "child", "env_suffix": "CHILD", "parent": "ours", "override_bounds": True,
                              "reach_constraint": {"enabled": False}}, "mimicgen")
    assert scene["parent"]["name"] == "ours"
    assert effective_bounds(scene, "square")["nut"]["reference"] == (0.0, 0.0, 0.82)
    assert reach_constraint(scene) is None


def test_table_clip_is_enforced():
    with pytest.raises(ValueError):
        load_scene_chain({"name": "bad", "env_suffix": "X", "table_clip_m": 0.1,
                          "bounds": {"square": {"nut": {"x": [-0.5, 0.5], "y": [0, 0], "z_rot": [0, 0], "reference": "table_offset"}}}})


def test_bad_rotation_axis_is_rejected():
    with pytest.raises(ValueError):
        load_scene_chain({"name": "bad", "env_suffix": "X",
                          "bounds": {"t": {"o": {"x": [0, 0], "y": [0, 0], "z_rot": [0, 0], "rotation_axis": "w", "reference": "table_offset"}}}})
