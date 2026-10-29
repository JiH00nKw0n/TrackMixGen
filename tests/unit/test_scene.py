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


def test_scene_file_helpers_and_reach_constraint_lookup(tmp_path):
    from tmg import CONFIGS_ROOT
    from tmg.generation.simulators import robosuite, scenes
    from tmg.generation.simulators.robosuite import effective_reach_constraint, interface_name

    assert scenes.generator_config_dir("mimicgen") == CONFIGS_ROOT / "experiment" / "generator" / "mimicgen"
    assert scenes.scene_details_path("ours", "mimicgen") == scenes.generator_config_dir("mimicgen") / "scene" / "ours.yaml"
    assert robosuite.scene_details_path is scenes.scene_details_path and robosuite.load_scene_file is scenes.load_scene_file
    assert robosuite.scene_chain is scenes.scene_chain
    ours = load_scene_chain({"name": "ours"}, "mimicgen")
    assert effective_reach_constraint(ours) == reach_constraint(ours)
    child = load_scene_chain({"name": "child", "env_suffix": "C", "parent": "ours", "reach_constraint": {"enabled": False}}, "mimicgen")
    assert reach_constraint(child) is None and effective_reach_constraint(child)["radius_m"] == 0.85   # inherited from ours
    assert effective_reach_constraint(load_scene_chain({"name": "official_d1"}, "mimicgen")) is None
    nearer = load_scene_chain({"name": "near", "env_suffix": "N", "parent": "ours",
                               "reach_constraint": {"enabled": True, "radius_m": 0.5, "base_xy": [-0.4, 0.1]}}, "mimicgen")
    assert effective_reach_constraint(nearer) == {"radius_m": 0.5, "base_xy": (-0.4, 0.1), "max_rejects": 500}   # the leaf wins
    grandchild = load_scene_chain({"name": "gc", "env_suffix": "G", "parent": {"name": "mid", "env_suffix": "M", "parent": "ours",
                                                                             "reach_constraint": {"enabled": False}}}, "mimicgen")
    assert effective_reach_constraint(grandchild)["radius_m"] == 0.85                                           # two levels up
    assert interface_name({"env_prefix": "Square"}) == "MG_Square"
    assert interface_name({"env_prefix": "Square", "datagen_interface": "MG_Custom"}) == "MG_Custom"
    # the generic loader takes the validation as a callable; robosuite's wrapper applies validate_scene
    seen = []
    scenes.load_scene_chain({"name": "x", "env_suffix": "X"}, validate=seen.append)
    assert seen == [{"name": "x", "env_suffix": "X"}]
    with pytest.raises(FileNotFoundError):
        scenes.load_scene_file("no_such_scene", "mimicgen", root=tmp_path)
