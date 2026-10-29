"""Low-dim object observation terms of the Isaac Lab tasks (tmg.generation.simulators.isaaclab_lowdim)."""
import sys
import types

import numpy as np
import pytest

from tmg.generation.simulators.isaaclab_lowdim import add_lowdim_terms, select_nodes


def _rope_rest(n=120, seed=0):
    """A straight rope along a diagonal of the table, 1 cm thick (nodes jittered around the axis)."""
    rng = np.random.default_rng(seed)
    s = np.linspace(0.0, 0.6, n)
    axis = np.array([0.8, 0.6, 0.0])
    return s[:, None] * axis[None, :] + rng.normal(scale=0.004, size=(n, 3)) + np.array([0.4, -0.1, 0.02])


def test_line_selection_orders_nodes_along_the_rope_and_spans_both_ends():
    rest = _rope_rest()
    idx = select_nodes(rest, "line", [16])
    assert idx.shape == (16,) and len(set(idx.tolist())) == 16          # 16 distinct nodes
    along = rest[idx] @ np.array([0.8, 0.6, 0.0])
    assert np.all(np.diff(along) > 0)                                   # in order along the rope's axis (sign fixed: +x)
    assert idx[0] == 0 and idx[-1] == len(rest) - 1                     # the two rope ends are included
    assert np.array_equal(select_nodes(rest + 0.3, "line", [16]), idx)  # a moved rest pose picks the same nodes


def test_grid_selection_picks_a_six_by_six_grid_including_the_corners():
    xs, ys = np.meshgrid(np.linspace(0, 0.4, 25), np.linspace(0, 0.3, 19), indexing="ij")
    rest = np.stack([xs.ravel() + 0.5, ys.ravel() - 0.15, np.full(xs.size, 0.01)], axis=1)
    idx = select_nodes(rest, "grid", [6, 6])
    assert idx.shape == (36,) and len(set(idx.tolist())) == 36
    picked = rest[idx]
    corners = {(round(x, 3), round(y, 3)) for x, y in picked[:, :2]}
    for corner in [(0.5, -0.15), (0.9, -0.15), (0.5, 0.15), (0.9, 0.15)]:
        assert corner in corners
    assert np.all(np.diff(picked.reshape(6, 6, 3)[:, 0, 0]) > 0)       # rows advance along the long side first
    with pytest.raises(ValueError):
        select_nodes(rest, "grid", [6])
    with pytest.raises(ValueError):
        select_nodes(rest[:10], "line", [16])


def test_add_lowdim_terms_writes_new_terms_into_the_policy_group(monkeypatch):
    class TermCfg:
        def __init__(self, func, params):
            self.func, self.params = func, params

    managers = types.ModuleType("isaaclab.managers")
    managers.ObservationTermCfg = TermCfg
    isaaclab = types.ModuleType("isaaclab")
    isaaclab.managers = managers
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)
    policy = types.SimpleNamespace(robot0_eef_pos=object())
    env_cfg = types.SimpleNamespace(observations=types.SimpleNamespace(policy=policy))
    task = {"lowdim_terms": {"rope_state": {"func": "tmg.generation.simulators.isaaclab_lowdim:deformable_nodes",
                                            "params": {"object": "object", "layout": "line", "counts": (16,), "relative": "ends",
                                                       "eef": {"sensor": "ee_frame"}}}}}
    added = add_lowdim_terms(env_cfg, task)
    assert added == {"rope_state": "tmg.generation.simulators.isaaclab_lowdim:deformable_nodes"}
    from tmg.generation.simulators.isaaclab_lowdim import deformable_nodes

    assert policy.rope_state.func is deformable_nodes and policy.rope_state.params["counts"] == [16]
    assert add_lowdim_terms(env_cfg, {}) == {}
    with pytest.raises(KeyError):
        add_lowdim_terms(env_cfg, {"lowdim_terms": {"robot0_eef_pos": {"func": "x:y"}}})


def test_task_configs_declare_the_decided_low_dim_terms():
    import pathlib

    import yaml

    root = pathlib.Path(__file__).resolve().parents[2] / "configs/experiment/generator"
    expect = {"softmimicgen/task/rope.yaml": ("rope_state", "deformable_nodes", {"layout": "line", "counts": [16], "relative": "ends"}),
              "softmimicgen/task/towel.yaml": ("towel_state", "deformable_nodes", {"layout": "grid", "counts": [6, 6], "relative": "centroid"}),
              "dexmimicgen/task/microwave.yaml": ("microwave_state", "scene_object_state", {"objects": [{"name": "microwave", "joints": ["microjoint"]}]}),
              "dexmimicgen/task/bottle_door.yaml": ("bottle_door_state", "scene_object_state",
                                                    {"objects": [{"name": "ranch_dressing_hope_robolab"}, {"name": "refrigerator", "joints": ["fridge_door_joint"]}]})}
    for rel, (name, func, params) in expect.items():
        cfg = yaml.safe_load((root / rel).read_text())
        assert cfg["object_state"] == [name]
        term = cfg["lowdim_terms"][name]
        assert term["func"].endswith(":" + func)
        for k, v in params.items():
            assert term["params"][k] == v
    pick = yaml.safe_load((root / "dexmimicgen/task/pick_place.yaml").read_text())
    assert pick["object_state"] == ["object"] and "lowdim_terms" not in pick     # the official term already exists
