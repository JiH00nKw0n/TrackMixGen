"""Policy-side units: wire encoding, evaluation summaries, checkpoint selection, and (with torch) the
normalizer, the EMA schedule and the BC-RNN network shapes."""
import json

import numpy as np
import pytest

from tmg.core.wire import pack, unpack
from tmg.evaluation.rollout import episode_seed, summarize, wilson


def test_wire_roundtrip_keeps_arrays_and_scalars():
    msg = {"type": "act", "obs": {"observation.images.cam": np.zeros((3, 8, 8), np.uint8), "observation.state": np.arange(5, dtype=np.float32)},
           "seed": 7, "flag": True, "none": None, "list": [1, 2.5, "x"]}
    back = unpack(pack(msg))
    assert back["seed"] == 7 and back["flag"] is True and back["none"] is None and back["list"] == [1, 2.5, "x"]
    assert back["obs"]["observation.images.cam"].dtype == np.uint8 and back["obs"]["observation.images.cam"].shape == (3, 8, 8)
    assert np.array_equal(back["obs"]["observation.state"], np.arange(5, dtype=np.float32))


def test_wilson_and_summary():
    lo, hi = wilson(5, 10)
    assert (lo, hi) == pytest.approx((0.23658, 0.76342), abs=1e-4)      # z = 1.96: centre 0.5, half width 0.26342
    rows = [{"split": "val", "success": i % 2 == 0, "steps": 10 + i, "error": None} for i in range(10)]
    rows += [{"split": "test", "success": False, "steps": 400, "error": "boom"}]
    s = summarize(rows)
    assert s["by_split"]["val"]["n_success"] == 5 and s["by_split"]["test"]["errors"] == 1 and s["all"]["n"] == 11
    # the committed rule SeedSequence([evaluation.seed, training_seed, reset_index]), spelt out
    assert episode_seed(7, 1, 3) == int(np.random.SeedSequence([7, 1, 3]).generate_state(1)[0]) != episode_seed(7, 1, 4)
    assert episode_seed(202600, 5, 0) == 3655276531 and episode_seed(202600, 5, 2) == 2855745476


def test_select_checkpoint_rules(tmp_path):
    torch = pytest.importorskip("torch")  # noqa: F841  trainer imports torch
    from tmg.policy.trainer import CHECKPOINT_JSON, CHECKPOINTS_DIR, VALIDATION_JSONL, select_checkpoint

    for u in (10000, 20000, 30000):
        d = tmp_path / CHECKPOINTS_DIR / f"{u:07d}"
        d.mkdir(parents=True)
        (d / CHECKPOINT_JSON).write_text(json.dumps({"update": u}))
    assert select_checkpoint(tmp_path, "last")["checkpoint"].endswith("0030000")
    with pytest.raises(FileNotFoundError):
        select_checkpoint(tmp_path, "best_validation")
    rows = [{"checkpoint": str(tmp_path / CHECKPOINTS_DIR / f"{u:07d}"), "success_rate": r} for u, r in ((10000, 0.4), (20000, 0.6), (30000, 0.6))]
    (tmp_path / VALIDATION_JSONL).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    chosen = select_checkpoint(tmp_path, "best_validation")
    assert chosen["checkpoint"].endswith("0020000") and chosen["success_rate"] == 0.6   # earliest on ties


def test_normalizer_modes_and_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")
    from tmg.policy.normalize import Normalizer

    stats = {"observation.state": {"min": np.array([0.0, -1.0]), "max": np.array([2.0, 1.0]), "mean": np.array([1.0, 0.0]), "std": np.array([0.5, 0.5])},
             "action": {"min": np.array([-1.0]), "max": np.array([1.0]), "mean": np.array([0.0]), "std": np.array([1.0])},
             "observation.images.cam": {"min": np.zeros((3, 1, 1)), "max": np.ones((3, 1, 1)), "mean": np.full((3, 1, 1), 0.5), "std": np.full((3, 1, 1), 0.25)}}
    n = Normalizer(stats, {"STATE": "min_max", "ACTION": "mean_std", "VISUAL": "identity"})
    batch = {"observation.state": torch.tensor([[2.0, -1.0]]), "action": torch.tensor([[0.5]]), "observation.images.cam": torch.ones(1, 3, 2, 2),
             "action_is_pad": torch.tensor([[False]])}
    out = n(batch)
    assert torch.allclose(out["observation.state"], torch.tensor([[1.0, -1.0]]))
    assert torch.allclose(out["action"], torch.tensor([[0.5]]), atol=1e-6) and torch.equal(out["observation.images.cam"], batch["observation.images.cam"])
    assert torch.allclose(n.unnormalize_action(out["action"]), batch["action"], atol=1e-6)
    n.save(tmp_path / "stats.json")
    again = Normalizer.load(tmp_path / "stats.json")
    assert again.modes == n.modes and torch.allclose(again(batch)["observation.state"], out["observation.state"])
    with pytest.raises(ValueError):
        Normalizer(stats, {"STATE": "quantile"})


def test_ema_decay_schedule_and_update():
    torch = pytest.importorskip("torch")
    from tmg.policy.ema import EMA

    model = torch.nn.Linear(2, 1)
    ema = EMA(model)
    assert ema.decay() == 0.0
    assert ema.update(model) == 0.0                        # official schedule: the first two updates copy the weights
    with torch.no_grad():
        model.weight.fill_(2.0)
    assert ema.update(model) == 0.0 and torch.all(ema.model.weight == 2.0)
    with torch.no_grad():
        model.weight.fill_(3.0)
    for _ in range(3):
        d = ema.update(model)
    assert 0.0 < d < 0.9999 and torch.all(ema.model.weight > 2.0) and torch.all(ema.model.weight < 3.0)
    assert abs(ema.decay() - min(0.9999, 1 - (1 + ema.step - 1) ** -0.75)) < 1e-9


def test_bc_rnn_net_shapes_and_rollout_state():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from tmg.policy.nets import BCRNNNet, BCRNNNetConfig

    cfg = BCRNNNetConfig(cameras=("a", "b"), image_shape=(3, 32, 32), state_dim=4, action_dim=7, crop_shape=(28, 28),
                         rnn_hidden_dim=16, visual_feature_dim=8)
    net = BCRNNNet(cfg)
    images = {"a": torch.rand(2, 5, 3, 32, 32), "b": torch.rand(2, 5, 3, 32, 32)}
    state, actions = torch.rand(2, 5, 4), torch.rand(2, 5, 7) * 2 - 1
    loss = net.loss(images, state, actions)
    assert loss.ndim == 0 and torch.isfinite(loss)
    net.eval()
    rnn_state = net.init_state(2, "cpu")
    a, rnn_state = net.forward_step({k: v[:, 0] for k, v in images.items()}, state[:, 0], rnn_state)
    assert a.shape == (2, 7) and a.abs().max() <= 1.0 + 1e-3 and rnn_state[0].shape == (2, 2, 16)


def test_bc_rnn_low_dim_net_has_no_encoders():
    torch = pytest.importorskip("torch")
    from tmg.policy.nets import BCRNNNet, BCRNNNetConfig

    net = BCRNNNet(BCRNNNetConfig(cameras=(), image_shape=None, state_dim=19, action_dim=7, rnn_hidden_dim=16))
    assert len(net.encoders) == 0 and net.rnn.input_size == 19
    state, actions = torch.rand(2, 5, 19), torch.rand(2, 5, 7) * 2 - 1
    assert torch.isfinite(net.loss({}, state, actions))
    net.eval()
    a, _ = net.forward_step({}, state[:, 0], net.init_state(2, "cpu"))
    assert a.shape == (2, 7)


def test_policy_params_by_observation():
    pytest.importorskip("lerobot")
    from tmg.policy.lerobot_policies import policy_features, policy_params

    cfg = {"name": "bc_rnn", "params": {"rnn_hidden_dim": 1000, "seq_length": 10},
           "by_observation": {"image": {}, "low_dim": {"rnn_hidden_dim": 400}, "none": None}}
    assert policy_params(cfg, "image")["rnn_hidden_dim"] == 1000
    assert policy_params(cfg, "low_dim") == {"rnn_hidden_dim": 400, "seq_length": 10}
    with pytest.raises(ValueError):
        policy_params(cfg, "none")
    with pytest.raises(ValueError):
        policy_params(cfg, "other")
    inputs, _ = policy_features({"cameras": [], "image_shape": None, "state_dim": 5, "env_state_dim": 14, "action_dim": 7})
    assert set(inputs) == {"observation.state", "observation.environment_state"} and inputs["observation.environment_state"].shape == (14,)
    inputs, _ = policy_features({"cameras": ["a"], "image_shape": (3, 84, 84), "state_dim": 5, "env_state_dim": 0, "action_dim": 7})
    assert set(inputs) == {"observation.images.a", "observation.state"}


def test_dataset_features_and_budget_check_the_observation():
    pytest.importorskip("torch")
    from tmg.data.loader import TrajectoryStore
    from tmg.policy.trainer import dataset_features, policy_budget

    lengths = np.array([3])
    image_store = TrajectoryStore(images={"a": [np.zeros((3, 8, 8, 3), np.uint8)]}, state=[np.zeros((3, 5), np.float32)],
                                  actions=[np.zeros((3, 7), np.float32)], lengths=lengths, cameras=["a"])
    low_store = TrajectoryStore(images={}, state=[np.zeros((3, 5), np.float32)], actions=[np.zeros((3, 7), np.float32)],
                                lengths=lengths, cameras=[], env_state=[np.zeros((3, 14), np.float32)])
    assert dataset_features(image_store, "image")["image_shape"] == (3, 8, 8)
    assert dataset_features(low_store, "low_dim") == {"cameras": [], "image_shape": None, "state_dim": 5, "env_state_dim": 14, "action_dim": 7}
    with pytest.raises(ValueError):
        dataset_features(image_store, "low_dim")
    with pytest.raises(ValueError):
        dataset_features(low_store, "image")
    training = {"by_policy": {"bc_rnn": {"image": {"batch_size": 16, "updates": 100}, "low_dim": {"batch_size": 100, "updates": 200}}}}
    assert policy_budget(training, "bc_rnn", "low_dim", None) == (100, 200) and policy_budget(training, "bc_rnn", "image", 7) == (16, 7)
    with pytest.raises(KeyError):
        policy_budget(training, "pi0", "image", None)


def test_select_scenes_limit_and_shards():
    from tmg.evaluation.rollout import select_scenes

    scenes = [{"index": i} for i in range(10)]
    assert [s["index"] for s in select_scenes(scenes, limit=4)] == [0, 1, 2, 3]
    shards = [select_scenes(scenes, shard=[k, 3]) for k in range(3)]
    assert sorted(s["index"] for sh in shards for s in sh) == list(range(10))
    assert [s["index"] for s in shards[1]] == [1, 4, 7]
    with pytest.raises(ValueError):
        select_scenes(scenes, shard=[3, 3])


def test_sync_controllers_reanchors_every_robot():
    from types import SimpleNamespace

    import numpy as np

    from tmg.evaluation.rollout import sync_controllers

    calls = []

    class Controller:
        def update_initial_joints(self, q):
            calls.append(("init", np.asarray(q).tolist()))

        def update(self, force=False):
            calls.append(("update", force))

        def reset_goal(self):
            calls.append(("goal",))

    qpos = np.arange(10.0)
    robot = SimpleNamespace(controller=Controller(), sim=SimpleNamespace(data=SimpleNamespace(qpos=qpos)), _ref_joint_pos_indexes=[2, 3, 4])
    env = SimpleNamespace(env=SimpleNamespace(robots=[robot]))
    sync_controllers(env)
    assert calls == [("init", [2.0, 3.0, 4.0]), ("update", True), ("goal",)]
    sync_controllers(SimpleNamespace(env=SimpleNamespace(robots=[])))     # no robots: nothing happens


def test_check_server_rejects_another_checkpoint():
    from tmg.evaluation.rollout import check_server

    check_server({"checkpoint": "/data/runs/x/checkpoints/0060000"}, "/data/runs/x/checkpoints/0060000/")
    with pytest.raises(RuntimeError):
        check_server({"checkpoint": "/data/runs/x/checkpoints/0010000"}, "/data/runs/x/checkpoints/0060000")


def test_eval_output_carries_split_and_horizon():
    from pathlib import Path

    from tmg.evaluation.rollout import eval_output

    ckpt = Path("/data/runs/r/checkpoints/0040000")
    assert eval_output(ckpt, "test", 600) == Path("/data/runs/r/eval/test_h600/0040000")
    assert eval_output(ckpt, "val") == Path("/data/runs/r/eval/val/0040000")


def test_adjusted_env_meta_keeps_controller_and_sets_scene_and_cameras():
    from tmg.evaluation.envs import adjusted_env_meta

    meta = {"env_name": "Coffee_D0", "env_version": "1.5.1", "type": 1,
            "env_kwargs": {"env_name": "Coffee_D0", "controller_configs": {"type": "IK_POSE", "control_delta": False}, "control_freq": 20,
                           "use_camera_obs": False, "has_offscreen_renderer": False, "robots": "Panda"}}
    out = adjusted_env_meta(meta, "Coffee_OURS", ["agentview", "robot0_eye_in_hand"], [84, 84])
    assert out["env_name"] == "Coffee_OURS" and "env_name" not in out["env_kwargs"]
    assert out["env_kwargs"]["controller_configs"] == meta["env_kwargs"]["controller_configs"]      # untouched
    assert out["env_kwargs"]["camera_names"] == ["agentview", "robot0_eye_in_hand"] and out["env_kwargs"]["camera_heights"] == 84
    assert out["env_kwargs"]["use_camera_obs"] is True and out["env_kwargs"]["has_offscreen_renderer"] is True
    low = adjusted_env_meta(meta, "Coffee_OURS", [], None)
    assert low["env_kwargs"]["use_camera_obs"] is False and "camera_names" not in low["env_kwargs"]
    assert meta["env_kwargs"]["use_camera_obs"] is False                                            # input not modified


def test_policy_input_accepts_both_image_layouts():
    import numpy as np

    from tmg.evaluation.rollout import policy_input

    observation = {"cameras": ["agentview"], "robot_state": ["robot0_eef_pos"]}
    hwc = (np.random.default_rng(0).random((84, 84, 3)) * 255).astype(np.uint8)
    raw = policy_input({"agentview_image": hwc, "robot0_eef_pos": np.zeros(3)}, observation)
    processed = policy_input({"agentview_image": hwc.transpose(2, 0, 1).astype(np.float32) / 255.0, "robot0_eef_pos": np.zeros(3)}, observation)
    assert raw["observation.images.agentview"].shape == (3, 84, 84)
    assert np.array_equal(raw["observation.images.agentview"], hwc.transpose(2, 0, 1))
    assert np.abs(processed["observation.images.agentview"].astype(int) - raw["observation.images.agentview"].astype(int)).max() <= 1


def test_crop_shape_auto_follows_the_image_side():
    pytest.importorskip("lerobot")
    from tmg.policy.lerobot_policies import resolve_crop

    assert resolve_crop({"crop_shape": "auto"}, {"image_shape": (3, 84, 84)})["crop_shape"] == [76, 76]
    assert resolve_crop({"crop_shape": "auto"}, {"image_shape": (3, 128, 128)})["crop_shape"] == [116, 116]
    assert resolve_crop({"crop_shape": "auto"}, {"image_shape": None})["crop_shape"] is None
    assert resolve_crop({"crop_shape": [70, 70]}, {"image_shape": (3, 84, 84)})["crop_shape"] == [70, 70]


def test_policy_specs_keep_the_old_names_types_and_pipeline_rule():
    """Read from the source text (the module imports lerobot): the table's names in the order of the old ``POLICIES``
    tuple, the ``config.json`` types the old ``build_config`` branches wrote, and the pipeline flag only on pi0."""
    import ast
    import importlib.util
    from pathlib import Path

    tree = ast.parse(Path(importlib.util.find_spec("tmg.policy.lerobot_policies").origin).read_text())
    def target(node):
        return getattr(node, "target", None) or (node.targets[0] if isinstance(node, ast.Assign) else None)

    table = next(node.value for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
                 and getattr(target(node), "id", None) == "POLICY_SPECS")
    names = [k.value for k in table.keys]
    assert names == ["bc_rnn", "diffusion_policy", "pi0"]
    specs = {}
    for key, call in zip(names, table.values):
        args = [a.value for a in call.args]
        kwargs = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}
        specs[key] = (args, kwargs)
    assert specs["bc_rnn"] == (["bc_rnn", "tmg_bc_rnn", "tmg.policy.bc_rnn:BCRNNConfig", "tmg.policy.bc_rnn:BCRNNPolicy"], {})
    assert specs["diffusion_policy"] == (["diffusion_policy", "diffusion", "lerobot.policies.diffusion.configuration_diffusion:DiffusionConfig",
                                          "lerobot.policies.diffusion.modeling_diffusion:DiffusionPolicy"],
                                         {"tuple_params": ("crop_shape", "down_dims"), "optimizer": "adamw", "per_sample_crop": True})
    assert specs["pi0"] == (["pi0", "pi0", "lerobot.policies.pi0.configuration_pi0:PI0Config", "lerobot.policies.pi0.modeling_pi0:PI0Policy"],
                            {"uses_pipeline": True})


def test_uses_pipeline_only_for_pi0():
    pytest.importorskip("lerobot")
    from tmg.policy.io import NormalizerIO, PipelineIO, io_class, uses_pipeline
    from tmg.policy.lerobot_policies import POLICIES, POLICY_SPECS, policy_spec

    assert POLICIES == ("bc_rnn", "diffusion_policy", "pi0") == tuple(POLICY_SPECS)
    assert [uses_pipeline(n) for n in POLICIES] == [False, False, True] and uses_pipeline("unknown") is False
    assert io_class("pi0") is PipelineIO and io_class("bc_rnn") is NormalizerIO
    assert [policy_spec(n).config_type for n in POLICIES] == ["tmg_bc_rnn", "diffusion", "pi0"]
    with pytest.raises(ValueError):
        policy_spec("unknown")
