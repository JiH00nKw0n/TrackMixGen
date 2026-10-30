"""BC-RNN supports uncropped images through training and saved-policy evaluation."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")


def _policy(crop, image_shape=(3, 128, 128)):
    from tmg.policy.lerobot_policies import build_policy

    cfg = {"name": "bc_rnn", "params": {
        "crop_shape": crop, "seq_length": 2, "rnn_hidden_dim": 16,
        "rnn_num_layers": 1, "visual_feature_dim": 8, "num_keypoints": 4,
    }, "by_observation": {"image": {}}}
    features = {"cameras": ["front", "side"], "image_shape": image_shape,
                "state_dim": 4, "env_state_dim": 0, "action_dim": 3}
    return build_policy(cfg, features, "cpu")


@pytest.mark.parametrize("crop,side", [(None, 128), ([28, 28], 32)])
def test_image_training_and_saved_rollout(tmp_path, crop, side):
    from tmg.policy.lerobot_policies import build_optimizer, load_policy

    policy = _policy(crop, (3, side, side))
    batch = {f"observation.images.{cam}": torch.rand(2, 2, 3, side, side)
             for cam in ("front", "side")}
    batch.update({"observation.state": torch.rand(2, 2, 4), "action": torch.rand(2, 2, 3)})
    sizes = []
    hook = policy.net.encoders["front"].trunk.register_forward_pre_hook(
        lambda module, args: sizes.append(tuple(args[0].shape[-2:])))
    optimizer = build_optimizer("bc_rnn", policy.config, policy)
    policy.train()
    loss, _ = policy(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)
    optimizer.step()
    hook.remove()
    assert sizes == [(side, side) if crop is None else tuple(crop)]
    assert type(optimizer) is torch.optim.Adam
    if crop is None:
        assert policy.config.crop_shape is None and policy.net.cfg.crop_shape is None
        images = batch["observation.images.front"][:, 0]
        for training in (True, False):
            policy.train(training)
            assert torch.equal(policy.net.encoders["front"].crop(images), images)

    policy.save_pretrained(tmp_path / "pretrained_model")
    restored = load_policy(tmp_path, "cpu")
    if crop is None:
        assert restored.config.crop_shape is None
    else:
        assert tuple(restored.config.crop_shape) == tuple(policy.config.crop_shape)
    step = {k: v[:, 0] for k, v in batch.items() if k != "action"}
    policy.reset()
    restored.reset()
    torch.manual_seed(19)
    expected = policy.select_action(step)
    torch.manual_seed(19)
    actual = restored.select_action(step)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("generator,task", [
    ("softmimicgen", "rope"), ("softmimicgen", "towel"),
    ("dexmimicgen", "pick_place"), ("dexmimicgen", "microwave"),
    ("dexmimicgen", "bottle_door"),
])
def test_campaign_config_constructs_uncropped_bc_rnn(tmp_path, monkeypatch, generator, task):
    from tmg.config import compose, resolved
    from tmg.policy.lerobot_policies import build_config, policy_class, task_policy_overrides

    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    cfg = resolved(compose("train", [f"generator={generator}", f"task={task}",
        "scene=ours", "method=none", "generation=pool_isaac", "observation=image", "policy=bc_rnn"]))
    features = {"cameras": ["front", "side"], "image_shape": (3, 128, 128),
                "state_dim": 36, "env_state_dim": 0, "action_dim": 36}
    config = build_config(cfg["policy"], features, "cpu", overrides=task_policy_overrides(cfg["task"], "bc_rnn"))
    assert config.crop_shape is None
    config.rnn_hidden_dim = 16
    config.rnn_num_layers = 1
    policy = policy_class("bc_rnn")(config)
    assert policy.net.cfg.crop_shape is None
    assert cfg["task"]["horizon"] == (400 if generator == "softmimicgen" else 600)
