"""Pod integration of the upstream DP fix, with CPU models smaller than campaign models."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")


@pytest.mark.parametrize("crop", [[76, 76], None])
def test_saved_policy_keeps_evaluation_outputs_after_crop_replacement(tmp_path, crop):
    from tmg.policy.augment import use_per_sample_crop
    from tmg.policy.lerobot_policies import build_config, load_policy, policy_class

    cfg = {"name": "diffusion_policy", "params": {
        "crop_shape": crop, "crop_is_random": True, "n_obs_steps": 2,
        "horizon": 16, "n_action_steps": 8, "down_dims": [32, 64],
        "use_group_norm": True, "pretrained_backbone_weights": None,
    }, "by_observation": {"image": {}}}
    features = {"cameras": ["agentview"], "image_shape": (3, 84, 84),
                "state_dim": 9, "env_state_dim": 0, "action_dim": 7}
    policy = policy_class("diffusion_policy")(build_config(cfg, features, "cpu")).eval()
    encoder = next(m for m in policy.modules() if hasattr(m, "do_crop"))
    images = torch.rand(4, 3, 84, 84)
    with torch.no_grad():
        expected = encoder(images)
    keys_before = set(policy.state_dict())
    assert use_per_sample_crop(policy) == (1 if crop else 0)
    assert set(policy.state_dict()) == keys_before
    with torch.no_grad():
        assert torch.equal(encoder(images), expected)
    # Exercise the exact checkpoint loader used by the evaluation policy server.
    policy.save_pretrained(tmp_path / "pretrained_model")
    restored = load_policy(tmp_path, "cpu").eval()
    assert set(restored.state_dict()) == keys_before
    restored_encoder = next(m for m in restored.modules() if hasattr(m, "do_crop"))
    with torch.no_grad():
        assert torch.equal(restored_encoder(images), expected)


@pytest.mark.parametrize("generator,task", [
    ("softmimicgen", "rope"), ("softmimicgen", "towel"),
    ("dexmimicgen", "pick_place"), ("dexmimicgen", "microwave"),
    ("dexmimicgen", "bottle_door"),
])
def test_campaign_task_disables_crop_but_uses_adamw(tmp_path, monkeypatch, generator, task):
    from tmg.config import compose, resolved
    from tmg.policy.lerobot_policies import build_config, build_optimizer, task_policy_overrides

    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    cfg = resolved(compose("train", [f"generator={generator}", f"task={task}",
        "scene=ours", "method=none", "generation=pool_isaac", "observation=image", "policy=diffusion_policy"]))
    features = {"cameras": ["front", "side"], "image_shape": (3, 128, 128),
                "state_dim": 36, "env_state_dim": 0, "action_dim": 36}
    config = build_config(cfg["policy"], features, "cpu", overrides=task_policy_overrides(cfg["task"], "diffusion_policy"))
    assert config.crop_shape is None
    assert cfg["task"]["horizon"] == (400 if generator == "softmimicgen" else 600)
    class TinyPolicy(torch.nn.Linear):
        def get_optim_params(self):
            return self.parameters()
    optimizer = build_optimizer("diffusion_policy", config, TinyPolicy(2, 2))
    assert type(optimizer) is torch.optim.AdamW
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)
