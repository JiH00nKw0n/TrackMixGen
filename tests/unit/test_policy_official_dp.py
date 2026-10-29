"""Diffusion Policy matches the official implementation where LeRobot differs: per-sample random crops in training,
center crop in evaluation, and AdamW with the preset's values. Needs torch and lerobot (train image)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lerobot")


def test_per_sample_crop_draws_one_location_per_image():
    from tmg.policy.augment import PerSampleRandomCrop

    torch.manual_seed(0)
    # image k holds the value 1000 * k + (row * 10 + col), so a crop's top-left value gives its location
    rows, cols = torch.meshgrid(torch.arange(10), torch.arange(10), indexing="ij")
    base = (rows * 10 + cols).float()
    x = torch.stack([base + 1000 * k for k in range(64)])[:, None].repeat(1, 3, 1, 1)          # (64, 3, 10, 10)
    out = PerSampleRandomCrop((4, 6))(x)
    assert out.shape == (64, 3, 4, 6)
    corner = out[:, 0, 0, 0] - 1000 * torch.arange(64)
    top, left = (corner // 10).long(), (corner % 10).long()
    assert top.min() >= 0 and top.max() <= 6 and left.min() >= 0 and left.max() <= 4
    assert len({(int(t), int(l)) for t, l in zip(top, left)}) > 10                           # not one shared location
    for k in (0, 17, 63):                                                                     # each crop is a contiguous window
        t, l = int(top[k]), int(left[k])
        assert torch.equal(out[k, 2], x[k, 2, t:t + 4, l:l + 6])


def _diffusion_policy():
    from tmg.policy.lerobot_policies import build_policy

    cfg = {"name": "diffusion_policy", "params": {"crop_shape": [76, 76], "crop_is_random": True, "n_obs_steps": 2,
                                                  "horizon": 16, "n_action_steps": 8, "use_group_norm": True,
                                                  "pretrained_backbone_weights": None},
           "by_observation": {"image": {}}}
    features = {"cameras": ["agentview"], "image_shape": (3, 84, 84), "state_dim": 9, "env_state_dim": 0, "action_dim": 7}
    return build_policy(cfg, features, "cpu", "image")


def test_diffusion_policy_uses_the_per_sample_crop_in_training_and_center_crop_in_eval():
    from tmg.policy.augment import PerSampleRandomCrop

    policy = _diffusion_policy()
    encoders = [m for m in policy.modules() if hasattr(m, "maybe_random_crop") and getattr(m, "do_crop", False)]
    assert encoders and all(isinstance(m.maybe_random_crop, PerSampleRandomCrop) for m in encoders)
    assert all(type(m.center_crop).__name__ == "CenterCrop" for m in encoders)
    assert all(not any(True for _ in m.maybe_random_crop.parameters()) for m in encoders)    # no weights: same checkpoints


def test_diffusion_policy_trains_with_adamw_and_the_preset_values():
    from tmg.policy.lerobot_policies import build_optimizer

    policy = _diffusion_policy()
    opt = build_optimizer("diffusion_policy", policy.config, policy)
    assert isinstance(opt, torch.optim.AdamW)
    group = opt.param_groups[0]
    assert group["lr"] == pytest.approx(1e-4) and tuple(group["betas"]) == pytest.approx((0.95, 0.999))
    assert group["weight_decay"] == pytest.approx(1e-6) and group["eps"] == pytest.approx(1e-8)
    assert np.isclose(sum(p.numel() for g in opt.param_groups for p in g["params"]),
                      sum(p.numel() for p in policy.get_optim_params()))


def test_bc_rnn_keeps_its_preset_optimizer():
    from tmg.policy.lerobot_policies import build_optimizer, build_policy

    cfg = {"name": "bc_rnn", "params": {"rnn_hidden_dim": 32, "seq_length": 4, "optimizer_lr": 1e-3},
           "by_observation": {"low_dim": {}}}
    features = {"cameras": [], "image_shape": None, "state_dim": 5, "env_state_dim": 14, "action_dim": 7}
    policy = build_policy(cfg, features, "cpu", "low_dim")
    opt = build_optimizer("bc_rnn", policy.config, policy)
    assert type(opt) is torch.optim.Adam and opt.param_groups[0]["lr"] == pytest.approx(1e-3)
