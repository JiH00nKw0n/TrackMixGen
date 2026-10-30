"""Construction of policies from our ``policy`` config block and the dataset's features.

``build_policy(policy_cfg, features, device)`` returns a LeRobot ``PreTrainedPolicy``:

- ``bc_rnn``: ``tmg.policy.bc_rnn.BCRNNPolicy``;
- ``diffusion_policy``: LeRobot's ``DiffusionPolicy`` with the official image U-Net values written into
  ``DiffusionConfig`` (the config file lists only the fields that differ from LeRobot's defaults);
- ``pi0``: LeRobot's ``PI0Policy`` from the hub weights (stage 6).

``features`` is ``{"cameras": [...], "image_shape": (3, H, W) or None, "state_dim": D, "env_state_dim": E,
"action_dim": A}`` as read from the dataset: an image dataset has cameras and no object state (E = 0), a low-dim
dataset has no cameras and E > 0 (``observation.environment_state``, LeRobot's ENV feature). ``observation``
(``image`` | ``low_dim``) picks the ``by_observation`` block of the policy config, whose values override
``params`` (BC-RNN hidden size, Diffusion Policy U-Net widths). The delta indices the loader must use come from the built config
(``observation_delta_indices`` / ``action_delta_indices``), so the dataset stacking always matches the policy.
``load_policy(checkpoint_dir)`` restores any of them from ``config.json`` + ``model.safetensors``.
"""
from __future__ import annotations

from pathlib import Path

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature

import tmg.policy.bc_rnn  # noqa: F401  registers the tmg_bc_rnn config type before any from_pretrained

POLICIES = ("bc_rnn", "diffusion_policy", "pi0")


def policy_features(features: dict) -> tuple[dict, dict]:
    """LeRobot ``input_features`` and ``output_features`` from the dataset feature summary."""
    inputs = {}
    if features.get("cameras"):
        c, h, w = (int(v) for v in features["image_shape"])
        inputs = {f"observation.images.{cam}": PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w)) for cam in features["cameras"]}
    inputs["observation.state"] = PolicyFeature(type=FeatureType.STATE, shape=(int(features["state_dim"]),))
    if int(features.get("env_state_dim", 0) or 0) > 0:
        inputs["observation.environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(int(features["env_state_dim"]),))
    outputs = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(int(features["action_dim"]),))}
    return inputs, outputs


def policy_params(policy_cfg: dict, observation: str) -> dict:
    """``params`` of the policy config with the ``by_observation[observation]`` block applied on top.

    Raises:
        ValueError: the policy has no form for that observation (``by_observation.<name>: null``, e.g. π0 low-dim).
    """
    params = dict(policy_cfg.get("params") or {})
    blocks = policy_cfg.get("by_observation") or {}
    if observation not in blocks:
        raise ValueError(f"policy {policy_cfg.get('name')!r} lists no by_observation block for {observation!r}")
    if blocks[observation] is None:
        raise ValueError(f"policy {policy_cfg.get('name')!r} has no {observation} form (by_observation.{observation} is null)")
    params.update(dict(blocks[observation]))
    return params


CROP_RATIO = 76 / 84   # robomimic's random-crop ratio: 76 of 84 pixels (and 116 of 128)


def resolve_crop(params: dict, features: dict) -> dict:
    """Turn ``crop_shape: auto`` into pixels: the robomimic ratio of the dataset's image side (84 -> 76, 128 -> 116).

    A fixed ``[76, 76]`` would crop 128x128 Isaac Lab images to 59% of their side instead of 90% (decision 2026-10-29).
    """
    params = dict(params)
    if params.get("crop_shape") == "auto":
        shape = features.get("image_shape")
        if shape is None:
            params["crop_shape"] = None
        else:
            _, h, w = (int(v) for v in shape)
            params["crop_shape"] = [round(h * CROP_RATIO), round(w * CROP_RATIO)]
    return params


def task_policy_overrides(task: dict, policy_name: str) -> dict:
    """``task.policy_overrides.<policy name>``: per-task policy values (for example the Isaac Lab tasks set ``crop_shape:
    null`` for both image policies, the project rule of 2026-10-29, although SoftMimicGen's official BC-RNN crops 128x128
    images to 76x76; MimicGen's tasks keep the 76x76 crop of the policy config). Empty when the task lists none."""
    blocks = (task or {}).get("policy_overrides") or {}
    return dict(blocks.get(policy_name) or {})


def build_config(policy_cfg: dict, features: dict, device: str, observation: str = "image", overrides: dict | None = None) -> PreTrainedConfig:
    """The LeRobot config object of ``policy_cfg['name']`` with our values, the task's overrides and the dataset features."""
    name = str(policy_cfg["name"])
    inputs, outputs = policy_features(features)
    params = resolve_crop({**policy_params(policy_cfg, observation), **(overrides or {})}, features)
    if name == "bc_rnn":
        from tmg.policy.bc_rnn import BCRNNConfig

        return BCRNNConfig(input_features=inputs, output_features=outputs, device=device, **params)
    if name == "diffusion_policy":
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

        for key in ("crop_shape", "down_dims"):
            if key in params and params[key] is not None:
                params[key] = tuple(params[key])
        return DiffusionConfig(input_features=inputs, output_features=outputs, device=device, **params)
    if name == "pi0":
        from lerobot.policies.pi0.configuration_pi0 import PI0Config

        return PI0Config(input_features=inputs, output_features=outputs, device=device, **params)
    raise ValueError(f"unknown policy {name!r}; known: {POLICIES}")


def policy_class(name: str):
    """The ``PreTrainedPolicy`` subclass of a policy name."""
    if name == "bc_rnn":
        from tmg.policy.bc_rnn import BCRNNPolicy

        return BCRNNPolicy
    if name == "diffusion_policy":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    if name == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy

        return PI0Policy
    raise ValueError(f"unknown policy {name!r}; known: {POLICIES}")


def build_policy(policy_cfg: dict, features: dict, device: str, observation: str = "image", overrides: dict | None = None):
    """A freshly initialized policy (or, for π0, the pretrained hub weights named by ``params.pretrained_path``)."""
    name = str(policy_cfg["name"])
    config = build_config(policy_cfg, features, device, observation, overrides)
    cls = policy_class(name)
    pretrained = (policy_cfg.get("pretrained_path") or None)
    policy = cls.from_pretrained(pretrained, config=config) if pretrained else cls(config)
    if name == "diffusion_policy":
        from tmg.policy.augment import use_per_sample_crop

        use_per_sample_crop(policy)
    return policy.to(device)


def build_optimizer(name: str, config, policy):
    """Use AdamW for Diffusion Policy with the preset's values; keep other policies' presets.

    Backported from main 4d7926d (published as 6917f3e) without the unrelated registry refactor.
    """
    preset = config.get_optimizer_preset()
    if name != "diffusion_policy":
        return preset.build(policy.get_optim_params())
    import torch

    return torch.optim.AdamW(policy.get_optim_params(), lr=preset.lr, betas=tuple(preset.betas), eps=preset.eps,
                             weight_decay=preset.weight_decay)


def normalization_modes(config: PreTrainedConfig) -> dict[str, str]:
    """``{"VISUAL"|"STATE"|"ENV"|"ACTION": "identity"|"mean_std"|"min_max"}`` from a LeRobot config's mapping
    (``ENV``, the object state, falls back to the ``STATE`` mode in ``tmg.policy.normalize`` when the mapping has none)."""
    table = {"IDENTITY": "identity", "MEAN_STD": "mean_std", "MIN_MAX": "min_max"}
    out = {}
    for key, mode in getattr(config, "normalization_mapping", {}).items():
        value = getattr(mode, "value", str(mode))
        if value not in table:
            raise ValueError(f"normalization mode {value} of {key} is not supported by tmg.policy.normalize")
        out[str(key)] = table[value]
    return out


def load_policy(checkpoint_dir: Path, device: str):
    """Restore a policy saved by ``trainer`` (``pretrained_model/`` with ``config.json`` and the weights)."""
    folder = Path(checkpoint_dir) / "pretrained_model"
    config = PreTrainedConfig.from_pretrained(str(folder))
    config.device = device
    cls = policy_class(policy_name_of(config))
    return cls.from_pretrained(str(folder), config=config).to(device)


def policy_name_of(config: PreTrainedConfig) -> str:
    """Our policy name of a LeRobot config type."""
    table = {"tmg_bc_rnn": "bc_rnn", "diffusion": "diffusion_policy", "pi0": "pi0"}
    if config.type not in table:
        raise ValueError(f"checkpoint config type {config.type!r} is not one of ours")
    return table[config.type]
