"""Construction of policies from our ``policy`` config block and the dataset's features.

``build_policy(policy_cfg, features, device)`` returns a LeRobot ``PreTrainedPolicy``:

- ``bc_rnn``: ``tmg.policy.bc_rnn.BCRNNPolicy``;
- ``diffusion_policy``: LeRobot's ``DiffusionPolicy`` with the official image U-Net values written into
  ``DiffusionConfig`` (the config file lists only the fields that differ from LeRobot's defaults);
- ``pi0``: LeRobot's ``PI0Policy`` from the hub weights (stage 6).

The three are one table, ``POLICY_SPECS``: our policy name, the LeRobot config type that ``config.json`` stores
(``tmg_bc_rnn``, ``diffusion``, ``pi0``; what ``load_policy`` reads back), the config and policy classes (imported
only when used, since the LeRobot modules are heavy), and whether the policy needs LeRobot's processor pipelines
(``tmg.policy.io``). ``POLICIES`` is the tuple of names in table order.

``features`` is ``{"cameras": [...], "image_shape": (3, H, W) or None, "state_dim": D, "env_state_dim": E,
"action_dim": A}`` as read from the dataset: an image dataset has cameras and no object state (E = 0), a low-dim
dataset has no cameras and E > 0 (``observation.environment_state``, LeRobot's ENV feature). ``observation``
(``image`` | ``low_dim``) picks the ``by_observation`` block of the policy config, whose values override
``params`` (BC-RNN hidden size, Diffusion Policy U-Net widths). The delta indices the loader must use come from the built config
(``observation_delta_indices`` / ``action_delta_indices``), so the dataset stacking always matches the policy.
``load_policy(checkpoint_dir)`` restores any of them from ``config.json`` + ``model.safetensors``.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

from lerobot.configs import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature

import tmg.policy.bc_rnn  # noqa: F401  registers the tmg_bc_rnn config type before any from_pretrained
from tmg.core.features import ACTION, ENV_STATE, STATE, image_feature
from tmg.core.layout import PRETRAINED_DIR

# Importing tmg.policy.bc_rnn above is a required side effect, not an unused import: its ``@PreTrainedConfig.register_subclass``
# decorator makes LeRobot's ``PreTrainedConfig.from_pretrained`` recognise ``"type": "tmg_bc_rnn"`` in a checkpoint's
# config.json. Without it ``load_policy`` of a BC-RNN checkpoint fails inside LeRobot.


@dataclass(frozen=True)
class PolicySpec:
    """One row of ``POLICY_SPECS``: how a policy name maps onto LeRobot."""

    name: str                       # our name (configs/experiment/policy/<name>.yaml, run folder component)
    config_type: str                # LeRobot config type stored as "type" in a checkpoint's config.json
    config_class: str               # "module:Class" of the PreTrainedConfig subclass
    policy_class: str               # "module:Class" of the PreTrainedPolicy subclass
    tuple_params: tuple[str, ...] = ()   # params the config class declares as tuples (lists in yaml)
    uses_pipeline: bool = False     # LeRobot's pre/post processor pipelines (tokenizer) instead of our normalizer
    optimizer: str = "preset"       # preset: the config's optimizer preset; adamw: AdamW with the preset's values
    per_sample_crop: bool = False   # crop every training image at its own location (tmg.policy.augment)


POLICY_SPECS: dict[str, PolicySpec] = {
    "bc_rnn": PolicySpec("bc_rnn", "tmg_bc_rnn", "tmg.policy.bc_rnn:BCRNNConfig", "tmg.policy.bc_rnn:BCRNNPolicy"),
    "diffusion_policy": PolicySpec("diffusion_policy", "diffusion",
                                   "lerobot.policies.diffusion.configuration_diffusion:DiffusionConfig",
                                   "lerobot.policies.diffusion.modeling_diffusion:DiffusionPolicy",
                                   tuple_params=("crop_shape", "down_dims"),
                                   # the official Diffusion Policy trains with AdamW and per-sample random crops; LeRobot's
                                   # preset is Adam (coupled L2 decay) and its RandomCrop draws one location per batch
                                   optimizer="adamw", per_sample_crop=True),
    "pi0": PolicySpec("pi0", "pi0", "lerobot.policies.pi0.configuration_pi0:PI0Config",
                      "lerobot.policies.pi0.modeling_pi0:PI0Policy", uses_pipeline=True),
}
POLICIES = tuple(POLICY_SPECS)      # ("bc_rnn", "diffusion_policy", "pi0")


def policy_spec(name: str) -> PolicySpec:
    """The spec of our policy name.

    Raises:
        ValueError: the name is not one of ``POLICIES``.
    """
    if name not in POLICY_SPECS:
        raise ValueError(f"unknown policy {name!r}; known: {POLICIES}")
    return POLICY_SPECS[name]


def _load_target(target: str):
    """The object ``"module:attribute"`` names, imported on first use."""
    module, _, attribute = target.partition(":")
    return getattr(importlib.import_module(module), attribute)


def policy_features(features: dict) -> tuple[dict, dict]:
    """LeRobot ``input_features`` and ``output_features`` from the dataset feature summary."""
    inputs = {}
    if features.get("cameras"):
        c, h, w = (int(v) for v in features["image_shape"])
        inputs = {image_feature(cam): PolicyFeature(type=FeatureType.VISUAL, shape=(c, h, w)) for cam in features["cameras"]}
    inputs[STATE] = PolicyFeature(type=FeatureType.STATE, shape=(int(features["state_dim"]),))
    if int(features.get("env_state_dim", 0) or 0) > 0:
        inputs[ENV_STATE] = PolicyFeature(type=FeatureType.ENV, shape=(int(features["env_state_dim"]),))
    outputs = {ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(int(features["action_dim"]),))}
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
    """``task.policy_overrides.<policy name>``: per-task policy values taken from the task's official policy configs (for
    example SoftMimicGen trains its image Diffusion Policy without a crop and its BC-RNN with a 76x76 crop on 128x128
    images; Isaac Lab's GR1 image policies crop to 90% of the side). Empty when the task lists none."""
    blocks = (task or {}).get("policy_overrides") or {}
    return dict(blocks.get(policy_name) or {})


def build_config(policy_cfg: dict, features: dict, device: str, observation: str = "image", overrides: dict | None = None) -> PreTrainedConfig:
    """The LeRobot config object of ``policy_cfg['name']`` with our values, the task's overrides and the dataset features."""
    spec = policy_spec(str(policy_cfg["name"]))
    inputs, outputs = policy_features(features)
    params = resolve_crop({**policy_params(policy_cfg, observation), **(overrides or {})}, features)
    for key in spec.tuple_params:
        if key in params and params[key] is not None:
            params[key] = tuple(params[key])
    return _load_target(spec.config_class)(input_features=inputs, output_features=outputs, device=device, **params)


def policy_class(name: str):
    """The ``PreTrainedPolicy`` subclass of a policy name."""
    return _load_target(policy_spec(name).policy_class)


def build_policy(policy_cfg: dict, features: dict, device: str, observation: str = "image", overrides: dict | None = None):
    """A freshly initialized policy (or, for π0, the pretrained hub weights named by ``params.pretrained_path``)."""
    name = str(policy_cfg["name"])
    config = build_config(policy_cfg, features, device, observation, overrides)
    cls = policy_class(name)
    pretrained = (policy_cfg.get("pretrained_path") or None)
    policy = cls.from_pretrained(pretrained, config=config) if pretrained else cls(config)
    if policy_spec(name).per_sample_crop:
        from tmg.policy.augment import use_per_sample_crop

        use_per_sample_crop(policy)
    return policy.to(device)


def build_optimizer(name: str, config, policy):
    """The policy's optimizer: the config's LeRobot preset, or AdamW with the preset's learning rate, betas, eps and
    weight decay when the spec says ``adamw`` (the official Diffusion Policy's optimizer)."""
    preset = config.get_optimizer_preset()
    if policy_spec(name).optimizer != "adamw":
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
    folder = Path(checkpoint_dir) / PRETRAINED_DIR
    config = PreTrainedConfig.from_pretrained(str(folder))
    config.device = device
    cls = policy_class(policy_name_of(config))
    return cls.from_pretrained(str(folder), config=config).to(device)


def policy_name_of(config: PreTrainedConfig) -> str:
    """Our policy name of a LeRobot config type (``config.json``'s ``type``)."""
    for spec in POLICY_SPECS.values():
        if spec.config_type == config.type:
            return spec.name
    raise ValueError(f"checkpoint config type {config.type!r} is not one of ours")
