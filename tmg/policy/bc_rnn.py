"""BC-RNN in LeRobot's ``PreTrainedPolicy`` format.

LeRobot 0.6.1 has no BC-RNN, so this module wraps ``tmg.policy.nets.BCRNNNet`` in the interface the training
loop and the policy server use for every policy: ``forward(batch) -> (loss, info)`` on sequence batches and
``select_action(batch) -> action`` on single-step batches. ``save_pretrained`` / ``from_pretrained`` come from
the base class (``config.json`` + ``model.safetensors``).

Batch layout (LeRobot keys, produced by ``tmg.data.loader`` with the delta indices below):

    observation.images.<cam>   (B, T, 3, H, W) float in [0, 1]      image form only
    observation.state          (B, T, D)                              robot state
    observation.environment_state (B, T, E)                           low-dim form only: object state, concatenated
                                                                      to the robot state before the LSTM
    action                     (B, T, A)

with ``T = seq_length`` (10) consecutive steps starting at the sampled step, edge-padded at the end of a demo
as robomimic's ``pad_seq_length`` does. The rollout keeps the LSTM state across calls and reinitializes it
every ``seq_length`` steps (robomimic ``BC_RNN.get_action`` with ``rnn.horizon``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from lerobot.configs import NormalizationMode, PreTrainedConfig
from lerobot.optim import AdamConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from torch import Tensor

from tmg.core.features import ACTION, ENV_STATE, STATE, camera_of, image_feature
from tmg.policy.nets import BCRNNNet, BCRNNNetConfig

POLICY_TYPE = "tmg_bc_rnn"


@PreTrainedConfig.register_subclass(POLICY_TYPE)
@dataclass
class BCRNNConfig(PreTrainedConfig):
    """BC-RNN hyperparameters (MimicGen's BC-RNN image settings; see ``tmg.policy.nets``)."""

    seq_length: int = 10                     # training sequence length and rollout horizon of the hidden state
    crop_shape: tuple[int, int] | None = (76, 76)   # None: no crop (the Isaac Lab tasks' policy_overrides)
    num_keypoints: int = 32
    visual_feature_dim: int = 64
    rnn_hidden_dim: int = 1000
    rnn_num_layers: int = 2
    num_modes: int = 5
    min_std: float = 1e-4
    std_activation: str = "softplus"
    low_noise_eval: bool = True
    normalization_mapping: dict[str, NormalizationMode] = field(default_factory=lambda: {
        "VISUAL": NormalizationMode.IDENTITY, "STATE": NormalizationMode.IDENTITY, "ACTION": NormalizationMode.IDENTITY})
    optimizer_lr: float = 1e-4

    def __post_init__(self):
        super().__post_init__()
        if self.seq_length < 1:
            raise ValueError("seq_length must be >= 1")

    def get_optimizer_preset(self) -> AdamConfig:
        """Adam, lr 1e-4, no weight decay (robomimic ``optim_params.policy``)."""
        return AdamConfig(lr=self.optimizer_lr, weight_decay=0.0)

    def get_scheduler_preset(self):
        return None

    def validate_features(self) -> None:
        if not self.image_features and self.env_state_feature is None:
            raise ValueError("BC-RNN needs observation.images.* (image form) or observation.environment_state (low-dim form)")
        if self.robot_state_feature is None:
            raise ValueError("BC-RNN needs observation.state")
        if self.action_feature is None:
            raise ValueError("BC-RNN needs the action feature")

    @property
    def observation_delta_indices(self) -> list:
        return list(range(self.seq_length))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.seq_length))

    @property
    def reward_delta_indices(self) -> None:
        return None


class BCRNNPolicy(PreTrainedPolicy):
    """LeRobot-format BC-RNN (see the module docstring)."""

    config_class = BCRNNConfig
    name = POLICY_TYPE

    def __init__(self, config: BCRNNConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        cameras = tuple(camera_of(k) for k in config.image_features)
        shape = next(iter(config.image_features.values())).shape if config.image_features else None
        env_dim = int(config.env_state_feature.shape[0]) if config.env_state_feature is not None else 0
        self.net = BCRNNNet(BCRNNNetConfig(
            cameras=cameras, image_shape=tuple(int(v) for v in shape) if shape is not None else None,
            state_dim=int(config.robot_state_feature.shape[0]) + env_dim,
            action_dim=int(config.action_feature.shape[0]),
            crop_shape=tuple(config.crop_shape) if config.crop_shape is not None else None,
            num_keypoints=config.num_keypoints, visual_feature_dim=config.visual_feature_dim,
            rnn_hidden_dim=config.rnn_hidden_dim, rnn_num_layers=config.rnn_num_layers, num_modes=config.num_modes,
            min_std=config.min_std, std_activation=config.std_activation, low_noise_eval=config.low_noise_eval))
        self.reset()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def _save_pretrained(self, save_directory, state_dict=None) -> None:
        """Write ``model.safetensors`` from a cloned state dict. On CUDA, cuDNN packs the LSTM weights into one
        shared buffer; safetensors' ``save_model`` refuses that (no single tensor covers the storage), so each
        tensor is cloned into its own storage first. ``from_pretrained`` (``load_model``) reads the file as usual."""
        from pathlib import Path

        from safetensors.torch import save_file

        save_directory = Path(save_directory)
        self.config._save_pretrained(save_directory)                 # config.json, as LeRobot's own _save_pretrained does
        state = {k: v.detach().cpu().contiguous().clone() for k, v in (state_dict or self.state_dict()).items()}
        save_file(state, str(save_directory / "model.safetensors"), metadata={"format": "pt"})

    def reset(self) -> None:
        """Forget the rollout state (call at the start of every episode)."""
        self._rnn_state = None
        self._counter = 0

    def _images(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        return {cam: batch[image_feature(cam)] for cam in self.net.cfg.cameras}

    def _state(self, batch: dict[str, Tensor]) -> Tensor:
        """Robot state, with the object state appended on the last axis in the low-dim form."""
        state = batch[STATE]
        if self.config.env_state_feature is not None:
            state = torch.cat([state, batch[ENV_STATE]], dim=-1)
        return state

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Negative log-likelihood of the action sequence; ``info`` carries the loss as a float."""
        loss = self.net.loss(self._images(batch), self._state(batch), batch[ACTION])
        return loss, {"nll": float(loss.detach())}

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """One action (B, A) for a single-step batch; the hidden state rolls over ``seq_length`` steps."""
        self.eval()
        state = self._state(batch)
        if self._rnn_state is None or self._counter % self.config.seq_length == 0:
            self._rnn_state = self.net.init_state(state.shape[0], state.device)
        self._counter += 1
        action, self._rnn_state = self.net.forward_step(self._images(batch), state, self._rnn_state)
        return action

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs) -> Tensor:
        """BC-RNN predicts one step at a time; the chunk is that step with a length-1 time axis."""
        return self.select_action(batch)[:, None, :]
