"""Torch modules of the BC-RNN port (robomimic ``BC_RNN_GMM``, MimicGen's image settings or robomimic's low-dim ones).

The structure follows robomimic d309eae exactly so that the model is the one the MimicGen paper trained:

- per camera a ``VisualCore``: ``CropRandomizer`` (76x76 random crop of the 84x84 image in training, centre
  crop in evaluation), ResNet18 convolutional trunk without the final pooling and classifier (not
  pretrained), ``SpatialSoftmax`` with 32 keypoints (temperature 1, no keypoint noise), flatten, Linear to a
  64-dimensional feature followed by ReLU (robomimic's observation encoder activation);
- the camera features and the low-dimensional state are concatenated per step and fed to an LSTM
  (2 layers, hidden 1000 for images, 400 low-dim); without cameras (low-dim) the state alone (robot state plus
  object state, concatenated by the policy) is the LSTM input, as in robomimic's bc_rnn low-dim template;
- a Gaussian-mixture head per step: 5 modes, means through tanh, scales ``softplus(x) + min_std`` with
  ``min_std`` 1e-4, mixture logits; the training loss is the negative mean log-likelihood of the action;
- in evaluation (``low_noise_eval``) the scales are fixed to 1e-4 and the action is a sample of the mixture,
  which is one mode's mean up to that tiny noise, as robomimic does.

These modules take already normalized float images in [0, 1] with layout (B, T, 3, H, W).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributions as D
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class BCRNNNetConfig:
    """Shapes and sizes of the BC-RNN network (values from MimicGen's BC-RNN image settings)."""

    cameras: tuple[str, ...]                 # () for the low-dim model
    image_shape: tuple[int, int, int] | None # (3, H, W) of the dataset images; None without cameras
    state_dim: int                           # robot state (plus object state, low-dim) width
    action_dim: int
    crop_shape: tuple[int, int] = (76, 76)
    num_keypoints: int = 32
    visual_feature_dim: int = 64
    rnn_hidden_dim: int = 1000
    rnn_num_layers: int = 2
    num_modes: int = 5
    min_std: float = 1e-4
    std_activation: str = "softplus"
    low_noise_eval: bool = True


class CropRandomizer(nn.Module):
    """Random crop in training, centre crop in evaluation (robomimic ``CropRandomizer`` with one crop)."""

    def __init__(self, crop_shape: tuple[int, int]):
        super().__init__()
        self.crop_h, self.crop_w = int(crop_shape[0]), int(crop_shape[1])

    def forward(self, images: Tensor) -> Tensor:
        """``images``: (N, C, H, W) -> (N, C, crop_h, crop_w)."""
        n, _, h, w = images.shape
        if self.training:
            top = torch.randint(0, h - self.crop_h, (n,), device=images.device)      # robomimic draws in [0, H - crop_h)
            left = torch.randint(0, w - self.crop_w, (n,), device=images.device)
            rows = top[:, None] + torch.arange(self.crop_h, device=images.device)[None, :]
            cols = left[:, None] + torch.arange(self.crop_w, device=images.device)[None, :]
            return images[torch.arange(n, device=images.device)[:, None, None, None], torch.arange(images.shape[1], device=images.device)[None, :, None, None],
                          rows[:, None, :, None], cols[:, None, None, :]]
        top, left = (h - self.crop_h) // 2, (w - self.crop_w) // 2
        return images[:, :, top:top + self.crop_h, left:left + self.crop_w]


class SpatialSoftmax(nn.Module):
    """Expected (x, y) coordinate of each of ``num_kp`` softmax attention maps (robomimic ``SpatialSoftmax``)."""

    def __init__(self, in_channels: int, height: int, width: int, num_kp: int = 32, temperature: float = 1.0):
        super().__init__()
        self.kp = nn.Conv2d(in_channels, num_kp, kernel_size=1)
        self.num_kp, self.h, self.w = num_kp, height, width
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        ys, xs = torch.meshgrid(torch.linspace(-1.0, 1.0, height), torch.linspace(-1.0, 1.0, width), indexing="ij")
        self.register_buffer("pos_x", xs.reshape(1, height * width))
        self.register_buffer("pos_y", ys.reshape(1, height * width))

    def forward(self, features: Tensor) -> Tensor:
        """``features``: (N, C, h, w) -> keypoints (N, num_kp, 2)."""
        x = self.kp(features).reshape(-1, self.h * self.w)
        attention = F.softmax(x / self.temperature, dim=-1)
        ex = (self.pos_x * attention).sum(-1, keepdim=True)
        ey = (self.pos_y * attention).sum(-1, keepdim=True)
        return torch.cat([ex, ey], dim=1).view(-1, self.num_kp, 2)


class VisualCore(nn.Module):
    """Crop -> ResNet18 trunk -> spatial softmax -> flatten -> linear feature (robomimic ``VisualCore``)."""

    def __init__(self, image_shape: tuple[int, int, int], crop_shape: tuple[int, int], num_kp: int, feature_dim: int):
        super().__init__()
        from torchvision.models import resnet18

        self.crop = CropRandomizer(crop_shape)
        trunk = resnet18(weights=None)
        self.trunk = nn.Sequential(*list(trunk.children())[:-2])
        with torch.no_grad():
            self.trunk.eval()                  # probe the output shape without touching BatchNorm statistics
            c, h, w = self.trunk(torch.zeros(1, image_shape[0], *crop_shape)).shape[1:]
            self.trunk.train()
        self.pool = SpatialSoftmax(c, h, w, num_kp=num_kp)
        self.out = nn.Linear(num_kp * 2, feature_dim)
        self.feature_dim = feature_dim

    def forward(self, images: Tensor) -> Tensor:
        """``images``: (N, 3, H, W) float in [0, 1] -> (N, feature_dim)."""
        x = self.crop(images)
        x = self.trunk(x)
        x = self.pool(x).flatten(1)
        return F.relu(self.out(x))          # robomimic's ObservationEncoder applies ReLU to every encoder output


class GMMHead(nn.Module):
    """Per-step Gaussian mixture over actions from the LSTM output (robomimic ``RNNGMMActorNetwork`` head)."""

    def __init__(self, in_dim: int, action_dim: int, num_modes: int, min_std: float, std_activation: str, low_noise_eval: bool):
        super().__init__()
        self.mean = nn.Linear(in_dim, num_modes * action_dim)
        self.scale = nn.Linear(in_dim, num_modes * action_dim)
        self.logits = nn.Linear(in_dim, num_modes)
        self.num_modes, self.action_dim = num_modes, action_dim
        self.min_std, self.low_noise_eval = float(min_std), bool(low_noise_eval)
        if std_activation not in ("softplus", "exp"):
            raise ValueError(f"std_activation must be softplus or exp, got {std_activation!r}")
        self.std_activation = std_activation

    def forward(self, h: Tensor) -> D.MixtureSameFamily:
        """``h``: (..., in_dim) -> mixture with batch shape (...) and event shape (action_dim,)."""
        means = torch.tanh(self.mean(h)).reshape(*h.shape[:-1], self.num_modes, self.action_dim)
        raw = self.scale(h).reshape(*h.shape[:-1], self.num_modes, self.action_dim)
        if self.low_noise_eval and not self.training:
            scales = torch.full_like(means, 1e-4)
        else:
            scales = (F.softplus(raw) if self.std_activation == "softplus" else torch.exp(raw)) + self.min_std
        components = D.Independent(D.Normal(means, scales), 1)
        return D.MixtureSameFamily(D.Categorical(logits=self.logits(h)), components)


class BCRNNNet(nn.Module):
    """The whole BC-RNN model: per-camera encoders, LSTM over the sequence, GMM head per step."""

    def __init__(self, cfg: BCRNNNetConfig):
        super().__init__()
        self.cfg = cfg
        self.encoders = nn.ModuleDict({cam: VisualCore(cfg.image_shape, cfg.crop_shape, cfg.num_keypoints, cfg.visual_feature_dim)
                                       for cam in cfg.cameras})
        in_dim = cfg.visual_feature_dim * len(cfg.cameras) + cfg.state_dim
        self.rnn = nn.LSTM(in_dim, cfg.rnn_hidden_dim, num_layers=cfg.rnn_num_layers, batch_first=True)
        self.head = GMMHead(cfg.rnn_hidden_dim, cfg.action_dim, cfg.num_modes, cfg.min_std, cfg.std_activation, cfg.low_noise_eval)

    def encode(self, images: dict[str, Tensor], state: Tensor) -> Tensor:
        """Per-step input vector: ``images[cam]`` (B, T, 3, H, W), ``state`` (B, T, D) -> (B, T, in_dim)."""
        b, t = state.shape[:2]
        feats = []
        for cam in self.cfg.cameras:
            x = images[cam].reshape(b * t, *images[cam].shape[2:])
            feats.append(self.encoders[cam](x).reshape(b, t, -1))
        feats.append(state)
        return torch.cat(feats, dim=-1)

    def init_state(self, batch_size: int, device) -> tuple[Tensor, Tensor]:
        """Zero LSTM state (h, c), each (num_layers, B, hidden)."""
        shape = (self.cfg.rnn_num_layers, batch_size, self.cfg.rnn_hidden_dim)
        return torch.zeros(shape, device=device), torch.zeros(shape, device=device)

    def forward_train(self, images: dict[str, Tensor], state: Tensor) -> D.MixtureSameFamily:
        """Mixture per step of a sequence batch: batch shape (B, T)."""
        x = self.encode(images, state)
        h, _ = self.rnn(x)
        return self.head(h)

    def forward_step(self, images: dict[str, Tensor], state: Tensor, rnn_state: tuple[Tensor, Tensor]):
        """One control step: inputs without the time axis, returns (action (B, A), new rnn state)."""
        x = self.encode({k: v[:, None] for k, v in images.items()}, state[:, None])
        h, rnn_state = self.rnn(x, rnn_state)
        dist = self.head(h[:, 0])
        return dist.sample(), rnn_state

    def loss(self, images: dict[str, Tensor], state: Tensor, actions: Tensor) -> Tensor:
        """Negative mean log-likelihood of ``actions`` (B, T, A) under the per-step mixtures."""
        dist = self.forward_train(images, state)
        return -dist.log_prob(actions).mean()
