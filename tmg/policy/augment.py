"""Image augmentation that matches the official Diffusion Policy where LeRobot differs (train image only).

The official Diffusion Policy (real-stanford/diffusion_policy, robomimic ``CropRandomizer``) crops every image of a
training batch at its own random location. LeRobot's ``DiffusionRgbEncoder`` uses one
``torchvision.transforms.RandomCrop`` on the whole ``(batch * steps * cameras)`` tensor, which draws a single location
for the batch. ``use_per_sample_crop`` swaps that module for ``PerSampleRandomCrop`` after the policy is built.
Evaluation is unchanged (LeRobot center-crops in eval mode), and the module has no weights, so checkpoints keep their
format.
"""
from __future__ import annotations

import torch
from torch import nn


class PerSampleRandomCrop(nn.Module):
    """Random ``size`` crop of every image of an ``(N, C, H, W)`` tensor at its own location."""

    def __init__(self, size):
        super().__init__()
        self.h, self.w = (int(v) for v in size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, _, height, width = x.shape
        if height < self.h or width < self.w:
            raise ValueError(f"crop {self.h}x{self.w} larger than the image {height}x{width}")
        top = torch.randint(0, height - self.h + 1, (n,), device=x.device)
        left = torch.randint(0, width - self.w + 1, (n,), device=x.device)
        rows = top[:, None] + torch.arange(self.h, device=x.device)            # (n, h)
        cols = left[:, None] + torch.arange(self.w, device=x.device)           # (n, w)
        batch = torch.arange(n, device=x.device)[:, None, None]
        out = x[batch, :, rows[:, :, None], cols[:, None, :]]                   # (n, h, w, C): advanced indices first
        return out.permute(0, 3, 1, 2).contiguous()


def use_per_sample_crop(policy: nn.Module) -> int:
    """Replace every image encoder's batch-wide ``RandomCrop`` (``maybe_random_crop``) by ``PerSampleRandomCrop`` of the
    same size; returns the number of encoders changed (0 when the policy has no random crop)."""
    changed = 0
    for module in policy.modules():
        crop = getattr(module, "maybe_random_crop", None)
        if crop is not None and type(crop).__name__ == "RandomCrop" and getattr(module, "do_crop", False):
            size = crop.size if isinstance(crop.size, (tuple, list)) else (crop.size, crop.size)
            module.maybe_random_crop = PerSampleRandomCrop(size)
            changed += 1
    return changed
