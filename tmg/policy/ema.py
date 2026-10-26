"""Exponential moving average of a model's parameters (Diffusion Policy evaluates the EMA weights).

Decay schedule of the official Diffusion Policy ``EMAModel``: at optimizer step ``s`` (counted from 1 after
``update_after_step``), ``decay = 1 - (1 + s / inv_gamma) ** (-power)`` clipped to ``[min_value, max_value]``;
defaults ``inv_gamma`` 1, ``power`` 0.75, ``min_value`` 0, ``max_value`` 0.9999. Buffers are copied, not averaged.
"""
from __future__ import annotations

import copy

import torch
from torch import nn


class EMA:
    """Keeps ``self.model``, an averaged copy of ``model`` updated after every optimizer step."""

    def __init__(self, model: nn.Module, *, inv_gamma: float = 1.0, power: float = 0.75, min_value: float = 0.0,
                 max_value: float = 0.9999, update_after_step: int = 0):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.inv_gamma, self.power = float(inv_gamma), float(power)
        self.min_value, self.max_value = float(min_value), float(max_value)
        self.update_after_step = int(update_after_step)
        self.step = 0

    def decay(self) -> float:
        """Decay used by the next ``update``."""
        s = max(0, self.step - self.update_after_step - 1)   # official EMAModel: evaluated before the step count grows
        if s <= 0:
            return 0.0
        value = 1.0 - (1.0 + s / self.inv_gamma) ** (-self.power)
        return float(min(max(value, self.min_value), self.max_value))

    @torch.no_grad()
    def update(self, model: nn.Module) -> float:
        """Blend ``model``'s parameters into the average; returns the decay used."""
        d = self.decay()
        self.step += 1
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            if p.requires_grad:
                ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
            else:
                ema_p.copy_(p)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)
        return d

    def state_dict(self) -> dict:
        return {"step": self.step, "model": self.model.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.step = int(state["step"])
        self.model.load_state_dict(state["model"])
