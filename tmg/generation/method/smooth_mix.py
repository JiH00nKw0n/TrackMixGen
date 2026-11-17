"""Smoothing without mixing (question 6, experiment A of the mixup hypothesis report, 2026-11-07).

Blending several sources halves the second difference of the position actions (pre-noise "jerk": 0.112 for the
top-1 source in MimicGen Threading, 0.059-0.075 for the blended conditions) and also pulls grasp and insertion poses
towards the mean pose. ``SmoothMixer`` (``mixing.type: smooth``) separates the two: it keeps the selected source's
segment (no other source contributes) and only low-pass filters its target positions over time with a zero-phase
Gaussian kernel of ``sigma`` steps (``scipy.ndimage.gaussian_filter1d``, edges held by ``mode: nearest``). Target
rotations, the gripper commands and the segment start are unchanged. The record sets ``mixed: true`` because the
generators replace a segment only when the mixer says so (``donors: []`` marks that no other source contributed). The filter is linear in the positions, so
filtering before the generator's rigid transform equals filtering after it.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.ndimage import gaussian_filter1d

from tmg.generation.method.interfaces import MixContext, MixedSegment


class SmoothMixer:
    """See the module docstring."""

    def __init__(self, sigma: float):
        if sigma <= 0:
            raise ValueError(f"mixing.sigma must be positive, got {sigma}")
        self.sigma = float(sigma)

    @classmethod
    def from_config(cls, cfg: Mapping, *, chunk_seed: int) -> SmoothMixer:
        return cls(sigma=float(cfg.get("sigma", 2.0)))

    def describe(self) -> dict:
        return {"type": "smooth", "sigma": self.sigma}

    def mix(self, context: MixContext) -> MixedSegment:
        targets = np.array(context.primary_targets, dtype=float, copy=True)
        record = {"subtask": int(context.subtask_ind), "source_a": int(context.primary), "mixed": False, "smoothed": False}
        if len(targets) >= 3:
            targets[:, :3, 3] = gaussian_filter1d(targets[:, :3, 3], self.sigma, axis=0, mode="nearest")
            record.update(mixed=True, smoothed=True, sigma=self.sigma, donors=[])
        return MixedSegment(context.primary_eef_start, targets, context.primary_gripper, record)
