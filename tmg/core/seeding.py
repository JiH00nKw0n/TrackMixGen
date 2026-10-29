"""Seeding of every random source a process uses (``random``, numpy's global generator, torch and, optionally, a
simulator environment).

``numpy.random.seed`` accepts only 32-bit seeds, so the seed is reduced modulo 2**32 for it; the others take the
seed as given. torch is imported inside the function: this module is shared with images that have no torch, and
the seeding is only called where torch is present (training, Isaac Lab evaluation).
"""
from __future__ import annotations

import random

import numpy as np


def seed_all(seed: int, env=None) -> None:
    """Seed ``env`` (when given, through its ``seed`` method), torch, numpy and ``random`` with ``seed``."""
    import torch

    seed = int(seed)
    if env is not None:
        env.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    random.seed(seed)
