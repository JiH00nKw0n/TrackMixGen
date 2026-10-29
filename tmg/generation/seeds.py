"""Derived random streams of one chunk, named in one place so every generator derives them the same way.

A chunk is one generation process with its own seed (``chunk.seed``). The official generators draw their own randomness from
that seed; the streams below are ours and must not change value, because finished pools were produced with them.
"""
from __future__ import annotations

import numpy as np

PRIVATE_SEED_OFFSET = 7919      # chunk_seed + offset seeds the private noise stream (tracking noise that the official code does not draw)
SOURCE_PICK_KEY = 515151        # spawn key of the per-attempt source-demonstration pick (selection=ours)


def private_noise_seed(chunk_seed: int) -> int:
    """Seed of the private noise stream of a chunk."""
    return int(chunk_seed) + PRIVATE_SEED_OFFSET


def source_pick_rng(chunk_seed: int) -> np.random.Generator:
    """Generator that picks the source demonstration of every attempt when all sources are used."""
    return np.random.default_rng(np.random.SeedSequence([int(chunk_seed), SOURCE_PICK_KEY]))
