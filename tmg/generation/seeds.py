"""Derived random streams of one chunk, named in one place so every generator derives them the same way.

A chunk is one generation process with its own seed (``chunk.seed``). The official generators draw their own randomness from
that seed; the streams below are ours and must not change value, because finished pools were produced with them.
"""
from __future__ import annotations

import numpy as np

SOURCE_PICK_KEY = 515151        # spawn key of the per-attempt source-demonstration pick (selection=ours)


def source_pick_rng(chunk_seed: int) -> np.random.Generator:
    """Generator that picks the source demonstration of every attempt when all sources are used."""
    return np.random.default_rng(np.random.SeedSequence([int(chunk_seed), SOURCE_PICK_KEY]))
