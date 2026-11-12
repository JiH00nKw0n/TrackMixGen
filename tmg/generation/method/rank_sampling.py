"""Rank-based source sampling (``sources.type: rank_based``): the top-1, top-3 and top-5 baselines.

``RankBasedSampling`` replaces the generator's choice of source demonstration by a uniform draw among the ``top_k``
best sources of the task's ranking file (``configs/experiment/generator/<generator>/ranking/<task>.yaml``, written by
``tmg rank_sources``; the logistic mixer reads the separate coefficient file instead). MimicGen asks once per subtask where
it selects a source (every subtask with ``selection=ours``), so the order of that subtask is used. The generator's own
selection still runs first, so its random draws are taken as before; only its result is replaced. Each draw comes from the
private stream ``SeedSequence([chunk_seed, call, seed_offset])``, so a chunk is reproducible.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from tmg.generation.method.ranking import SourceRanking, load_ranking

DEFAULT_SEED_OFFSET = 424242


class RankBasedSampling:
    """Uniform choice among the ``top_k`` best-ranked sources of a subtask (see the module docstring)."""

    def __init__(self, *, ranking: SourceRanking, top_k: int, chunk_seed: int, seed_offset: int = DEFAULT_SEED_OFFSET):
        if int(top_k) < 1:
            raise ValueError(f"sources.top_k must be at least 1, got {top_k}")
        self.ranking = ranking
        self.top_k = int(top_k)
        self.chunk_seed = int(chunk_seed)
        self.seed_offset = int(seed_offset)
        self.calls = 0

    @classmethod
    def from_config(cls, cfg: Mapping, *, chunk_seed: int) -> RankBasedSampling:
        """Build from the ``method.sources`` block (``ranking``: ``<generator>/<task>``, ``top_k``, optional ``seed_offset``)."""
        return cls(ranking=load_ranking(str(cfg["ranking"])), top_k=int(cfg["top_k"]), chunk_seed=chunk_seed,
                   seed_offset=int(cfg.get("seed_offset", DEFAULT_SEED_OFFSET)))

    def allowed(self, subtask_ind: int) -> list[int]:
        """The ``top_k`` best sources of ``subtask_ind``, best first."""
        return [int(i) for i in self.ranking.for_subtask(subtask_ind).order[: self.top_k]]

    def pick(self, subtask_ind: int) -> int:
        """One source of ``allowed(subtask_ind)``, drawn uniformly from the private stream."""
        allowed = self.allowed(subtask_ind)
        rng = np.random.default_rng(np.random.SeedSequence([self.chunk_seed, self.calls, self.seed_offset]))
        self.calls += 1
        return allowed[int(rng.integers(len(allowed)))]

    def describe(self) -> dict:
        return {"type": "rank_based", "top_k": self.top_k, "ranking": self.ranking.describe()}


def rank_based_sampling(cfg: Mapping, *, chunk_seed: int) -> RankBasedSampling:
    """Function form of ``RankBasedSampling.from_config``."""
    return RankBasedSampling.from_config(cfg, chunk_seed=chunk_seed)
