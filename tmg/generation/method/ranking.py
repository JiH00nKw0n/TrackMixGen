"""Source rankings: which source demonstrations generate well, ordered, per task and per subtask.

A ranking file ``configs/experiment/generator/<generator>/ranking/<task>.yaml`` (next to the generator config,
because each generator has its own source demonstrations) stores only an ORDER of source indices (highest
generation success rate first) with the measurements behind it (attempts, successes, rate), either once for
the whole task (``default``) or per subtask index (``subtasks``). It does not store a good/bad split: the
method config decides at run time how many top sources count as high-success (``good_fraction`` or
``good_count``), so the split can change without touching the measured ranking.

``ranking_from_attempts`` builds such a ranking from the ``attempts.jsonl`` records of a baseline pool: for
every subtask, the source selected for that subtask is credited with the attempt's full-task success. This is
generator-independent because every adapter writes the same record format (``tmg.core.records``).
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from tmg import CONFIGS_ROOT

GENERATOR_DIR = "experiment/generator"
RANKING_DIR = "ranking"


@dataclass(frozen=True)
class RankedSources:
    """One ordered list of sources with the measurements behind it."""

    order: tuple[int, ...]
    attempts: dict[int, int] = field(default_factory=dict)
    successes: dict[int, int] = field(default_factory=dict)
    rate: dict[int, float] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, m: Mapping, where: str) -> RankedSources:
        """Validate and build from a YAML mapping (``order`` required; counts optional)."""
        if "order" not in m or not m["order"]:
            raise KeyError(f"{where}: ranking block lacks a non-empty 'order'")
        order = tuple(int(v) for v in m["order"])
        if len(set(order)) != len(order):
            raise ValueError(f"{where}: ranking order repeats a source")
        ints = {k: {int(a): int(b) for a, b in (m.get(k) or {}).items()} for k in ("attempts", "successes")}
        rate = {int(a): float(b) for a, b in (m.get("rate") or {}).items()}
        return cls(order=order, attempts=ints["attempts"], successes=ints["successes"], rate=rate)

    def to_mapping(self) -> dict:
        """Plain mapping in the file layout (order first, then the counts in that order)."""
        out: dict = {"order": list(self.order)}
        for key in ("attempts", "successes", "rate"):
            values = getattr(self, key)
            if values:
                out[key] = {s: values[s] for s in self.order if s in values}
        return out

    def split(self, *, good_fraction: float | None = None, good_count: int | None = None) -> tuple[set[int], set[int]]:
        """Top sources as ``good`` and the rest as ``bad``.

        Args:
            good_fraction: share of the sources counted as good, rounded up (``ceil(n * fraction)``).
            good_count: explicit number of top sources; takes precedence over ``good_fraction``.

        Returns:
            ``(good, bad)`` sets of source indices; both non-empty, or ``ValueError``.
        """
        n = len(self.order)
        if good_count is None:
            if good_fraction is None:
                raise ValueError("a split needs good_fraction or good_count")
            if not 0 < float(good_fraction) < 1:
                raise ValueError(f"good_fraction must lie strictly between 0 and 1, got {good_fraction}")
            good_count = int(math.ceil(n * float(good_fraction)))
        if not 0 < good_count < n:
            raise ValueError(f"good_count {good_count} must leave both groups non-empty for {n} sources")
        return set(self.order[:good_count]), set(self.order[good_count:])


@dataclass(frozen=True)
class SourceRanking:
    """A task's ranking: a ``default`` order and/or per-subtask orders, with provenance text."""

    name: str                               # "<generator>/<task>"
    generator: str
    task: str
    basis: str
    default: RankedSources | None
    subtasks: dict[int, RankedSources]
    path: str

    def for_subtask(self, subtask_ind: int) -> RankedSources:
        """The order of ``subtask_ind``, else the task-wide default."""
        if subtask_ind in self.subtasks:
            return self.subtasks[subtask_ind]
        if self.default is None:
            raise KeyError(f"ranking {self.name}: no order for subtask {subtask_ind} and no default")
        return self.default

    def describe(self) -> dict:
        """JSON-able summary for logs and provenance."""
        return {"name": self.name, "path": self.path, "basis": self.basis,
                "default": list(self.default.order) if self.default else None,
                "subtasks": {k: list(v.order) for k, v in sorted(self.subtasks.items())}}


def ranking_path(name: str, root: Path | None = None) -> Path:
    """``configs/experiment/generator/<generator>/ranking/<task>.yaml`` for a ``<generator>/<task>`` name."""
    generator, _, task = name.partition("/")
    if not task:
        raise ValueError(f"ranking name must be <generator>/<task>, got {name!r}")
    return Path(root or CONFIGS_ROOT) / GENERATOR_DIR / generator / RANKING_DIR / f"{task}.yaml"


def load_ranking(name: str, root: Path | None = None) -> SourceRanking:
    """Read and validate a ranking file by its ``<generator>/<task>`` name."""
    path = ranking_path(name, root)
    if not path.exists():
        raise FileNotFoundError(f"source ranking {path} not found; run `tmg rank_sources` on a baseline pool first")
    data = yaml.safe_load(path.read_text()) or {}
    default = RankedSources.from_mapping(data["default"], f"{path}:default") if data.get("default") else None
    subtasks = {int(k): RankedSources.from_mapping(v, f"{path}:subtasks/{k}") for k, v in (data.get("subtasks") or {}).items()}
    if default is None and not subtasks:
        raise KeyError(f"{path}: a ranking needs 'default' or 'subtasks'")
    return SourceRanking(name=name, generator=str(data.get("generator", "")), task=str(data.get("task", "")),
                         basis=str(data.get("basis", "")), default=default, subtasks=subtasks, path=str(path))


def ranking_from_attempts(records: Iterable[Mapping], *, num_sources: int | None = None) -> dict[int, RankedSources]:
    """Per-subtask rankings from ``attempts.jsonl`` lines.

    Args:
        records: attempt dicts with ``success`` and ``src_demo_inds`` (source selected for each subtask).
        num_sources: when given, sources never selected are appended at the end with zero counts.

    Returns:
        ``{subtask_ind: RankedSources}`` ordered by success rate (ties: lower source index first).
    """
    counts: dict[int, dict[int, list[int]]] = {}
    for r in records:
        inds = r.get("src_demo_inds") or []
        ok = int(bool(r.get("success", False)))
        for subtask_ind, src in enumerate(inds):
            c = counts.setdefault(int(subtask_ind), {}).setdefault(int(src), [0, 0])
            c[0] += 1
            c[1] += ok
    out = {}
    for subtask_ind, by_src in sorted(counts.items()):
        if num_sources is not None:
            for s in range(int(num_sources)):
                by_src.setdefault(s, [0, 0])
        rate = {s: (v[1] / v[0] if v[0] else 0.0) for s, v in by_src.items()}
        order = tuple(sorted(by_src, key=lambda s: (-rate[s], s)))
        out[subtask_ind] = RankedSources(order=order, attempts={s: v[0] for s, v in by_src.items()},
                                         successes={s: v[1] for s, v in by_src.items()},
                                         rate={s: round(rate[s], 4) for s in by_src})
    return out


def write_ranking(path: Path, *, generator: str, task: str, basis: str, subtasks: Mapping[int, RankedSources],
                  default: RankedSources | None = None) -> Path:
    """Write a ranking file in the layout ``load_ranking`` reads (comment header, then the mapping)."""
    doc: dict = {"generator": generator, "task": task, "basis": basis,
                 "split_note": "The ranking only orders the sources. Which prefix counts as high-success is "
                               "method.mixing.good_fraction."}
    if default is not None:
        doc["default"] = default.to_mapping()
    if subtasks:
        doc["subtasks"] = {int(k): v.to_mapping() for k, v in sorted(subtasks.items())}
    header = (f"# Source ranking of {generator} {task}. Sources ordered from the highest to the lowest full-task generation\n"
              "# success rate per subtask (ties: lower source index first). The good/bad split is not stored here; the\n"
              "# method config chooses how many top sources count as high-success (method.mixing.good_fraction).\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(doc, sort_keys=False, default_flow_style=None, width=120))
    return path
