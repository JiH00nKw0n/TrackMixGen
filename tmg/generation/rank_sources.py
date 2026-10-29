"""``tmg rank_sources``: measure each source demonstration's generation success in a baseline pool and write
the task's ranking file.

Input: a finished pool of the configured (generator, task, scene, generation) with ``method=none`` (the pool
folder is the same as ``tmg generate``'s output for those choices). The stage reads every chunk's
``records/attempts.jsonl``, credits the source selected for each subtask with the attempt's full-task success,
orders the sources per subtask (``tmg.generation.method.ranking.ranking_from_attempts``) and writes:

- ``<output>/source_ranking.yaml`` and ``<output>/RANK_SOURCES.json`` (summary with the chunk list);
- ``configs/experiment/generator/<generator>/ranking/<task>.yaml`` when ``write_config`` is true, so the ranking becomes part of the
  committed configuration that ``method.mixing.ranking`` names.

The stage is generator-independent; it only needs the common attempt records.
"""
from __future__ import annotations

from pathlib import Path

from tmg.config import dump_yaml, resolved
from tmg.core.chunks import CHUNKS_DIR, POOL_DONE, RESOLVED_CONFIG, chunk_is_done, read_json, write_json
from tmg.core.provenance import write_provenance
from tmg.core.records import RECORDS_DIR, read_attempts
from tmg.generation.method.ranking import ranking_from_attempts, ranking_path, write_ranking

RANKING_FILE = "source_ranking.yaml"
SUMMARY = "RANK_SOURCES.json"


def rank_sources(cfg) -> dict:
    """Run the stage described in the module docstring.

    Args:
        cfg: the composed ``rank_sources`` config (``pool``, ``output``, ``write_config``, ``min_attempts``).

    Returns:
        The summary written to ``RANK_SOURCES.json`` (chunks read, attempts, per-subtask orders, files written).

    Raises:
        FileNotFoundError: the pool has no finished chunk with records.
        ValueError: fewer attempts than ``min_attempts``.
    """
    plain = resolved(cfg)
    pool = Path(plain["pool"])
    output = Path(plain["output"])
    output.mkdir(parents=True, exist_ok=True)
    dump_yaml(plain, output / RESOLVED_CONFIG)
    write_provenance(output, extra={"command": "rank_sources", "output": str(output), "pool": str(pool)})

    chunks = sorted(p for p in (pool / CHUNKS_DIR).glob("seed*") if chunk_is_done(p))
    records = []
    for chunk in chunks:
        records.extend(read_attempts(chunk / RECORDS_DIR))
    if not records:
        raise FileNotFoundError(f"no finished chunk with attempt records under {pool}")
    if len(records) < int(plain["min_attempts"]):
        raise ValueError(f"{len(records)} attempts in {pool}, fewer than min_attempts={plain['min_attempts']}")
    task, generator = plain["task"], plain["generator"]
    subtasks = ranking_from_attempts(records, num_sources=task.get("num_sources"))
    basis = (f"Full-task generation success rate (successes / attempts) of each source demonstration, per subtask, "
             f"measured by tmg rank_sources on the baseline pool {pool} ({len(records)} attempts, "
             f"{len(chunks)} chunks, scene {plain['scene']['name']}, generation {plain['generation']['name']}). "
             f"The source of a subtask is the demonstration the generator selected for that subtask.")
    written = [str(write_ranking(output / RANKING_FILE, generator=generator["name"], task=task["name"], basis=basis,
                                 subtasks=subtasks))]
    if plain.get("write_config", True):
        written.append(str(write_ranking(ranking_path(f"{generator['name']}/{task['name']}"), generator=generator["name"],
                                         task=task["name"], basis=basis, subtasks=subtasks)))
    done = read_json(pool / POOL_DONE) if (pool / POOL_DONE).exists() else None
    summary = {"pool": str(pool), "pool_done": done is not None, "chunks": [c.name for c in chunks], "attempts": len(records),
               "successes": int(sum(bool(r.get("success")) for r in records)),
               "subtasks": {k: {"order": list(v.order), "rate": v.rate, "attempts": v.attempts} for k, v in subtasks.items()},
               "written": written}
    write_json(output / SUMMARY, summary)
    print(f"[tmg] rank_sources {generator['name']}/{task['name']}: {len(records)} attempts from {len(chunks)} chunks")
    for k, v in subtasks.items():
        print(f"[tmg]   subtask {k}: order {list(v.order)} rates {[v.rate[s] for s in v.order]}")
    print(f"[tmg] written: {', '.join(written)}")
    return summary
