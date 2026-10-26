"""Chunk plan of a pool and the on-disk markers that make a pool resumable.

A pool is generated in chunks; a chunk is one generator process with one seed and one attempt count.
Chunk folder: ``<pool>/chunks/seed<seed>/``. A chunk is complete when its ``CHUNK_DONE.json`` exists
(written by the runner after it verified the generator's own completion marker), and a pool is complete
when ``DONE.json`` exists at the pool root.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

CHUNKS_DIR = "chunks"
CHUNK_DONE = "CHUNK_DONE.json"
CHUNK_SPEC = "chunk_spec.json"
POOL_DONE = "DONE.json"
POOL_PLAN = "plan.json"
RESOLVED_CONFIG = "resolved_config.yaml"
LOGS_DIR = "logs"


@dataclass(frozen=True)
class Chunk:
    """One generator process worth of work: ``attempts`` attempts under one random ``seed``.

    Chunks of a pool are numbered by ``index`` and get consecutive seeds (``seed_offset + index + 1``), so the
    same pool config always yields the same chunk plan and a pool can be resumed chunk by chunk.
    """
    index: int
    seed: int
    attempts: int

    @property
    def name(self) -> str:
        """Folder name of the chunk under ``<pool>/chunks/`` (``seed<seed>``)."""
        return f"seed{self.seed}"

    def folder(self, pool: Path) -> Path:
        """Absolute folder of this chunk inside ``pool``."""
        return Path(pool) / CHUNKS_DIR / self.name


def plan_chunks(attempts: int, chunk: int, seed_offset: int) -> list[Chunk]:
    """Split ``attempts`` attempts into chunks of at most ``chunk`` attempts.

    Args:
        attempts: total attempts of the pool (>= 1).
        chunk: attempts per chunk (>= 1); the last chunk may be smaller.
        seed_offset: chunk i gets seed ``seed_offset + i + 1``.

    Returns:
        The chunks in index order.
    """
    if attempts < 1 or chunk < 1:
        raise ValueError(f"attempts and chunk must be >= 1, got {attempts}, {chunk}")
    out = []
    remaining = int(attempts)
    index = 0
    while remaining > 0:
        n = min(int(chunk), remaining)
        out.append(Chunk(index=index, seed=int(seed_offset) + index + 1, attempts=n))
        remaining -= n
        index += 1
    return out


def write_json(path: Path, data) -> Path:
    """Write ``data`` as pretty JSON atomically (temp file then rename) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return path


def read_json(path: Path):
    """Load a JSON file."""
    return json.loads(Path(path).read_text())


def chunk_is_done(folder: Path) -> bool:
    """True when the runner wrote the chunk's completion marker (``CHUNK_DONE.json``)."""
    return (Path(folder) / CHUNK_DONE).is_file()


def clear_partial_chunk(folder: Path) -> bool:
    """Delete a chunk folder that has no completion marker (a rerun restarts the chunk from attempt 0)."""
    folder = Path(folder)
    if folder.exists() and not chunk_is_done(folder):
        shutil.rmtree(folder)
        return True
    return False


def plan_as_dict(chunks: list[Chunk]) -> list[dict]:
    """The chunk plan as a JSON-able list of dicts (index, seed, attempts)."""
    return [asdict(c) for c in chunks]
