"""Pieces every generator adapter and runtime needs and none owns: the records summary of a finished chunk, the
demonstration count of an hdf5 file, and the ``RECORD_ERROR.json`` marker a runtime leaves when writing a record fails.

Only stdlib, numpy and h5py are imported here, so the runner (which has no official package installed) can use it.
"""
from __future__ import annotations

import json
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import h5py

from tmg.core.records import ATTEMPTS_JSONL, RECORD_ERROR, AttemptRecorder, read_attempts, validate_records


def count_hdf5_demos(path: Path) -> int:
    """Number of groups under ``data`` of an hdf5 file in robomimic layout (0 when the file has no ``data`` group)."""
    with h5py.File(path, "r") as f:
        return len(f["data"].keys()) if "data" in f else 0


def records_summary(records_dir: Path, attempts: int, *, check_ok: bool) -> dict | None:
    """The attempt lines of a chunk with their success and failure counts, once the chunk recorded every attempt.

    Args:
        records_dir: the chunk's ``records/`` folder.
        attempts: planned attempt count of the chunk.
        check_ok: whether the verdict of ``validate_records`` decides the result. MimicGen checks it: a chunk whose
            records are incomplete or unreadable (or that left ``RECORD_ERROR.json``) gives ``{"error": ...}``, and
            only a missing ``attempts.jsonl`` counts as "not finished". CP-Gen does not: the chunk is "not finished"
            while fewer than ``attempts`` lines exist, and the validation result is ignored once they do (kept as the
            adapters behaved before this helper existed; tightening it is a separate decision).

    Returns:
        None when the chunk has not finished, ``{"error": str}`` when ``check_ok`` and the records are bad, otherwise
        ``{"lines": [...], "successes": int, "failures": int}``.
    """
    records_dir = Path(records_dir)
    if check_ok:
        if not (records_dir / ATTEMPTS_JSONL).exists():
            return None
        checked = validate_records(records_dir, attempts)
        if not checked["ok"]:
            return {"error": checked["error"]}
        lines = checked["lines"]
    else:
        lines = read_attempts(records_dir)
        if len(lines) < int(attempts):
            return None
        validate_records(records_dir, attempts)
    successes = sum(1 for line in lines if line.get("success"))
    return {"lines": lines, "successes": successes, "failures": len(lines) - successes}


def write_record_error(records_dir: Path | None, *, attempt: int, error: BaseException) -> None:
    """Leave ``RECORD_ERROR.json`` in ``records_dir`` (created if needed) so ``validate_records`` fails the chunk visibly.

    Nothing is written when ``records_dir`` is None (records disabled). The payload is the same for every generator:
    the attempt index, ``repr`` of the exception, the traceback and the wall-clock time.
    """
    if records_dir is None:
        return
    folder = Path(records_dir)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / RECORD_ERROR).write_text(json.dumps(
        {"attempt": int(attempt), "error": repr(error), "traceback": traceback.format_exc(), "time": time.time()}, indent=2))


@contextmanager
def record_error_guard(records_dir: Path | None, rec: AttemptRecorder):
    """Run the body (a record flush); on any exception write ``RECORD_ERROR.json`` for ``rec.attempt`` and re-raise.

    The exception is re-raised so the chunk process dies with it instead of generating on without records.
    """
    try:
        yield
    except Exception as error:
        write_record_error(records_dir, attempt=rec.attempt, error=error)
        raise
