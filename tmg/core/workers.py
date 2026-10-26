"""Worker count for chunk processes."""
from __future__ import annotations

import os


def resolve_workers(value, max_workers: int) -> tuple[int, str]:
    """Number of chunk processes to run at once.

    Args:
        value: ``"auto"`` (or None) for ``min(cpu_count, max_workers)``, else an explicit integer.
        max_workers: cap used by ``auto``.

    Returns:
        ``(count, explanation)`` where the explanation says how the count was chosen (printed by the runner).
    """
    if value == "auto" or value is None:
        n = min(os.cpu_count() or 1, int(max_workers))
        return n, f"auto -> {n} (cpu_count={os.cpu_count()}, max_workers={max_workers})"
    n = int(value)
    if n < 1:
        raise ValueError(f"workers must be >= 1, got {value}")
    return n, f"workers={n}"
