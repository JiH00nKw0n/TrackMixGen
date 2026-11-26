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


ISAACLAB_MAX_WORKERS = 3     # default cap on the Isaac Sim chunk processes one pool runs at once


def cap_simulator_workers(n: int, explained: str, simulator: str, cluster: dict, requested=None) -> tuple[int, str]:
    """The chunk processes an Isaac Lab pool runs at once.

    Every chunk is one Isaac Sim process of about 15 GB of host memory; 12 at once ran the Isaac server out of memory
    (2026-11). The cap ``cluster.isaaclab_max_workers`` (default ``ISAACLAB_MAX_WORKERS``) bounds the chunk processes of one
    ``run_pool`` call, i.e. of one ``make generate``: it is per pool runner, not per GPU, so two pools started at once
    (two GPUs, or two shards) run up to twice as many. ``workers: auto`` is capped; an explicit ``workers: N`` above the
    cap is a deliberate choice (a machine with more memory) and is kept, with a warning in the explanation the runner
    prints. Other simulators keep ``n``."""
    if simulator != "isaaclab":
        return n, explained
    cap = int(cluster.get("isaaclab_max_workers", ISAACLAB_MAX_WORKERS))
    if cap < 1:
        raise ValueError(f"cluster.isaaclab_max_workers must be >= 1, got {cap}")
    if n <= cap:
        return n, explained
    if requested not in (None, "auto"):
        return n, (f"{explained} (WARNING: above cluster.isaaclab_max_workers={cap}; each Isaac Sim process needs about "
                   f"15 GB of RAM)")
    return cap, f"{explained}, capped at {cap} Isaac Sim processes per pool (cluster.isaaclab_max_workers)"
