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


def cap_simulator_workers(n: int, explained: str, simulator: str, cluster: dict, requested=None) -> tuple[int, str]:
    """The chunk processes a pool runs at once, capped per simulator by the machine's config.

    ``cluster.max_workers_by_simulator`` maps a generator's simulator (``generator.simulator``) to the most chunk
    processes of that simulator one ``run_pool`` call (one ``make generate``) runs at once; a simulator it does not name
    keeps ``n``. The Isaac server caps Isaac Lab there: every chunk is one Isaac Sim process of about 15 GB of host memory
    and 12 at once ran it out of memory (2026-11). The cap is per pool runner, not per GPU, so two pools started at once
    (two GPUs, or two shards) run up to twice as many. ``workers: auto`` is capped; an explicit ``workers: N`` above the
    cap is a deliberate choice (a machine with more memory) and is kept, with a warning in the explanation the runner
    prints."""
    caps = dict(cluster.get("max_workers_by_simulator") or {})
    if simulator not in caps or caps[simulator] is None:
        return n, explained
    cap = int(caps[simulator])
    if cap < 1:
        raise ValueError(f"cluster.max_workers_by_simulator.{simulator} must be >= 1, got {cap}")
    if n <= cap:
        return n, explained
    if requested not in (None, "auto"):
        return n, f"{explained} (WARNING: above cluster.max_workers_by_simulator.{simulator}={cap})"
    return cap, f"{explained}, capped at {cap} {simulator} processes per pool (cluster.max_workers_by_simulator)"
