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


def cap_simulator_workers(n: int, explained: str, simulator: str, cluster: dict, requested=None,
                          default_cap: int | None = None) -> tuple[int, str]:
    """The chunk processes a pool runs at once, capped for the generator's simulator.

    The cap is the machine's ``cluster.max_workers_by_simulator[simulator]`` when it names one, else the generator's own
    ``default_cap`` (``generator.default_max_workers``; the Isaac Lab generators declare it: every chunk is one Isaac Sim
    process of about 15 GB of host memory, and 12 at once ran the Isaac server out of memory, 2026-11). A machine with
    more memory raises it in its cluster file; a generator that declares none and a machine that names none leave ``n``.
    The cap bounds the chunk processes of one ``run_pool`` call (one ``make generate``): it is per pool runner, not per
    GPU, so two pools started at once (two GPUs, or two shards) run up to twice as many. ``workers: auto`` is capped; an
    explicit ``workers: N`` above the cap is a deliberate choice and is kept, with a warning in the explanation the
    runner prints."""
    by_machine = dict(cluster.get("max_workers_by_simulator") or {}).get(simulator)
    cap, source = ((by_machine, f"cluster.max_workers_by_simulator.{simulator}") if by_machine is not None
                   else (default_cap, "generator.default_max_workers"))
    if cap is None:
        return n, explained
    cap = int(cap)
    if cap < 1:
        raise ValueError(f"{source} must be >= 1, got {cap}")
    if n <= cap:
        return n, explained
    if requested not in (None, "auto"):
        return n, f"{explained} (WARNING: above {source}={cap})"
    return cap, f"{explained}, capped at {cap} {simulator or 'chunk'} processes per pool ({source})"
