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


ISAACLAB_MAX_WORKERS = 3     # default cap on concurrent Isaac Sim chunk processes (one container, one GPU)


def cap_simulator_workers(n: int, explained: str, simulator: str, cluster: dict) -> tuple[int, str]:
    """Cap the chunk processes of an Isaac Lab generator at ``cluster.isaaclab_max_workers`` (default
    ``ISAACLAB_MAX_WORKERS``): every chunk is one Isaac Sim process of about 15 GB of host memory, and 12 at once ran the Isaac
    server out of memory (2026-11). The container sees one GPU (``ISAAC_GPU``), so the cap is per GPU. Other simulators keep
    ``n``."""
    if simulator != "isaaclab":
        return n, explained
    cap = int(cluster.get("isaaclab_max_workers", ISAACLAB_MAX_WORKERS))
    if cap < 1:
        raise ValueError(f"cluster.isaaclab_max_workers must be >= 1, got {cap}")
    if n <= cap:
        return n, explained
    return cap, f"{explained}, capped at {cap} Isaac Sim processes (cluster.isaaclab_max_workers)"
