"""Action-vector layout helpers shared by the adapters."""
from __future__ import annotations


def split_by_prefix(names, prefixes: dict) -> dict:
    """Column indices of ``names`` grouped by the first matching prefix, e.g. {"left": "L_", "right": "R_"}.
    Every name must match exactly one group; the result keeps the group order of ``prefixes``."""
    groups = {group: [] for group in prefixes}
    for i, name in enumerate(names):
        hits = [g for g, p in prefixes.items() if str(name).startswith(p)]
        if len(hits) != 1:
            raise ValueError(f"joint {name!r} matches {len(hits)} groups of {prefixes}, expected exactly one")
        groups[hits[0]].append(i)
    return groups
