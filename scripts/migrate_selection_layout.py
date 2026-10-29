#!/usr/bin/env python3
"""Move an existing data root to the folder layout with the source-selection rule (2026-10-29).

    python3 scripts/migrate_selection_layout.py --data-root <root> [--apply]

Old layout: pools|datasets|runs/<generator>/<task>/<scene>/<method>/...  and reports/rank_sources/<generator>/<task>/<scene>/<generation>
New layout: .../<scene>/<selection>/<method>/...                          and reports/rank_sources/<generator>/<task>/<scene>/<selection>/<generation>

The selection of every existing folder is known from its history: the MimicGen pools of 2026-10 (ranking pools, low-dim
exploration pools, test pilots) were made with our rule (a source chosen at random for every subtask) and go under ``ours``;
the SoftMimicGen, DexMimicGen and CP-Gen pools used their generators' official rules and go under ``official``. Folders whose
name already is ``official`` or ``ours`` are left alone, as are ``pools/_verify_baseline`` and any generator folder that is not
one of the four. Without ``--apply`` the script only prints the moves.
"""
from __future__ import annotations

import argparse
from pathlib import Path

SELECTION_OF = {"mimicgen": "ours", "softmimicgen": "official", "dexmimicgen": "official", "cpgen": "official"}
SELECTIONS = set(SELECTION_OF.values())


def _moves(base: Path) -> list[tuple[Path, Path]]:
    """The moves under one ``<base>/<generator>/<task>/<scene>/<child>`` tree: every child folder of a scene that is not
    already a selection name goes to ``<scene>/<selection of the generator>/<child>``."""
    moves = []
    for gen_dir in sorted(base.glob("*")) if base.exists() else []:
        selection = SELECTION_OF.get(gen_dir.name)
        if selection is None or not gen_dir.is_dir():
            continue
        for task_dir in sorted(p for p in gen_dir.iterdir() if p.is_dir()):
            for scene_dir in sorted(p for p in task_dir.iterdir() if p.is_dir()):
                for child in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
                    if child.name in SELECTIONS:
                        continue
                    moves.append((child, scene_dir / selection / child.name))
    return moves


def plan(root: Path) -> list[tuple[Path, Path]]:
    """Every move of the data root: the three result kinds, then the ranking reports."""
    moves = []
    for kind in ("pools", "datasets", "runs"):
        moves += _moves(root / kind)
    moves += _moves(root / "reports" / "rank_sources")
    return moves


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--apply", action="store_true", help="move the folders (default: print the plan only)")
    a = ap.parse_args()
    moves = plan(Path(a.data_root))
    for src, dst in moves:
        print(f"{'MOVE' if a.apply else 'plan'}  {src}  ->  {dst}")
        if a.apply:
            if dst.exists():
                raise FileExistsError(f"{dst} exists; resolve by hand")
            dst.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dst)
    print(f"{len(moves)} folder(s) {'moved' if a.apply else 'to move'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
