"""Working inside a CP-Gen checkout: its modules importable and its relative config paths resolvable.

CP-Gen's code (``demo_aug``, ``scripts.dataset.mp4_from_h5``) is not an installed package but a checkout
(``generator.source_root``, ``TMG_CPGEN_SRC``), and it opens ``demo_aug/configs/...`` relative to the working directory
while writing scratch files (``eef_and_env_model.xml``, ``updated_xml.xml``) there too. ``windows.py`` keeps its own
simpler version of this for the ``tmg sources`` step, which needs the imports only.
"""
from __future__ import annotations

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


def checkout_root(source_root: str | Path) -> Path:
    """``source_root`` as a path, after checking it is a CP-Gen checkout (has ``demo_aug/``)."""
    root = Path(source_root)
    if not (root / "demo_aug").exists():
        raise FileNotFoundError(f"CP-Gen checkout {root} has no demo_aug/ (generator.source_root, TMG_CPGEN_SRC)")
    return root


def mirror_checkout(root: Path, workdir: Path) -> Path:
    """Make ``workdir`` look like the checkout's top level with one symlink per entry (existing entries are kept).

    The checkout is read-only in the images and shared by the chunk processes, so each chunk works in its own
    mirror where CP-Gen can write its scratch files without touching the checkout or another chunk.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    for entry in root.iterdir():
        link = workdir / entry.name
        if not link.exists() and not link.is_symlink():
            link.symlink_to(entry)
    return workdir


@contextmanager
def cpgen_checkout(source_root: str | Path, workdir: Path | None = None) -> Iterator[Path]:
    """Put the checkout first on ``sys.path`` and work in ``workdir`` (a mirror of the checkout) or in the checkout itself.

    Yields the working directory. The previous working directory is restored on exit; the ``sys.path`` entry stays,
    since the imported CP-Gen modules keep referring to it.
    """
    root = checkout_root(source_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    target = root if workdir is None else mirror_checkout(root, Path(workdir))
    previous = os.getcwd()
    os.chdir(target)
    try:
        yield target
    finally:
        os.chdir(previous)
