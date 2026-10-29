"""Steps every chunk process (``entrypoint.py``) performs before handing over to the official generator.

A chunk process is launched as ``python -m <entrypoint> --spec <chunk>/chunk_spec.json``. It reads the spec, registers
the scene's environment class (robosuite generators), installs the grafts unless the spec disables them, and runs the
official script or function. The helpers here are the generator-independent parts; the official import and call stay
in each entrypoint.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

DISABLED_MESSAGE = "[tmg] grafts DISABLED: running the official generator without records (identity check)"


def load_spec(argv=None) -> dict:
    """Parse ``--spec <path>`` from ``argv`` (``sys.argv[1:]`` when None) and return the chunk spec as a dict."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    return json.loads(Path(args.spec).read_text())


def register_scene_env(spec: dict) -> str:
    """Register the robosuite environment class of the spec's (task, scene) and log its name."""
    from tmg.generation.simulators.robosuite import register_env_class

    env_name = register_env_class(spec["task"], spec["scene"])
    print(f"[tmg] environment class: {env_name}", flush=True)
    return env_name


def install_runtime(spec: dict, apply: Callable[..., dict], **extra) -> dict | None:
    """Install a generator's grafts for the chunk, or log that they are disabled.

    Args:
        spec: the chunk spec. ``extra.disable_graft`` (set by ``verify_baseline``) skips the installation.
        apply: the generator runtime's ``apply(method, *, chunk_seed, records_dir, graft_hashes, **extra)``.
        extra: further keyword arguments of ``apply`` (CP-Gen's ``all_sources``).

    Returns:
        ``apply``'s description of the installed grafts (also logged), or None when they were disabled.
    """
    if spec["extra"].get("disable_graft"):
        print(DISABLED_MESSAGE, flush=True)
        return None
    from tmg.generation.method import build_method

    chunk_seed = int(spec["chunk"]["seed"])
    method = build_method(spec["method"], chunk_seed=chunk_seed)
    info = apply(method, chunk_seed=chunk_seed, records_dir=Path(spec["records_dir"]),
                 graft_hashes=dict(spec["extra"].get("graft_hashes") or {}), **extra)
    print(f"[tmg] grafts installed: {json.dumps(info, indent=1)}", flush=True)
    return info
