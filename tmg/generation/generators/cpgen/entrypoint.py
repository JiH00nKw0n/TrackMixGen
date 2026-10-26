"""Chunk process of the CP-Gen adapter.

    python -m tmg.generation.generators.cpgen.entrypoint --spec <chunk>/chunk_spec.json

Steps: make the CP-Gen checkout the working directory (its code opens ``demo_aug/configs/...`` relative to
it) and put it on ``sys.path`` (``scripts.dataset.mp4_from_h5``), register the scene's environment class,
install the grafts (``runtime.apply``) unless the spec disables them for the identity check, then call
``demo_aug.generate.main`` with the chunk's ``Config``. CP-Gen's ``main`` raises after the attempts when no
attempt succeeded (it tries to render observations of a merged file that does not exist); that case ends the
chunk normally once every attempt is recorded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    from tmg.core import egl

    egl.apply()
    extra = spec["extra"]
    root = Path(extra["source_root"])
    if not (root / "demo_aug").exists():
        raise FileNotFoundError(f"CP-Gen checkout {root} has no demo_aug/ (generator.source_root, TMG_CPGEN_SRC)")
    sys.path.insert(0, str(root))
    os.chdir(root)

    from tmg.core.records import RECORDS_DIR, read_attempts
    from tmg.generation.generators.cpgen import runtime
    from tmg.generation.method import build_method
    from tmg.generation.simulators.robosuite import register_env_class

    env_name = register_env_class(spec["task"], spec["scene"])
    print(f"[tmg] environment class: {env_name}", flush=True)
    if extra.get("disable_graft"):
        print("[tmg] grafts DISABLED: running the official generator without records (identity check)", flush=True)
    else:
        method = build_method(spec["method"], chunk_seed=int(spec["chunk"]["seed"]))
        info = runtime.apply(method, chunk_seed=int(spec["chunk"]["seed"]), records_dir=Path(spec["records_dir"]),
                             graft_hashes=dict(extra.get("graft_hashes") or {}))
        print(f"[tmg] grafts installed: {json.dumps(info, indent=1)}", flush=True)
    import demo_aug.generate as official

    config = dict(extra["cpgen_config"])
    config["save_dir"] = Path(config["save_dir"])
    cfg = official.Config(**config)
    print(f"[tmg] CP-Gen main: seed {cfg.seed}, {cfg.n_demos} attempts, env {cfg.env_name}, source {cfg.demo_path} "
          f"[{cfg.load_demos_start_idx}, {cfg.load_demos_end_idx})", flush=True)
    try:
        official.main(cfg)
    except Exception:
        records = read_attempts(Path(spec["folder"]) / RECORDS_DIR)
        attempts = int(spec["chunk"]["attempts"])
        if len(records) >= attempts and not any(r.get("success") for r in records):
            print(f"[tmg] CP-Gen main failed after all {attempts} attempts with no success (expected: nothing to render):\n"
                  f"{traceback.format_exc()}", flush=True)
            return
        raise


if __name__ == "__main__":
    main()
