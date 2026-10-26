"""Chunk process of the MimicGen adapter.

    python -m tmg.generation.generators.mimicgen.entrypoint --spec <chunk>/chunk_spec.json

Registers the scene's environment class, installs the grafts (records and, when configured, the method),
then runs MimicGen's own ``generate_dataset`` script with ``--auto-remove-exp`` so no prompt can block.
MimicGen exits 0 even when generation fails, so the runner judges success by the output files.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

GENERATE_MODULE = "mimicgen.scripts.generate_dataset"


def main(argv=None):
    """Run one chunk: read the spec, register the scene's environment class, install the grafts (unless the
    spec disables them for the identity check) and run MimicGen's ``generate_dataset`` main.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())

    from tmg.generation.generators.mimicgen import runtime
    from tmg.generation.method import build_method
    from tmg.generation.simulators.robosuite import register_env_class

    env_name = register_env_class(spec["task"], spec["scene"])
    print(f"[tmg] environment class: {env_name}", flush=True)
    if spec["extra"].get("disable_graft"):
        print("[tmg] grafts DISABLED: running the official generator without records (identity check)", flush=True)
    else:
        method = build_method(spec["method"], chunk_seed=int(spec["chunk"]["seed"]))
        info = runtime.apply(method, chunk_seed=int(spec["chunk"]["seed"]), records_dir=Path(spec["records_dir"]),
                             graft_hashes=dict(spec["extra"].get("graft_hashes") or {}))
        print(f"[tmg] grafts installed: {json.dumps(info, indent=1)}", flush=True)
    sys.argv = ["generate_dataset.py", "--config", spec["generator_config"], "--auto-remove-exp"]
    runpy.run_module(GENERATE_MODULE, run_name="__main__")


if __name__ == "__main__":
    main()
