"""Chunk process of the MimicGen adapter.

    python -m tmg.generation.generators.mimicgen.entrypoint --spec <chunk>/chunk_spec.json

Registers the scene's environment class, installs the grafts (records and, when configured, the method),
then runs MimicGen's own ``generate_dataset`` script with ``--auto-remove-exp`` so no prompt can block.
MimicGen exits 0 even when generation fails, so the runner judges success by the output files.
"""
from __future__ import annotations

import runpy
import sys

from tmg.generation.generators._entry import install_runtime, load_spec, register_scene_env

GENERATE_MODULE = "mimicgen.scripts.generate_dataset"


def main(argv=None):
    """Run one chunk: read the spec, register the scene's environment class, install the grafts (unless the
    spec disables them for the identity check) and run MimicGen's ``generate_dataset`` main.
    """
    spec = load_spec(argv)
    from tmg.generation.generators.mimicgen import runtime

    register_scene_env(spec)
    install_runtime(spec, runtime.apply)
    sys.argv = ["generate_dataset.py", "--config", spec["generator_config"], "--auto-remove-exp"]
    runpy.run_module(GENERATE_MODULE, run_name="__main__")


if __name__ == "__main__":
    main()
