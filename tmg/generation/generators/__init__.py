"""Generator adapters. ``make_generator(cfg)`` picks the adapter named by ``cfg.generator.name``."""
from __future__ import annotations

import importlib

from tmg.generation.base import SyntheticDataGenerator

# generator.name -> "module:Class" of its adapter; imported on use because each adapter module imports its generator's
# simulator helpers and the runner runs where the official packages are absent
ADAPTERS: dict[str, str] = {
    "mimicgen": "tmg.generation.generators.mimicgen.adapter:MimicGenGenerator",
    "softmimicgen": "tmg.generation.generators.isaaclab.adapter:SoftMimicGenGenerator",
    "dexmimicgen": "tmg.generation.generators.autodata.adapter:DexMimicGenGenerator",     # the AutoData stack runs every GR1 task
    "cpgen": "tmg.generation.generators.cpgen.adapter:CPGenGenerator",
}


def adapter_class(name: str) -> type[SyntheticDataGenerator]:
    """The adapter class registered for generator ``name`` (``KeyError`` with the available names otherwise)."""
    if name not in ADAPTERS:
        raise KeyError(f"no generator adapter named {name!r} (available: {', '.join(ADAPTERS)})")
    module_name, class_name = ADAPTERS[name].split(":")
    return getattr(importlib.import_module(module_name), class_name)


def make_generator(cfg: dict) -> SyntheticDataGenerator:
    """Instantiate the adapter for ``cfg['generator']['name']`` with the plain resolved config."""
    return adapter_class(str(cfg["generator"]["name"]))(cfg)
