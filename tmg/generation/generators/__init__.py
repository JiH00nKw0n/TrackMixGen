"""Generator adapters. ``make_generator(cfg)`` picks the adapter named by ``cfg.generator.name``."""
from __future__ import annotations

from tmg.generation.base import SyntheticDataGenerator


def make_generator(cfg: dict) -> SyntheticDataGenerator:
    """Instantiate the adapter for ``cfg['generator']['name']`` with the plain resolved config."""
    name = str(cfg["generator"]["name"])
    if name == "mimicgen":
        from tmg.generation.generators.mimicgen.adapter import MimicGenGenerator

        return MimicGenGenerator(cfg)
    if name == "softmimicgen":
        from tmg.generation.generators.isaaclab.adapter import SoftMimicGenGenerator

        return SoftMimicGenGenerator(cfg)
    if name == "dexmimicgen":
        from tmg.generation.generators.isaaclab.adapter import DexMimicGenGenerator

        return DexMimicGenGenerator(cfg)
    if name == "cpgen":
        from tmg.generation.generators.cpgen.adapter import CPGenGenerator

        return CPGenGenerator(cfg)
    if name == "autodata":
        from tmg.generation.generators.autodata.adapter import AutoDataGenerator

        return AutoDataGenerator(cfg)
    raise KeyError(f"no generator adapter named {name!r} (available: mimicgen, softmimicgen, dexmimicgen, cpgen, autodata)")
