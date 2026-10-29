"""Shared part of the adapters that run on robosuite (MimicGen, CP-Gen): scene chain resolution and the environment
class name of (task, scene)."""
from __future__ import annotations

from tmg.generation.base import SyntheticDataGenerator
from tmg.generation.simulators.robosuite import env_class_name, load_scene_chain


class RobosuiteGenerator(SyntheticDataGenerator):
    """Base of the robosuite adapters; subclasses add the official package's config, command and result reading."""

    def scene_chain(self) -> dict:
        """The scene config with its parent chain inlined (``simulators.robosuite.load_scene_chain``)."""
        return load_scene_chain(self.scene, self.generator_cfg["name"])

    def env_name(self) -> str:
        """Name of the robosuite environment class of (task, scene), e.g. ``Square_D1`` or ``Square_OURS``."""
        return env_class_name(self.task, self.scene_chain())
