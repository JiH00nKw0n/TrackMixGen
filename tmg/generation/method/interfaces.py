"""Interfaces of the generation method: what a generator adapter calls, and what a method implementation provides.

``MixContext`` and ``MixedSegment`` are the input and output of a ``SegmentMixer`` (called once per subtask, before the
generator transforms the source segment to the current scene). ``Method`` bundles the configured mixer and the
action-label choice; ``build_method`` makes it from a ``method`` config.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


@dataclass
class MixContext:
    """What a mixer sees for one subtask of one attempt. Poses are 4x4 in the WORLD frame of the source
    demos; ``src_object_poses[i]`` is source i's reference-object pose at its segment start, so a mixer
    expresses segments in the object frame itself."""
    subtask_ind: int
    primary: int                                   # selected source index (A)
    primary_eef_start: np.ndarray                  # (4,4) first eef pose of A's segment
    primary_targets: np.ndarray                    # (T,4,4) target poses of A's segment
    primary_gripper: np.ndarray                    # (T,G) gripper commands of A's segment
    src_object_poses: list                         # per source: (4,4) reference object pose at segment start
    src_segments: list                             # per source: (eef_start (4,4), targets (T_i,4,4), gripper (T_i,G))
    num_sources: int
    has_reference_object: bool
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))   # unused by the mixers, see README
    extra: dict = field(default_factory=dict)


@dataclass
class MixedSegment:
    """Output of a mixer: the segment the generator should transform and execute, plus a record."""
    eef_start: np.ndarray
    targets: np.ndarray
    gripper: np.ndarray
    record: dict                                   # written to the attempt record (mixed, donor, ...)


class SegmentMixer(Protocol):
    """Replaces a subtask's source segment before the generator transforms it to the current scene."""
    def mix(self, context: MixContext) -> MixedSegment:
        """Return the segment to execute for ``context`` (the selected source's segment, changed or not) with a
        record of what was done."""
        ...

    def begin_attempt(self) -> None:
        """Called by the runtime when an attempt starts (a mixer with per-attempt random choices draws them anew)."""
        ...


@dataclass
class Method:
    """The configured method: a mixer (or None) and the action label kind."""
    name: str
    mixer: SegmentMixer | None
    labels: str = "executed"                       # executed | pre_noise

    @property
    def is_identity(self) -> bool:
        """True when the method changes nothing (no mixer, executed labels)."""
        return self.mixer is None and self.labels == "executed"


class MixerType(Protocol):
    """A mixer implementation as ``build_method`` sees it: a class with ``from_config``."""
    def from_config(self, cfg: Mapping, *, chunk_seed: int, method_name: str) -> SegmentMixer:
        """Build the mixer from the ``method.mixing`` block; ``chunk_seed`` seeds its private random choices and
        ``method_name`` is the method's name (a mixer may tie a setting to the folder name its output goes to)."""
        ...


_MIXERS: dict[str, MixerType] = {}                                              # mixing.type -> class; filled lazily


def _mixer_types() -> dict[str, MixerType]:
    """Mixer implementations by ``method.mixing.type``."""
    if not _MIXERS:
        # imported here, not at the top: the mixer module imports MixContext and MixedSegment from this module
        from tmg.generation.method.skillmix import SkillMixer

        _MIXERS.update(skillmix=SkillMixer)
    return _MIXERS


def build_method(cfg: Mapping, *, chunk_seed: int) -> Method:
    """Build the method of a ``method`` config block (plain mapping).

    ``mixing.type`` picks the mixer implementation; a new variant is a class added to ``_mixer_types`` plus a method
    config file naming it. A ``tracking`` or ``sources`` block (path following, rank-based source sampling) is refused:
    those methods were removed from the repository.
    """
    name = str(cfg.get("name", "none"))
    for removed in ("tracking", "sources"):
        block = cfg.get(removed)
        if block and (removed == "sources" or dict(block).get("enabled", False)):
            raise ValueError(f"method {name!r}: method.{removed} is no longer supported (removed with TrackMix, "
                             f"rank-based sampling and path following)")
    mixing = dict(cfg.get("mixing") or {})
    labels = str(cfg.get("labels", "executed"))
    if labels not in ("executed", "pre_noise"):
        raise ValueError(f"method.labels must be executed or pre_noise, got {labels!r}")
    mixer = None
    if mixing.get("enabled", False):
        kind = str(mixing.get("type", "skillmix"))
        types = _mixer_types()
        if kind not in types:
            raise ValueError(f"method.mixing.type {kind!r} unknown; known: {sorted(types)}")
        mixer = types[kind].from_config(mixing, chunk_seed=chunk_seed, method_name=name)
    return Method(name=name, mixer=mixer, labels=labels)
