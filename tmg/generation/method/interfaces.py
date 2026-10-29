"""Interfaces of the TrackMix method: what a generator adapter calls, and what a method implementation
provides.

``MixContext`` and ``MixedSegment`` are the input and output of a ``SegmentMixer`` (called once per subtask,
before the generator transforms the source segment to the current scene). ``PathFollower`` is the object that
decides the command of every control step while a trajectory executes. ``Method`` bundles the configured
mixer, the follower factory and the action-label choice; ``build_method`` makes it from a ``method`` config.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from tmg.generation.method.tracking import ErrorAdaptiveFollower


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
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng(0))   # unused by GuidedMixer, see README
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


class PathFollower(Protocol):
    """Decides the command of each control step while one trajectory (all waypoint sequences of a subtask,
    all end effectors) executes. One instance per trajectory; see ``tmg.generation.method.tracking``."""

    def start_path(self, eefs: list, noises: list, kinds: list, previous_states: dict | None = None) -> None:
        """Load the flattened path. ``eefs`` are ``EndEffectorPath`` objects (one per end effector, equal
        length n), ``noises`` and ``kinds`` have one entry per waypoint, ``previous_states`` maps an end
        effector to its last gripper state before this path (None on the first path of an attempt)."""
        ...

    def finished(self) -> bool:
        """True once the path variable reached the last waypoint."""
        ...

    def final_gripper_states(self) -> dict:
        """Gripper state per end effector at the end of the path, for the next path's event detection."""
        ...

    def next_target(self, measured: dict) -> tuple[dict, dict, Any, dict]:
        """Advance once. ``measured`` maps end-effector names to measured 4x4 poses. Returns (target poses by
        end effector, gripper commands by end effector, noise scale of the commanded waypoint, record), where
        ``record`` has the keys documented in ``tmg.generation.method.tracking.ErrorAdaptiveFollower``."""
        ...


@dataclass
class Method:
    """The configured method: a mixer (or None), a follower factory (or None), the action label kind and a source
    selection rule (or None: the generator's own choice)."""
    name: str
    mixer: SegmentMixer | None
    follower_factory: Any | None                   # callable() -> PathFollower, one per segment
    labels: str = "executed"                       # executed | pre_noise
    sources: Any | None = None                     # TopKSources (method.sources.top_k) or None

    @property
    def is_identity(self) -> bool:
        """True when the method changes nothing (no mixer, no follower, executed labels, the generator's sources)."""
        return self.mixer is None and self.follower_factory is None and self.labels == "executed" and self.sources is None


class MixerType(Protocol):
    """A mixer implementation as ``build_method`` sees it: a class with ``from_config``."""
    def from_config(self, cfg: Mapping, *, chunk_seed: int) -> SegmentMixer:
        """Build the mixer from the ``method.mixing`` block; ``chunk_seed`` seeds its private random choices."""
        ...


class FollowerType(Protocol):
    """A follower implementation as ``build_method`` sees it: a class with ``factory``."""
    def factory(self, cfg: Mapping) -> Callable[[], PathFollower]:
        """Return the zero-argument constructor of the ``method.tracking`` block (one follower per trajectory)."""
        ...


_MIXERS: dict[str, MixerType] = {}                                              # mixing.type -> class; filled lazily
_FOLLOWERS: dict[str, FollowerType] = {"error_adaptive": ErrorAdaptiveFollower}  # tracking.type -> class


def _mixer_types() -> dict[str, MixerType]:
    """Mixer implementations by ``method.mixing.type``."""
    if not _MIXERS:
        # imported here, not at the top: guided_mix imports MixContext and MixedSegment from this module
        from tmg.generation.method.geometric_mix import JunctionMixer, TransformMixer
        from tmg.generation.method.guided_mix import GuidedMixer

        _MIXERS.update(guided=GuidedMixer, transform=TransformMixer, junction=JunctionMixer)
    return _MIXERS


def _follower_types() -> dict[str, FollowerType]:
    """Follower implementations by ``method.tracking.type``."""
    return _FOLLOWERS


def build_method(cfg: Mapping, *, chunk_seed: int) -> Method:
    """Build the method of a ``method`` config block (plain mapping).

    ``mixing.type`` (default ``guided``) and ``tracking.type`` (default ``error_adaptive``) pick the implementation;
    ``sources.top_k`` restricts the generator's source choice to the best-ranked sources (``TopKSources``);
    a new variant is a class added to ``_mixer_types`` or ``_FOLLOWERS`` plus a method config file naming it.
    """
    name = str(cfg.get("name", "none"))
    mixing = dict(cfg.get("mixing") or {})
    tracking = dict(cfg.get("tracking") or {})
    labels = str(cfg.get("labels", "executed"))
    if labels not in ("executed", "pre_noise"):
        raise ValueError(f"method.labels must be executed or pre_noise, got {labels!r}")
    mixer = None
    if mixing.get("enabled", False):
        kind = str(mixing.get("type", "guided"))
        types = _mixer_types()
        if kind not in types:
            raise ValueError(f"method.mixing.type {kind!r} unknown; known: {sorted(types)}")
        mixer = types[kind].from_config(mixing, chunk_seed=chunk_seed)
    factory = None
    if tracking.get("enabled", False):
        kind = str(tracking.get("type", "error_adaptive"))
        types = _follower_types()
        if kind not in types:
            raise ValueError(f"method.tracking.type {kind!r} unknown; known: {sorted(types)}")
        factory = types[kind].factory(tracking)
    sources = None
    source_cfg = dict(cfg.get("sources") or {})
    if source_cfg.get("top_k") is not None:
        from tmg.generation.method.source_select import TopKSources

        sources = TopKSources.from_config(source_cfg, chunk_seed=chunk_seed)
    return Method(name=name, mixer=mixer, follower_factory=factory, labels=labels, sources=sources)
