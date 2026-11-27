"""Interfaces of the generation method: what a generator adapter calls, and what a method implementation provides.

``MixContext`` and ``MixedSegment`` are the input and output of a ``SegmentMixer`` (called once per subtask, before the
generator transforms the source segment to the current scene). ``Method`` bundles the configured mixer and the
action-label choice; ``build_method`` makes it from a ``method`` config.

Three value objects are shared by every runtime, the mixer, the attempt records and the fit (``tmg fit_coefficients``):

- ``Arms`` and ``SubtaskId``: which end effectors number their own subtasks, and the id of one subtask. A generator with
  one arm (MimicGen, CP-Gen, SoftMimicGen) has ids ``k``; one with two or more arms (DexMimicGen's GR1) numbers the
  subtasks of every arm from 0, so its ids are ``(eef, k)``, written ``<eef>/<k>``. Rule-file blocks, random draws and the
  fit's blocks are keyed by ``SubtaskId``; the arms come from the generator's layout (the generator config), never from
  the records.
- ``SubtaskScene``: the scene of one subtask as SkillMix scores it (reference object now and in every source, the hand and
  what it carries now and in every source). A runtime fills it (``MixContext.scene``), the mixer turns it into features
  (``success_model.source_features``), ``SubtaskScene.record`` writes the selected source's part into the attempt record and
  ``success_model.scene_of_record`` reads it back for the fit, so the mixer and the fit compute the features with one
  function.
- ``HeldRule`` / ``HeldTest``: how a runtime decides whether the hand holds the carried object (distance limits from the
  generator config, recorded with every record and in the rule file).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


# arms and subtask ids -------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SubtaskId:
    """Subtask ``k`` of end effector ``eef`` (None for a generator with one arm)."""
    k: int
    eef: str | None = None

    @property
    def key(self):
        """The key in a rule file and in the fit summary: ``k`` (single arm) or ``"<eef>/<k>"``."""
        return int(self.k) if self.eef is None else f"{self.eef}/{int(self.k)}"

    @classmethod
    def parse(cls, key) -> SubtaskId:
        """From a rule-file key: ``0`` or ``"0"`` (single arm) or ``"right/2"``."""
        text = str(key)
        eef, sep, k = text.rpartition("/")
        if not sep:
            return cls(int(text))
        if not eef:
            raise ValueError(f"subtask key {key!r}: expected <k> or <eef>/<k>")
        return cls(int(k), eef)

    @property
    def order(self) -> tuple:
        """Sort key: by end effector, then by subtask."""
        return (self.eef or "", int(self.k))

    def __str__(self) -> str:
        return str(self.key)


@dataclass(frozen=True)
class Arms:
    """The end effectors of a generator as SkillMix sees them. ``names`` empty: one arm whose records name no end effector
    (MimicGen, CP-Gen). One name: one arm whose records name it (SoftMimicGen's ``robot0``); ids stay ``k``. Two or more:
    every arm numbers its subtasks from 0 and ids are ``(eef, k)``."""
    names: tuple[str, ...] = ()

    @classmethod
    def of(cls, arms) -> Arms:
        """From an ``Arms``, a sequence of end-effector names, or None (one unnamed arm)."""
        if isinstance(arms, Arms):
            return arms
        return cls(tuple(str(a) for a in (arms or ())))

    @property
    def per_arm(self) -> bool:
        return len(self.names) > 1

    def subtask(self, k: int, eef: str | None = None) -> SubtaskId:
        """The id of subtask ``k`` of the end effector a runtime or a record names (``eef``)."""
        if not self.per_arm:
            return SubtaskId(int(k))
        if eef is None or str(eef) not in self.names:
            raise KeyError(f"subtask {k} of end effector {eef!r}: the generator's arms are {list(self.names)}")
        return SubtaskId(int(k), str(eef))

    @property
    def record_eefs(self) -> tuple:
        """The ``eef`` values of the subtask records to read per arm: the names, or ``(None,)`` (every record)."""
        return self.names or (None,)


# the scene of a subtask -----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class HeldRule:
    """When a hand counts as holding an object: its origin within ``rigid_m`` of the end effector (rigid object), or its
    nearest simulation node within ``nodes_m`` (deformable object). ``version`` names the definition; the values come
    from the generator config (``scene_reading.held_rule``)."""
    version: int
    rigid_m: float
    nodes_m: float

    @classmethod
    def from_mapping(cls, m: Mapping | None) -> HeldRule | None:
        if m is None:
            return None
        return cls(version=int(m["version"]), rigid_m=float(m["rigid_m"]), nodes_m=float(m["nodes_m"]))

    def as_dict(self) -> dict:
        return {"version": self.version, "rigid_m": self.rigid_m, "nodes_m": self.nodes_m}

    def limit(self, deformable: bool) -> float:
        return self.nodes_m if deformable else self.rigid_m

    def holds(self, distance: float, deformable: bool) -> bool:
        return float(distance) <= self.limit(deformable)


@dataclass
class HeldTest:
    """The hold test of one subtask: the distances the rule was applied to, now and in every source at its segment start."""
    rule: HeldRule
    deformable: bool
    now_m: float
    sources_m: list

    @property
    def now(self) -> bool:
        return self.rule.holds(self.now_m, self.deformable)

    @property
    def sources(self) -> list[bool]:
        return [self.rule.holds(d, self.deformable) for d in self.sources_m]


@dataclass
class SubtaskScene:
    """The scene of one subtask of one end effector as SkillMix scores it. Poses are 4x4 world poses (or the 12-number
    record layout); ``None`` is "not known" or, with a hold test, "not in the hand".

    ``carried_object`` is the reference object of the end effector's previous subtask (None for its first subtask):
    the in-hand features compare its pose in the hand now and in every source. ``carried_pose`` and
    ``source_carried_poses[j]`` are None where that pose is not known (CP-Gen: a constraint that does not involve the
    object) or where the hand does not hold the object (``held``, the Isaac Lab runtimes)."""
    subtask: int
    eef: str | None = None                           # the end effector the runtime names, None when it names none
    reference_object: str | None = None
    current_object_pose: np.ndarray | None = None     # reference object now (a deformable object: its nodal frame)
    source_object_poses: list | None = None           # per source at its segment start; None: MixContext.src_object_poses
    eef_pose: np.ndarray | None = None                # end effector now
    source_eef_poses: list | None = None              # per source at its segment start
    carried_object: str | None = None
    carried_pose: np.ndarray | None = None            # carried object now
    source_carried_poses: list | None = None          # per source: the carried object at its segment start
    held: HeldTest | None = None                      # the hold test, when the runtime makes one and something is carried
    held_rule: HeldRule | None = None                 # the runtime's hold rule (also when nothing is carried)
    object_frame: dict | None = None                  # deformable reference object: frame definition and reference hash

    @property
    def num_sources(self) -> int:
        return len(self.source_object_poses or self.source_eef_poses or [])

    def record(self, selected: int) -> dict:
        """The subtask-record fields of the selected source, what ``success_model.scene_of_record`` reads back. ``objects``
        and ``src_objects`` hold the carried object, and only where its pose is known (and it is held), so the fit's
        in-hand features are the mixer's."""
        objects, src_objects = {}, {}
        if self.carried_object is not None and self.carried_pose is not None:
            objects[self.carried_object] = self.carried_pose
        carried_src = None if self.source_carried_poses is None else self.source_carried_poses[selected]
        if self.carried_object is not None and carried_src is not None:
            src_objects[self.carried_object] = carried_src
        out = {"cur_obj_pose": self.current_object_pose,
               "src_obj_pose": None if self.source_object_poses is None else self.source_object_poses[selected],
               "eef_pose": self.eef_pose, "src_eef_pose": None if self.source_eef_poses is None else self.source_eef_poses[selected],
               "objects": objects, "src_objects": src_objects,
               "carried_object": self.carried_object if self.carried_pose is not None else None,
               "carried_candidate": self.carried_object}
        if self.held is not None:
            out["held_distance"] = {"now": self.held.now_m, "source": self.held.sources_m[selected]}
            out["held"] = {"now": self.held.now, "source": self.held.sources[selected], "deformable": self.held.deformable}
        if self.held_rule is not None:
            out["held_rule"] = self.held_rule.as_dict()
        if self.object_frame is not None:
            out["object_frame"] = self.object_frame
        return out


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
    scene: SubtaskScene | None = None              # what the mixer scores (None: the mixer leaves the segment as it is)


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
    def from_config(self, cfg: Mapping, *, chunk_seed: int, method_name: str, arms: Arms) -> SegmentMixer:
        """Build the mixer from the ``method.mixing`` block; ``chunk_seed`` seeds its private random choices,
        ``method_name`` is the method's name (a mixer may tie a setting to the folder name its output goes to) and
        ``arms`` are the generator's end effectors."""
        ...


_MIXERS: dict[str, MixerType] = {}                                              # mixing.type -> class; filled lazily


def _mixer_types() -> dict[str, MixerType]:
    """Mixer implementations by ``method.mixing.type``."""
    if not _MIXERS:
        # imported here, not at the top: the mixer module imports MixContext and MixedSegment from this module
        from tmg.generation.method.skillmix import SkillMixer

        _MIXERS.update(skillmix=SkillMixer)
    return _MIXERS


def build_method(cfg: Mapping, *, chunk_seed: int, arms: Arms | Sequence[str] | None = None) -> Method:
    """Build the method of a ``method`` config block (plain mapping). ``arms``: the generator's end effectors from its
    layout (``Arms.of``; None for a generator with one unnamed arm), handed to the mixer when it is made.

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
        mixer = types[kind].from_config(mixing, chunk_seed=chunk_seed, method_name=name, arms=Arms.of(arms))
    return Method(name=name, mixer=mixer, labels=labels)
