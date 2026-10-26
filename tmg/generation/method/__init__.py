"""The TrackMix method as two hook interfaces a generator adapter calls.

``SegmentMixer.mix`` runs after a subtask's source segment is extracted in its reference-object frame and
before the generator transforms it to the current scene. ``PathFollower`` decides the target of every control
step during execution. ``build_method(cfg)`` returns the pair configured by a ``method`` config group.
"""
from tmg.generation.method.interfaces import (  # noqa: F401
    Method,
    MixContext,
    MixedSegment,
    PathFollower,
    SegmentMixer,
    build_method,
)
