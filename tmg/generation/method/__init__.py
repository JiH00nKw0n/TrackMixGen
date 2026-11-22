"""The generation method as a hook interface a generator adapter calls.

``SegmentMixer.mix`` runs after a subtask's source segment is extracted in its reference-object frame and before the
generator transforms it to the current scene. ``build_method(cfg)`` returns the method configured by a ``method`` config
group (its mixer, or None for the generator as released, and the action-label kind).
"""
from tmg.generation.method.interfaces import (  # noqa: F401
    Method,
    MixContext,
    MixedSegment,
    SegmentMixer,
    build_method,
)
