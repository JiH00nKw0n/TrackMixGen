"""AutoData adapter (NVIDIA isaac-sim/AutoData, the DexMimicGen rewrite on Isaac Lab-Arena).

- ``adapter.py``: ``DexMimicGenGenerator``, the Isaac Lab adapter's ``prepare_chunk`` hooks for AutoData's script (task descriptor,
  Arena callback, cameras).
- ``runtime.py``: ``AutoDataRuntime`` (the Isaac Lab runtime's hooks for ``autodata_core``) and ``AutoDataTrackingLoop``.
"""
