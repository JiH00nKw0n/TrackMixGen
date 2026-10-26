"""Synthetic demonstration generation.

Folder layout (what each piece owns):

- ``base.py``: the contract every generator adapter implements (``SyntheticDataGenerator``) and the chunk
  description the runner exchanges with it (``ChunkSpec``).
- ``execution.py``: generator-independent bookkeeping while an official executor walks a trajectory step by
  step (segments, the shared path follower, random-number accounting, action labels).
- ``runner.py``: one pool = many chunks run as subprocesses, with resume, provenance and completion markers.
- ``verify.py``: the baseline identity check (official generator run vs adapter run with the method off).
- ``method/``: the TrackMix method as two hook interfaces (segment mixing, path following) plus detectors.
- ``simulators/``: simulator-level helpers (scene ranges applied to environment classes).
- ``graft.py``: AST-level insertion of callbacks into official functions with source-hash verification, shared by
  every adapter.
- ``generators/``: one adapter per official generator package (MimicGen today; Isaac Lab ones to come).
"""
