"""Datasets for policy training, independent of the generator that made the demonstrations.

- ``dataset.py``: ``tmg dataset``, which selects successful attempts of a generation pool and writes one
  common hdf5 (images, robot state, action labels, executed and pre-noise actions, states, subtask ids).
- ``loader.py``: the in-memory frame dataset the training loop uses for every policy backend; it builds
  LeRobot-style batches (``observation.images.<camera>``, ``observation.state``, ``action``,
  ``action_is_pad``) with temporal stacking by delta indices and computes normalization statistics.
- ``lerobot_export.py`` (stage 6): conversion of the common hdf5 to a LeRobot dataset for π0.
"""
