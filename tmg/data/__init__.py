"""Datasets for policy training, independent of the generator that made the demonstrations.

- ``dataset.py``: ``tmg dataset``, which selects successful attempts of a generation pool and writes one
  common hdf5 (images, robot state, action labels, executed and pre-noise actions, states, subtask ids).
- ``loader.py``: the in-memory frame dataset the training loop uses for every policy backend; it builds
  LeRobot-style batches (``observation.images.<camera>``, ``observation.state``, ``action``,
  ``action_is_pad``) with temporal stacking by delta indices and computes normalization statistics.
- ``images.py``: the resize shared by the dataset writer and the evaluators, the uint8 frame conversion, and the
  rule that finds a camera's key in an ``obs`` group.

π0 trains from the same common hdf5 through ``loader.py`` (task strings in the items); there is no separate LeRobot
export step.
"""
