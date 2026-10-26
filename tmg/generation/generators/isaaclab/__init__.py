"""Shared adapter for the Isaac Lab mimic generators (SoftMimicGen and DexMimicGen).

Both official packages (``softmimicgen.datagen`` on the SoftMimicGen fork of Isaac Lab, ``isaaclab_mimic.datagen``
on the Isaac Lab commit DexMimicGen ships with) have the same shape: an asynchronous ``DataGenerator.generate``
that builds one trajectory per end effector and subtask (``generate_trajectory`` for Soft,
``generate_eef_subtask_trajectory`` + ``merge_eef_subtask_trajectory`` for Dex) and then executes one
``MultiWaypoint`` per control step, with ``generation.py`` driving the environment from an action queue and
Isaac Lab's recorder writing the hdf5.

- ``runtime.py``: the ``_tmg`` callbacks grafted into those functions (records, source mixing, pre-noise labels)
  and the shared-path follower loop used when tracking is on.
- ``adapter.py``: ``IsaacLabGenerator`` (chunk spec, command and result validation) and the two thin subclasses
  ``SoftMimicGenGenerator`` and ``DexMimicGenGenerator`` with their package names and end-effector layout.
- ``entrypoint.py``: the chunk process inside the Isaac Sim image (app launcher, scene ranges, grafts, official main).
- ``sources.py``: download of the released source demonstrations.
"""
