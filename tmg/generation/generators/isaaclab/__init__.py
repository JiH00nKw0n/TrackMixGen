"""Shared adapter for the Isaac Lab mimic generators (SoftMimicGen here; DexMimicGen's AutoData stack builds on it).

The official packages have the same shape: an asynchronous ``DataGenerator.generate`` that builds one trajectory per
end effector and subtask (``generate_trajectory`` for SoftMimicGen's ``softmimicgen.datagen``,
``generate_eef_subtask_trajectory`` + ``merge_eef_subtask_trajectory`` for Isaac Lab Mimic's ``isaaclab_mimic.datagen`` and
for AutoData's ``autodata_core``) and then executes one ``MultiWaypoint`` per control step, with ``generate_dataset.py``
driving the environment from an action queue and Isaac Lab's recorder writing the hdf5.

- ``runtime.py``: the ``_tmg`` callbacks grafted into those functions (records, source mixing, pre-noise labels) with the
  package-specific steps as hooks, and the Isaac Lab hooks of the tracking loop.
- ``tracking_loop.py``: the shared-path follower loop used when tracking is on (``TrackingLoop`` and its pure helpers).
- ``adapter.py``: ``IsaacLabGenerator`` (chunk spec template, command and result validation) and ``SoftMimicGenGenerator``.
- ``entrypoint.py``: the chunk process inside the Isaac Sim image (app launcher, scene ranges, grafts, official main);
  ``..autodata`` reuses it through ``spec["extra"]["runtime_module"]``.
- ``sources.py``: download of the released source demonstrations.
"""
