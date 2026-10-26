"""Adapter for NVlabs/mimicgen (robosuite + MuJoCo).

- ``adapter.py``: ``MimicGenGenerator``, the ``SyntheticDataGenerator`` implementation (sources, chunk
  config, chunk command, result validation).
- ``config_builder.py``: builds MimicGen's generation config json from its task template and our protocol.
- ``runtime.py``: the callbacks grafted into ``DataGenerator.generate`` and ``WaypointTrajectory.execute``.
- ``entrypoint.py``: the chunk process (register scene class, install grafts, run the official script).
- ``sources.py``: download and prepare the released human source demonstrations.
"""
