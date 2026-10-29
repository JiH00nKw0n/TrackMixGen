"""Adapter for NVlabs/mimicgen (robosuite + MuJoCo).

- ``adapter.py``: ``MimicGenGenerator``, the ``SyntheticDataGenerator`` implementation (sources, chunk
  config, result validation); the robosuite part it shares with CP-Gen is ``..robosuite_base.RobosuiteGenerator``.
- ``config_builder.py``: builds MimicGen's generation config json from its task template and our protocol.
- ``runtime.py``: the callbacks grafted into ``DataGenerator.generate`` and ``WaypointTrajectory.execute``.
- ``entrypoint.py``: the chunk process (register scene class, install grafts, run the official script).
- ``sources.py``: download and prepare the released human source demonstrations (the download itself and the
  manifest helpers are ``.._sources``; the other generators import ``download`` and ``file_md5`` from here).
"""
