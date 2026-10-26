"""Environment registrations needed on top of the official Isaac Lab tasks.

The official GR1 pick-and-place mimic task records no camera images. ``gr1_pickplace_cameras`` registers a copy
of its config with two 128x128 RGB cameras (front and side) added to the scene and to the policy observations,
under the gym id ``TMG-PickPlace-GR1T2-Images-Mimic-v0``. Importing a module here requires a running Isaac Sim
(the chunk entrypoint imports it after the official script's app launcher).
"""
