"""Environment registrations needed on top of the official Isaac Lab tasks.

The official GR1 pick-and-place mimic task records no camera images. ``gr1_pickplace_cameras`` registers a copy
of its config with two 128x128 RGB cameras (front and side) added to the scene and read by their own observation
group ``camera_obs`` (not the ``policy`` group, so the official recorder does not store them), under the gym id
``TMG-PickPlace-GR1T2-Images-Mimic-v0``. Importing that module registers the environment when Isaac Lab is installed
(``register()`` runs at import; the chunk entrypoint imports it after the official script's app launcher) and only
offers the camera helpers elsewhere.
"""
