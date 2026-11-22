"""Stand-in for the ``egl_probe`` package, mounted over ``egl_probe/__init__.py`` in the robosuite and CP-Gen images when
they render on the GPU (Makefile, ``RENDER=egl``).

robomimic calls ``egl_probe.get_available_devices()`` to choose an EGL device. The real package renders a test frame on
every EGL device of the host (all GPUs, other users' included) and stalled with about 190 chunk processes on the A100
(2026-11-21). The Makefile already chose the device (``MUJOCO_EGL_DEVICE_ID``, the EGL index of the rendering GPU), so this
returns that device only.
"""
import os


def get_available_devices():
    return [int(os.environ["MUJOCO_EGL_DEVICE_ID"])]
