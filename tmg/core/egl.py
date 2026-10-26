"""Offscreen-rendering device variables for robosuite 1.5 (the CP-Gen image).

robosuite 1.5 reads the EGL device index from ``MUJOCO_EGL_DEVICE_ID`` (else from ``CUDA_VISIBLE_DEVICES``) and
asserts at import time that the former appears in the latter. On a host where only Mesa's software EGL device can
be initialised (A100: the GPUs' DRM devices are not openable by the user), the EGL index is not a GPU index, so
the assertion would fail. ``apply`` appends the EGL index to ``CUDA_VISIBLE_DEVICES``; CUDA ignores every entry
from the first invalid index on, so the GPUs the process sees do not change as long as the EGL index is not a
valid GPU index (cluster ``egl_device_id`` is documented with that constraint).
"""
from __future__ import annotations

import os


def apply(environ: dict | None = None) -> dict:
    """Make ``CUDA_VISIBLE_DEVICES`` list ``MUJOCO_EGL_DEVICE_ID`` (both from ``environ``, default ``os.environ``).

    Returns:
        The environment mapping that was adjusted (``environ`` or ``os.environ``). No change when either variable
        is unset or the EGL index is already listed.
    """
    env = os.environ if environ is None else environ
    egl = env.get("MUJOCO_EGL_DEVICE_ID")
    cuda = env.get("CUDA_VISIBLE_DEVICES")
    if egl and cuda and egl not in [x.strip() for x in cuda.split(",")]:
        env["CUDA_VISIBLE_DEVICES"] = f"{cuda},{egl}"
    return env
