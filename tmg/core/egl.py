"""Offscreen-rendering device variables for robosuite 1.5 (the CP-Gen image).

robosuite 1.5 reads the EGL device index from ``MUJOCO_EGL_DEVICE_ID`` (else from ``CUDA_VISIBLE_DEVICES``) and
asserts at import time that the former appears in the latter. The EGL index is often not the CUDA index of the same GPU:
with GPU rendering (Makefile ``RENDER=egl``) EGL numbers the GPUs in its own order (A100: GPU 0 is EGL device 2, cluster
``egl_device_of_gpu``), and with Mesa software rendering on the A100 (``RENDER=mesa``) the index is Mesa's software device
(8), which is no GPU at all. ``apply`` appends the EGL index to ``CUDA_VISIBLE_DEVICES`` so the assertion passes. The GPU
the process computes on stays the first entry (``cuda:0``): CUDA ignores an index that is no valid GPU (Mesa's 8), and a
valid one (GPU rendering on the A100) only becomes a second visible device that nothing uses.
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
