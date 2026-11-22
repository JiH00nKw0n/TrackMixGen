"""Offscreen rendering of the robosuite images as the launcher set it up: which renderer (``renderer``), and the device
variables robosuite 1.5 needs (``apply``, the CP-Gen image).

The Makefile chooses the renderer (``RENDER``: egl = the GPU through NVIDIA EGL, mesa = Mesa's software EGL on the CPU) and
passes it into the container as ``TMG_RENDER``. The two do not draw the same pixels, so everything that produces or
consumes images under one of them records it (``tmg.generation.base.ChunkSpec``).

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

RENDER_ENV = "TMG_RENDER"
RENDERERS = ("egl", "mesa")


def renderer(environ: dict | None = None) -> str | None:
    """The offscreen renderer of this process (``TMG_RENDER``), or None when the launcher did not say: a process started
    without the Makefile, or an image that does not render through robosuite (Isaac Lab)."""
    value = (os.environ if environ is None else environ).get(RENDER_ENV) or None
    if value is not None and value not in RENDERERS:
        raise ValueError(f"{RENDER_ENV} must be one of {RENDERERS}, got {value!r}")
    return value


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
