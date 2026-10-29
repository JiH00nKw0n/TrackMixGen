"""Image helpers shared by the dataset writer and the evaluators (the same resize at training and evaluation time)."""
from __future__ import annotations

import numpy as np


def resize_frames(frames: np.ndarray, size) -> np.ndarray:
    """Resize ``(T, H, W, 3)`` or ``(H, W, 3)`` uint8 frames to ``size = (height, width)`` with PIL's bilinear filter.

    Returns the input unchanged when it already has that size. Used when a generator records larger images than the
    policies train on (Arena's 512x512 head camera -> 128x128): the dataset step and the evaluator call the same function.
    """
    frames = np.asarray(frames)
    h, w = (int(v) for v in size)
    single = frames.ndim == 3
    batch = frames[None] if single else frames
    if batch.shape[1] == h and batch.shape[2] == w:
        return frames
    from PIL import Image
    out = np.empty((batch.shape[0], h, w, batch.shape[3]), dtype=np.uint8)
    for i, frame in enumerate(batch):
        out[i] = np.asarray(Image.fromarray(np.ascontiguousarray(frame.astype(np.uint8))).resize((w, h), Image.BILINEAR))
    return out[0] if single else out


def to_uint8_frame(frame: np.ndarray) -> np.ndarray:
    """A frame as uint8: unchanged when it already is; floats in [0, 1] are scaled by 255 (robomimic's processed
    observations, Isaac Lab camera terms), anything else is cast. Values are clipped only on the [0, 1] path."""
    frame = np.asarray(frame)
    if frame.dtype == np.uint8:
        return frame
    return (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1.0 else frame.astype(np.uint8)


def camera_obs_key(obs_group, cam: str, pattern: str | None = None) -> str | None:
    """The key under which an ``obs`` group keeps camera ``cam``. Two lookup rules, kept apart on purpose:

    - ``pattern`` given (a generator's demonstration file, ``DemoLayout.image_key``): ``pattern.format(cam=cam)``, else
      Arena's ``camera_obs/<cam>`` group; None when the group has neither (the caller reports the missing camera).
    - no ``pattern`` (the common hdf5 of ``tmg dataset``): the camera's own name when present (Isaac Lab generators
      store ``agentview_image`` under that name), else MimicGen's ``<cam>_image``, returned without checking so the
      caller's read raises on a missing camera as it always did.
    """
    if pattern is None:
        return cam if cam in obs_group else f"{cam}_image"
    return next((k for k in (pattern.format(cam=cam), f"camera_obs/{cam}") if k in obs_group), None)
