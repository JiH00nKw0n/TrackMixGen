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
