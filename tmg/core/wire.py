"""Message encoding between the policy server and the evaluation client: msgpack with numpy arrays.

An array travels as ``{"__nd__": True, "dtype": "<f4", "shape": [..], "data": bytes}``; everything else is
plain msgpack. ``pack`` and ``unpack`` are symmetric and keep dicts, lists, strings, numbers, booleans, None.
"""
from __future__ import annotations

import msgpack
import numpy as np

MAX_MESSAGE = 64 * 1024 * 1024   # bytes per websocket message, server and client alike (a few 84x84 cameras fit easily)


def _encode(obj):
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {"__nd__": True, "dtype": arr.dtype.str, "shape": list(arr.shape), "data": arr.tobytes()}
    if isinstance(obj, (np.generic,)):
        return obj.item()
    raise TypeError(f"cannot pack {type(obj).__name__}")


def _decode(obj):
    if isinstance(obj, dict) and obj.get("__nd__"):
        return np.frombuffer(obj["data"], dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()
    return obj


def pack(message) -> bytes:
    """Serialize a message (dict/list/scalars/numpy arrays) to bytes."""
    return msgpack.packb(message, default=_encode, use_bin_type=True)


def unpack(data: bytes):
    """Inverse of ``pack``."""
    return msgpack.unpackb(data, object_hook=_decode, raw=False)
