"""Synchronous websocket client of the policy server (protocol in ``tmg.policy.server``)."""
from __future__ import annotations

import inspect

import numpy as np

from tmg.core.wire import MAX_MESSAGE, pack, unpack


def keepalive_off(connect) -> dict:
    """``{"ping_interval": None}`` when ``connect`` (``websockets.sync.client.connect``) takes that argument, else ``{}``.

    The policy server sends no keepalive pings and the client must not either (``tmg.policy.server``). The sync client of
    websockets 14 and later sends them unless ``ping_interval=None``; the sync client of websockets 13 (13.1 in the
    robosuite image, ``requirements/robosuite.lock``) has no keepalive and no such argument, and passing it raised
    ``TypeError`` on every connection, so every evaluation in that image failed (2026-11-21). Only that argument depends
    on the installed version; any other ``TypeError`` of ``connect`` still surfaces.
    """
    try:
        params = inspect.signature(connect).parameters
    except (TypeError, ValueError):          # no signature to read: pass the argument as before
        return {"ping_interval": None}
    return {"ping_interval": None} if "ping_interval" in params else {}


class PolicyClient:
    """``with PolicyClient(url) as policy: policy.reset(); a = policy.act(obs)``."""

    def __init__(self, url: str, *, timeout_s: float = 120.0):
        self.url = url
        self.timeout_s = timeout_s
        self._ws = None

    def __enter__(self):
        from websockets.sync.client import connect

        self._ws = connect(self.url, max_size=MAX_MESSAGE, open_timeout=self.timeout_s, **keepalive_off(connect))
        return self

    def __exit__(self, *exc):
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    def _call(self, message: dict) -> dict:
        self._ws.send(pack(message))
        try:
            reply = unpack(self._ws.recv(timeout=self.timeout_s))
        except TimeoutError:      # a late reply would answer the next request: start a fresh connection (2026-11-02)
            self.__exit__()
            self.__enter__()
            raise
        if "error" in reply:
            raise RuntimeError(f"policy server: {reply['error']}")
        return reply

    def info(self) -> dict:
        return self._call({"type": "info"})

    def reset(self, seed: int | None = None, task: str | None = None) -> None:
        self._call({"type": "reset", "seed": seed, "task": task})

    def act(self, obs: dict) -> np.ndarray:
        return np.asarray(self._call({"type": "act", "obs": obs})["action"], dtype=np.float32)
