"""Synchronous websocket client of the policy server (protocol in ``tmg.policy.server``)."""
from __future__ import annotations

import numpy as np

from tmg.core.wire import MAX_MESSAGE, pack, unpack


class PolicyClient:
    """``with PolicyClient(url) as policy: policy.reset(); a = policy.act(obs)``."""

    def __init__(self, url: str, *, timeout_s: float = 120.0):
        self.url = url
        self.timeout_s = timeout_s
        self._ws = None

    def __enter__(self):
        from websockets.sync.client import connect

        self._ws = connect(self.url, max_size=MAX_MESSAGE, open_timeout=self.timeout_s)
        return self

    def __exit__(self, *exc):
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    def _call(self, message: dict) -> dict:
        self._ws.send(pack(message))
        reply = unpack(self._ws.recv(timeout=self.timeout_s))
        if "error" in reply:
            raise RuntimeError(f"policy server: {reply['error']}")
        return reply

    def info(self) -> dict:
        return self._call({"type": "info"})

    def reset(self, seed: int | None = None, task: str | None = None) -> None:
        self._call({"type": "reset", "seed": seed, "task": task})

    def act(self, obs: dict) -> np.ndarray:
        return np.asarray(self._call({"type": "act", "obs": obs})["action"], dtype=np.float32)
