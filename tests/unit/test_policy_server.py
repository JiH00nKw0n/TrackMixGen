"""``tmg serve`` keeps answering keepalive pings while a request is computed (2026-11-01: a Diffusion Policy action that took
over 40 s on a busy T4 left the client's ping unanswered, the client closed the connection and every scene failed)."""
import asyncio
import threading
import time

import numpy as np
import pytest

websockets = pytest.importorskip("websockets")

from tmg.core.wire import MAX_MESSAGE, pack, unpack
from tmg.policy.server import PolicyRunner


class SlowRunner(PolicyRunner):
    """A runner without a checkpoint whose ``act`` blocks longer than the client's ping timeout."""

    def __init__(self, delay: float):
        self.delay = delay
        self._lock = asyncio.Lock()

    def act(self, obs):
        time.sleep(self.delay)
        return np.zeros(3, np.float32)


async def blocking_handle(runner, websocket):
    """The handler before 2026-11-02: the request ran on the event loop."""
    async for raw in websocket:
        await websocket.send(pack(runner.answer(unpack(raw))))


def serve_in_thread(handler) -> int:
    """Start a websocket server on its own event loop in a daemon thread; returns its port."""
    ready, box = threading.Event(), {}

    async def main():
        async with websockets.serve(handler, "127.0.0.1", 0, max_size=MAX_MESSAGE) as server:
            box["port"] = server.sockets[0].getsockname()[1]
            ready.set()
            await asyncio.Future()

    threading.Thread(target=lambda: asyncio.run(main()), daemon=True).start()
    assert ready.wait(10)
    return box["port"]


def act_once(port: int):
    """One act request from a client that pings every 0.2 s and gives up after 0.5 s without a pong."""
    async def main():
        async with websockets.connect(f"ws://127.0.0.1:{port}", ping_interval=0.2, ping_timeout=0.5, max_size=MAX_MESSAGE) as ws:
            await ws.send(pack({"type": "act", "obs": {}}))
            return unpack(await asyncio.wait_for(ws.recv(), timeout=10))
    return asyncio.run(main())


def test_answer_dispatches_the_three_request_types():
    runner = SlowRunner(0.0)
    runner.reset = lambda seed=None, task=None: None
    runner.info = lambda: {"policy": "x"}
    assert runner.answer({"type": "info"}) == {"policy": "x"}
    assert runner.answer({"type": "reset", "seed": 1}) == {"ok": True}
    assert runner.answer({"type": "act", "obs": {}})["action"].shape == (3,)
    assert "unknown request type" in runner.answer({"type": "bogus"})["error"]


def test_a_slow_action_keeps_the_connection():
    runner = SlowRunner(delay=1.5)
    reply = act_once(serve_in_thread(runner.handle))
    assert np.asarray(reply["action"]).shape == (3,)


def test_the_old_blocking_handler_lost_the_connection():
    runner = SlowRunner(delay=1.5)
    port = serve_in_thread(lambda ws: blocking_handle(runner, ws))
    with pytest.raises(websockets.exceptions.ConnectionClosed):
        act_once(port)
