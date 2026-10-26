"""``tmg serve``: a websocket policy server around one checkpoint.

The evaluation loop runs in the simulator image and cannot import the policy stack (different Python and
torch), so it talks to this server. Protocol (msgpack, ``tmg.core.wire``), one request per message:

    {"type": "info"}                      -> {"policy": name, "checkpoint": path, "cameras": [...], "state_dim": D,
                                              "env_state_dim": E (0 for image policies), "action_dim": A}
    {"type": "reset", "seed": int|None, "task": str|None} -> {"ok": true}  episode start: clears queues/hidden state; the
                                             task instruction (π0) is kept for the episode
    {"type": "act", "obs": {...}}         -> {"action": (A,) float32}

``obs`` carries ``observation.images.<cam>`` as (3, H, W) uint8, ``observation.state`` as (D,) float32 and, for a
low-dim policy, ``observation.environment_state`` as (E,) float32, exactly the dataset layout. The server adds the batch axis and runs the checkpoint's ``PolicyIO`` (our
normalizer, or LeRobot's pipelines for π0) before ``select_action`` and after it. One connection serves one
episode at a time; connections are handled sequentially per policy instance (the hidden state is per policy).
"""
from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import torch

from tmg.config import resolved
from tmg.core.wire import pack, unpack
from tmg.policy.io import load_io
from tmg.policy.lerobot_policies import load_policy, policy_name_of
from tmg.policy.trainer import PRETRAINED_DIR


class PolicyRunner:
    """Holds the policy and answers the three request types."""

    def __init__(self, checkpoint: Path, device: str):
        self.checkpoint = Path(checkpoint)
        self.device = device
        self.policy = load_policy(self.checkpoint, device)
        self.policy.eval()
        cfg = self.policy.config
        self.policy_name = policy_name_of(cfg)
        self.io = load_io(self.checkpoint, self.policy_name, cfg, device)
        self.task = None
        self.cameras = [k.split("observation.images.", 1)[1] for k in cfg.image_features]
        self.state_dim = int(cfg.robot_state_feature.shape[0])
        self.env_state_dim = int(cfg.env_state_feature.shape[0]) if cfg.env_state_feature is not None else 0
        self.action_dim = int(cfg.action_feature.shape[0])
        self._lock = asyncio.Lock()

    def info(self) -> dict:
        return {"policy": self.policy.name, "checkpoint": str(self.checkpoint), "cameras": self.cameras,
                "state_dim": self.state_dim, "env_state_dim": self.env_state_dim, "action_dim": self.action_dim,
                "device": self.device}

    def reset(self, seed: int | None = None, task: str | None = None) -> None:
        self.task = task
        if seed is not None:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed) % (2**32))
        self.policy.reset()

    @torch.no_grad()
    def act(self, obs: dict) -> np.ndarray:
        """One action for one observation dict (layout in the module docstring)."""
        batch = {}
        for cam in self.cameras:
            batch[f"observation.images.{cam}"] = torch.as_tensor(obs[f"observation.images.{cam}"])[None]
        batch["observation.state"] = torch.as_tensor(obs["observation.state"], dtype=torch.float32)[None]
        if self.env_state_dim:
            batch["observation.environment_state"] = torch.as_tensor(obs["observation.environment_state"], dtype=torch.float32)[None]
        if self.task is not None:
            batch["task"] = [self.task]
        batch = self.io.prepare(batch)
        action = self.policy.select_action(batch)
        action = self.io.postprocess(action)
        return action[0].detach().cpu().numpy().astype(np.float32)

    async def handle(self, websocket):
        """Serve one connection until it closes."""
        async for raw in websocket:
            req = unpack(raw)
            kind = req.get("type")
            async with self._lock:
                if kind == "info":
                    reply = self.info()
                elif kind == "reset":
                    self.reset(req.get("seed"), req.get("task"))
                    reply = {"ok": True}
                elif kind == "act":
                    reply = {"action": self.act(req["obs"])}
                else:
                    reply = {"error": f"unknown request type {kind!r}"}
            await websocket.send(pack(reply))


async def serve_forever(runner: PolicyRunner, host: str, port: int) -> None:
    import websockets

    async with websockets.serve(runner.handle, host, port, max_size=64 * 1024 * 1024):
        print(f"[tmg] policy server {runner.policy.name} from {runner.checkpoint} on ws://{host}:{port}", flush=True)
        await asyncio.Future()


def serve(cfg) -> None:
    """Run the server for ``cfg.checkpoint`` on ``cfg.server.host:port`` until killed."""
    plain = resolved(cfg)
    checkpoint = Path(plain["checkpoint"])
    if not (checkpoint / PRETRAINED_DIR).exists():
        raise FileNotFoundError(f"{checkpoint} has no {PRETRAINED_DIR}/")
    runner = PolicyRunner(checkpoint, str(plain.get("device", "cuda")))
    asyncio.run(serve_forever(runner, str(plain["server"]["host"]), int(plain["server"]["port"])))
