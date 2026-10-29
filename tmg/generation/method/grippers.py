"""Gripper event detection: when does a commanded gripper or hand change between open and closed?

The path follower never skips such an event and slows down in front of it, so the definition of an event is
part of the method. Two detectors cover the generators in scope:

- ``sign``: one scalar command per step, positive = close, negative = open, zero carries the previous state
  (robosuite grippers in MimicGen and the SoftMimicGen Franka hand).
- ``closure``: a hand with many finger joints (GR1 in DexMimicGen). The closure is the mean of a few commanded
  joint angles (negated so that larger means more closed), normalised by its 10th and 90th percentiles over the
  path, and an event fires when it crosses 0.65 upward (close) or 0.35 downward (open); the hysteresis keeps
  jitter from creating events. Ported from the 2026-10 Dex experiment (dex_hand_alignment.py).

Both return the indices of the steps whose state differs from the step before, plus the state at the start,
so a path can also be compared with the last state of the previous path.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

CLOSE_THRESHOLD = 0.65      # normalised closure at or above which a many-joint hand counts as closed
OPEN_THRESHOLD = 0.35       # normalised closure at or below which it counts as open again (hysteresis)


class IncompatibleGripperEvents(ValueError):
    """Two gripper command sequences cannot be aligned on the same open/close events."""


def gripper_states(commands) -> np.ndarray:
    """Boolean closed/open per step of scalar sign commands, carrying the state through zero commands."""
    commands = np.asarray(commands, dtype=float).reshape(-1)
    if commands.size == 0 or not np.isfinite(commands).all():
        raise ValueError("gripper commands must be a finite non-empty sequence")
    if commands[0] == 0:
        raise IncompatibleGripperEvents("initial zero command has no known open/closed state")
    state = commands[0] > 0
    out = np.empty(len(commands), dtype=bool)
    for i, value in enumerate(commands):
        if value != 0:
            state = value > 0
        out[i] = state
    return out


def change_indices(states) -> np.ndarray:
    """Indices of the steps whose state differs from the step before (the first step of every new state)."""
    states = np.asarray(states, dtype=bool)
    return np.flatnonzero(states[1:] != states[:-1]) + 1


class GripperEventDetector(Protocol):
    """Protocol of the open/closed state rule a follower uses to find gripper events."""
    def states(self, commands: np.ndarray) -> np.ndarray:
        """Boolean closed/open state per step for commands of shape (T, G)."""
        ...

    def state_of(self, command: np.ndarray, previous: bool | None) -> bool | None:
        """State implied by one command vector (None when the command carries the previous state)."""
        ...


class SignGripper:
    """Scalar commands: positive closes, negative opens, zero keeps the previous state."""
    name = "sign"

    def states(self, commands):
        """Closed/open per step from the first command column."""
        g = np.asarray(commands, dtype=float).reshape(len(commands), -1)[:, 0]
        return gripper_states(g)

    def state_of(self, command, previous):
        """State of one command: ``previous`` when the command is zero."""
        v = float(np.asarray(command, dtype=float).reshape(-1)[0])
        return previous if v == 0 else v > 0


class ClosureGripper:
    """Many-joint hands (GR1): closure = negated mean of ``columns``, normalised by its 10th/90th percentiles
    over the path, with ``CLOSE_THRESHOLD`` / ``OPEN_THRESHOLD`` hysteresis.
    """
    name = "closure"

    def __init__(self, columns, close_threshold: float = CLOSE_THRESHOLD, open_threshold: float = OPEN_THRESHOLD):
        """Args:
            columns: indices of the finger joints whose mean flexion defines the closure.
            close_threshold: normalised closure at or above which the hand counts as closed.
            open_threshold: normalised closure at or below which it counts as open again.
        """
        self.columns = [int(c) for c in columns]
        self.close_threshold = float(close_threshold)
        self.open_threshold = float(open_threshold)

    def closure(self, commands) -> np.ndarray:
        """Raw closure per step (negated mean of the chosen columns)."""
        g = np.asarray(commands, dtype=float).reshape(len(commands), -1)
        return -g[:, self.columns].mean(axis=1)

    def states(self, commands):
        """Closed/open per step with hysteresis; all open when the path has no closure range."""
        closure = self.closure(commands)
        lo, hi = np.percentile(closure, [10, 90])
        if hi - lo < 1e-6:
            return np.zeros(len(closure), dtype=bool)
        z = (closure - lo) / (hi - lo)
        closed = bool(z[0] >= self.close_threshold)
        out = np.empty(len(z), dtype=bool)
        out[0] = closed
        for i, v in enumerate(z[1:], 1):
            if not closed and v >= self.close_threshold:
                closed = True
            elif closed and v <= self.open_threshold:
                closed = False
            out[i] = closed
        return out

    def state_of(self, command, previous):
        """A single command carries no scale, so the previous state is kept."""
        return previous      # a single command has no scale; the state is only known over a path


def event_indices(detector: GripperEventDetector, commands, previous_state: bool | None) -> list[int]:
    """Step indices whose state differs from the step before; index 0 counts when ``previous_state`` differs."""
    states = detector.states(commands)
    out = [int(i) for i in change_indices(states)]
    if previous_state is not None and len(states) and bool(states[0]) != bool(previous_state):
        out.insert(0, 0)
    return out


def build_detector(cfg) -> GripperEventDetector:
    """Build a detector from ``method.tracking.gripper_events`` (``{type: sign}`` or ``{type: closure, columns: [...]}``)."""
    cfg = dict(cfg or {"type": "sign"})
    kind = str(cfg.get("type", "sign"))
    if kind == "sign":
        return SignGripper()
    if kind == "closure":
        return ClosureGripper(cfg["columns"], cfg.get("close_threshold", CLOSE_THRESHOLD),
                              cfg.get("open_threshold", OPEN_THRESHOLD))
    raise KeyError(f"unknown gripper event detector {kind!r} (sign | closure)")
