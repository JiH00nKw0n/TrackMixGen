"""Transit Pacing: error-bounded path following on the bridge (transit) waypoints only.

The motivation (the success filter keeps scenes close to the source because the tracking error builds up over the
transit before a skill and is largest when the skill starts) puts the problem in the transit, not in the skill. This
follower therefore leaves every skill waypoint at the official speed (one waypoint per control step, ds = 1) and paces
only the bridge waypoints, without hand-set gains:

- One-step response model of the controller, fitted on generation records (2026-11-17, MimicGen Adaptive Pacing pools of
  Coffee, Square, Stack Three and Threading, bridge steps): x_{t+1} - x_t = k (u_t - x_t), with k_pos ~ 0.05-0.07 for the
  position and k_rot ~ 0.08-0.15 for the rotation (rotation vector of the target in the hand frame).
- On a bridge waypoint the follower takes the largest advance ds in [0, 1] whose predicted next error stays within the
  transit bound: (1 - k_pos) |p(s + ds) - x_t| <= eps_pos and (1 - k_rot) angle(R(s + ds), R_t) <= eps_rot. The target
  thus runs ahead of the hand by at most what the controller can close, which is the time-optimal advance under that
  bound. ds = 0 (wait) when even the current target violates it.
- At the last bridge waypoint before a skill (the bridge end), the follower waits until the measured error is within the
  settle bound (eps_settle_pos, eps_settle_rot) before the skill starts, so the error at the skill start is bounded
  directly.
- Gripper events are never skipped (as in Adaptive Pacing), and once a path has used as many extra steps as it has
  waypoints the follower advances one waypoint per step (``extra_steps_cap: segment``), so a path always ends.

The bounds are powers of ten chosen from the measured lag (2026-11-17): the bridge target ran 93-173 mm ahead of the hand
(median), so eps_pos = 0.1 m keeps the bridge at a usable speed (about k_pos * 0.1 m per step), and the settle bound
0.01 m makes the skill start within one centimetre.
"""
from __future__ import annotations

import math

import numpy as np

from tmg.core.poses import pose_gap
from tmg.generation.method.grippers import build_detector
from tmg.generation.method.tracking import EPS, KIND_BRIDGE, ErrorAdaptiveFollower, clamp_advance

GRID = np.linspace(1.0, 0.0, 41)          # candidate advances, largest first (1/40 waypoint resolution)


class TransitPacingFollower(ErrorAdaptiveFollower):
    """``tracking.type: transit_pacing`` (see the module docstring)."""

    def __init__(self, *, k_pos: float, k_rot: float, eps_pos: float, eps_rot: float, eps_settle_pos: float,
                 eps_settle_rot: float, extra_steps_cap: str, detector=None):
        super().__init__(ds_min=1.0, regions={"default": {"alpha_p": 0.0, "alpha_r": 0.0}}, approach_window=0,
                         extra_steps_cap=extra_steps_cap, detector=detector)
        if not (0.0 < k_pos < 1.0 and 0.0 < k_rot < 1.0):
            raise ValueError("tracking.k_pos and k_rot must be in (0, 1)")
        self.k_pos, self.k_rot = float(k_pos), float(k_rot)
        self.eps_pos, self.eps_rot = float(eps_pos), float(eps_rot)
        self.eps_settle_pos, self.eps_settle_rot = float(eps_settle_pos), float(eps_settle_rot)

    @classmethod
    def factory(cls, cfg: dict):
        """Zero-argument constructor for the ``method.tracking`` block (one follower per trajectory)."""
        cfg = dict(cfg)

        def make():
            return cls(k_pos=cfg["k_pos"], k_rot=cfg["k_rot"], eps_pos=cfg["eps_pos"], eps_rot=cfg["eps_rot"],
                       eps_settle_pos=cfg["eps_settle_pos"], eps_settle_rot=cfg["eps_settle_rot"],
                       extra_steps_cap=cfg.get("extra_steps_cap", "segment"),
                       detector=build_detector(cfg.get("gripper_events")))

        make.describe = lambda: {"type": "transit_pacing", **{k: v for k, v in cfg.items() if k != "enabled"}}
        return make

    # rule -----------------------------------------------------------------------------------------------
    def _bridge_end(self, k: int) -> bool:
        """True when waypoint k is a bridge waypoint followed by a skill waypoint (or the path end)."""
        return self.kinds[k] == KIND_BRIDGE and (k + 1 >= self.n or self.kinds[k + 1] != KIND_BRIDGE)

    def _predicted_ok(self, s: float, measured: dict) -> bool:
        """Whether the predicted next error toward target p(s) stays within the transit bound for every end effector."""
        for name, target in self.targets(s).items():
            p, r = pose_gap(target, measured[name])
            if (1.0 - self.k_pos) * p > self.eps_pos + EPS or (1.0 - self.k_rot) * r > self.eps_rot + EPS:
                return False
        return True

    def advance(self, s: float, measured: dict, e_pos: float, e_rot: float) -> float:
        """Proposed advance from path variable ``s`` (before the event clamp)."""
        k = int(math.floor(s + EPS))
        if self.kinds[k] != KIND_BRIDGE:
            return 1.0                                        # skill: official speed
        if self._bridge_end(k) and abs(s - k) < EPS:          # at the bridge end: settle before the skill starts
            return 1.0 if (e_pos <= self.eps_settle_pos and e_rot <= self.eps_settle_rot) else 0.0
        nxt = [j for j in range(k + 1, self.n) if self._bridge_end(j)]
        limit = float(nxt[0]) if nxt else self.s_end          # never run past the bridge end in one step
        for ds in GRID:
            cand = min(s + float(ds), limit)
            if self._predicted_ok(cand, measured):
                return cand - s
        return 0.0

    def next_target(self, measured: dict):
        """As :meth:`ErrorAdaptiveFollower.next_target` with the transit rule; the record's ``region`` is
        ``bridge``, ``settle`` or ``skill``."""
        if self._finished:
            raise IndexError("path already finished")
        before = self.s
        k_before = 0 if before is None else int(math.floor(before + EPS))
        e_pos, e_rot = self.error(0.0 if before is None else before, measured)
        capped = self.cap == "segment" and self.extra >= self.n
        if before is None:
            s, ds_proposed, region = 0.0, 0.0, "start"
        elif capped:
            s = min(float(math.floor(before + EPS) + 1), self.s_end)
            ds_proposed, region = s - before, "capped"
        else:
            ds_proposed = self.advance(before, measured, e_pos, e_rot)
            region = ("skill" if self.kinds[k_before] != KIND_BRIDGE else
                      "settle" if self._bridge_end(k_before) and abs(before - k_before) < EPS else "bridge")
            s = clamp_advance(before, ds_proposed, self.events, self.s_end) if ds_proposed > 0 else before
        k = int(math.floor(s + EPS))
        new_slots = list(range(self.next_slot, k + 1))
        self.next_slot = max(self.next_slot, k + 1)
        self.steps += 1
        self.extra = self.steps - (k + 1)
        self._finished = s >= self.s_end - EPS
        if self._finished:
            new_slots += list(range(self.next_slot, self.n))
            self.next_slot = self.n
        self.s = s
        record = {"s_before": -1.0 if before is None else float(before), "s_after": float(s),
                  "ds_proposed": float(ds_proposed), "ds_actual": float(s - (0.0 if before is None else before)),
                  "k": k, "region": region, "capped": bool(capped), "new_slots": new_slots,
                  "e_pos": float(e_pos), "e_rot": float(e_rot), "kind": self.kinds[k]}
        return self.targets(s), self.grippers_at(k), self._noises[k], record
