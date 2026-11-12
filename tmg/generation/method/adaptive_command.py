"""Adaptive Command: path following that corrects the next command from the current tracking error within the robot's
one-step reach (tracking.type: adaptive_command). Proposal: reports/tracking_method_b_proposal_20261113.md (방법론 B).

Adaptive Pacing (``ErrorAdaptiveFollower``, tracking.type adaptive_pacing / error_adaptive) slows the progress along the
path when the hand lags and always commands the reference path point. Adaptive Command instead keeps the progress at one
waypoint per step whenever the next reference point is reachable in one step and changes the COMMAND so that the hand
lands on it.

Per end effector, with the measured position x_t and orientation R_t, the previous step's motion dx_t = x_t - x_{t-1}
and w_t = Log(R_t R_{t-1}^T), and the reference path p(s), Q(s):

1. One-step response model (fitted per task from source demonstrations; Coffee: a_p 0.629, b_p 0.0936, a_r 0.712,
   b_r 0.0663):  x_{t+1} = x_t + a_p dx_t + b_p D_p u_p,   w_{t+1} = a_r w_t + b_r D_r u_r,   R_{t+1} = Exp(w_{t+1}) R_t,
   with the action u in [-1, 1]^6 and the controller's per-step limits D_p (0.05 m) and D_r (0.5 rad).
2. Desired next pose for a path value s, keeping (1 - kappa) of the current error e_t = x_t - p(s_t),
   E_t = R_t Q(s_t)^T:  x_des(s) = p(s) + (1 - kappa) e_t,  R_des(s) = Exp((1 - kappa) Log E_t) Q(s).
3. Unclipped action that reaches it: u~_p(s) = (x_des(s) - x_t - a_p dx_t) / (b_p D_p),
   u~_r(s) = (Log(R_des(s) R_t^T) - a_r w_t) / (b_r D_r). The part clipping cuts off is the shortfall rho(s).
4. Advance: the largest s in [s_t + ds_min, min(s_t + 1, next gripper event, end)] (searched from ds 1 downward in steps
   of ``ds_step``) whose shortfall is at most eps_p and eps_r for every end effector; ds_min when none.
5. Action: u* = clip((u~(s_{t+1}) + lambda u*_{t-1}) / (1 + lambda)) (the closed form of the box-constrained least
   squares with a smoothness term, the model being axis-separable).
6. Command pose sent to the controller: x_cmd = x_t + D_p u*_p, R_cmd = Exp(D_r u*_r) R_t. The official
   ``target_pose_to_action`` computes (target - current) / D_p, so the executed pre-noise action is exactly u*.

Safeguards: when the previous prediction missed the measured position by more than ``residual_max`` (the hand is blocked
by contact, the model does not hold) the step advances by 1 and commands the reference point unchanged, as the official
generator does. Before a gripper event the hand must arrive within ``arrive_pos``/``arrive_rot`` of p(event) (at most
``arrive_max_wait`` extra steps, the gripper keeping its pre-event command) before the event is crossed. Bookkeeping
(``k``, ``new_slots``, the extra-step cap) is the same as Adaptive Pacing's, so the runtime's noise accounting and the
pre-noise labels work unchanged; each record also carries ``ref`` (the reference path point, recorded as r_pos/r_rot).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from tmg.core.poses import pose_gap
from tmg.generation.method.grippers import GripperEventDetector, SignGripper, build_detector
from tmg.generation.method.tracking import EPS, ErrorAdaptiveFollower, clamp_advance

DEFAULTS = dict(a_p=0.629, b_p=0.0936, a_r=0.712, b_r=0.0663, max_dpos=0.05, max_drot=0.5, kappa=0.5, smooth=1.0,
                ds_min=0.25, ds_step=0.05, eps_pos=0.002, eps_rot_deg=1.0, residual_max=0.003, arrive_pos=0.005,
                arrive_rot_deg=3.0, arrive_max_wait=5)


def rot_log(R) -> np.ndarray:
    """Axis-angle vector of a rotation matrix."""
    return Rotation.from_matrix(np.asarray(R, dtype=float)).as_rotvec()


def rot_exp(v) -> np.ndarray:
    """Rotation matrix of an axis-angle vector."""
    return Rotation.from_rotvec(np.asarray(v, dtype=float)).as_matrix()


class AdaptiveCommandFollower(ErrorAdaptiveFollower):
    """The ``PathFollower`` of the module docstring. Shares path loading, regions, events and gripper handling with
    Adaptive Pacing; only the advance and the commanded pose differ."""

    def __init__(self, *, detector: GripperEventDetector | None = None, extra_steps_cap: str = "segment",
                 approach_window: int = 10, **values):
        unknown = set(values) - set(DEFAULTS)
        if unknown:
            raise KeyError(f"tracking (adaptive_command): unknown keys {sorted(unknown)}")
        v = {**DEFAULTS, **values}
        super().__init__(ds_min=v["ds_min"], regions={"default": {"alpha_p": 0.0, "alpha_r": 0.0}},
                         approach_window=approach_window, extra_steps_cap=extra_steps_cap, detector=detector or SignGripper())
        self.a_p, self.b_p, self.a_r, self.b_r = (float(v[k]) for k in ("a_p", "b_p", "a_r", "b_r"))
        self.dp, self.dr = float(v["max_dpos"]), float(v["max_drot"])
        if not (0 < self.b_p and 0 < self.b_r and self.dp > 0 and self.dr > 0):
            raise ValueError("b_p, b_r, max_dpos and max_drot must be positive")
        self.kappa = float(v["kappa"])
        if not 0.0 < self.kappa <= 1.0:
            raise ValueError("kappa must lie in (0, 1]")
        self.smooth = float(v["smooth"])
        self.ds_step = float(v["ds_step"])
        self.eps_pos, self.eps_rot = float(v["eps_pos"]), math.radians(float(v["eps_rot_deg"]))
        self.residual_max = float(v["residual_max"])
        self.arrive_pos, self.arrive_rot = float(v["arrive_pos"]), math.radians(float(v["arrive_rot_deg"]))
        self.arrive_max_wait = int(v["arrive_max_wait"])

    @classmethod
    def factory(cls, cfg: dict):
        cfg = dict(cfg)
        values = {k: cfg[k] for k in DEFAULTS if k in cfg}

        def make():
            return cls(detector=build_detector(cfg.get("gripper_events")), extra_steps_cap=cfg.get("extra_steps_cap", "segment"),
                       approach_window=cfg.get("approach_window", 10), **values)

        make.describe = lambda: {"type": "adaptive_command", **{k: v for k, v in cfg.items() if k != "enabled"}}
        return make

    def start_path(self, eefs, noises, kinds, previous_states=None):
        super().start_path(eefs, noises, kinds, previous_states)
        self._prev = {}          # eef -> previous measured pose (4x4)
        self._u = {}             # eef -> previous action (6,)
        self._pred = {}          # eef -> predicted position for this step
        self._wait = 0

    # model ------------------------------------------------------------------------------------------
    def _state(self, name: str, measured: np.ndarray):
        x, R = measured[:3, 3], measured[:3, :3]
        prev = self._prev.get(name)
        if prev is None:
            return x, R, np.zeros(3), np.zeros(3)
        return x, R, x - prev[:3, 3], rot_log(R @ prev[:3, :3].T)

    def _unclipped(self, s: float, s_now: float, name: str, measured: np.ndarray):
        """(u~ (6,), shortfall pos (m), shortfall rot (rad)) of reaching the desired pose at path value s."""
        x, R, dx, w = self._state(name, measured)
        ref_now = self.targets(s_now)[name]
        ref = self.targets(s)[name]
        e = x - ref_now[:3, 3]
        E = R @ ref_now[:3, :3].T
        x_des = ref[:3, 3] + (1.0 - self.kappa) * e
        R_des = rot_exp((1.0 - self.kappa) * rot_log(E)) @ ref[:3, :3]
        up = (x_des - x - self.a_p * dx) / (self.b_p * self.dp)
        ur = (rot_log(R_des @ R.T) - self.a_r * w) / (self.b_r * self.dr)
        short_p = self.b_p * self.dp * float(np.linalg.norm(up - np.clip(up, -1, 1)))
        short_r = self.b_r * self.dr * float(np.linalg.norm(ur - np.clip(ur, -1, 1)))
        return np.concatenate([up, ur]), short_p, short_r

    def _feasible(self, s: float, s_now: float, measured: dict) -> bool:
        for name in measured:
            _, sp, sr = self._unclipped(s, s_now, name, measured[name])
            if sp > self.eps_pos + EPS or sr > self.eps_rot + EPS:
                return False
        return True

    def _command(self, s: float, s_now: float, name: str, measured: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        u_tilde, _, _ = self._unclipped(s, s_now, name, measured)
        prev_u = self._u.get(name, np.zeros(6))
        u = np.clip((u_tilde + self.smooth * prev_u) / (1.0 + self.smooth), -1.0, 1.0)
        x, R = measured[:3, 3], measured[:3, :3]
        cmd = np.eye(4)
        cmd[:3, 3] = x + self.dp * u[:3]
        cmd[:3, :3] = rot_exp(self.dr * u[3:]) @ R
        return cmd, u

    # one step ---------------------------------------------------------------------------------------
    def next_target(self, measured: dict):
        if self._finished:
            raise IndexError("path already finished")
        measured = {name: np.asarray(measured[name], dtype=float) for name in (e.name for e in self.eefs)}
        before = self.s
        s_now = 0.0 if before is None else before
        k_before = int(math.floor(s_now + EPS))
        e_pos, e_rot = self.error(s_now, measured)
        residual = max((float(np.linalg.norm(measured[n][:3, 3] - p)) for n, p in self._pred.items()), default=0.0)
        fallback = before is not None and residual > self.residual_max
        capped = self.cap == "segment" and self.extra >= self.n
        waiting = False
        if before is None:
            s, ds_proposed = 0.0, 0.0
        elif capped:
            s = min(float(math.floor(before + EPS) + 1), self.s_end)
            ds_proposed = s - before
        elif fallback:
            ds_proposed = 1.0
            s = clamp_advance(before, 1.0, self.events, self.s_end)
        else:
            s, ds_proposed = None, self.ds_min
            ds = 1.0
            while ds >= self.ds_min - EPS:
                cand = clamp_advance(before, ds, self.events, self.s_end)
                if self._feasible(cand, s_now, measured):
                    s, ds_proposed = cand, ds
                    break
                ds = round(ds - self.ds_step, 10)
            if s is None:
                s = clamp_advance(before, self.ds_min, self.events, self.s_end)
        # arrival check before crossing a gripper event: hold just short of it, pre-event gripper command
        if before is not None and any(abs(s - ev) < EPS and ev > before + EPS for ev in self.events):
            gaps = [pose_gap(self.targets(s)[n], measured[n]) for n in measured]
            far = any(gp > self.arrive_pos or gr > self.arrive_rot for gp, gr in gaps)
            if far and self._wait < self.arrive_max_wait:
                self._wait += 1
                waiting = True
                s = max(before, s - 1e-6)
            else:
                self._wait = 0
        k = int(math.floor(s + EPS))
        refs = self.targets(s)
        if fallback or before is None:
            targets = refs
            actions = {n: None for n in measured}
        else:
            targets, actions = {}, {}
            for n in measured:
                targets[n], actions[n] = self._command(s, s_now, n, measured[n])
        # bookkeeping shared with Adaptive Pacing
        new_slots = list(range(self.next_slot, k + 1))
        self.next_slot = max(self.next_slot, k + 1)
        self.steps += 1
        self.extra = self.steps - (k + 1)
        self._finished = s >= self.s_end - EPS
        if self._finished:
            new_slots += list(range(self.next_slot, self.n))
            self.next_slot = self.n
        # model state for the next step
        self._pred = {}
        for n in measured:
            x, R, dx, _ = self._state(n, measured[n])
            u = actions[n] if actions[n] is not None else np.clip((targets[n][:3, 3] - x) / self.dp, -1, 1).tolist() + [0, 0, 0]
            u = np.asarray(u, dtype=float)
            self._pred[n] = x + self.a_p * dx + self.b_p * self.dp * u[:3]
            self._u[n] = u
            self._prev[n] = measured[n]
        self.s = s
        record = {"s_before": -1.0 if before is None else float(before), "s_after": float(s),
                  "ds_proposed": float(ds_proposed), "ds_actual": float(s - s_now) if before is not None else 0.0,
                  "k": k, "region": self.region[k_before], "capped": bool(capped), "new_slots": new_slots,
                  "e_pos": float(e_pos), "e_rot": float(e_rot), "kind": self.kinds[k], "ref": refs,
                  "fallback": bool(fallback), "waiting": bool(waiting), "residual": float(residual)}
        return targets, self.grippers_at(k), self._noises[k], record
