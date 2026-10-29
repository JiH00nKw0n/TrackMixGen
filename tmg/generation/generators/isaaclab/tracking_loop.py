"""The shared-path tracking loop that replaces ``DataGenerator.generate`` when ``method.tracking.enabled`` is on.

The official loop advances one waypoint per step for each end effector separately, so a follower that moves both hands
along one path variable cannot be hosted by grafts. ``TrackingLoop.run`` is that replacement loop; everything that differs
between the official packages (how the environment is reset, which builder makes a trajectory, where the subtask settings
live, how a waypoint carries its hand command, the ``MultiWaypoint.execute`` signature and the recorder calls at the end) is a
hook that ``isaaclab.runtime.IsaacLabTrackingLoop`` and ``autodata.runtime.AutoDataTrackingLoop`` fill in.

The loop (a port of the validated 2026-10-26 bimanual implementation into this repository's follower):

1. ``randomize_subtask_boundaries`` once; for each end effector whose trajectory is missing, call the official builder for
   its current subtask (all end effectors share the selected source; checked).
2. Cut every end effector's trajectory to the common source-time interval ``[start, min(end))`` (``cut_interval``); an end
   effector whose official bridge belongs to another interval gets a linear bridge of the same length so that both
   timelines have one waypoint per base step.
3. One follower for the interval over all end effectors (shared path variable, gripper events from the union of the hands,
   errors on the configured links). Each step builds a ``MultiWaypoint`` from the follower's targets, replays the official
   noise bundle of the base waypoint the step covers (or a private draw for extra steps) and executes it through the
   official ``MultiWaypoint.execute``.
4. When the interval ends, advance the end effectors whose subtask ended, and continue until every end effector finished
   its subtasks. Success and the official recorder's export happen exactly as in the official loop.

Random streams: ``NoiseLedger`` draws every official per-step bundle up front from the global torch generator (the values
the official loop would have drawn in order) and hands bundle ``slot`` to the step that commands base waypoint ``slot``;
extra steps draw from a private generator seeded with ``tmg.generation.seeds.private_noise_seed`` (chunk seed + 7919).
"""
from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from tmg.generation.method.tracking import EndEffectorPath

KIND_BRIDGE, KIND_SKILL = 0, 1


def _np(x) -> np.ndarray:
    return np.asarray(x.detach().cpu().numpy() if hasattr(x, "detach") else x, dtype=float)


# noise bookkeeping ----------------------------------------------------------------------------------
class NoiseLedger:
    """The official per-step noise draws, one bundle per base waypoint, plus a private stream for extra steps.

    ``target_eef_pose_to_action`` draws ``torch.randn_like`` once per shape in ``shapes`` at every step. To keep
    the official random stream, the ledger draws all ``n`` bundles up front from the global generator (the same
    values the official loop would have drawn in order) and hands bundle ``slot`` to the step that commands base
    waypoint ``slot``; steps the follower adds draw from a private generator seeded from the chunk seed.
    """

    def __init__(self, n: int, shapes: tuple[int, ...], device, dtype, private_seed: int):
        import torch

        self.torch, self.shapes, self.device, self.dtype = torch, tuple(shapes), device, dtype
        self.original = [[torch.randn(k, device=device, dtype=dtype) for k in self.shapes] for _ in range(int(n))]
        self.private = torch.Generator(device=device)
        self.private.manual_seed(int(private_seed))
        self.used: set[int] = set()
        self.extra = 0

    def bundle(self, slot: int | None) -> list:
        if slot is not None:
            if slot in self.used or not 0 <= slot < len(self.original):
                raise RuntimeError(f"noise slot {slot} reused or out of range")
            self.used.add(slot)
            return self.original[slot]
        self.extra += 1
        return [self.torch.randn(k, device=self.device, dtype=self.dtype, generator=self.private) for k in self.shapes]


@contextmanager
def replay_randn_like(torch, values: list):
    """Replace ``torch.randn_like`` with the next value of ``values`` for each call while the block runs."""
    calls = [0]
    old = torch.randn_like

    def draw(x, *a, **k):
        if calls[0] >= len(values) or tuple(x.shape) != tuple(values[calls[0]].shape):
            raise RuntimeError("unexpected randn_like call while replaying a noise bundle")
        v = values[calls[0]]
        calls[0] += 1
        return v.clone()

    torch.randn_like = draw
    try:
        yield draw
    finally:
        torch.randn_like = old


_wrap_randn_like = replay_randn_like       # former name


def ledger_size(bounds: dict, steps: dict) -> int:
    """Base waypoints the official loop would command for one source: per end effector the sum over its subtasks of the
    source steps, the bridge steps (interpolation + fixed) and one extra for the first subtask; the largest count wins.

    Args:
        bounds: end effector -> ``(subtasks, 2)`` boundaries of the selected source.
        steps: end effector -> ``[(num_interpolation_steps, num_fixed_steps), ...]`` per subtask.
    """
    def count(e):
        return sum(int(b[1] - b[0]) + int(i == 0) + int(interp) + int(fixed)
                   for i, (b, (interp, fixed)) in enumerate(zip(bounds[e], steps[e])))

    return max(count(e) for e in bounds)


# intervals ------------------------------------------------------------------------------------------
@dataclass
class Interval:
    """A source-time interval [start, end) during which every end effector has an official trajectory."""

    start: int
    end: int
    subtasks: dict[str, int]
    waypoints: dict[str, list]           # eef -> list of official Waypoint objects (bridge + body)
    kinds: list[int]                     # 0 bridge, 1 skill, per waypoint
    source_indices: list[int]


def activated(raw: list, begin: int, end: int, subtask: int) -> dict:
    """The bookkeeping entry of a freshly built trajectory: its waypoints, source span and bridge length."""
    begin, end = int(begin), int(end)
    return {"raw": list(raw), "begin": begin, "end": end, "prefix": len(raw) - (end - begin), "subtask": int(subtask)}


def completed_last(completed: dict, eefs: list[str]) -> dict:
    """Last executed waypoint per end effector (for bridges of the next interval)."""
    return {e: completed[e][-1] for e in eefs if e in completed}


def advance_completed(active: dict, completed: dict, indices: dict, eefs: list[str], end: int) -> None:
    """Move every end effector whose trajectory ends at ``end`` from ``active`` to ``completed`` and step its subtask index."""
    for eef in eefs:
        if active[eef]["end"] == end:
            completed[eef] = active[eef]["raw"]
            del active[eef]
            indices[eef] += 1


def linear_bridge_poses(a: np.ndarray, b: np.ndarray, n: int) -> list[np.ndarray]:
    """``n`` poses from ``a`` towards ``b``: positions interpolate linearly, the rotation switches to ``b``'s at t >= 0.5."""
    out = []
    for j in range(int(n)):
        t = (j + 1) / (n + 1)
        pose = a.copy()
        pose[:3, 3] = (1 - t) * a[:3, 3] + t * b[:3, 3]
        pose[:3, :3] = b[:3, :3] if t >= 0.5 else a[:3, :3]
        out.append(pose)
    return out


def cut_interval(active: dict, eefs: list[str], start: int, Waypoint, previous_last: dict, clone_hand: Callable) -> Interval:
    """Cut the active trajectories to the common source-time interval (step 2 of the module docstring).

    Args:
        active: end effector -> entry of ``activated``.
        eefs: end effectors in the official order.
        start: first source step of the interval.
        Waypoint: the official waypoint class (bridge waypoints are built from it).
        previous_last: end effector -> last executed waypoint (``completed_last``), the start of a linear bridge.
        clone_hand: waypoint -> copy of its hand command, for the linear bridge's waypoints.
    """
    end = min(active[e]["end"] for e in eefs)
    prefixes = {e: active[e]["prefix"] if active[e]["begin"] == start else 0 for e in eefs}
    prefix = max(prefixes.values())
    waypoints, kinds = {}, None
    for e in eefs:
        seg = active[e]
        first = seg["prefix"] + start - seg["begin"]
        body = seg["raw"][first:first + end - start]
        if len(body) != end - start:
            raise RuntimeError(f"{e}: trajectory shorter than its source interval")
        if prefixes[e]:
            if prefixes[e] != prefix:
                raise NotImplementedError("unequal simultaneous bridge lengths are not supported")
            bridge = seg["raw"][:prefix]
        elif prefix:
            prior = previous_last.get(e)
            if prior is None:
                raise RuntimeError(f"{e}: no previous waypoint to bridge from")
            import torch

            template = body[0]
            bridge = [Waypoint(torch.as_tensor(pose, device=template.pose.device, dtype=template.pose.dtype), clone_hand(template),
                               template.noise) for pose in linear_bridge_poses(_np(prior.pose), _np(template.pose), prefix)]
        else:
            bridge = []
        waypoints[e] = bridge + body
        kinds = [KIND_BRIDGE] * len(bridge) + [KIND_SKILL] * len(body)
    return Interval(start=start, end=end, subtasks={e: active[e]["subtask"] for e in eefs}, waypoints=waypoints,
                    kinds=kinds or [], source_indices=[start] * prefix + list(range(start, end)))


# the loop -------------------------------------------------------------------------------------------
@dataclass
class Attempt:
    """Per-attempt state the hooks share: the official generator, the environment index and what the reset produced."""

    generator: Any
    env_id: int
    success_term: Any
    env_action_queue: Any
    eefs: list[str]
    initial_state: Any = None
    env_ids: Any = None                  # the environment index as the recorder wants it (a tensor)
    extra: dict = field(default_factory=dict)


class TrackingLoop:
    """The shared-path loop (module docstring) with one hook per package-specific step.

    ``run`` is the loop; a subclass fills the hooks and its module's ``make_tracking_generate`` wraps ``begin`` + ``run`` in a
    coroutine with the official ``generate`` signature.
    """

    def __init__(self, runtime, follower_factory: Callable, private_seed: int):
        self.runtime = runtime
        self.follower_factory = follower_factory
        self.private_seed = int(private_seed)

    # hooks ---------------------------------------------------------------------------------------
    async def begin(self, generator, env_id: int, success_term, env_reset_queue, env_action_queue) -> Attempt:
        """Reset the environment as the official ``generate`` does and open the attempt (``runtime.begin_attempt``)."""
        raise NotImplementedError

    def lock(self, attempt: Attempt):
        """The async context manager guarding the source pool while trajectories are built."""
        raise NotImplementedError

    def build(self, attempt: Attempt, eef: str, subtask_ind: int, boundaries: dict, selected: dict, previous) -> list:
        """The official builder's trajectory (bridge + transformed source steps) for ``eef``'s subtask; sets ``selected[eef]``."""
        raise NotImplementedError

    def subtask_steps(self, attempt: Attempt, eef: str) -> list[tuple[int, int]]:
        """``(num_interpolation_steps, num_fixed_steps)`` per subtask of ``eef``."""
        raise NotImplementedError

    def num_subtasks(self, attempt: Attempt, eef: str) -> int:
        raise NotImplementedError

    def waypoint_classes(self, attempt: Attempt) -> tuple:
        """``(Waypoint, MultiWaypoint)`` of the official package."""
        raise NotImplementedError

    def waypoint_gripper(self, waypoint, eef: str) -> np.ndarray:
        """The hand command of an official waypoint as a numpy array."""
        raise NotImplementedError

    def clone_hand(self, waypoint):
        """A copy of an official waypoint's hand command (for linear bridge waypoints)."""
        raise NotImplementedError

    def hand_command(self, base, eef: str, gripper: np.ndarray, device):
        """The hand argument of a new waypoint: ``gripper`` as a tensor of ``base``'s hand dtype (plus other channels)."""
        raise NotImplementedError

    async def execute(self, attempt: Attempt, multi_waypoint) -> bool:
        """Run the official ``MultiWaypoint.execute`` for one step; returns the step's success verdict."""
        raise NotImplementedError

    def export(self, attempt: Attempt, success: bool, export_demo: bool) -> dict:
        """Tell the recorder the verdict, export the episode, and return the ``results`` handed to ``runtime.end_attempt``."""
        raise NotImplementedError

    def result(self, attempt: Attempt, results: dict):
        """What the replacement ``generate`` returns."""
        return results

    # shared --------------------------------------------------------------------------------------
    def cut(self, active: dict, eefs: list[str], start: int, Waypoint, previous_last: dict) -> Interval:
        return cut_interval(active, eefs, start, Waypoint, previous_last, self.clone_hand)

    async def run(self, attempt: Attempt, export_demo: bool):
        """Drive one attempt from the (already reset) environment to the exported episode."""
        import torch

        runtime, eefs, env_id = self.runtime, attempt.eefs, attempt.env_id
        Waypoint, MultiWaypoint = self.waypoint_classes(attempt)
        async with self.lock(attempt):
            boundaries = attempt.generator.randomize_subtask_boundaries()
        selected = {eef: None for eef in eefs}
        active: dict[str, dict] = {}
        indices = {eef: 0 for eef in eefs}
        completed: dict[str, list] = {}
        success, offset, start = False, 0, 0
        previous_states: dict | None = None
        ledger: NoiseLedger | None = None
        try:
            while True:
                async with self.lock(attempt):
                    for eef in eefs:
                        if eef in active:
                            continue
                        i = indices[eef]
                        raw = self.build(attempt, eef, i, boundaries, selected, completed.get(eef))
                        begin, end = (int(v) for v in boundaries[eef][int(selected[eef]), i])
                        active[eef] = activated(raw, begin, end, i)
                sources = {int(selected[e]) for e in eefs}
                if len(sources) != 1:
                    raise NotImplementedError("tracking generate needs one shared source demonstration per attempt")
                source = sources.pop()
                interval = self.cut(active, eefs, start, Waypoint, previous_last=completed_last(completed, eefs))
                first = interval.waypoints[eefs[0]][0].pose
                if ledger is None:
                    n_total = ledger_size({e: boundaries[e][source] for e in eefs},
                                          {e: self.subtask_steps(attempt, e) for e in eefs})
                    ledger = NoiseLedger(n_total, runtime.layout.noise_draw_shapes, first.device, first.dtype, self.private_seed)
                paths = [EndEffectorPath(e, np.stack([_np(w.pose) for w in interval.waypoints[e]]),
                                         np.stack([self.waypoint_gripper(w, e) for w in interval.waypoints[e]])) for e in eefs]
                noises = [float(interval.waypoints[eefs[0]][k].noise or 0.0) for k in range(len(interval.source_indices))]
                follower = self.follower_factory()
                follower.start_path(paths, noises, interval.kinds, previous_states)
                runtime.begin_segment(KIND_SKILL)
                runtime.rec.add_extra(interval_start=start, interval_end=interval.end, interval_subtasks=interval.subtasks)
                while not follower.finished():
                    measured = runtime.measured_poses(runtime.action_owner(attempt.generator), env_id, eefs)
                    targets, grippers, _noise, row = follower.next_target(measured)
                    k = int(row["k"])
                    slot = offset + row["new_slots"][0] if row["new_slots"] else None
                    bundle = ledger.bundle(slot)
                    waypoints = {e: Waypoint(torch.as_tensor(targets[e], device=first.device, dtype=first.dtype),
                                             self.hand_command(interval.waypoints[e][k], e, grippers[e], first.device),
                                             interval.waypoints[e][k].noise) for e in eefs}
                    for extra_slot in row["new_slots"][1:]:
                        ledger.used.add(offset + extra_slot)       # waypoints skipped within one step consume their draw
                    runtime.note_track(row, slot_used=slot is not None)
                    with replay_randn_like(torch, bundle):
                        ok = await self.execute(attempt, MultiWaypoint(waypoints))
                    success = success or bool(ok)
                previous_states = follower.final_gripper_states()
                offset += len(interval.source_indices)
                advance_completed(active, completed, indices, eefs, interval.end)
                if all(indices[e] == self.num_subtasks(attempt, e) for e in eefs):
                    break
                start = interval.end
            results = self.export(attempt, success, export_demo)
            runtime.rec.add_extra(noise_slots_used=len(ledger.used) if ledger else 0,
                                  private_noise_steps=ledger.extra if ledger else 0)
            runtime.end_attempt(attempt.generator, env_id, results)
            return self.result(attempt, results)
        except BaseException:
            runtime._restore_label_wrapper()
            raise
