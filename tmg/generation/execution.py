"""Generator-independent execution bookkeeping shared by every adapter.

An adapter's official executor walks a trajectory (a list of waypoint sequences) one control step at a time.
``TrajectoryExecution`` owns everything about that walk that does not depend on the official package:

- the segment table (one record segment per waypoint sequence, bridge or skill),
- the path follower over the flattened trajectory, shared by every end effector of the robot, and its
  per-step records,
- the random-number accounting of the follower (one "global" draw per original waypoint, in waypoint order,
  applied to the step that commands that waypoint; private draws for the extra steps), so that an attempt
  with the follower consumes the official generator's random stream exactly as the baseline does,
- the action labels (executed, or pre-noise arm action clipped plus gripper commands) and the gripper states
  carried from one trajectory to the next for event detection.

Poses are numpy 4x4 arrays keyed by end-effector name; a single-arm generator uses one key. The adapter
supplies two callables: ``global_draw(size)`` drawing from the official generator's stream and
``private_draw(size)`` drawing from a private stream. An Isaac Lab adapter converts torch tensors before
calling in.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from tmg.core.records import SEG_BRIDGE, SEG_SKILL, AttemptRecorder
from tmg.generation.method.interfaces import Method
from tmg.generation.method.tracking import EndEffectorPath


@dataclass
class WaypointView:
    """Minimal view of one official waypoint: per end effector a 4x4 pose and a gripper command vector."""
    poses: dict
    grippers: dict
    noise: float | None


@dataclass
class StepTarget:
    """What the official executor should command at one step."""
    poses: dict
    grippers: dict
    noise_scale: float | None       # None: let the official executor draw its own noise (identity)
    noise_vector: np.ndarray | None  # the vector to add when noise_scale is not None
    record: dict | None             # follower record for the step (None when the follower is off)


@dataclass(frozen=True)
class Segment:
    """One record segment of a trajectory: the waypoint indices ``[start, end)`` it covers, its recorder id and kind."""
    start: int
    end: int
    seg_id: int
    kind: int                       # SEG_BRIDGE or SEG_SKILL


class SegmentTable:
    """The record segments of one trajectory in path order (one per waypoint sequence).

    Maps a waypoint index of the flattened path to the recorder segment it belongs to, which is what every
    adapter needs when a follower walks the whole trajectory as one path.
    """
    def __init__(self):
        self.rows: list[Segment] = []

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Segment:
        return self.rows[index]

    def add(self, length: int, seg_id: int, kind: int) -> Segment:
        """Append a segment of ``length`` waypoints after the last one."""
        start = self.total
        row = Segment(start, start + int(length), int(seg_id), int(kind))
        self.rows.append(row)
        return row

    @property
    def total(self) -> int:
        """Number of waypoints covered so far."""
        return self.rows[-1].end if self.rows else 0

    def kinds_per_waypoint(self) -> list[int]:
        """The kind of every waypoint of the path, in order."""
        return [row.kind for row in self.rows for _ in range(row.end - row.start)]

    def seg_id_of(self, k: int) -> int:
        """Record-segment id of waypoint index ``k`` (last segment when out of range, -1 when empty)."""
        for row in self.rows:
            if row.start <= k < row.end:
                return row.seg_id
        return self.rows[-1].seg_id if self.rows else -1


def slot_sources(record: dict) -> tuple[bool, str]:
    """Whether a follower step commands an original waypoint for the first time, and the name of the noise stream
    it draws from.

    A step whose commanded waypoint ``k`` is among the waypoints it newly covers (``new_slots``) is a slot: it takes
    the official generator's draw for that waypoint (``global``). Every other step is an extra step and draws from
    the private stream (``private``).
    """
    slot = record["k"] in record["new_slots"]
    return slot, "global" if slot else "private"


class TrajectoryExecution:
    """Step-by-step bookkeeping for one attempt's trajectories (see the module docstring).

    Call order per trajectory: :meth:`begin_trajectory`, then for each waypoint sequence :meth:`enter_sequence`
    and, for each step of :meth:`steps_of_sequence`, :meth:`target`, :meth:`apply_noise`, :meth:`before_step`,
    :meth:`after_step`. :meth:`label_actions` gives the action array to store once the attempt ends.
    """
    def __init__(self, method: Method, recorder: AttemptRecorder, *, global_draw: Callable[[int], np.ndarray],
                 private_draw: Callable[[int], np.ndarray], action_dim: int = 6):
        """Args:
            method: the configured method (mixer, follower factory, label kind).
            recorder: the attempt recorder that receives every step.
            global_draw: ``f(size) -> array`` drawing from the official generator's random stream.
            private_draw: ``f(size) -> array`` drawing from a private stream (extra follower steps).
            action_dim: size of one noise vector (arm action dimension).
        """
        self.method = method
        self.rec = recorder
        self.global_draw = global_draw
        self.private_draw = private_draw
        self.action_dim = int(action_dim)
        self.begin_attempt()

    # attempt --------------------------------------------------------------------------------------
    def begin_attempt(self):
        """Reset gripper states, labels and trajectory state for a new attempt."""
        self.gripper_states: dict = {}     # end effector -> last known closed/open state (bool)
        self.labels: list[np.ndarray] = []
        self._reset_trajectory()

    def _reset_trajectory(self):
        """Forget the follower, the segment table and the pending step."""
        self.follower = None
        self.segments = SegmentTable()
        self.seg_id = -1
        self._current: StepTarget | None = None
        self._pending: dict | None = None
        self._pre: np.ndarray | None = None

    # trajectory -----------------------------------------------------------------------------------
    def begin_trajectory(self, sequences: Sequence[Sequence[WaypointView]]):
        """Called once per trajectory, before its first sequence runs."""
        self._reset_trajectory()
        n = len(sequences)
        for i, seq in enumerate(sequences):
            kind = SEG_SKILL if i == n - 1 else SEG_BRIDGE
            self.segments.add(len(seq), self.rec.begin_segment(kind), kind)
        if self.method.follower_factory is not None and self.segments.total > 0:
            flat = [w for seq in sequences for w in seq]
            kinds = self.segments.kinds_per_waypoint()
            names = list(flat[0].poses)
            eefs = [EndEffectorPath(name, np.stack([w.poses[name] for w in flat]),
                                    np.stack([np.asarray(w.grippers[name], dtype=float).reshape(-1) for w in flat]))
                    for name in names]
            self.follower = self.method.follower_factory()
            self.follower.start_path(eefs, [w.noise for w in flat], kinds, self.gripper_states)

    def enter_sequence(self, index: int):
        """Make sequence ``index`` of the current trajectory the one whose steps are recorded next."""
        self.seg_id = self.segments[index].seg_id

    def steps_of_sequence(self, index: int, length: int):
        """Iteration the official executor should perform for sequence ``index``."""
        if self.follower is None:
            return range(length)
        if index != 0:
            return range(0)          # the whole trajectory is walked from the first sequence
        return self._follower_steps()

    def _follower_steps(self):
        """Yield step indices 0, 1, ... until the follower reports the path finished; then store the final
        gripper states for the next trajectory.
        """
        j = 0
        while True:
            yield j
            j += 1
            if self.follower is None or self.follower.finished():
                self.gripper_states.update(self.follower.final_gripper_states())
                return

    def _step_noise(self, record: dict) -> np.ndarray:
        """The noise vector of a follower step, and the draw accounting behind it.

        One global draw is taken for every waypoint the step newly covers, in waypoint order, so the official
        stream advances exactly once per original waypoint whatever the step count. The step uses the draw of the
        waypoint it commands when that is one of them, else a private draw. ``record`` gets ``noise_source`` and
        ``slot`` (see :func:`slot_sources`).
        """
        draws = {k: self.global_draw(self.action_dim) for k in record["new_slots"]}
        slot, source = slot_sources(record)
        vector = draws[record["k"]] if slot else self.private_draw(self.action_dim)
        record["noise_source"] = source
        record["slot"] = slot
        return vector

    # step -----------------------------------------------------------------------------------------
    def target(self, original: WaypointView | None, measured: dict | None) -> StepTarget:
        """The command of the next step: the original waypoint (identity) or the follower's target."""
        if self.follower is None:
            self._current = StepTarget(original.poses, original.grippers, None, None, None)
            return self._current
        poses, grippers, noise, record = self.follower.next_target(
            {k: np.asarray(v, dtype=float) for k, v in measured.items()})
        vector = self._step_noise(record)
        self.seg_id = self.segments.seg_id_of(record["k"])
        self._current = StepTarget(poses, grippers, 0.0 if noise is None else float(noise), vector, record)
        return self._current

    def apply_noise(self, action: np.ndarray) -> np.ndarray:
        """Add the step's noise to ``action`` in place and clip, exactly as the official executor does for its
        own draws. Returns the pre-noise copy. No-op in identity mode (the official line draws)."""
        pre = np.array(action, dtype=float)
        cur = self._current
        if cur is not None and cur.noise_scale is not None:
            if cur.noise_scale != 0.0:
                action += cur.noise_scale * cur.noise_vector[: action.shape[0]]
            np.clip(action, -1.0, 1.0, out=action)
        self._pre = pre
        return pre

    def before_step(self, *, target_poses: dict, measured_poses: dict, executed_action, grippers: dict,
                    noise_of_waypoint):
        """Record what is known before the environment steps; ``after_step`` completes the record."""
        gripper_vec = np.concatenate([np.asarray(grippers[k], dtype=float).reshape(-1) for k in grippers]) \
            if grippers else np.zeros(0)
        pre = self._pre
        if pre is None:
            pre = np.array(executed_action[: len(executed_action) - gripper_vec.size], dtype=float)
        label = np.concatenate([np.clip(pre, -1.0, 1.0), gripper_vec])
        self.labels.append(label)
        cur = self._current
        noise = noise_of_waypoint if (cur is None or cur.noise_scale is None) else cur.noise_scale
        self._pending = dict(w_poses=target_poses, e_poses=measured_poses, a_pre=pre, a_label=label,
                             a_exec=np.asarray(executed_action), noise=noise, seg_id=self.seg_id,
                             track=None if cur is None else cur.record)
        if self.follower is None:
            # identity mode: keep the per-eef gripper state from the scalar sign rule for the next trajectory
            for name, g in grippers.items():
                v = np.asarray(g, dtype=float).reshape(-1)
                if v.size and v[0] != 0:
                    self.gripper_states[name] = bool(v[0] > 0)
        self._pre = None

    def after_step(self, *, task_success: bool, term_signals: dict | None):
        """Complete the pending step with what is known after the environment stepped.

        Args:
            task_success: task predicate of the new state.
            term_signals: subtask termination signals of the new state (``{name: bool}``) or None.
        """
        pending, self._pending = self._pending, None
        if pending is None:
            raise RuntimeError("after_step called without before_step")
        self.rec.add_step(term_signals=term_signals, **pending)
        self.rec.add_success(bool(task_success))

    # labels ---------------------------------------------------------------------------------------
    def label_actions(self, executed: np.ndarray) -> np.ndarray:
        """The action array a dataset should store: executed actions, or the pre-noise labels."""
        if self.method.labels != "pre_noise":
            return executed
        if len(self.labels) != len(executed):
            raise RuntimeError(f"pre-noise labels ({len(self.labels)}) do not match executed actions ({len(executed)})")
        return np.array(self.labels) if self.labels else executed
