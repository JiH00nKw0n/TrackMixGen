"""Generator-independent execution bookkeeping shared by the robosuite adapters.

An adapter's official executor walks a trajectory (a list of waypoint sequences) one control step at a time.
``TrajectoryExecution`` owns everything about that walk that does not depend on the official package:

- the segment table (one record segment per waypoint sequence, bridge or skill),
- the action labels (executed, or pre-noise arm action clipped plus gripper commands),
- feeding every step to the attempt recorder.

It changes nothing the official executor does: every waypoint is executed once, in order, with the official noise draw.
Poses are numpy 4x4 arrays keyed by end-effector name; a single-arm generator uses one key.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tmg.core.records import SEG_BRIDGE, SEG_SKILL, AttemptRecorder
from tmg.generation.method.interfaces import Method


@dataclass
class WaypointView:
    """Minimal view of one official waypoint: per end effector a 4x4 pose and a gripper command vector."""
    poses: dict
    grippers: dict
    noise: float | None


@dataclass(frozen=True)
class Segment:
    """One record segment of a trajectory: the waypoint indices ``[start, end)`` it covers, its recorder id and kind."""
    start: int
    end: int
    seg_id: int
    kind: int                       # SEG_BRIDGE or SEG_SKILL


class SegmentTable:
    """The record segments of one trajectory in path order (one per waypoint sequence).

    Maps a waypoint index of the flattened path to the recorder segment it belongs to.
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

    def seg_id_of(self, k: int) -> int:
        """Record-segment id of waypoint index ``k`` (last segment when out of range, -1 when empty)."""
        for row in self.rows:
            if row.start <= k < row.end:
                return row.seg_id
        return self.rows[-1].seg_id if self.rows else -1


class TrajectoryExecution:
    """Step-by-step bookkeeping for one attempt's trajectories (see the module docstring).

    Call order per trajectory: :meth:`begin_trajectory`, then for each waypoint sequence :meth:`enter_sequence`
    and, for each of its steps, :meth:`keep_pre_noise`, :meth:`before_step`, :meth:`after_step`. :meth:`label_actions` gives the action array to store once the attempt ends.
    """
    def __init__(self, method: Method, recorder: AttemptRecorder):
        """Args:
            method: the configured method (its label kind).
            recorder: the attempt recorder that receives every step.
        """
        self.method = method
        self.rec = recorder
        self.begin_attempt()

    # attempt --------------------------------------------------------------------------------------
    def begin_attempt(self):
        """Reset labels and trajectory state for a new attempt."""
        self.labels: list[np.ndarray] = []
        self._reset_trajectory()

    def _reset_trajectory(self):
        """Forget the segment table and the pending step."""
        self.segments = SegmentTable()
        self.seg_id = -1
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

    def enter_sequence(self, index: int):
        """Make sequence ``index`` of the current trajectory the one whose steps are recorded next."""
        self.seg_id = self.segments[index].seg_id

    # step -----------------------------------------------------------------------------------------
    def keep_pre_noise(self, action: np.ndarray) -> np.ndarray:
        """Keep the pre-noise copy of the arm action (the official line adds its own noise afterwards)."""
        pre = np.array(action, dtype=float)
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
        self._pending = dict(w_poses=target_poses, e_poses=measured_poses, a_pre=pre, a_label=label,
                             a_exec=np.asarray(executed_action), noise=noise_of_waypoint, seg_id=self.seg_id)
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
