"""Attempt records written during generation (one folder per chunk, generator-independent).

Layout under ``<chunk>/records/``::

    attempts.jsonl        one line per attempt: scalars and per-subtask metadata (keys below)
    attempt_000012.npz    per-step arrays of attempt 12 (arrays below)

The recorder is fed by the generator adapters; it never imports a simulator. Everything is written when the
attempt ends (``flush``), so a process that dies mid-attempt leaves the previous attempts intact. A rerun of
the same chunk starts from attempt 0 again, so ``configure`` wipes stale partial records of that chunk.

Per-step arrays (T steps, S segments)::

    eef_names (E,)                 end-effector names in the order of the E axis below (one for a single arm)
    w_pos (T,E,3) w_rot (T,E,9)    planned target pose of each end effector at the step (world frame)
    e_pos (T,E,3) e_rot (T,E,9)    measured end-effector pose before the step
    a_pre (T,A)                    arm action before noise and before clipping (normalized units)
    a_label (T,A+G)                action label the dataset will use: clipped pre-noise arm + gripper
    a_exec (T,A+G)                 action actually executed (noisy, clipped) + gripper
    noise (T,)                     noise scale of the step
    seg_of_step (T,)               segment id of each step
    seg_sub (S,) seg_kind (S,) seg_len (S,)   subtask index, kind (0 bridge, 1 skill), executed length per segment
    term__<signal> (T,)            subtask termination signals of the state AFTER the step
    succ_task (T,)                 task success predicate of the state AFTER the step
    track_s_before, track_s_after (T,)   path follower: path variable before and after the step (NaN when off)
    track_ds_proposed, track_ds_actual (T,)  advance the rule proposed and the advance after event/end clamping
    track_k (T,)                   waypoint index the step commanded (-1 when off)
    track_region (T,)              region code of the waypoint before the step (0 rest, 1 approach, 2 bridge; -1 off)
    track_capped (T,)              1 when the extra-step cap forced base speed
    track_slot (T,)                1 when the step applied an original waypoint's global noise draw, 0 for a private draw
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tmg.core.poses import pose_to_list

RECORDS_DIR = "records"
ATTEMPTS_JSONL = "attempts.jsonl"
NPZ_FORMAT = "attempt_%06d.npz"
NPZ_GLOB = "attempt_*.npz"
SEG_BRIDGE = 0
SEG_SKILL = 1
REGION_CODES = {"rest": 0, "approach": 1, "bridge": 2}
POSE_DECIMALS = 6


def _round_pose(pose) -> list[float]:
    """4x4 pose as 12 floats rounded to ``POSE_DECIMALS`` for the jsonl line."""
    return [round(v, POSE_DECIMALS) for v in pose_to_list(pose)]


def _jsonable(value):
    """Recursively convert numpy values to JSON-able Python values (4x4 arrays become pose lists)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.ndarray):
        if value.shape == (4, 4):
            return _round_pose(value)
        return [float(x) for x in value.reshape(-1)]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


class AttemptRecorder:
    """Accumulates one attempt and writes it as one npz and one jsonl line."""

    def __init__(self):
        """Create an idle recorder; call :meth:`configure` to choose the folder."""
        self.dir: Path | None = None
        self.attempt = -1
        self.n_success = 0
        self.n_failure = 0
        self._active = False
        self._reset()

    def _reset(self):
        """Clear the per-attempt buffers."""
        self.w_pos, self.w_rot, self.e_pos, self.e_rot = [], [], [], []
        self.eef_names: list[str] = []
        self.a_pre, self.a_label, self.a_exec, self.noise = [], [], [], []
        self.seg_of_step, self.seg_sub, self.seg_kind, self.seg_len = [], [], [], []
        self.term = {}
        self.succ = []
        self.track = {k: [] for k in ("s_before", "s_after", "ds_proposed", "ds_actual", "k", "region", "capped", "slot")}
        self.subtasks = []
        self.objects_init = {}
        self.scene_meta = None
        self.extra = {}
        self.cur_subtask = -1

    # lifecycle ------------------------------------------------------------------------------------
    def configure(self, folder: Path | None):
        """Choose the records folder (``None`` disables writing) and wipe stale records of an earlier,
        unfinished run of the same chunk, since a rerun restarts from attempt 0.
        """
        if folder is None:
            self.dir = None
            return
        self.dir = Path(folder)
        self.dir.mkdir(parents=True, exist_ok=True)
        stale = self.dir / ATTEMPTS_JSONL
        if stale.exists():
            for f in self.dir.glob(NPZ_GLOB):
                f.unlink()
            stale.unlink()

    @property
    def enabled(self) -> bool:
        """True when a folder is configured, so :meth:`flush` writes files."""
        return self.dir is not None

    def begin_attempt(self):
        """Start a new attempt (increments the attempt index and clears the buffers)."""
        self.attempt += 1
        self._active = True
        self._reset()

    # generator-level events -----------------------------------------------------------------------
    def set_objects_init(self, poses: dict):
        """Record the pose of every datagen object at the start of the attempt (``{name: 4x4}``)."""
        if self._active:
            self.objects_init = {k: _round_pose(v) for k, v in poses.items()}

    def set_scene_meta(self, meta):
        """Record scene-level metadata written by the environment class (reach rejects, base distances)."""
        if self._active:
            self.scene_meta = None if meta is None else _jsonable(meta)

    def add_subtask(self, **meta):
        """Append one subtask's metadata (keyword arguments become the jsonl keys); ``subtask_ind`` also
        becomes the subtask of the segments recorded next.
        """
        if not self._active:
            return
        self.subtasks.append(_jsonable(meta))
        self.cur_subtask = int(meta.get("subtask_ind", self.cur_subtask))

    def add_extra(self, **fields):
        """Add attempt-level keys to the jsonl line (for example the label kind)."""
        if self._active:
            self.extra.update(_jsonable(fields))

    # executor-level events ------------------------------------------------------------------------
    def begin_segment(self, kind: int) -> int:
        """Open a new segment of ``kind`` (``SEG_BRIDGE`` or ``SEG_SKILL``) for the current subtask.

        Returns:
            The segment id steps refer to in ``seg_of_step`` (-1 when no attempt is active).
        """
        if not self._active:
            return -1
        self.seg_sub.append(self.cur_subtask)
        self.seg_kind.append(int(kind))
        self.seg_len.append(0)
        return len(self.seg_len) - 1

    def add_step(self, *, w_poses, e_poses, a_pre, a_label, a_exec, noise, seg_id, term_signals,
                 track=None):
        """``w_poses`` and ``e_poses`` map end-effector names to 4x4 poses; the key order of the first step
        fixes the E axis of the arrays."""
        if not self._active:
            return
        if not self.eef_names:
            self.eef_names = list(w_poses)
        w = np.stack([np.asarray(w_poses[n], dtype=float) for n in self.eef_names])
        e = np.stack([np.asarray(e_poses[n], dtype=float) for n in self.eef_names])
        self.w_pos.append(w[:, :3, 3].astype(np.float32))
        self.w_rot.append(w[:, :3, :3].reshape(len(self.eef_names), 9).astype(np.float32))
        self.e_pos.append(e[:, :3, 3].astype(np.float32))
        self.e_rot.append(e[:, :3, :3].reshape(len(self.eef_names), 9).astype(np.float32))
        self.a_pre.append(np.asarray(a_pre, dtype=np.float32).reshape(-1))
        self.a_label.append(np.asarray(a_label, dtype=np.float32).reshape(-1))
        self.a_exec.append(np.asarray(a_exec, dtype=np.float32).reshape(-1))
        self.noise.append(np.float32(0.0 if noise is None else float(noise)))
        self.seg_of_step.append(np.int16(seg_id))
        if 0 <= seg_id < len(self.seg_len):
            self.seg_len[seg_id] += 1
        if term_signals:
            for name, value in term_signals.items():
                self.term.setdefault(name, []).append(1 if value else 0)
        t = self.track
        if track is None:
            for key in ("s_before", "s_after", "ds_proposed", "ds_actual"):
                t[key].append(np.nan)
            t["k"].append(-1)
            t["region"].append(-1)
            t["capped"].append(0)
            t["slot"].append(1)
        else:
            for key in ("s_before", "s_after", "ds_proposed", "ds_actual"):
                t[key].append(float(track[key]))
            t["k"].append(int(track["k"]))
            t["region"].append(int(REGION_CODES.get(track["region"], -1)))
            t["capped"].append(1 if track.get("capped") else 0)
            t["slot"].append(1 if track.get("slot") else 0)

    def add_success(self, flag: bool):
        """Append the task-success predicate of the state after the step."""
        if self._active:
            self.succ.append(1 if flag else 0)

    # finalize -------------------------------------------------------------------------------------
    def flush(self, *, success: bool, src_demo_inds) -> dict | None:
        """End the attempt: write ``attempt_%06d.npz`` and one ``attempts.jsonl`` line.

        Args:
            success: the generator's success verdict for the attempt.
            src_demo_inds: source demonstration index chosen for each executed subtask.

        Returns:
            The jsonl line as a dict (also when writing is disabled), or None when no attempt was active.
        """
        if not self._active:
            return None
        self._active = False
        if success:
            self.n_success += 1
        else:
            self.n_failure += 1
        succ = np.asarray(self.succ, dtype=np.uint8)
        idx = np.flatnonzero(succ)
        term_first = {}
        for name, sig in self.term.items():
            s = np.flatnonzero(np.asarray(sig))
            term_first[name] = int(s[0]) if s.size else -1
        line = dict(
            attempt=self.attempt,
            success=bool(success),
            n_steps=len(self.w_pos),
            src_demo_inds=[int(i) for i in src_demo_inds],
            subtasks=self.subtasks,
            objects_init=self.objects_init,
            scene_meta=self.scene_meta,
            succ_first=int(idx[0]) if idx.size else -1,
            succ_last=int(succ[-1]) if succ.size else 0,
            term_first=term_first,
            n_success_so_far=self.n_success,
            n_failure_so_far=self.n_failure,
            npz=NPZ_FORMAT % self.attempt,
            eef_names=list(self.eef_names),
        )
        line.update(self.extra)
        if self.dir is None:
            return line
        n_eef = max(len(self.eef_names), 1)
        arrays = dict(
            eef_names=np.array(self.eef_names, dtype=str),
            w_pos=np.array(self.w_pos, dtype=np.float32).reshape(-1, n_eef, 3),
            w_rot=np.array(self.w_rot, dtype=np.float32).reshape(-1, n_eef, 9),
            e_pos=np.array(self.e_pos, dtype=np.float32).reshape(-1, n_eef, 3),
            e_rot=np.array(self.e_rot, dtype=np.float32).reshape(-1, n_eef, 9),
            a_pre=np.array(self.a_pre, dtype=np.float32),
            a_label=np.array(self.a_label, dtype=np.float32),
            a_exec=np.array(self.a_exec, dtype=np.float32),
            noise=np.array(self.noise, dtype=np.float32),
            seg_of_step=np.array(self.seg_of_step, dtype=np.int16),
            seg_sub=np.array(self.seg_sub, dtype=np.int16),
            seg_kind=np.array(self.seg_kind, dtype=np.uint8),
            seg_len=np.array(self.seg_len, dtype=np.int32),
            succ_task=succ,
            track_s_before=np.array(self.track["s_before"], dtype=np.float32),
            track_s_after=np.array(self.track["s_after"], dtype=np.float32),
            track_ds_proposed=np.array(self.track["ds_proposed"], dtype=np.float32),
            track_ds_actual=np.array(self.track["ds_actual"], dtype=np.float32),
            track_k=np.array(self.track["k"], dtype=np.int32),
            track_region=np.array(self.track["region"], dtype=np.int8),
            track_capped=np.array(self.track["capped"], dtype=np.uint8),
            track_slot=np.array(self.track["slot"], dtype=np.uint8),
        )
        for name, sig in self.term.items():
            arrays["term__" + name] = np.array(sig, dtype=np.uint8)
        np.savez_compressed(self.dir / line["npz"], **arrays)
        with open(self.dir / ATTEMPTS_JSONL, "a") as f:
            f.write(json.dumps(line) + "\n")
        return line


RECORD_ERROR = "RECORD_ERROR.json"


def validate_records(folder: Path, expected_attempts: int) -> dict:
    """Check a chunk's records: every attempt line present, every npz readable with the right step count.
    Returns {"ok": bool, "error": str | None, "lines": [...], "successes": int, "failures": int}."""
    folder = Path(folder)
    if (folder / RECORD_ERROR).exists():
        return {"ok": False, "error": f"record writing failed, see {folder / RECORD_ERROR}", "lines": []}
    lines = read_attempts(folder)
    if len(lines) != int(expected_attempts):
        return {"ok": False, "error": f"{len(lines)} attempt records, {expected_attempts} planned", "lines": lines}
    for line in lines:
        npz = folder / line["npz"]
        if not npz.exists():
            return {"ok": False, "error": f"missing step arrays {npz}", "lines": lines}
        try:
            with np.load(npz) as arrays:
                n = arrays["w_pos"].shape[0]
        except Exception as error:  # noqa: BLE001
            return {"ok": False, "error": f"unreadable step arrays {npz}: {error!r}", "lines": lines}
        if n != line["n_steps"]:
            return {"ok": False, "error": f"{npz}: {n} steps recorded, line says {line['n_steps']}", "lines": lines}
    successes = sum(1 for l in lines if l["success"])
    return {"ok": True, "error": None, "lines": lines, "successes": successes, "failures": len(lines) - successes}


def read_attempts(folder: Path) -> list[dict]:
    """All ``attempts.jsonl`` lines of a records folder as dicts (empty list when absent)."""
    path = Path(folder) / ATTEMPTS_JSONL
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
