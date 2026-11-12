"""Per-subtask success model of the logistic mixer, and its coefficient file.

Given that an attempt has reached subtask k, the probability that subtask k succeeds when source demonstration j is
adapted to the current scene is a linear logistic regression (paper, Method, "Success model"):

    P(Y_k = 1 | j, scene) = sigmoid( alpha_kj + a_k |dx_kj| + b_k |dy_kj| + c_k rot_kj
                                     + [k >= 1] ( d_k inhand_pos_kj + e_k inhand_rot_kj ) )

- ``dx_kj``, ``dy_kj`` (m): the x and y offsets (world frame, the table plane) between the reference object of subtask k
  in the current scene and in source j at the start of its subtask-k segment;
- ``rot_kj`` (rad): the angle of the relative rotation between those two object orientations;
- ``inhand_pos_kj`` (m), ``inhand_rot_kj`` (rad): from the second subtask on (index k >= 1), the position distance and the
  rotation angle between the in-hand pose now and in source j at its segment start. The in-hand pose is the pose of the
  previous subtask's reference object in the end-effector frame (for a grasp-then-place task: the grasped object).

``alpha_kj`` is source j's own log-odds; there is no separate intercept (the J source terms contain it). The slopes are
shared by all sources of the subtask. Subtask k therefore has J + 3 coefficients if k = 0 and J + 5 otherwise.

Fit: maximum likelihood (Newton's method) on the attempts of a uniform-source baseline pool (the 3,000-attempt
``rank_human_demos`` pool that ``tmg rank_sources`` also reads) that reached subtask k, with Y = subtask k's success.
A tiny fixed L2 term (``STABILIZER``, the same for every task and coefficient, never tuned) keeps the estimate finite
when a source never (or always) succeeds; it changes the other coefficients by far less than their standard errors.

Subtask success comes from the records (MimicGen): for every subtask but the last, the subtask's termination signal
(``task.subtask_term_signals``, MimicGen's ``subtask_term_signal`` of the task template, recorded per step as
``term__<signal>``) turned on at some step up to the end of that subtask's execution; the last subtask succeeds when the
attempt succeeds. An attempt reaches subtask k when every earlier subtask succeeded. CP-Gen stops an attempt at its first
failed subtask (constraint), so there subtask k succeeded when the attempt went on to subtask k + 1 (``reached_outcomes``).

The coefficients live in ``configs/experiment/generator/<generator>/coefficients/<task>.yaml`` (written by
``tmg fit_coefficients``), separate from the ranking file ``.../ranking/<task>.yaml`` that ``RankBasedSampling`` reads.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from tmg.core.poses import rotation_angle
from tmg.generation.simulators.scenes import generator_config_dir

COEFFICIENTS_DIR = "coefficients"
SCENE_FEATURES = ("dx", "dy", "rotation")
INHAND_FEATURES = ("inhand_position", "inhand_rotation")
STABILIZER = 1e-3
UNITS = {"dx": "m", "dy": "m", "rotation": "rad", "inhand_position": "m", "inhand_rotation": "rad"}


# features ---------------------------------------------------------------------------------------------------------
def pose12(v) -> np.ndarray:
    """A 4x4 pose from the 12-number record layout (position, then the row-major rotation), or a 4x4 as is."""
    v = np.asarray(v, dtype=float)
    if v.shape == (4, 4):
        return v
    out = np.eye(4)
    out[:3, 3], out[:3, :3] = v[:3], v[3:12].reshape(3, 3)
    return out


def scene_features(current_pose, source_pose) -> np.ndarray:
    """``(|dx|, |dy|, rotation)`` between the current and a source's reference-object pose (4x4 world poses)."""
    c, s = pose12(current_pose), pose12(source_pose)
    return np.array([abs(c[0, 3] - s[0, 3]), abs(c[1, 3] - s[1, 3]), float(rotation_angle(c[:3, :3], s[:3, :3]))])


def in_hand(eef_pose, object_pose) -> np.ndarray:
    """The object's pose in the end-effector frame (4x4)."""
    e, o = pose12(eef_pose), pose12(object_pose)
    return np.linalg.inv(e) @ o


def inhand_features(current_in_hand, source_in_hand) -> np.ndarray:
    """``(position distance, rotation angle)`` between two in-hand poses (4x4, end-effector frame)."""
    c, s = np.asarray(current_in_hand, dtype=float), np.asarray(source_in_hand, dtype=float)
    return np.array([float(np.linalg.norm(c[:3, 3] - s[:3, 3])), float(rotation_angle(c[:3, :3], s[:3, :3]))])


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# model ------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SubtaskModel:
    """The coefficients of one subtask."""
    alpha: np.ndarray            # (J,) per-source log-odds
    slopes: dict                 # feature name -> slope (scene features; in-hand features from the second subtask on)

    def logits(self, scene: np.ndarray, inhand: np.ndarray | None = None) -> np.ndarray:
        """Per-source logits. ``scene`` (J, 3) holds ``scene_features`` of every source; ``inhand`` (J, 2) the
        ``inhand_features`` (required when the subtask has in-hand slopes)."""
        z = self.alpha + np.asarray(scene, dtype=float) @ np.array([self.slopes[f] for f in SCENE_FEATURES])
        if any(f in self.slopes for f in INHAND_FEATURES):
            if inhand is None:
                raise ValueError("this subtask's model has in-hand terms; pass the in-hand features")
            z = z + np.asarray(inhand, dtype=float) @ np.array([self.slopes[f] for f in INHAND_FEATURES])
        return z

    def predict(self, scene, inhand=None) -> np.ndarray:
        """Per-source success probabilities of the subtask."""
        return sigmoid(self.logits(scene, inhand))


@dataclass(frozen=True)
class SuccessModel:
    """A task's coefficient file."""
    name: str                    # "<generator>/<task>"
    num_sources: int
    subtasks: dict               # subtask index -> SubtaskModel
    carried: dict                # subtask index -> the object whose in-hand pose the subtask uses (k >= 1)
    path: str = ""

    def for_subtask(self, k: int) -> SubtaskModel:
        if int(k) not in self.subtasks:
            raise KeyError(f"coefficients {self.name}: no subtask {k}")
        return self.subtasks[int(k)]

    @classmethod
    def from_mapping(cls, data: Mapping, *, name: str = "", path: str = "") -> SuccessModel:
        n = int(data["num_sources"])
        subtasks, carried = {}, {}
        for k, block in (data.get("subtasks") or {}).items():
            alpha = np.array([float(block["alpha"][j]) for j in range(n)])
            subtasks[int(k)] = SubtaskModel(alpha=alpha, slopes={f: float(v) for f, v in block["slopes"].items()})
            if block.get("carried_object") is not None:
                carried[int(k)] = str(block["carried_object"])
        if not subtasks:
            raise KeyError(f"{path or name}: coefficient file has no subtasks")
        return cls(name=name or f"{data.get('generator', '')}/{data.get('task', '')}", num_sources=n, subtasks=subtasks,
                   carried=carried, path=path)

    def describe(self) -> dict:
        return {"name": self.name, "path": self.path, "num_sources": self.num_sources, "subtasks": sorted(self.subtasks)}


def coefficients_path(name: str, root: Path | None = None) -> Path:
    """``configs/experiment/generator/<generator>/coefficients/<task>.yaml`` for a ``<generator>/<task>`` name."""
    generator, _, task = str(name).partition("/")
    if not task:
        raise ValueError(f"coefficient name must be <generator>/<task>, got {name!r}")
    return generator_config_dir(generator, root) / COEFFICIENTS_DIR / f"{task}.yaml"


def load_coefficients(name: str, root: Path | None = None) -> SuccessModel:
    """Read a coefficient file by its ``<generator>/<task>`` name (or a path ending in .yaml)."""
    path = Path(name) if str(name).endswith(".yaml") else coefficients_path(name, root)
    if not path.exists():
        raise FileNotFoundError(
            f"coefficient file {path} not found. Build it from scratch with `make coefficients GENERATOR=<gen> TASK=<task>` "
            f"(3,000-attempt uniform-source pool, then tmg rank_sources and tmg fit_coefficients); `make generate` does this "
            f"by itself for a method whose mixing.type is logistic.")
    return SuccessModel.from_mapping(yaml.safe_load(path.read_text()) or {}, name=str(name), path=str(path))


# fitting ----------------------------------------------------------------------------------------------------------
def fit_logistic(X: np.ndarray, y: np.ndarray, stabilizer: float = STABILIZER) -> np.ndarray:
    """Maximum-likelihood logistic coefficients (no intercept column added) by Newton's method with step halving."""
    beta = np.zeros(X.shape[1])
    pen = np.full(X.shape[1], float(stabilizer))

    def loss(b):
        z = X @ b
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * pen @ (b * b))

    cur = loss(beta)
    for _ in range(200):
        p = sigmoid(X @ beta)
        grad = X.T @ (p - y) + pen * beta
        hess = X.T @ (X * (p * (1 - p))[:, None]) + np.diag(pen)
        step = np.linalg.solve(hess, grad)
        t, new = 1.0, cur
        while t > 1e-6:
            new = loss(beta - t * step)
            if new <= cur + 1e-12:
                break
            t /= 2
        beta = beta - t * step
        done = abs(cur - new) < 1e-10
        cur = new
        if done:
            break
    return beta


def subtask_outcomes(attempt: Mapping, steps: Mapping, signals: Sequence) -> list[bool]:
    """Success of every subtask of one attempt (module docstring). ``steps`` holds the attempt's per-step arrays
    (``seg_of_step``, ``seg_sub`` and ``term__<signal>``); ``signals`` the termination signal of every subtask but the last
    (None for the last)."""
    num = len(signals)
    seg_sub = np.asarray(steps["seg_sub"]).astype(int)
    sub_of_step = seg_sub[np.asarray(steps["seg_of_step"]).astype(int)] if len(seg_sub) else np.zeros(0, dtype=int)
    out = []
    for k, signal in enumerate(signals):
        if k == num - 1 or signal is None:
            out.append(bool(attempt["success"]) if k == num - 1 else False)
            continue
        key = f"term__{signal}"
        if key not in steps:
            raise KeyError(f"attempt {attempt.get('attempt')}: no per-step signal {key!r} in the record")
        fired = np.asarray(steps[key]).astype(bool)
        upto = sub_of_step <= k
        out.append(bool(np.any(fired[: len(upto)] & upto[: len(fired)])))
    return out


def source_in_hand(source_infos: Sequence[Mapping], j: int, start: int, carried: str) -> np.ndarray:
    """Source j's in-hand pose of ``carried`` at step ``start`` (``source_infos[j]``: eef_pose (T,4,4), object_poses)."""
    info = source_infos[j]
    return in_hand(info["eef_pose"][start], info["object_poses"][carried][start])


def _inhand_row(s: Mapping, j: int, carried: str, source_infos) -> np.ndarray:
    """In-hand features of one subtask record. The source's in-hand pose comes from the record itself when it carries the
    source's end-effector and object poses (CP-Gen: ``src_eef_pose``, ``src_objects``), else from ``source_infos``
    (MimicGen ``datagen_info``). When the carried object's source pose is not available (a CP-Gen constraint that does not
    involve it), both features are 0 for that row, and the model uses 0 for it at mixing time as well."""
    objects = s.get("objects") or {}
    if carried not in objects:
        return np.zeros(2)
    now = in_hand(s["eef_pose"], objects[carried])
    if "src_eef_pose" in s:
        src_objects = s.get("src_objects") or {}
        if carried not in src_objects:
            return np.zeros(2)
        return inhand_features(now, in_hand(s["src_eef_pose"], src_objects[carried]))
    if source_infos is None:
        raise ValueError("the in-hand features of subtask >= 1 need the source demonstrations' datagen_info")
    return inhand_features(now, source_in_hand(source_infos, j, int(s["src_subtask_start"]), carried))


def reached_outcomes(attempt: Mapping, num_subtasks: int) -> list[bool]:
    """Subtask success of a generator that stops an attempt at its first failed subtask (CP-Gen): subtask k succeeded when
    the attempt went on to subtask k + 1; the last subtask succeeded when the attempt succeeded."""
    n = len(attempt.get("subtasks") or [])
    return [n > k + 1 if k < num_subtasks - 1 else bool(attempt["success"]) and n >= num_subtasks for k in range(num_subtasks)]


def design_rows(attempts: Sequence[Mapping], outcomes: Sequence[Sequence[bool]], k: int, num_sources: int,
                source_infos: Sequence[Mapping] | None) -> tuple[np.ndarray, np.ndarray, str | None]:
    """The design matrix and labels of subtask k: one row per attempt that reached it.
    Columns: J source indicators, the three scene features, and for k >= 1 the two in-hand features."""
    rows, ys, carried = [], [], None
    for a, out in zip(attempts, outcomes):
        if not all(out[:k]):
            continue
        subs = sorted(a["subtasks"], key=lambda s: int(s["subtask_ind"]))
        if len(subs) <= k:
            continue
        s = subs[k]
        j = int(s["src_demo_ind"])
        x = np.zeros(num_sources)
        x[j] = 1.0
        parts = [x, scene_features(s["cur_obj_pose"], s["src_obj_pose"])]
        if k >= 1:
            carried = str(subs[k - 1]["object_name"])
            parts.append(_inhand_row(s, j, carried, source_infos))
        rows.append(np.concatenate(parts))
        ys.append(float(out[k]))
    return np.array(rows), np.array(ys), carried


def fit_subtask(X: np.ndarray, y: np.ndarray, num_sources: int, k: int) -> dict:
    """Fit one subtask and return its block of the coefficient file (with the counts behind it)."""
    names = list(SCENE_FEATURES) + (list(INHAND_FEATURES) if k >= 1 else [])
    if len(y) == 0:
        raise ValueError(f"subtask {k}: no attempt reached it")
    varies = [c for c in range(X.shape[1]) if c < num_sources or X[:, c].std() > 1e-9]
    beta = np.zeros(X.shape[1])
    beta[varies] = fit_logistic(X[:, varies], y)
    p = np.clip(sigmoid(X @ beta), 1e-9, 1 - 1e-9)
    src = X[:, :num_sources].argmax(axis=1)
    return {
        "reached": int(len(y)), "successes": int(y.sum()),
        "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
        "alpha": {j: round(float(beta[j]), 6) for j in range(num_sources)},
        "slopes": {f: round(float(beta[num_sources + i]), 6) for i, f in enumerate(names)},
        "source_counts": {j: {"reached": int((src == j).sum()), "successes": int(y[src == j].sum())} for j in range(num_sources)},
    }
