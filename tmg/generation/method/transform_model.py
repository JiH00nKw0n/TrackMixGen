"""Success model of the convex mixer: full-task generation success as a logistic regression on each subtask's source
and transform size, fitted once per task on the baseline ranking pool (the 3,000 attempts whose sources were drawn
uniformly at random) by maximum likelihood (Newton's method). The same fit runs for every task; nothing is chosen by
hand. (A least-squares fit of the same terms, ``--link linear``, predicted as well on the 2026-11-06 check but fell
below 0 for some sources; the logistic model keeps every prediction a probability.)

    logit P(success) = b0 + sum over subtasks s of ( c_s[source of s] + d_s * displacement_s + r_s * rotation_s )

``displacement_s`` is the distance (m) between subtask s's reference object in the current scene and in the source
scene at the segment start, ``rotation_s`` the angle (rad) of their relative rotation (the transform the generator
applies). ``c_s[0]`` is 0 (source 0 is the reference level). A feature that never varies in the pool (a subtask object
that every source demonstration placed at the same pose, or that never turns) gets coefficient 0.

For mixing one subtask the other subtasks are unknown, so their terms are replaced by their mean over the pool:
``p_k = sigmoid(offset_s + c_s[k] + d_s * displacement_k + r_s * rotation_k)`` (the linear link: the same sum clipped to
[0, 1]). The file stores the ``link``, and ``offset_s`` (b0 plus the other subtasks' mean contributions), ``source_s``,
``displacement_s`` and ``rotation_s`` per subtask.

With ``--location`` every subtask also gets source-by-location terms: for each source k, coefficients on the
standardized current pose of the subtask's reference object (x, y, sin yaw, cos yaw), so a source can rank high in one
part of the workspace and low in another. These terms carry an L2 penalty whose strength is chosen by five-fold
cross-validated log loss over a fixed grid (``LOCATION_PENALTIES``); the other terms are not penalized.

With ``--link decay`` (user proposal 2026-11-07, from the chapter 4 fragility pool where replaying a source succeeds at
zero transform and fails more as the transform grows, at a source-specific rate) each subtask's success starts at 1 and
falls by the same ratio per unit of transform, at a rate of its own for every source; one shared factor multiplies the
success of an attempt whose subtasks use different sources (stitching sources is a failure cause that does not depend on
the transform size):

    P(success) = mixed_factor^[sources differ] * prod over subtasks s of exp(-(a_s[k] * displacement_s + b_s[k] * rotation_s))

with every rate >= 0, fitted by maximum likelihood (L-BFGS-B). For one subtask the prediction is
``p_k = exp(-(a_s[k] * displacement_k + b_s[k] * rotation_k))``; the factors shared by all sources cancel in the weights.

With ``--source <source hdf5>`` (user request 2026-11-07) every subtask after the first also gets one in-hand term: the
distance (m) between the carried object's position in the gripper frame now (the previous subtask's reference object,
at the start of this subtask) and in source k's demonstration at the start of its segment. It enters the logistic sum as
``+ inhand_s * gap_k`` and the decay exponent as ``- inhand_s * gap_k`` (``inhand_s >= 0``), so a source whose plan
assumes the object sits in the hand where it does now is predicted to succeed more. The generator passes the gaps of the
current attempt in ``MixContext.extra["inhand_gaps"]``.

    python -m tmg.generation.method.transform_model --pool <ranking pool dir> --out <model.yaml> [--location | --link decay] [--source <hdf5>]
"""
from __future__ import annotations

import argparse
import glob
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from tmg.core.poses import rotation_angle


def transform_size(current_pose, source_pose) -> tuple[float, float]:
    """``(displacement m, rotation rad)`` between two 4x4 object poses."""
    c, s = np.asarray(current_pose, dtype=float), np.asarray(source_pose, dtype=float)
    return float(np.linalg.norm(c[:3, 3] - s[:3, 3])), float(rotation_angle(c[:3, :3], s[:3, :3]))


def _pose12(v) -> np.ndarray:
    """A 4x4 pose from the 12-number record layout (position, then the row-major rotation)."""
    v = np.asarray(v, dtype=float)
    out = np.eye(4)
    out[:3, 3], out[:3, :3] = v[:3], v[3:12].reshape(3, 3)
    return out


def _attempt_rows(pool: Path) -> list[dict]:
    files = sorted(glob.glob(str(pool / "chunks" / "*" / "records" / "attempts.jsonl")))
    if not files:
        raise FileNotFoundError(f"no chunks/*/records/attempts.jsonl under {pool}")
    return [{**json.loads(line), "_records": str(Path(f).parent)} for f in files for line in open(f)]


LOCATION_PENALTIES = (1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0)


def location_features(pose) -> np.ndarray:
    """(x, y, sin yaw, cos yaw) of a 4x4 object pose."""
    pose = np.asarray(pose, dtype=float)
    yaw = np.arctan2(pose[1, 0], pose[0, 0])
    return np.array([pose[0, 3], pose[1, 3], np.sin(yaw), np.cos(yaw)])


def _logistic(X: np.ndarray, y: np.ndarray, penalty: np.ndarray | None = None) -> np.ndarray:
    """Logistic regression coefficients by Newton's method; ``penalty`` is a per-coefficient L2 weight (None: none)."""
    beta = np.zeros(X.shape[1])
    pen = np.zeros(X.shape[1]) if penalty is None else np.asarray(penalty, dtype=float)
    for _ in range(100):
        p = 1 / (1 + np.exp(-X @ beta))
        hess = X.T @ (X * (p * (1 - p))[:, None]) + np.diag(pen)
        step = np.linalg.solve(hess + 1e-9 * np.eye(len(beta)), X.T @ (y - p) - pen * beta)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def _cv_log_loss(X: np.ndarray, y: np.ndarray, penalty: np.ndarray, folds: int = 5) -> float:
    idx = np.random.default_rng(0).permutation(len(y))
    total = 0.0
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        p = np.clip(1 / (1 + np.exp(-X[te] @ _logistic(X[tr], y[tr], penalty))), 1e-6, 1 - 1e-6)
        total += -np.sum(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p))
    return total / len(y)


def source_inhand(source_path) -> list[dict]:
    """Per source demonstration: object name -> (T, 3) position of that object in the gripper frame at every step
    (MimicGen ``datagen_info``)."""
    import h5py
    out = []
    with h5py.File(source_path, "r") as f:
        for name in sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1])):
            info = f["data"][name]["datagen_info"]
            eef = np.asarray(info["eef_pose"], dtype=float)
            out.append({obj: np.einsum("tji,tj->ti", eef[:, :3, :3], np.asarray(info["object_poses"][obj], dtype=float)[:, :3, 3] - eef[:, :3, 3])
                        for obj in info["object_poses"].keys()})
    return out


def inhand_gaps(attempts: list[dict], sources: list[dict]) -> np.ndarray:
    """(attempts, subtasks) distance (m) between the carried object's in-hand position at the start of each subtask
    and the selected source's at the start of its segment; 0 for the first subtask."""
    num_subtasks = len(attempts[0]["subtasks"])
    gaps = np.zeros((len(attempts), num_subtasks))
    for i, a in enumerate(attempts):
        subs = sorted(a["subtasks"], key=lambda s: int(s["subtask_ind"]))
        for si in range(1, num_subtasks):
            carried = subs[si - 1]["object_name"]
            eef, obj = _pose12(subs[si]["eef_pose"]), _pose12(subs[si]["objects"][carried])
            now = eef[:3, :3].T @ (obj[:3, 3] - eef[:3, 3])
            src = sources[int(subs[si]["src_demo_ind"])][carried][int(subs[si]["src_subtask_start"])]
            gaps[i, si] = float(np.linalg.norm(now - src))
    return gaps


def _flat12(pose) -> list[float]:
    """The 12-number record layout of a 4x4 pose (inverse of ``_pose12``)."""
    pose = np.asarray(pose, dtype=float)
    return [float(v) for v in pose[:3, 3]] + [float(v) for v in pose[:3, :3].reshape(-1)]


def _hand_in_object(eef_pose, object_pose) -> np.ndarray:
    """End-effector position in an object's frame (the Isaac Lab runtime's in-hand relation, ``IsaacLabRuntime._hand_in_object``)."""
    eef_pose, object_pose = np.asarray(eef_pose, dtype=float), np.asarray(object_pose, dtype=float)
    return object_pose[:3, :3].T @ (eef_pose[:3, 3] - object_pose[:3, 3])


def isaaclab_rows(attempts: list[dict], source_path, pool: Path, inhand: bool = False) -> tuple[list[dict], np.ndarray | None]:
    """Ranking-pool rows of an Isaac Lab generator (AutoData, Isaac Lab Mimic) in the layout ``fit`` reads.

    Their records list one subtask entry per (subtask, end effector) and carry no source object pose. Each attempt keeps
    the entries of the end effector with the most subtasks (Microwave: the right hand), the selected source's reference-object
    pose at the segment start is read from the source hdf5 (``obs/datagen_info``), and with ``inhand`` the in-hand gap of
    every later subtask is the distance between the hand's position in the frame of the previous subtask's object when the
    subtask is built and the selected source's at its segment start. The hand position then is the measured wrist position
    of the per-step record (``e_pos``, whose origin is the eef frame's) at the step where the subtask starts: the earlier
    subtasks' interpolation, fixed and segment steps, plus the eef pose the first subtask prepends. The object must be the
    current subtask's reference object (its pose is the recorded ``cur_obj_pose``); others raise.
    """
    import h5py
    src = []
    with h5py.File(source_path, "r") as f:
        for name in sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1])):
            info = f["data"][name]["obs"]["datagen_info"]
            src.append({"eef": {k: np.asarray(v) for k, v in info["eef_pose"].items()},
                        "obj": {k: np.asarray(v) for k, v in info["object_pose"].items()}})
    counts = {}
    for s in attempts[0]["subtasks"]:
        counts[s["eef"]] = counts.get(s["eef"], 0) + 1
    eef = max(sorted(counts), key=lambda e: counts[e])
    rows, gaps = [], (np.zeros((len(attempts), counts[eef])) if inhand else None)
    for i, a in enumerate(attempts):
        subs = sorted((dict(s) for s in a["subtasks"] if s["eef"] == eef), key=lambda s: int(s["subtask_ind"]))
        for s in subs:
            k, t = int(s["src_demo_ind"]), int(s["src_subtask_start"])
            s["src_obj_pose"] = _flat12(src[k]["obj"][s["object_name"]][t])
        rows.append({**a, "subtasks": subs})
        if not inhand:
            continue
        e_pos = None
        step = 0
        for si, s in enumerate(subs):
            if si > 0:
                carried = subs[si - 1]["object_name"]
                if carried != s["object_name"]:
                    raise NotImplementedError(f"in-hand gap of {carried!r} while subtask {si} references {s['object_name']!r}")
                if e_pos is None:
                    z = np.load(Path(a["_records"]) / a["npz"])
                    e_pos = z["e_pos"][:, list(z["eef_names"]).index(eef)]
                obj = _pose12(s["cur_obj_pose"])
                now_eef = np.eye(4)
                now_eef[:3, 3] = e_pos[min(step, len(e_pos) - 1)]
                k, t = int(s["src_demo_ind"]), int(s["src_subtask_start"])
                gaps[i, si] = float(np.linalg.norm(_hand_in_object(now_eef, obj) - _hand_in_object(src[k]["eef"][eef][t], src[k]["obj"][carried][t])))
            step += int(s["num_interpolation_steps"]) + int(s["num_fixed_steps"]) + int(s["src_subtask_end"]) - int(s["src_subtask_start"]) + (1 if si == 0 else 0)
    return rows, gaps


def shared_subtasks(attempts: list[dict]) -> bool:
    """Whether every attempt used one source with one reference-object pose for all its subtasks, so the source,
    transform and location terms of the subtasks are the same columns (Microwave: one source per attempt, the microwave
    body does not move). ``fit`` then keeps one copy of them; only the in-hand terms differ between subtasks."""
    for a in attempts:
        subs = a["subtasks"]
        if len({int(s["src_demo_ind"]) for s in subs}) > 1:
            return False
        if any(not np.allclose(s["cur_obj_pose"], subs[0]["cur_obj_pose"], atol=1e-4) or not np.allclose(s["src_obj_pose"], subs[0]["src_obj_pose"], atol=1e-4)
               for s in subs):
            return False
    return True


def fit(attempts: list[dict], num_sources: int, link: str = "logistic", location: bool = False, gaps: np.ndarray | None = None) -> dict:
    """The model mapping of the module docstring, with the fit's size and its in-sample R^2 (of the probabilities).
    When ``shared_subtasks`` holds, the subtask-0 terms are fitted once and every subtask gets them (plus its in-hand term)."""
    if link not in ("logistic", "linear"):
        raise ValueError(f"link must be logistic or linear, got {link!r}")
    if location and link != "logistic":
        raise ValueError("location terms need the logistic link")
    num_subtasks = len(attempts[0]["subtasks"])
    if num_subtasks > 1 and shared_subtasks(attempts):
        one = [{**a, "subtasks": [s for s in a["subtasks"] if int(s["subtask_ind"]) == 0]} for a in attempts]
        model = fit(one, num_sources, link, location)
        y = np.array([float(a["success"]) for a in attempts])
        base0 = model["subtasks"][0]
        if gaps is not None:            # the in-hand terms on top of the shared linear predictor
            z0 = np.array([_shared_z(base0, a["subtasks"][0], num_sources) for a in one])
            G = np.asarray(gaps, dtype=float)[:, 1:]
            X = np.column_stack([z0, G])
            beta = _logistic(X, y) if link == "logistic" else np.linalg.lstsq(X, y, rcond=None)[0]
            scale, inh = float(beta[0]), beta[1:]
            mean_g = G.mean(axis=0)
            pred = X @ beta
            resid = y - (1 / (1 + np.exp(-pred)) if link == "logistic" else pred)
            model["r2"] = round(float(1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))), 4)
            base0 = _scaled(base0, scale)
        model["subtasks"] = {}
        for s in range(num_subtasks):
            c = dict(base0)
            if gaps is not None:
                c["offset"] = round(float(base0["offset"] + sum(inh[j] * mean_g[j] for j in range(num_subtasks - 1) if j != s - 1)), 6)
                if s > 0:
                    c["inhand"] = round(float(inh[s - 1]), 6)
            model["subtasks"][s] = c
        model["shared_subtasks"] = True
        return model
    base = num_sources - 1 + 2                        # source dummies (source 0 is the reference) + displacement + rotation
    nloc = 4 * num_sources if location else 0         # source-by-location terms (standardized x, y, sin yaw, cos yaw)
    ninh = 1 if gaps is not None else 0               # the in-hand term (constant 0 for the first subtask, so dropped there)
    width = base + nloc + ninh
    X = np.zeros((len(attempts), 1 + num_subtasks * width))
    X[:, 0] = 1.0
    y = np.zeros(len(attempts))
    raw_loc = np.zeros((len(attempts), num_subtasks, 4))
    src_of = np.zeros((len(attempts), num_subtasks), dtype=int)
    for i, a in enumerate(attempts):
        y[i] = float(a["success"])
        for s in a["subtasks"]:
            si = int(s["subtask_ind"])
            o = 1 + si * width
            k = int(s["src_demo_ind"])
            if k > 0:
                X[i, o + k - 1] = 1.0
            X[i, o + base - 2], X[i, o + base - 1] = transform_size(_pose12(s["cur_obj_pose"]), _pose12(s["src_obj_pose"]))
            raw_loc[i, si] = location_features(_pose12(s["cur_obj_pose"]))
            src_of[i, si] = k
            if ninh:
                X[i, o + base + nloc] = gaps[i, si]
    loc_mean = raw_loc.mean(axis=0)
    loc_std = raw_loc.std(axis=0) + 1e-9
    if location:
        for si in range(num_subtasks):
            z = (raw_loc[:, si] - loc_mean[si]) / loc_std[si]
            for i in range(len(y)):
                o = 1 + si * width + base + 4 * src_of[i, si]
                X[i, o:o + 4] = z[i]
    varies = [0] + [j for j in range(1, X.shape[1]) if X[:, j].std() > 1e-9]
    beta = np.zeros(X.shape[1])
    chosen_penalty = None
    if link == "logistic":
        is_loc = np.array([base <= ((j - 1) % width) < base + nloc if j > 0 else False for j in range(X.shape[1])])[varies]
        if location:
            scores = {lam: _cv_log_loss(X[:, varies], y, lam * is_loc) for lam in LOCATION_PENALTIES}
            chosen_penalty = min(scores, key=scores.get)
            beta[varies] = _logistic(X[:, varies], y, chosen_penalty * is_loc)
        else:
            beta[varies] = _logistic(X[:, varies], y)
    else:
        beta[varies] = np.linalg.lstsq(X[:, varies], y, rcond=None)[0]
    resid = y - (1 / (1 + np.exp(-X @ beta)) if link == "logistic" else X @ beta)
    mean = X.mean(axis=0)
    contrib = [float(beta[1 + s * width:1 + (s + 1) * width] @ mean[1 + s * width:1 + (s + 1) * width]) for s in range(num_subtasks)]
    subtasks = {}
    for s in range(num_subtasks):
        o = 1 + s * width
        subtasks[s] = {"offset": round(float(beta[0] + sum(contrib) - contrib[s]), 6),
                       "source": [0.0] + [round(float(b), 6) for b in beta[o:o + num_sources - 1]],
                       "displacement": round(float(beta[o + base - 2]), 6),
                       "rotation": round(float(beta[o + base - 1]), 6)}
        if location:
            subtasks[s]["location"] = {"mean": [round(float(v), 6) for v in loc_mean[s]],
                                       "std": [round(float(v), 6) for v in loc_std[s]],
                                       "coef": [[round(float(v), 6) for v in beta[o + base + 4 * k:o + base + 4 * k + 4]]
                                                for k in range(num_sources)]}
        if ninh and s > 0:
            subtasks[s]["inhand"] = round(float(beta[o + base + nloc]), 6)
    extra = {"location_penalty": chosen_penalty} if location else {}
    return {"link": link, **extra, "attempts": len(y), "success_rate": round(float(y.mean()), 4), "num_sources": num_sources,
            "r2": round(float(1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))), 4), "subtasks": subtasks}


def _shared_z(c: dict, sub: dict, num_sources: int) -> float:
    """The linear predictor of one fitted subtask block ``c`` for a record's subtask entry (its own source)."""
    cur = _pose12(sub["cur_obj_pose"])
    d, r = transform_size(cur, _pose12(sub["src_obj_pose"]))
    k = int(sub["src_demo_ind"])
    z = c["offset"] + c["source"][k] + c["displacement"] * d + c["rotation"] * r
    if "location" in c:
        loc = c["location"]
        z += np.asarray(loc["coef"][k]) @ ((location_features(cur) - np.asarray(loc["mean"])) / np.asarray(loc["std"]))
    return float(z)


def _scaled(c: dict, scale: float) -> dict:
    """A fitted subtask block with every coefficient multiplied by ``scale`` (the in-hand refit's weight on its predictor)."""
    out = {"offset": round(c["offset"] * scale, 6), "source": [round(v * scale, 6) for v in c["source"]],
           "displacement": round(c["displacement"] * scale, 6), "rotation": round(c["rotation"] * scale, 6)}
    if "location" in c:
        out["location"] = {**c["location"], "coef": [[round(v * scale, 6) for v in row] for row in c["location"]["coef"]]}
    return out


DECAY_SCALE = (0.1, float(np.radians(30)), 0.01)   # the fit works in units of 10 cm, 30 degrees and 1 cm (in-hand gap); the file stores per m and per rad


def fit_decay(attempts: list[dict], num_sources: int, gaps: np.ndarray | None = None) -> dict:
    """The ``decay`` model of the module docstring, with the fit's size and in-sample R^2 (of the probabilities)."""
    from scipy.optimize import minimize
    num_subtasks = len(attempts[0]["subtasks"])
    n = len(attempts)
    src = np.zeros((n, num_subtasks), dtype=int)
    disp = np.zeros((n, num_subtasks))
    rot = np.zeros((n, num_subtasks))
    y = np.array([float(a["success"]) for a in attempts])
    for i, a in enumerate(attempts):
        for s in a["subtasks"]:
            si = int(s["subtask_ind"])
            src[i, si] = int(s["src_demo_ind"])
            d, r = transform_size(_pose12(s["cur_obj_pose"]), _pose12(s["src_obj_pose"]))
            disp[i, si], rot[i, si] = d / DECAY_SCALE[0], r / DECAY_SCALE[1]
    shared = num_subtasks > 1 and shared_subtasks(attempts)
    if shared:                                  # one copy of the source rates (see shared_subtasks); later subtasks: in-hand only
        disp[:, 1:], rot[:, 1:] = 0.0, 0.0
    mixed = np.array([len(set(row)) > 1 for row in src], dtype=float)
    gap = np.zeros((n, num_subtasks)) if gaps is None else np.asarray(gaps, dtype=float) / DECAY_SCALE[2]
    ninh = 0 if gaps is None else num_subtasks         # one in-hand rate per subtask (unused for the first)

    def prob(theta):
        log_p = np.zeros(n)
        for si in range(num_subtasks):
            a = theta[2 * num_sources * si:2 * num_sources * si + num_sources]
            b = theta[2 * num_sources * si + num_sources:2 * num_sources * (si + 1)]
            log_p -= a[src[:, si]] * disp[:, si] + b[src[:, si]] * rot[:, si]
            if ninh:
                log_p -= theta[2 * num_sources * num_subtasks + si] * gap[:, si]
        return np.clip(np.exp(log_p) * np.where(mixed == 1, theta[-1], 1.0), 1e-6, 1 - 1e-6)

    def loss(theta):
        p = prob(theta)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    size = 2 * num_sources * num_subtasks + ninh + 1
    start = np.r_[np.full(size - 1, 0.2), 0.5]
    theta = minimize(loss, start, method="L-BFGS-B", bounds=[(0.0, 20.0)] * (size - 1) + [(1e-3, 1.0)]).x
    resid = y - prob(theta)
    subtasks = {}
    for si in range(num_subtasks):
        o = 2 * num_sources * (0 if shared else si)
        subtasks[si] = {"decay_displacement": [round(float(v) / DECAY_SCALE[0], 6) for v in theta[o:o + num_sources]],
                        "decay_rotation": [round(float(v) / DECAY_SCALE[1], 6) for v in theta[o + num_sources:o + 2 * num_sources]]}
        if ninh and si > 0:
            subtasks[si]["inhand"] = round(float(theta[2 * num_sources * num_subtasks + si]) / DECAY_SCALE[2], 6)
    extra = {"shared_subtasks": True} if shared else {}
    return {"link": "decay", **extra, "mixed_factor": round(float(theta[-1]), 6), "attempts": n, "success_rate": round(float(y.mean()), 4),
            "num_sources": num_sources, "r2": round(float(1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))), 4),
            "subtasks": subtasks}


@dataclass(frozen=True)
class TransformSuccessModel:
    """The fitted model of one task (module docstring)."""
    subtasks: dict
    num_sources: int
    link: str = "linear"

    @classmethod
    def load(cls, path) -> TransformSuccessModel:
        m = yaml.safe_load(Path(path).read_text())
        return cls(subtasks={int(k): v for k, v in m["subtasks"].items()}, num_sources=int(m["num_sources"]),
                   link=str(m.get("link", "linear")))

    def predict(self, subtask: int, displacements, rotations, current_pose=None, inhand_gaps=None) -> np.ndarray:
        """Predicted full-task success of every source for one subtask (a probability). ``current_pose`` (the current
        reference-object pose) feeds the source-by-location terms of a model fitted with ``--location``; ``inhand_gaps``
        (per source, m) the in-hand term of a model fitted with ``--source``."""
        c = self.subtasks[int(subtask)]
        inhand = 0.0
        if "inhand" in c:
            if inhand_gaps is None:
                raise ValueError("this model has an in-hand term: pass inhand_gaps")
            inhand = c["inhand"] * np.asarray(inhand_gaps, dtype=float)
        if self.link == "decay":
            return np.exp(-(np.asarray(c["decay_displacement"], dtype=float) * np.asarray(displacements, dtype=float)
                            + np.asarray(c["decay_rotation"], dtype=float) * np.asarray(rotations, dtype=float) + inhand))
        z = (c["offset"] + np.asarray(c["source"], dtype=float) + c["displacement"] * np.asarray(displacements, dtype=float)
             + c["rotation"] * np.asarray(rotations, dtype=float))
        if "location" in c:
            if current_pose is None:
                raise ValueError("this model has source-by-location terms: pass current_pose")
            loc = c["location"]
            zf = (location_features(current_pose) - np.asarray(loc["mean"])) / np.asarray(loc["std"])
            z = z + np.asarray(loc["coef"], dtype=float) @ zf
        z = z + inhand
        return 1 / (1 + np.exp(-z)) if self.link == "logistic" else np.clip(z, 0.0, 1.0)


def main(argv=None):
    p = argparse.ArgumentParser(description="Fit the convex mixer's success model on a baseline ranking pool.")
    p.add_argument("--pool", required=True, help="baseline ranking pool folder (chunks/*/records/attempts.jsonl)")
    p.add_argument("--out", required=True)
    p.add_argument("--num-sources", type=int, default=10)
    p.add_argument("--link", choices=("logistic", "linear", "decay"), default="logistic")
    p.add_argument("--location", action="store_true", help="add source-by-location terms (scene-adaptive weights)")
    p.add_argument("--source", help="the task's source hdf5 (MimicGen datagen_info): add the in-hand term")
    p.add_argument("--isaaclab-source", help="an Isaac Lab generator's source hdf5 (obs/datagen_info): read the pool's records "
                   "with isaaclab_rows")
    p.add_argument("--inhand", action="store_true", help="with --isaaclab-source: add the in-hand term")
    a = p.parse_args(argv)
    rows = _attempt_rows(Path(a.pool))
    if a.isaaclab_source:
        rows, gaps = isaaclab_rows(rows, a.isaaclab_source, Path(a.pool), inhand=a.inhand)
    else:
        gaps = inhand_gaps(rows, source_inhand(a.source)) if a.source else None
    model = fit_decay(rows, a.num_sources, gaps) if a.link == "decay" else fit(rows, a.num_sources, a.link, a.location, gaps)
    model = {"basis": f"tmg.generation.method.transform_model fit on {a.pool}", **model}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(yaml.safe_dump(model, sort_keys=False))
    print(f"[tmg] transform model: {model['attempts']} attempts, R^2 {model['r2']}, written to {a.out}")


if __name__ == "__main__":
    main()
