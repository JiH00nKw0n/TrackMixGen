"""``tmg fit_coefficients``: fit the SkillMix model of a task and write its rule file.

Input: finished pools of (generator, task, scene, selection=ours) made with the same code and settings:

- calibration pools (``calibration_pools``): method ``skillmix`` with ``mixing.rule: calibration`` under the method name
  ``skillmix_calibration``. Every subtask of every attempt is blended with random Dirichlet weights (recorded under
  ``subtasks[k].mix``: ``weights``, ``scene_features``, ``inhand_features``), so the pools show how task success changes
  with the weights;
- baseline pools (``baseline_pools``): method ``none``, one source per subtask (a one-hot weight).

and, for MimicGen, the task's source demonstrations (their ``datagen_info`` gives each source's in-hand pose). Rows of
subtask k are the attempts that reached k, labelled with the attempt's task success
(``tmg.generation.method.success_model``). Columns: the weights ``w`` (J), the weighted features ``sum_j w_j phi_j`` (five)
and the spread ``H(w) = -sum_j w_j log w_j`` (0 for a baseline row). The model

    P(task succeeds | reached k, w) = sigmoid( c0 + sum_j w_j a_j + g . sum_j w_j phi_j + d H(w) )

is a logistic regression on standardized columns with an L2 penalty chosen per subtask by 4-fold cross-validation over
``PENALTIES``, folds grouped by chunk (attempts of one chunk share their scenes); the reported gain is the 5-fold
chunk-grouped cross-validated log-loss gain over the base rate, on all rows and on the calibration rows. The rule file
keeps ``a``, ``g``, ``d`` and ``c0`` per subtask (``tmg.generation.method.skillmix`` turns them into the softmax weights).

Output: ``<output>/coefficients.yaml`` and ``<output>/FIT_COEFFICIENTS.json`` (summary), and when ``write_config`` is true
``configs/experiment/generator/<generator>/coefficients/<task>.yaml``, the file ``method.mixing.coefficients`` names.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml

from tmg.config import dump_yaml, resolved
from tmg.core.chunks import CHUNKS_DIR, RESOLVED_CONFIG, chunk_is_done, write_json
from tmg.core.provenance import write_provenance
from tmg.core.records import RECORDS_DIR, read_attempts
from tmg.generation.method.skillmix import rule_path
from tmg.generation.method.success_model import (
    FEATURES,
    UNITS,
    design_rows,
    reached_outcomes,
    subtask_outcomes,
)

COEFFICIENT_FILE = "coefficients.yaml"
SUMMARY = "FIT_COEFFICIENTS.json"
PENALTIES = (0.1, 1.0, 10.0, 100.0)          # candidate L2 penalties (standardized columns), chosen by 4-fold CV
BASELINE_GROUP = 10000                       # chunk groups of the baseline pools are offset by this (kept apart from the
                                             # calibration chunks, whose seeds are the same)


def read_source_infos(path) -> list[dict]:
    """Per source demonstration (``demo_0``, ``demo_1``, ... in index order): ``eef_pose`` (T,4,4) and
    ``object_poses`` {name: (T,4,4)} of MimicGen's ``datagen_info``."""
    import h5py
    out = []
    with h5py.File(path, "r") as f:
        for name in sorted(f["data"].keys(), key=lambda d: int(d.split("_")[1])):
            info = f["data"][name]["datagen_info"]
            out.append({"eef_pose": np.asarray(info["eef_pose"], dtype=float),
                        "object_poses": {k: np.asarray(v, dtype=float) for k, v in info["object_poses"].items()}})
    return out


# reading pools --------------------------------------------------------------------------------------------------------
def finished_chunks(pools: Sequence) -> list[Path]:
    """The finished chunks with attempt records of the pools, in pool order and chunk-name order; a chunk name seen in an
    earlier pool is skipped (a copy of the same chunk)."""
    seen, out = set(), []
    for pool in pools:
        for chunk in sorted((Path(pool) / CHUNKS_DIR).glob("seed*")):
            if chunk.name in seen or not chunk_is_done(chunk) or not (chunk / RECORDS_DIR / "attempts.jsonl").exists():
                continue
            seen.add(chunk.name)
            out.append(chunk)
    return out


def read_pools(pools: Sequence, signals) -> tuple[list[dict], list, list[int]]:
    """``(attempts, subtask outcomes (MimicGen) or None, chunk index of every attempt)``."""
    attempts, outcomes, groups = [], [], []
    for index, chunk in enumerate(finished_chunks(pools)):
        folder = chunk / RECORDS_DIR
        for a in read_attempts(folder):
            if signals:                    # MimicGen: the recorded subtask termination signals
                with np.load(folder / a["npz"]) as z:
                    steps = {k: z[k] for k in z.files if k in ("seg_sub", "seg_of_step") or k.startswith("term__")}
                outcomes.append(subtask_outcomes(a, steps, signals))
            else:
                outcomes.append(None)
            attempts.append(a)
            groups.append(index)
    return attempts, outcomes, groups


def entropy(w) -> float:
    """``H(w) = -sum_j w_j log w_j`` over the non-zero weights."""
    w = np.asarray(w, dtype=float)
    nz = w > 0
    return float(-(w[nz] * np.log(w[nz])).sum())


def mix_records(attempt: Mapping) -> dict[int, dict]:
    """The first mix record with weights of every subtask of an attempt (planned segments, then solves whose motion plan
    failed)."""
    out = {}
    for m in [s.get("mix") for s in attempt["subtasks"]] + list(attempt.get("unplanned_mixes") or []):
        if m and "weights" in m and int(m["subtask"]) not in out:
            out[int(m["subtask"])] = m
    return out


def baseline_rows(attempts, outcomes, groups, k: int, num_sources: int, infos) -> tuple[list, list, list]:
    """Rows of subtask k from a baseline pool: the one-hot source weight, its features, spread 0."""
    J = num_sources
    X, y, _ = design_rows(attempts, outcomes, k, J, infos)
    if len(y) == 0:
        return [], [], []
    g = [grp for a, o, grp in zip(attempts, outcomes, groups) if all(o[:k]) and len(a["subtasks"]) > k]
    inhand = X[:, J + 3:J + 5] if X.shape[1] > J + 3 else np.zeros((len(X), 2))
    rows = [np.concatenate([X[i, :J], X[i, J:J + 3], inhand[i], [0.0]]) for i in range(len(y))]
    return rows, list(y), [BASELINE_GROUP + v for v in g]


def calibration_rows(attempts, outcomes, groups, k: int, num_sources: int, mimicgen: bool) -> tuple[list, list, list]:
    """Rows of subtask k from a calibration pool: the recorded weights, the weighted features, their spread."""
    rows, ys, gs = [], [], []
    for a, o, grp in zip(attempts, outcomes, groups):
        mixes = mix_records(a)
        if k not in mixes or (mimicgen and not all(o[:k])):
            continue
        m = mixes[k]
        w = np.asarray(m["weights"], dtype=float)
        inhand = np.asarray(m["inhand_features"], dtype=float) if m.get("inhand_features") else np.zeros((num_sources, 2))
        phi = np.hstack([np.asarray(m["scene_features"], dtype=float), inhand])
        rows.append(np.concatenate([w, w @ phi, [entropy(w)]]))
        ys.append(float(bool(a["success"])))
        gs.append(grp)
    return rows, ys, gs


# the model ------------------------------------------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_penalized(X: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    """Logistic regression with an intercept (unpenalized) and an L2 penalty on the other coefficients, by Newton's method."""
    Xa = np.hstack([np.ones((len(X), 1)), X])
    b = np.zeros(Xa.shape[1])
    pen = np.full(len(b), float(penalty))
    pen[0] = 0.0
    for _ in range(80):
        p = _sigmoid(Xa @ b)
        grad = Xa.T @ (p - y) + pen * b
        hess = Xa.T @ (Xa * (p * (1 - p))[:, None]) + np.diag(pen + 1e-8)
        step = np.linalg.solve(hess, grad)
        b -= step
        if np.abs(step).max() < 1e-8:
            break
    return b


def _nll(p, y) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


class MixingModel:
    """The model of the module docstring on rows ``X``, labels ``y`` and chunk groups ``groups`` (0, 1, ...): standardized
    columns, the penalty of ``PENALTIES`` with the lowest 4-fold chunk-grouped log loss, then one fit on every row.
    ``coef`` are the coefficients of the raw columns and ``c0`` the intercept."""

    def __init__(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        folds = np.random.default_rng(1).permutation(groups.max() + 1)[groups] % 4
        best = None
        for penalty in PENALTIES:
            loss = 0.0
            for f in range(4):
                train, test = folds != f, folds == f
                if train.any() and test.any():
                    b = _fit_penalized(Z[train], y[train], penalty)
                    loss += _nll(_sigmoid(b[0] + Z[test] @ b[1:]), y[test])
            if best is None or loss < best[0]:
                best = (loss, penalty)
        b = _fit_penalized(Z, y, best[1])
        self.coef = b[1:] / sd
        self.c0 = float(b[0] - (b[1:] * mu / sd).sum())
        self.penalty = best[1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(self.c0 + X @ self.coef)


def cv_gain(X: np.ndarray, y: np.ndarray, groups: np.ndarray, calibration: np.ndarray) -> tuple[float, float]:
    """5-fold chunk-grouped cross-validated log-loss gain (%) of the model over the base rate: on all rows, and on the
    calibration rows only."""
    folds = np.random.default_rng(0).permutation(groups.max() + 1)[groups] % 5
    base_all = model_all = base_cal = model_cal = 0.0
    for f in range(5):
        train, test = folds != f, folds == f
        p = MixingModel(X[train], y[train], groups[train]).predict(X[test])
        base = np.full(test.sum(), y[train].mean())
        base_all += _nll(base, y[test])
        model_all += _nll(p, y[test])
        c = calibration[test]
        base_cal += _nll(base[c], y[test][c])
        model_cal += _nll(p[c], y[test][c])
    return (100 * (base_all - model_all) / base_all,
            100 * (base_cal - model_cal) / base_cal if base_cal else float("nan"))


def fit_subtask(calibration: tuple, baseline: tuple, num_sources: int, with_cv: bool = True) -> dict:
    """One subtask's block of the rule file from its calibration and baseline rows (``(rows, labels, groups)`` each)."""
    rows = list(baseline[0]) + list(calibration[0])
    if not rows:
        raise ValueError("no attempt reached this subtask")
    X, y = np.array(rows), np.array(list(baseline[1]) + list(calibration[1]))
    is_cal = np.array([False] * len(baseline[0]) + [True] * len(calibration[0]))
    _, groups = np.unique(np.array(list(baseline[2]) + list(calibration[2])), return_inverse=True)
    m = MixingModel(X, y, groups)
    J = num_sources
    block = {"a": [float(v) for v in m.coef[:J]], "g": {f: float(v) for f, v in zip(FEATURES, m.coef[J:J + 5])},
             "d": float(m.coef[J + 5]), "c0": m.c0, "penalty": m.penalty,
             "rows": {"calibration": int(is_cal.sum()), "baseline": int((~is_cal).sum())},
             "successes": int(y.sum())}
    if with_cv:
        gain_all, gain_cal = cv_gain(X, y, groups, is_cal)
        block["cv_log_loss_gain_percent"] = {"all_rows": round(gain_all, 1), "calibration_rows": round(gain_cal, 1)}
    return block


# rule file ------------------------------------------------------------------------------------------------------------
def write_rule(path: Path, *, generator: str, task: str, basis: str, num_sources: int, blocks: Mapping) -> Path:
    """Write a SkillMix softmax rule file (read by ``tmg.generation.method.skillmix.load_rule``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = (f"# SkillMix rule of {generator} {task} (tmg fit_coefficients). Per subtask k, with phi_j = (|dx|, |dy|, rotation,\n"
            f"# inhand_position, inhand_rotation) of source j against the current scene (in-hand terms 0 for k = 0):\n"
            f"#   P(task succeeds | reached k, weights w) = sigmoid(c0 + sum_j w_j s_j + d H(w)),  s_j = a[j] + g . phi_j\n"
            f"#   weights: softmax(s / d) over the sources compatible with the selected one when d > 0, the best one when d <= 0\n"
            f"# Units: {UNITS}.\n")
    data = {"generator": generator, "task": task, "rule": "softmax", "basis": basis, "num_sources": int(num_sources),
            "features": list(FEATURES), "units": UNITS, "subtasks": {int(k): dict(v) for k, v in sorted(blocks.items())}}
    path.write_text(head + yaml.safe_dump(data, sort_keys=False, width=120))
    return path


def fit_coefficients(cfg) -> dict:
    """Run the stage described in the module docstring and return the summary written to ``FIT_COEFFICIENTS.json``."""
    plain = resolved(cfg)
    calibration_pools = [Path(p) for p in plain["calibration_pools"]]
    baseline_pools = [Path(p) for p in plain["baseline_pools"]]
    output = Path(plain["output"])
    output.mkdir(parents=True, exist_ok=True)
    dump_yaml(plain, output / RESOLVED_CONFIG)
    write_provenance(output, extra={"command": "fit_coefficients", "output": str(output),
                                    "calibration_pools": [str(p) for p in calibration_pools],
                                    "baseline_pools": [str(p) for p in baseline_pools]})
    task, generator = plain["task"], plain["generator"]
    mimicgen = bool(task.get("subtask_term_signals"))
    signals = task.get("subtask_term_signals") if mimicgen else None
    cal = read_pools(calibration_pools, signals)
    base = read_pools(baseline_pools, signals)
    for name, (attempts, _, _), pools in (("calibration", cal, calibration_pools), ("baseline", base, baseline_pools)):
        if len(attempts) < int(plain["min_attempts"]):
            raise ValueError(f"{len(attempts)} attempts in the {name} pools {[str(p) for p in pools]}, fewer than "
                             f"min_attempts={plain['min_attempts']}")
    num_sources = int(task.get("num_sources") or 10)
    K = len(signals) if signals else max(len(a["subtasks"]) for a in cal[0] + base[0] if a["success"])
    if not mimicgen:                      # CP-Gen: an attempt stops at its first failed subtask
        cal = (cal[0], [reached_outcomes(a, K) for a in cal[0]], cal[2])
        base = (base[0], [reached_outcomes(a, K) for a in base[0]], base[2])
    infos = read_source_infos(plain["source"]) if mimicgen and K > 1 else None
    blocks = {}
    for k in range(K):
        blocks[k] = fit_subtask(calibration_rows(*cal, k, num_sources, mimicgen), baseline_rows(*base, k, num_sources, infos),
                                num_sources)
        b = blocks[k]
        print(f"[tmg] subtask {k}: rows calibration/baseline {b['rows']['calibration']}/{b['rows']['baseline']}, d {b['d']:+.3f} "
              f"({'softmax' if b['d'] > 0 else 'best source alone'}), CV gain {b['cv_log_loss_gain_percent']}", flush=True)
    basis = (f"Fit (tmg fit_coefficients) on the calibration pools {[str(p) for p in calibration_pools]} ({len(cal[0])} "
             f"attempts) and the baseline pools {[str(p) for p in baseline_pools]} ({len(base[0])} attempts); rows of "
             f"subtask k: the attempts that reached it, label = the attempt's task success.")
    kwargs = dict(generator=generator["name"], task=task["name"], basis=basis, num_sources=num_sources, blocks=blocks)
    written = [str(write_rule(output / COEFFICIENT_FILE, **kwargs))]
    if plain.get("write_config", True):
        written.append(str(write_rule(rule_path(f"{generator['name']}/{task['name']}"), **kwargs)))
    summary = {"calibration_pools": [str(p) for p in calibration_pools], "baseline_pools": [str(p) for p in baseline_pools],
               "calibration_attempts": len(cal[0]), "baseline_attempts": len(base[0]),
               "subtasks": {k: {key: b[key] for key in ("d", "penalty", "rows", "successes", "cv_log_loss_gain_percent")}
                            for k, b in blocks.items()},
               "written": written}
    write_json(output / SUMMARY, summary)
    print(f"[tmg] written: {', '.join(written)}")
    return summary
