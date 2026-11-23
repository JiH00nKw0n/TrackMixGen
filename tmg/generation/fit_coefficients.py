"""``tmg fit_coefficients``: fit the SkillMix model of a task and write its rule files.

Input: finished pools of (generator, task, scene, selection=ours) made with the same code and settings:

- randmix pools (``randmix_pools``): method ``skillmix`` with ``mixing.rule: randmix`` under the method name
  ``skillmix_randmix``. Every subtask of every attempt is blended with random Dirichlet weights (recorded under
  ``subtasks[k].mix``: ``weights``, ``scene_features``, ``inhand_features``), so the pools show how task success changes
  with the weights;
- unmix pools (``unmix_pools``): method ``none``, one source per subtask (a one-hot weight).

and, for MimicGen, the task's source demonstrations (their ``datagen_info`` gives each source's in-hand pose). Rows of
subtask k are the attempts that reached k, labelled with the attempt's task success
(``tmg.generation.method.success_model``). Columns: the weights ``w`` (J), the weighted features ``sum_j w_j phi_j`` (five)
and the spread of the weights (0 for a unmix row), measured once per fitted rule (``tmg.generation.method.skillmix``):
``H(w) = -sum_j w_j log w_j`` for the softmax rule, ``1 - sum_j w_j^2`` for the sparsemax rule. The model

    P(task succeeds | reached k, w) = sigmoid( c0 + sum_j w_j a_j + g . sum_j w_j phi_j + d spread(w) )

is a logistic regression on standardized columns with an L2 penalty chosen per subtask by cross-validation over
``PENALTIES`` (``PENALTY_FOLDS`` folds), folds grouped by chunk (attempts of one chunk share their scenes; a randmix
chunk and a unmix chunk are different groups even when they share a seed); the reported gain is the chunk-grouped
cross-validated log-loss gain over the base rate (``GAIN_FOLDS`` folds), on all rows and on the randmix rows. It is fitted
once per rule of ``rules`` (default: every fitted rule) on the same rows; each rule file keeps ``a``, ``g``, ``d`` and ``c0`` per subtask
(``tmg.generation.method.skillmix`` turns them into the weights of that rule).

Output: ``<output>/<rule>.yaml`` for every rule of ``rules`` and ``<output>/FIT_COEFFICIENTS.json`` (summary), and when
``write_config`` is true ``configs/experiment/generator/<generator>/coefficients/<rule>/<task>.yaml``, the files
``method.mixing.coefficients`` names.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml

from tmg.config import dump_yaml, resolved
from tmg.core.chunks import CHUNKS_DIR, RESOLVED_CONFIG, chunk_is_done, write_json
from tmg.core.layout import ATTEMPTS_JSONL
from tmg.core.provenance import write_provenance
from tmg.core.records import RECORDS_DIR, read_attempts
from tmg.generation.method.skillmix import FITTED, FITTED_RULES, rule_path
from tmg.generation.method.success_model import (
    FEATURES,
    UNITS,
    design_rows,
    reached_outcomes,
    subtask_outcomes,
)

SUMMARY = "FIT_COEFFICIENTS.json"
PENALTIES = (0.1, 1.0, 10.0, 100.0)          # candidate L2 penalties (standardized columns)
PENALTY_FOLDS, PENALTY_FOLD_SEED = 4, 1      # cross-validation that chooses the penalty: folds, seed of the chunk shuffle
GAIN_FOLDS, GAIN_FOLD_SEED = 5, 0            # cross-validation of the reported log-loss gain
NEWTON_STEPS, NEWTON_TOLERANCE = 80, 1e-8    # the logistic fit stops after this many steps or below this step size
RANDMIX, UNMIX = 0, 1                 # kind of pool a row comes from (first word of its cross-validation group)


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
@dataclass
class Attempts:
    """The attempt records of the pools of one kind, with the records folder and chunk index (0, 1, ... over the chunks
    read) of every attempt."""
    records: list[dict]
    folders: list[Path]
    chunks: list[int]

    def __len__(self) -> int:
        return len(self.records)


class Rows(NamedTuple):
    """Rows of one subtask from one kind of pool: the columns ``[w (J), sum_j w_j phi_j (5)]`` (the spread column is
    added per rule by ``with_spread``), the task-success labels and the chunk index of every row."""
    x: list
    y: list
    chunks: list


def finished_chunks(pools: Sequence) -> list[Path]:
    """The finished chunks with attempt records of the pools, in pool order and chunk-name order; a chunk name seen in an
    earlier pool is skipped (a copy of the same chunk)."""
    seen, out = set(), []
    for pool in pools:
        for chunk in sorted((Path(pool) / CHUNKS_DIR).glob("seed*")):
            if chunk.name in seen or not chunk_is_done(chunk) or not (chunk / RECORDS_DIR / ATTEMPTS_JSONL).exists():
                continue
            seen.add(chunk.name)
            out.append(chunk)
    return out


def read_pools(pools: Sequence) -> Attempts:
    """Every attempt record of the finished chunks of ``pools``."""
    out = Attempts([], [], [])
    for index, chunk in enumerate(finished_chunks(pools)):
        folder = chunk / RECORDS_DIR
        for a in read_attempts(folder):
            out.records.append(a)
            out.folders.append(folder)
            out.chunks.append(index)
    return out


def subtask_success(attempts: Attempts, signals: Sequence | None, num_subtasks: int) -> list[list[bool]]:
    """Success of every subtask of every attempt (``success_model`` module docstring): from the recorded termination
    signals when the task has them (``task.subtask_term_signals``, MimicGen), otherwise from how far the attempt went (a
    generator that stops an attempt at its first failed subtask, CP-Gen)."""
    if not signals:
        return [reached_outcomes(a, num_subtasks) for a in attempts.records]
    out = []
    for a, folder in zip(attempts.records, attempts.folders):
        with np.load(folder / a["npz"]) as z:
            steps = {k: z[k] for k in z.files if k in ("seg_sub", "seg_of_step") or k.startswith("term__")}
        out.append(subtask_outcomes(a, steps, signals))
    return out


def mix_records(attempt: Mapping) -> dict[int, dict]:
    """The first mix record with weights of every subtask of an attempt (planned segments, then solves whose motion plan
    failed)."""
    out = {}
    for m in [s.get("mix") for s in attempt["subtasks"]] + list(attempt.get("unplanned_mixes") or []):
        if m and "weights" in m and int(m["subtask"]) not in out:
            out[int(m["subtask"])] = m
    return out


def unmix_rows(attempts: Attempts, success: Sequence, k: int, num_sources: int, infos) -> Rows:
    """Rows of subtask k from a unmix pool: the one-hot source weight and its features."""
    X, y, kept = design_rows(attempts.records, success, k, num_sources, infos)
    return Rows([np.asarray(x, dtype=float) for x in X], list(y), [attempts.chunks[i] for i in kept])


def randmix_rows(attempts: Attempts, success: Sequence | None, k: int, num_sources: int) -> Rows:
    """Rows of subtask k from a randmix pool: the recorded weights and the weighted features. An attempt
    counts when it has a mix record of subtask k and, with ``success`` (termination signals), every earlier subtask
    succeeded. ``success`` None is the generator that stops at the first failed subtask: there the mix record itself shows
    that the attempt reached k, also when the motion plan to k then failed and k has no subtask record."""
    rows = Rows([], [], [])
    for index, a in enumerate(attempts.records):
        m = mix_records(a).get(k)
        if m is None or (success is not None and not all(success[index][:k])):
            continue
        w = np.asarray(m["weights"], dtype=float)
        inhand = np.asarray(m["inhand_features"], dtype=float) if m.get("inhand_features") else np.zeros((num_sources, 2))
        phi = np.hstack([np.asarray(m["scene_features"], dtype=float), inhand])
        rows.x.append(np.concatenate([w, w @ phi]))
        rows.y.append(float(bool(a["success"])))
        rows.chunks.append(attempts.chunks[index])
    return rows


# the model ------------------------------------------------------------------------------------------------------------
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _fit_penalized(X: np.ndarray, y: np.ndarray, penalty: float) -> np.ndarray:
    """Logistic regression with an intercept (unpenalized) and an L2 penalty on the other coefficients, by Newton's method."""
    Xa = np.hstack([np.ones((len(X), 1)), X])
    b = np.zeros(Xa.shape[1])
    pen = np.full(len(b), float(penalty))
    pen[0] = 0.0
    for _ in range(NEWTON_STEPS):
        p = _sigmoid(Xa @ b)
        grad = Xa.T @ (p - y) + pen * b
        hess = Xa.T @ (Xa * (p * (1 - p))[:, None]) + np.diag(pen + 1e-8)
        step = np.linalg.solve(hess, grad)
        b -= step
        if np.abs(step).max() < NEWTON_TOLERANCE:
            break
    return b


def _nll(p, y) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).sum())


def _folds(groups: np.ndarray, folds: int, seed: int) -> np.ndarray:
    """The fold of every row: the groups (0, 1, ...) shuffled with ``seed`` and dealt round-robin, so a group stays whole."""
    return np.random.default_rng(seed).permutation(groups.max() + 1)[groups] % folds


class MixingModel:
    """The model of the module docstring on rows ``X``, labels ``y`` and chunk groups ``groups`` (0, 1, ...): standardized
    columns, the penalty of ``PENALTIES`` with the lowest chunk-grouped cross-validated log loss, then one fit on every row.
    ``coef`` are the coefficients of the raw columns and ``c0`` the intercept."""

    def __init__(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray):
        mu, sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - mu) / sd
        folds = _folds(groups, PENALTY_FOLDS, PENALTY_FOLD_SEED)
        best = None
        for penalty in PENALTIES:
            loss = 0.0
            for f in range(PENALTY_FOLDS):
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


def cv_gain(X: np.ndarray, y: np.ndarray, groups: np.ndarray, randmix: np.ndarray) -> tuple[float, float]:
    """Chunk-grouped cross-validated log-loss gain (%) of the model over the base rate: on all rows, and on the
    randmix rows only."""
    folds = _folds(groups, GAIN_FOLDS, GAIN_FOLD_SEED)
    base_all = model_all = base_cal = model_cal = 0.0
    for f in range(GAIN_FOLDS):
        train, test = folds != f, folds == f
        p = MixingModel(X[train], y[train], groups[train]).predict(X[test])
        base = np.full(test.sum(), y[train].mean())
        base_all += _nll(base, y[test])
        model_all += _nll(p, y[test])
        c = randmix[test]
        base_cal += _nll(base[c], y[test][c])
        model_cal += _nll(p[c], y[test][c])
    return (100 * (base_all - model_all) / base_all,
            100 * (base_cal - model_cal) / base_cal if base_cal else float("nan"))


def with_spread(rows: Sequence, num_sources: int, rule: str) -> np.ndarray:
    """The design matrix of ``rows`` with the spread of their weights (first ``num_sources`` columns) under ``rule`` as
    the last column (0 for a one-hot weight under both rules)."""
    spread = FITTED[rule].spread
    return np.array([np.append(x, spread(np.asarray(x[:num_sources]))) for x in rows])


def fit_subtask(randmix: Rows, unmix: Rows, num_sources: int, rule: str, with_cv: bool = True) -> dict:
    """One subtask's block of the ``rule`` file from its randmix and unmix rows (``Rows``, or ``(x, y, chunks)``)."""
    randmix, unmix = Rows(*randmix), Rows(*unmix)
    rows = list(unmix.x) + list(randmix.x)
    if not rows:
        raise ValueError("no attempt reached this subtask")
    X, y = with_spread(rows, num_sources, rule), np.array(list(unmix.y) + list(randmix.y))
    is_randmix = np.array([False] * len(unmix.x) + [True] * len(randmix.x))
    # one cross-validation group per (kind of pool, chunk): the two kinds share their chunk seeds but not their attempts
    keys = [(UNMIX, c) for c in unmix.chunks] + [(RANDMIX, c) for c in randmix.chunks]
    rank = {key: i for i, key in enumerate(sorted(set(keys)))}
    groups = np.array([rank[key] for key in keys])
    m = MixingModel(X, y, groups)
    J, F = num_sources, len(FEATURES)
    block = {"a": [float(v) for v in m.coef[:J]], "g": {f: float(v) for f, v in zip(FEATURES, m.coef[J:J + F])},
             "d": float(m.coef[J + F]), "c0": m.c0, "penalty": m.penalty,
             "rows": {"randmix": int(is_randmix.sum()), "unmix": int((~is_randmix).sum())},
             "successes": int(y.sum())}
    if with_cv:
        gain_all, gain_cal = cv_gain(X, y, groups, is_randmix)
        block["cv_log_loss_gain_percent"] = {"all_rows": round(gain_all, 1), "randmix_rows": round(gain_cal, 1)}
    return block


def fit_blocks(randmix: Attempts, unmix: Attempts, *, num_sources: int, signals: Sequence | None, source,
               rules: Sequence = FITTED_RULES, with_cv: bool = True) -> dict[str, dict[int, dict]]:
    """``{rule: {subtask: block}}``: the rule-file block of every subtask under every rule of ``rules``, from the attempts
    of the randmix and unmix pools (the rows are read once and fitted once per rule). ``signals`` are the
    task's subtask termination signals (None for a generator that stops an attempt at its first failed subtask);
    ``source`` is the source-demonstration file whose ``datagen_info`` gives each source's in-hand pose, read when the
    records do not carry it (with ``signals``, from the second subtask on)."""
    successful = [a for a in randmix.records + unmix.records if a["success"]]
    K = len(signals) if signals else max(len(a["subtasks"]) for a in successful)
    infos = read_source_infos(source) if signals and K > 1 else None
    reached_unmix = subtask_success(unmix, signals, K)
    reached_randmix = subtask_success(randmix, signals, K) if signals else None
    rows = {k: (randmix_rows(randmix, reached_randmix, k, num_sources), unmix_rows(unmix, reached_unmix, k, num_sources, infos))
            for k in range(K)}
    return {rule: {k: fit_subtask(*rows[k], num_sources, rule, with_cv) for k in range(K)} for rule in rules}


# rule file ------------------------------------------------------------------------------------------------------------
def write_rule(path: Path, *, rule: str, generator: str, task: str, basis: str, num_sources: int, blocks: Mapping) -> Path:
    """Write a SkillMix rule file of ``rule`` (read by ``tmg.generation.method.skillmix.load_rule``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fitted = FITTED[rule]
    head = (f"# SkillMix {rule} rule of {generator} {task} (tmg fit_coefficients). Per subtask k, with phi_j = (|dx|, |dy|,\n"
            f"# rotation, inhand_position, inhand_rotation) of source j against the current scene (in-hand terms 0 for k = 0):\n"
            f"#   P(task succeeds | reached k, weights w) = sigmoid(c0 + sum_j w_j s_j + d spread(w)),\n"
            f"#   s_j = a[j] + g . phi_j,  spread(w) = {fitted.spread_formula}\n"
            f"#   weights: {fitted.weights_formula} over the sources compatible with the selected one when d > 0, the best one when d <= 0\n"
            f"# Units: {UNITS}.\n")
    data = {"generator": generator, "task": task, "rule": rule, "basis": basis, "num_sources": int(num_sources),
            "features": list(FEATURES), "units": UNITS, "subtasks": {int(k): dict(v) for k, v in sorted(blocks.items())}}
    path.write_text(head + yaml.safe_dump(data, sort_keys=False, width=120))
    return path


def fit_coefficients(cfg) -> dict:
    """Run the stage described in the module docstring and return the summary written to ``FIT_COEFFICIENTS.json``."""
    plain = resolved(cfg)
    pools = {"randmix": [str(p) for p in plain["randmix_pools"]], "unmix": [str(p) for p in plain["unmix_pools"]]}
    output = Path(plain["output"])
    output.mkdir(parents=True, exist_ok=True)
    dump_yaml(plain, output / RESOLVED_CONFIG)
    write_provenance(output, extra={"command": "fit_coefficients", "output": str(output),
                                    "randmix_pools": pools["randmix"], "unmix_pools": pools["unmix"]})
    task, generator = plain["task"], plain["generator"]
    attempts = {kind: read_pools(paths) for kind, paths in pools.items()}
    for kind, read in attempts.items():
        if len(read) < int(plain["min_attempts"]):
            raise ValueError(f"{len(read)} attempts in the {kind} pools {pools[kind]}, fewer than "
                             f"min_attempts={plain['min_attempts']}")
    num_sources = int(task["num_sources"])
    rules = [str(rule) for rule in plain["rules"]]
    if not rules or set(rules) - set(FITTED_RULES):
        raise ValueError(f"rules must name fitted rules ({FITTED_RULES}), got {rules}")
    fitted = fit_blocks(attempts["randmix"], attempts["unmix"], num_sources=num_sources,
                        signals=task.get("subtask_term_signals") or None, source=plain["source"], rules=rules)
    for rule, blocks in fitted.items():
        for k, b in blocks.items():
            print(f"[tmg] {rule} subtask {k}: rows randmix/unmix {b['rows']['randmix']}/{b['rows']['unmix']}, d {b['d']:+.3f} "
                  f"({'mixed' if b['d'] > 0 else 'best source alone'}), CV gain {b['cv_log_loss_gain_percent']}", flush=True)
    basis = (f"Fit (tmg fit_coefficients) on the randmix pools {pools['randmix']} ({len(attempts['randmix'])} "
             f"attempts) and the unmix pools {pools['unmix']} ({len(attempts['unmix'])} attempts); rows of "
             f"subtask k: the attempts that reached it, label = the attempt's task success.")
    name = f"{generator['name']}/{task['name']}"
    written = []
    for rule, blocks in fitted.items():
        kwargs = dict(rule=rule, generator=generator["name"], task=task["name"], basis=basis, num_sources=num_sources,
                      blocks=blocks)
        written.append(str(write_rule(output / f"{rule}.yaml", **kwargs)))
        if plain.get("write_config", True):
            written.append(str(write_rule(rule_path(name, rule), **kwargs)))
    summary = {"randmix_pools": pools["randmix"], "unmix_pools": pools["unmix"],
               "randmix_attempts": len(attempts["randmix"]), "unmix_attempts": len(attempts["unmix"]),
               "rules": {rule: {k: {key: b[key] for key in ("d", "penalty", "rows", "successes", "cv_log_loss_gain_percent")}
                                for k, b in blocks.items()} for rule, blocks in fitted.items()},
               "written": written}
    write_json(output / SUMMARY, summary)
    print(f"[tmg] written: {', '.join(written)}")
    return summary
