"""``tmg fit_coefficients``: fit the SkillMix model of a task and write its rule file.

Input: finished pools of (generator, task, scene, selection=ours) made with the same code and settings:

- randmix pools (``randmix_pools``): method ``skillmix`` with ``mixing.rule: randmix`` under the method name
  ``skillmix_randmix``. Every subtask of every attempt is blended with random Dirichlet weights (recorded under
  ``subtasks[k].mix``: ``weights``, ``scene_features``, ``inhand_features``), so the pools show how task success changes
  with the weights;
- unmix pools (``unmix_pools``): method ``none``, one source per subtask (a one-hot weight).

and, for MimicGen, the task's source demonstrations (their ``datagen_info`` gives each source's in-hand pose). Rows of
subtask k are the attempts that reached k, labelled with the attempt's task success
(``tmg.generation.method.success_model``). Columns: the weights ``w`` (J), the weighted features ``sum_j w_j phi_j`` (five)
and the spread ``H(w) = -sum_j w_j log w_j`` (0 for a unmix row). The model

    P(task succeeds | reached k, w) = sigmoid( c0 + sum_j w_j a_j + g . sum_j w_j phi_j + d H(w) )

is a logistic regression on standardized columns with an L2 penalty chosen per subtask by cross-validation over
``PENALTIES`` (``PENALTY_FOLDS`` folds), folds grouped by chunk (attempts of one chunk share their scenes; a randmix
chunk and a unmix chunk are different groups even when they share a seed); the reported gain is the chunk-grouped
cross-validated log-loss gain over the base rate (``GAIN_FOLDS`` folds), on all rows and on the randmix rows. The rule
file keeps ``a``, ``g``, ``d`` and ``c0`` per subtask (``tmg.generation.method.skillmix`` turns them into the softmax
weights).

Isaac Lab generators (``generator.simulator: isaaclab``, ``fit_isaaclab_blocks``): their records number the subtasks of every
end effector from 0 (the GR1's right arm 0-3 and left arm 0-1 in Bottle Placement), so a block is fitted per (end effector,
subtask) and written under the key ``<eef>/<k>`` when the task has more than one end effector (``skillmix.subtask_key``),
under ``k`` for a single arm (SoftMimicGen). Their subtask records carry the selected source's poses (``src_obj_pose``,
``src_eef_pose``, ``src_objects``; ``isaaclab.runtime.SubtaskScene``), so no source file is read. A subtask without a
reference object gets no block (the mixer leaves it unmixed). Which subtasks an attempt reached (``isaaclab_outcomes``):
subtask k of an end effector, other than its last, succeeded when its termination signal (the record's ``term_signal``)
first fired (``term_first``) before that end effector's subtask k + 1 started (its record's ``start_step``). A missing
signal raises; ``fit.allow_reached_fallback: true`` instead counts every recorded subtask as reached, so the row's label,
the task success, is the only outcome used. The fit prints, per block, how many attempts used which.

Output: ``<output>/coefficients.yaml`` and ``<output>/FIT_COEFFICIENTS.json`` (summary), and when ``write_config`` is true
``configs/experiment/generator/<generator>/coefficients/<task>.yaml``, the file ``method.mixing.coefficients`` names.
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
from tmg.generation.method.skillmix import rule_path, subtask_key
from tmg.generation.method.success_model import (
    FEATURES,
    UNITS,
    design_rows,
    reached_outcomes,
    subtask_outcomes,
)

COEFFICIENT_FILE = "coefficients.yaml"
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
    """Rows of one subtask from one kind of pool: the columns ``[w (J), sum_j w_j phi_j (5), H(w)]``, the task-success
    labels and the chunk index of every row."""
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


def entropy(w) -> float:
    """``H(w) = -sum_j w_j log w_j`` over the non-zero weights."""
    w = np.asarray(w, dtype=float)
    nz = w > 0
    return float(-(w[nz] * np.log(w[nz])).sum())


def mix_records(attempt: Mapping, eef: str | None = None) -> dict[int, dict]:
    """The first mix record with weights of every subtask of an attempt (planned segments, then solves whose motion plan
    failed); with ``eef``, of that end effector's subtasks only."""
    out = {}
    for m in [s.get("mix") for s in attempt["subtasks"]] + list(attempt.get("unplanned_mixes") or []):
        if m and "weights" in m and (eef is None or m.get("eef") == eef) and int(m["subtask"]) not in out:
            out[int(m["subtask"])] = m
    return out


def unmix_rows(attempts: Attempts, success: Sequence, k: int, num_sources: int, infos, eef: str | None = None) -> Rows:
    """Rows of subtask k (of end effector ``eef``) from a unmix pool: the one-hot source weight, its features, spread 0."""
    X, y, kept = design_rows(attempts.records, success, k, num_sources, infos, eef=eef)
    return Rows([np.append(x, 0.0) for x in X], list(y), [attempts.chunks[i] for i in kept])


def randmix_rows(attempts: Attempts, success: Sequence | None, k: int, num_sources: int, eef: str | None = None) -> Rows:
    """Rows of subtask k (of end effector ``eef``) from a randmix pool: the recorded weights, the weighted features, their
    spread. An attempt counts when it has a mix record of subtask k and, with ``success`` (termination signals), every
    earlier subtask succeeded. ``success`` None is the generator that stops at the first failed subtask: there the mix
    record itself shows that the attempt reached k, also when the motion plan to k then failed and k has no subtask
    record."""
    rows = Rows([], [], [])
    for index, a in enumerate(attempts.records):
        m = mix_records(a, eef).get(k)
        if m is None or (success is not None and not all(success[index][:k])):
            continue
        w = np.asarray(m["weights"], dtype=float)
        inhand = np.asarray(m["inhand_features"], dtype=float) if m.get("inhand_features") else np.zeros((num_sources, 2))
        phi = np.hstack([np.asarray(m["scene_features"], dtype=float), inhand])
        rows.x.append(np.concatenate([w, w @ phi, [entropy(w)]]))
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


def fit_subtask(randmix: Rows, unmix: Rows, num_sources: int, with_cv: bool = True) -> dict:
    """One subtask's block of the rule file from its randmix and unmix rows (``Rows``, or ``(x, y, chunks)``)."""
    randmix, unmix = Rows(*randmix), Rows(*unmix)
    rows = list(unmix.x) + list(randmix.x)
    if not rows:
        raise ValueError("no attempt reached this subtask")
    X, y = np.array(rows), np.array(list(unmix.y) + list(randmix.y))
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
               with_cv: bool = True) -> dict[int, dict]:
    """The rule-file block of every subtask from the attempts of the randmix and unmix pools. ``signals`` are the
    task's subtask termination signals (None for a generator that stops an attempt at its first failed subtask);
    ``source`` is the source-demonstration file whose ``datagen_info`` gives each source's in-hand pose, read when the
    records do not carry it (with ``signals``, from the second subtask on)."""
    successful = [a for a in randmix.records + unmix.records if a["success"]]
    K = len(signals) if signals else max(len(a["subtasks"]) for a in successful)
    infos = read_source_infos(source) if signals and K > 1 else None
    reached_unmix = subtask_success(unmix, signals, K)
    reached_randmix = subtask_success(randmix, signals, K) if signals else None
    return {k: fit_subtask(randmix_rows(randmix, reached_randmix, k, num_sources),
                           unmix_rows(unmix, reached_unmix, k, num_sources, infos), num_sources, with_cv)
            for k in range(K)}


# Isaac Lab generators ---------------------------------------------------------------------------------------------
def eef_subtasks(attempt: Mapping, eef: str) -> list[dict]:
    """The subtask records of one end effector of an attempt, by subtask index."""
    return sorted((s for s in attempt["subtasks"] if s.get("eef") == eef), key=lambda s: int(s["subtask_ind"]))


def isaaclab_subtasks(attempts: Sequence[Mapping]) -> dict[str, int]:
    """End effector -> number of subtasks, from the records (the largest subtask index recorded, plus one)."""
    out: dict[str, int] = {}
    for a in attempts:
        for s in a["subtasks"]:
            out[str(s["eef"])] = max(out.get(str(s["eef"]), 0), int(s["subtask_ind"]) + 1)
    return dict(sorted(out.items()))


def check_isaaclab_records(attempts: Sequence[Mapping]) -> dict | None:
    """Refuse records without the scene of their subtasks (made before the Isaac Lab runtimes recorded it, or whose scene
    could not be read: ``scene_error``) and deformable-object frames of different definitions or reference clouds.
    Returns the one deformable-object frame of the records (``{version, reference_sha256}``), or None."""
    missing, errors, frames = 0, [], set()
    for a in attempts:
        for s in a["subtasks"]:
            if s.get("scene_error"):
                errors.append(f"attempt {a.get('attempt')} {s.get('eef')}/{s.get('subtask_ind')}: {s['scene_error']}")
            elif (s.get("object_name") is not None and "src_obj_pose" not in s) or "start_step" not in s:
                missing += 1
            if s.get("object_frame"):
                frames.add((s["object_frame"]["version"], s["object_frame"]["reference_sha256"]))
    if errors:
        raise ValueError(f"{len(errors)} Isaac Lab subtask records have no scene (scene_error), e.g. {errors[0]}")
    if missing:
        raise ValueError(f"{missing} Isaac Lab subtask records lack src_obj_pose or start_step: the pools were made before "
                         f"the runtime recorded the subtask scene; regenerate them")
    if len(frames) > 1:
        raise ValueError(f"the records use different deformable-object frames {sorted(frames)}")
    if not frames:
        return None
    version, sha = next(iter(frames))
    return {"version": version, "reference_sha256": sha}


def isaaclab_outcomes(attempts: Attempts, eef: str, num_subtasks: int, allow_fallback: bool = False,
                      no_step_signals: bool = False) -> tuple[list[list[bool]], dict]:
    """Success of every subtask of one end effector in every attempt (module docstring), and how many attempts each
    subtask's outcome came from: ``{k: {"term_signal": n, "fallback": n}}`` (the last subtask: the task success).

    Subtask k (not the last) succeeded when its termination signal (``term_signal`` of its record) first fired
    (``term_first``) before the step at which the same end effector's subtask k + 1 started (``start_step`` of that record;
    the attempt's last step when k + 1 was never built), as MimicGen's ``subtask_outcomes`` counts it up to the end of
    subtask k. An attempt without steps (a plan that failed before execution) reached nothing past what it recorded. A
    recorded signal that the attempt's per-step signals do not have raises (a name mismatch, or a stack that records no
    per-step signals). Without a usable signal the fallback (a recorded subtask counts as reached, the label alone
    decides) is used only with ``allow_fallback`` (``fit.allow_reached_fallback``), or, for an attempt that recorded no
    per-step signal at all, when the generator declares that its environments publish none (``no_step_signals``:
    ``generator.per_step_subtask_signals: false``, DexMimicGen); otherwise that raises too."""
    out, used = [], {k: {"term_signal": 0, "fallback": 0} for k in range(num_subtasks - 1)}
    for a in attempts.records:
        subs = {int(s["subtask_ind"]): s for s in eef_subtasks(a, eef)}
        fired = a.get("term_first") or {}
        n_steps = int(a.get("n_steps", 0))
        starts = [int(subs[k]["start_step"]) for k in sorted(subs) if k >= 1]
        if n_steps and len(starts) > 0 and max(starts) == 0:
            raise ValueError(f"attempt {a.get('attempt')} {eef}: every later subtask starts at step 0, so the subtasks "
                             f"were not built when they started; the reached window cannot be measured")
        row = []
        for k in range(num_subtasks):
            if k == num_subtasks - 1:
                row.append(bool(a["success"]))
                continue
            signal = (subs.get(k) or {}).get("term_signal")
            where = f"attempt {a.get('attempt')} {eef}/{k}"
            if signal is not None and (str(signal) in fired or n_steps == 0):
                used[k]["term_signal"] += 1
                end = int(subs[k + 1]["start_step"]) if k + 1 in subs else n_steps
                first = int(fired.get(str(signal), -1))
                row.append(0 <= first < end)
                continue
            if not (allow_fallback or (no_step_signals and not fired)):
                why = (f"its term signal {signal!r} is not among the attempt's per-step signals {sorted(fired)}"
                       if signal is not None else "it has no term signal")
                raise ValueError(f"{where}: {why}; set fit.allow_reached_fallback=true to count a recorded subtask as "
                                 f"reached and fit on the task-success label alone")
            used[k]["fallback"] += 1
            row.append(k in subs)
        out.append(row)
    return out, used


def describe_outcomes(used: Mapping) -> str:
    """One line: where the outcome of each earlier subtask (which decides whether an attempt reached this one) came from."""
    parts = []
    for k, n in sorted(used.items()):
        kind = ("term signal" if n["fallback"] == 0 else
                "FALLBACK: a recorded subtask counts as reached, the label is the task success" if n["term_signal"] == 0 else
                "term signal where recorded, else the fallback")
        parts.append(f"earlier subtask {k}: {kind} ({n['term_signal']} attempts by signal, {n['fallback']} by fallback)")
    return "; ".join(parts) or "first subtask: every attempt reaches it"


def fit_isaaclab_blocks(randmix: Attempts, unmix: Attempts, *, num_sources: int, with_cv: bool = True,
                        allow_fallback: bool = False, no_step_signals: bool = False) -> tuple[dict, dict]:
    """The rule-file blocks of an Isaac Lab generator's task (module docstring), keyed by ``skillmix.subtask_key``, and per
    block where its reached signal came from (``allow_fallback``: ``isaaclab_outcomes``)."""
    check_isaaclab_records(randmix.records + unmix.records)
    counts = isaaclab_subtasks([a for a in randmix.records + unmix.records if a["success"]]
                               or randmix.records + unmix.records)
    two_arms = len(counts) > 1
    blocks, sources = {}, {}
    for eef, K in counts.items():
        reached_unmix, used_unmix = isaaclab_outcomes(unmix, eef, K, allow_fallback, no_step_signals)
        reached_randmix, used_randmix = isaaclab_outcomes(randmix, eef, K, allow_fallback, no_step_signals)
        for k in range(K):
            if not any(s.get("object_name") is not None for a in unmix.records for s in eef_subtasks(a, eef)
                       if int(s["subtask_ind"]) == k):
                continue                                    # no reference object: the mixer leaves the subtask alone
            key = subtask_key(k, eef if two_arms else None)
            blocks[key] = fit_subtask(randmix_rows(randmix, reached_randmix, k, num_sources, eef),
                                      unmix_rows(unmix, reached_unmix, k, num_sources, None, eef), num_sources, with_cv)
            before = {j: {kind: used_unmix[j][kind] + used_randmix[j][kind] for kind in ("term_signal", "fallback")}
                      for j in range(k)}
            sources[key] = describe_outcomes(before)
    if not blocks:
        raise ValueError("no subtask with a reference object in the Isaac Lab records")
    return blocks, sources


# rule file ------------------------------------------------------------------------------------------------------------
def write_rule(path: Path, *, generator: str, task: str, basis: str, num_sources: int, blocks: Mapping,
               object_frame: Mapping | None = None) -> Path:
    """Write a SkillMix softmax rule file (read by ``tmg.generation.method.skillmix.load_rule``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    head = (f"# SkillMix rule of {generator} {task} (tmg fit_coefficients). Per subtask k, with phi_j = (|dx|, |dy|, rotation,\n"
            f"# inhand_position, inhand_rotation) of source j against the current scene (in-hand terms 0 for k = 0):\n"
            f"#   P(task succeeds | reached k, weights w) = sigmoid(c0 + sum_j w_j s_j + d H(w)),  s_j = a[j] + g . phi_j\n"
            f"#   weights: softmax(s / d) over the sources compatible with the selected one when d > 0, the best one when d <= 0\n"
            f"# Units: {UNITS}.\n")
    keyed = any(isinstance(k, str) for k in blocks)
    if keyed:
        head += "# Subtask keys <eef>/<k>: subtask k of that end effector (the subtasks of every arm are numbered from 0).\n"
    data = {"generator": generator, "task": task, "rule": "softmax", "basis": basis, "num_sources": int(num_sources),
            "features": list(FEATURES), "units": UNITS,
            "subtasks": {(str(k) if keyed else int(k)): dict(blocks[k]) for k in sorted(blocks, key=str if keyed else int)}}
    if object_frame is not None:          # deformable reference object: the mixer checks the frame of the scene against it
        data["object_frame"] = dict(object_frame)
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
    reached, frame = {}, None
    if generator.get("simulator") == "isaaclab":
        frame = check_isaaclab_records(attempts["randmix"].records + attempts["unmix"].records)
        allow = bool((plain.get("fit") or {}).get("allow_reached_fallback", False))
        no_step_signals = generator.get("per_step_subtask_signals", True) is False
        if no_step_signals:
            print(f"[tmg] {generator['name']}: the environments publish no per-step subtask signals "
                  f"(generator.per_step_subtask_signals: false); attempts without signals count every recorded subtask "
                  f"as reached and are fitted on the task-success label alone", flush=True)
        blocks, reached = fit_isaaclab_blocks(attempts["randmix"], attempts["unmix"], num_sources=num_sources,
                                              allow_fallback=allow, no_step_signals=no_step_signals)
    else:
        blocks = fit_blocks(attempts["randmix"], attempts["unmix"], num_sources=num_sources,
                            signals=task.get("subtask_term_signals") or None, source=plain["source"])
    for k, b in blocks.items():
        print(f"[tmg] subtask {k}: rows randmix/unmix {b['rows']['randmix']}/{b['rows']['unmix']}, d {b['d']:+.3f} "
              f"({'softmax' if b['d'] > 0 else 'best source alone'}), CV gain {b['cv_log_loss_gain_percent']}", flush=True)
        if k in reached:
            print(f"[tmg] subtask {k}: reached from {reached[k]}", flush=True)
    basis = (f"Fit (tmg fit_coefficients) on the randmix pools {pools['randmix']} ({len(attempts['randmix'])} "
             f"attempts) and the unmix pools {pools['unmix']} ({len(attempts['unmix'])} attempts); rows of "
             f"subtask k: the attempts that reached it, label = the attempt's task success.")
    kwargs = dict(generator=generator["name"], task=task["name"], basis=basis, num_sources=num_sources, blocks=blocks,
                  object_frame=frame)
    written = [str(write_rule(output / COEFFICIENT_FILE, **kwargs))]
    if plain.get("write_config", True):
        written.append(str(write_rule(rule_path(f"{generator['name']}/{task['name']}"), **kwargs)))
    summary = {"randmix_pools": pools["randmix"], "unmix_pools": pools["unmix"],
               "randmix_attempts": len(attempts["randmix"]), "unmix_attempts": len(attempts["unmix"]),
               "subtasks": {k: {key: b[key] for key in ("d", "penalty", "rows", "successes", "cv_log_loss_gain_percent")}
                            for k, b in blocks.items()},
               **({"reached_from": reached} if reached else {}),
               "written": written}
    write_json(output / SUMMARY, summary)
    print(f"[tmg] written: {', '.join(written)}")
    return summary
