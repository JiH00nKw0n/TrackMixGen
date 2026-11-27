"""``tmg fit_coefficients``: fit the SkillMix model of a task and write its rule file.

Input: finished pools of (generator, task, scene, selection=ours) made with the same code and settings:

- randmix pools (``randmix_pools``): method ``skillmix`` with ``mixing.rule: randmix`` under the method name
  ``skillmix_randmix``. Every subtask of every attempt is blended with random Dirichlet weights (recorded under
  ``subtasks[k].mix``: ``weights``, ``scene_features``, ``inhand_features``, ``held``), so the pools show how task success
  changes with the weights;
- unmix pools (``unmix_pools``): method ``none``, one source per subtask (a one-hot weight).

Rows of a block are the attempts that reached its subtask, labelled with the attempt's task success
(``tmg.generation.method.success_model``). Columns: the weights ``w`` (J), the weighted features ``sum_j w_j phi_j`` and the
spread ``H(w) = -sum_j w_j log w_j`` (0 for a unmix row). The model

    P(task succeeds | reached k, w) = sigmoid( c0 + sum_j w_j a_j + g . sum_j w_j phi_j + d H(w) )

is a logistic regression on standardized columns with an L2 penalty chosen per subtask by cross-validation over
``PENALTIES`` (``PENALTY_FOLDS`` folds), folds grouped by chunk (attempts of one chunk share their scenes; a randmix
chunk and a unmix chunk are different groups even when they share a seed); the reported gain is the chunk-grouped
cross-validated log-loss gain over the base rate (``GAIN_FOLDS`` folds), on all rows and on the randmix rows. The rule
file keeps ``a``, ``g``, ``d`` and ``c0`` per block (``tmg.generation.method.skillmix`` turns them into the softmax
weights).

One fit path for every generator (``fit_blocks``). What differs between generators is declared by the generator adapter
as its fit layout (``SyntheticDataGenerator.fit_layout``, ``tmg.generation.method.fit_layout``): the arms (one block per
``SubtaskId``: ``k`` for one arm, ``<eef>/<k>`` when every arm numbers its subtasks from 0), the rule that says which
subtasks an attempt reached, where the selected source's poses come from, and the features. The records are first
brought to the current record version and checked (``tmg.generation.method.fit_records.prepare``): attempts whose scene
could not be read are left out, the records must name the layout's arms, and records measured with different
deformable-object frames or hold rules are refused. A subtask without a reference object gets no block (the mixer
leaves it unmixed). The fit prints, per block, where the reached verdicts came from and a sanity report
(``fit_records.sanity_report``, judged on the whole pool by ``judge_sanity``), and stops when the report has an error.
A randmix row whose mix record cannot say what a feature needs (``success_model.FeatureUnknown``: an unversioned record
without per-source hold flags that the pools' own hold tests could not fill) is left out and counted. A feature whose
column is the same in every row of a block cannot be fitted: its slope is reported, and a slope other than 0 is a
loud warning.

Output: ``<output>/coefficients.yaml`` and ``<output>/FIT_COEFFICIENTS.json`` (summary), and when ``write_config`` is true
``configs/experiment/generator/<generator>/coefficients/<task>.yaml``, the file ``method.mixing.coefficients`` names. The
rule file records what its records were measured with (``features``, ``object_frame``, ``held_rule``,
``donor_registration``); the mixer refuses a scene measured otherwise.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import yaml

from tmg.config import dump_yaml, resolved
from tmg.core.chunks import RESOLVED_CONFIG, write_json
from tmg.core.provenance import write_provenance
from tmg.generation.generators import make_generator
from tmg.generation.method.fit_layout import FitLayout
from tmg.generation.method.fit_records import (
    ASSUMED,
    SIGNAL,
    Attempts,
    complete_source_holds,
    describe_outcomes,
    judge_sanity,
    measured_with,
    mix_records,
    prepare,
    print_sanity,
    reached_rule,
    read_pools,
    sanity_report,
)
from tmg.generation.method.interfaces import HeldRule, SubtaskId
from tmg.generation.method.skillmix import rule_path
from tmg.generation.method.success_model import (
    FEATURES,
    HELD_FEATURE,
    UNITS,
    FeatureUnknown,
    design_rows,
    eef_subtasks,
    mix_record_features,
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


# rows -----------------------------------------------------------------------------------------------------------------
class Rows(NamedTuple):
    """Rows of one block from one kind of pool: the columns ``[w (J), sum_j w_j phi_j (F), H(w)]``, the task-success
    labels and the chunk index of every row."""
    x: list
    y: list
    chunks: list


def entropy(w) -> float:
    """``H(w) = -sum_j w_j log w_j`` over the non-zero weights."""
    w = np.asarray(w, dtype=float)
    nz = w > 0
    return float(-(w[nz] * np.log(w[nz])).sum())


def unmix_rows(attempts: Attempts, success: Sequence, k: int, num_sources: int, *, layout: FitLayout, eef: str | None = None,
               source_infos=None) -> Rows:
    """Rows of subtask k (of end effector ``eef``) from a unmix pool: the one-hot source weight, its features, spread 0."""
    X, y, kept = design_rows(attempts.records, success, k, num_sources, eef=eef, features=layout.features,
                             source_poses=layout.source_poses, source_infos=source_infos)
    return Rows([np.append(x, 0.0) for x in X], list(y), [attempts.chunks[i] for i in kept])


def randmix_rows(attempts: Attempts, success: Sequence | None, k: int, num_sources: int, *, features: Sequence[str] = FEATURES,
                 eef: str | None = None, unknown: list | None = None) -> Rows:
    """Rows of subtask k (of end effector ``eef``) from a randmix pool: the recorded weights, the weighted features, their
    spread. An attempt counts when it has a mix record of subtask k and, with ``success`` (the reached rule's verdicts),
    every earlier subtask succeeded. ``success`` None is the generator whose mix record itself shows that the attempt
    reached k (``ReachedRule.mix_record_proves_reach``), also when the motion plan to k then failed and k has no subtask
    record. An attempt whose mix record cannot say a feature (``FeatureUnknown``) gives no row; its index in ``attempts``
    is appended to ``unknown``."""
    rows = Rows([], [], [])
    for index, a in enumerate(attempts.records):
        m = mix_records(a, eef).get(k)
        if m is None or (success is not None and not all(success[index][:k])):
            continue
        w = np.asarray(m["weights"], dtype=float)
        try:
            phi = mix_record_features(m, features, num_sources)
        except FeatureUnknown:
            if unknown is not None:
                unknown.append(index)
            continue
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


def fit_subtask(randmix: Rows, unmix: Rows, num_sources: int, with_cv: bool = True, features: Sequence[str] = FEATURES) -> dict:
    """One block of the rule file from its randmix and unmix rows (``Rows``, or ``(x, y, chunks)``) with the feature
    columns ``features``."""
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
    J, F = num_sources, len(features)
    block = {"a": [float(v) for v in m.coef[:J]], "g": {f: float(v) for f, v in zip(features, m.coef[J:J + F])},
             "d": float(m.coef[J + F]), "c0": m.c0, "penalty": m.penalty,
             "rows": {"randmix": int(is_randmix.sum()), "unmix": int((~is_randmix).sum())},
             "successes": int(y.sum())}
    if with_cv:
        gain_all, gain_cal = cv_gain(X, y, groups, is_randmix)
        block["cv_log_loss_gain_percent"] = {"all_rows": round(gain_all, 1), "randmix_rows": round(gain_cal, 1)}
    return block


# the blocks of a task ----------------------------------------------------------------------------------------------
def has_reference_object(attempts: Attempts, eef: str | None, k: int) -> bool:
    """False when subtask k of ``eef`` was recorded and never with a reference object (the mixer leaves it unmixed)."""
    subs = [s for a in attempts.records for s in eef_subtasks(a, eef) if int(s["subtask_ind"]) == k]
    return not subs or any(s.get("object_name") is not None for s in subs)


def constant_features(randmix: Rows, unmix: Rows, num_sources: int, features: Sequence[str]) -> dict:
    """``{feature: value}`` of the features whose column is the same in every row of a block (nothing to fit there)."""
    randmix, unmix = Rows(*randmix), Rows(*unmix)
    rows = np.array(list(unmix.x) + list(randmix.x))
    if not len(rows):
        return {}
    columns = rows[:, num_sources:num_sources + len(features)]
    return {f: float(columns[0, i]) for i, f in enumerate(features) if np.all(columns[:, i] == columns[0, i])}


def fit_blocks(randmix: Attempts, unmix: Attempts, *, layout: FitLayout, task: Mapping, num_sources: int, source=None,
               with_cv: bool = True) -> tuple[dict, dict]:
    """The rule-file block of every subtask from the (prepared) attempts of the randmix and unmix pools, keyed by
    ``SubtaskId``, and per block what the fit says about it: ``reached`` (where the reached verdicts of the earlier
    subtasks came from), ``unknown_rows`` (randmix attempts left out because their mix record could not say a feature)
    and ``constant_features`` (module docstring). ``source``: the source-demonstration file, read when the layout takes
    the sources' poses from its ``datagen_info``."""
    rule = reached_rule(layout.reached)
    blocks, about = {}, {}
    for eef in layout.arms.record_eefs:
        K = rule.num_subtasks(randmix.records + unmix.records, eef, task)
        infos = read_source_infos(source) if layout.source_poses == "datagen_info" and K > 1 else None
        reached_unmix, used_unmix = rule.outcomes(unmix, eef, K, task)
        if rule.mix_record_proves_reach:
            reached_randmix, used_randmix = None, {k: {SIGNAL: len(randmix), ASSUMED: 0} for k in used_unmix}
        else:
            reached_randmix, used_randmix = rule.outcomes(randmix, eef, K, task)
        for k in range(K):
            if not has_reference_object(unmix, eef, k):
                continue
            key, unknown = layout.arms.subtask(k, eef), []
            mixed = randmix_rows(randmix, reached_randmix, k, num_sources, features=layout.features, eef=eef, unknown=unknown)
            single = unmix_rows(unmix, reached_unmix, k, num_sources, layout=layout, eef=eef, source_infos=infos)
            blocks[key] = fit_subtask(mixed, single, num_sources, with_cv, layout.features)
            about[key] = {"reached": describe_outcomes({j: {kind: used_unmix[j][kind] + used_randmix[j][kind] for kind in (SIGNAL, ASSUMED)}
                                                        for j in range(k)}),
                          "unknown_rows": len(unknown), "constant_features": constant_features(mixed, single, num_sources, layout.features)}
    if not blocks:
        raise ValueError("no subtask with a reference object in the records")
    return blocks, about


def print_block(key, block: Mapping, about: Mapping) -> None:
    """What the fit says about one block, the warnings loud."""
    print(f"[tmg] subtask {key}: rows randmix/unmix {block['rows']['randmix']}/{block['rows']['unmix']}, d {block['d']:+.3f} "
          f"({'softmax' if block['d'] > 0 else 'best source alone'}), CV gain {block.get('cv_log_loss_gain_percent')}", flush=True)
    print(f"[tmg] subtask {key}: reached from {about['reached']}", flush=True)
    if about["unknown_rows"]:
        print(f"[tmg] WARNING subtask {key}: {about['unknown_rows']} randmix attempts left out: their mix records do not say "
              f"which sources hold the carried object, and no hold test of these pools settles it", flush=True)
    for feature, value in about["constant_features"].items():
        slope = block["g"][feature]
        if slope != 0.0:
            print(f"[tmg] WARNING subtask {key}: feature {feature} is {value} in every row, yet its fitted slope is {slope:+.3g}: "
                  f"nothing in these pools supports that slope", flush=True)
        else:
            print(f"[tmg] subtask {key}: feature {feature} is {value} in every row (slope 0: these pools say nothing about it)",
                  flush=True)


# rule file ------------------------------------------------------------------------------------------------------------
def write_rule(path: Path, *, generator: str, task: str, basis: str, num_sources: int, blocks: Mapping,
               features: Sequence[str] = FEATURES, object_frame: Mapping | None = None, held_rule: HeldRule | None = None,
               donor_registration: Mapping | None = None) -> Path:
    """Write a SkillMix softmax rule file (read by ``tmg.generation.method.skillmix.load_rule``); ``blocks`` are keyed by
    ``SubtaskId``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blocks = {(k if isinstance(k, SubtaskId) else SubtaskId.parse(k)): v for k, v in blocks.items()}
    head = (f"# SkillMix rule of {generator} {task} (tmg fit_coefficients). Per subtask k, with phi_j = (|dx|, |dy|, rotation,\n"
            f"# inhand_position, inhand_rotation) of source j against the current scene (in-hand terms 0 for k = 0):\n"
            f"#   P(task succeeds | reached k, weights w) = sigmoid(c0 + sum_j w_j s_j + d H(w)),  s_j = a[j] + g . phi_j\n"
            f"#   weights: softmax(s / d) over the sources compatible with the selected one when d > 0, the best one when d <= 0\n"
            f"# Units: {{{', '.join(f'{f!r}: {UNITS[f]!r}' for f in features)}}}.\n")
    if HELD_FEATURE in features:
        head += (f"# {HELD_FEATURE}: 1 when exactly one of the two hands (now, source j) holds the carried object; the in-hand terms\n"
                 f"# are 0 then, and when neither holds it. held_rule: the distances within which a hand counts as holding it.\n")
    keyed = any(k.eef is not None for k in blocks)
    if keyed:
        head += "# Subtask keys <eef>/<k>: subtask k of that end effector (the subtasks of every arm are numbered from 0).\n"
    data = {"generator": generator, "task": task, "rule": "softmax", "basis": basis, "num_sources": int(num_sources),
            "features": list(features), "units": {f: UNITS[f] for f in features},
            "subtasks": {k.key: dict(blocks[k]) for k in sorted(blocks, key=lambda k: k.order)}}
    if object_frame is not None:          # deformable reference object: the mixer checks the frame of the scene against it
        data["object_frame"] = dict(object_frame)
    if held_rule is not None:             # the mixer checks the hold rule of the scene against it
        data["held_rule"] = held_rule.as_dict()
    if donor_registration is not None:    # the mixer checks how the scene's donors were registered against it
        data["donor_registration"] = dict(donor_registration)
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
    layout = make_generator(plain).fit_layout()
    attempts = {kind: read_pools(paths) for kind, paths in pools.items()}
    for kind, read in attempts.items():
        if len(read) < int(plain["min_attempts"]):
            raise ValueError(f"{len(read)} attempts in the {kind} pools {pools[kind]}, fewer than "
                             f"min_attempts={plain['min_attempts']}")
    num_sources = int(task["num_sources"])
    name = f"{generator['name']}/{task['name']}"
    print(f"[tmg] {name}: fit layout arms {list(layout.arms.names) or 'one arm'}, reached rule {layout.reached}, source poses "
          f"from {layout.source_poses}, features {list(layout.features)}", flush=True)
    prepared, sanity, problems = {}, {}, []
    for kind, read in attempts.items():
        attempts[kind], prepared[kind] = prepare(read, layout)
        about = prepared[kind]
        if about["without_scene"]:
            print(f"[tmg] WARNING {kind} pools: {about['without_scene']} of {about['attempts']} attempts left out, their "
                  f"subtask scene could not be read (e.g. {about['scene_error_examples'][0] if about['scene_error_examples'] else 'marked invalid'})",
                  flush=True)
        if about["unversioned_upgraded"]:
            print(f"[tmg] WARNING {kind} pools: {about['unversioned_upgraded']} attempt records carry no record_version "
                  f"(written before 2026-11-27): read as version 1 with the hold rule declared for unversioned records, "
                  f"{layout.unversioned_held_rule}, checked against their recorded hold distances"
                  + (f", and with the donor registration declared for them, {dict(layout.unversioned_donor_registration)} (not "
                     f"recorded in version 1: the constants of the code that made them)" if layout.unversioned_donor_registration else ""),
                  flush=True)
    holds = complete_source_holds(attempts["randmix"].records + attempts["unmix"].records)
    if holds["filled"] or holds["unknown"]:
        print(f"[tmg] WARNING {holds['filled'] + holds['unknown']} mix records do not say which sources hold the carried object "
              f"(unversioned records where the hand held nothing): {holds['filled']} filled from the hold tests these pools "
              f"recorded per (end effector/subtask/source: [held, tested]) {holds['evidence']}, {holds['unknown']} left "
              f"unknown (their rows are left out)", flush=True)
    for kind in attempts:
        sanity[kind] = sanity_report(attempts[kind], layout, task)
        # the scene errors are counted over everything read, before the unreadable attempts were left out
        sanity[kind].update(attempts=prepared[kind]["attempts"], without_scene=prepared[kind]["without_scene"])
        errors, warnings = judge_sanity(sanity[kind], layout.sanity)
        print_sanity(sanity[kind], f"{kind} pools", errors, warnings)
        sanity[kind].update(errors=errors, warnings=warnings)
        problems += [f"{kind} pools: {text}" for text in errors]
    if problems:
        raise ValueError("the records fail the sanity check of the fit: " + "; ".join(problems))
    measured = measured_with(attempts["randmix"].records + attempts["unmix"].records)
    blocks, about = fit_blocks(attempts["randmix"], attempts["unmix"], layout=layout, task=task, num_sources=num_sources,
                               source=plain.get("source"))
    for key, b in blocks.items():
        print_block(key, b, about[key])
    basis = (f"Fit (tmg fit_coefficients) on the randmix pools {pools['randmix']} ({len(attempts['randmix'])} "
             f"attempts) and the unmix pools {pools['unmix']} ({len(attempts['unmix'])} attempts); rows of "
             f"subtask k: the attempts that reached it, label = the attempt's task success.")
    kwargs = dict(generator=generator["name"], task=task["name"], basis=basis, num_sources=num_sources, blocks=blocks,
                  features=layout.features, object_frame=measured.object_frame, held_rule=measured.held_rule,
                  donor_registration=measured.donor_registration)
    written = [str(write_rule(output / COEFFICIENT_FILE, **kwargs))]
    if plain.get("write_config", True):
        written.append(str(write_rule(rule_path(name), **kwargs)))
    summary = {"randmix_pools": pools["randmix"], "unmix_pools": pools["unmix"],
               "randmix_attempts": len(attempts["randmix"]), "unmix_attempts": len(attempts["unmix"]),
               "subtasks": {key.key: {field: b[field] for field in ("d", "penalty", "rows", "successes", "cv_log_loss_gain_percent")}
                            for key, b in blocks.items()},
               "layout": {"arms": list(layout.arms.names), "reached": layout.reached, "source_poses": layout.source_poses,
                          "features": list(layout.features)},
               "reached_from": {str(key.key): a["reached"] for key, a in about.items()},
               "unknown_rows": {str(key.key): a["unknown_rows"] for key, a in about.items()},
               "constant_features": {str(key.key): a["constant_features"] for key, a in about.items()},
               "records": prepared, "source_holds": holds, "sanity": sanity,
               "written": written}
    write_json(output / SUMMARY, summary)
    print(f"[tmg] written: {', '.join(written)}")
    return summary
