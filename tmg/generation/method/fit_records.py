"""The attempt records as ``tmg fit_coefficients`` (and a chunk's sanity check) reads them: the pools, the record versions,
which subtasks an attempt reached, and the sanity report.

Record versions (``record_version`` of an attempt line, ``tmg.core.records.RECORD_VERSION``). A line without the field is
version 1: written before 2026-11-27 (commit 2fcfc0d and earlier). Version 2 adds, for a runtime with a hold test, the
hold rule of every subtask record (``held_rule``), the verdicts (``held``: now, selected source) and, in the mix record,
who holds the carried object (``held``: now, every source). ``prepare`` brings version-1 records of such a generator to
that shape with the hold rule the fit layout declares for them (``FitLayout.unversioned_held_rule``), says so loudly, and
checks the rule against the distances the records do carry (``held_distance``). A version-1 mix record has no per-source
hold flags: a source whose two recorded in-hand features are exactly 0 next to in-hand features of other sources did not
hold the object (the version-1 mixer wrote 0, 0 for exactly that case), and when the hand held nothing no source is taken
to hold it either.

Reached rules (``REACHED``, named by ``FitLayout.reached``). An attempt reaches subtask k when every earlier subtask of
the same end effector succeeded; the last subtask succeeds when the attempt does:

- ``step_signals``: subtask k succeeded when its termination signal (``task.subtask_term_signals``, recorded per step as
  ``term__<signal>`` in the attempt npz) turned on at some step up to the end of that subtask's execution (MimicGen);
- ``term_window``: subtask k succeeded when its recorded termination signal (``term_signal`` of its record) first fired
  (``term_first``) before the step at which the same end effector's subtask k + 1 started (``start_step`` of that record;
  the attempt's last step when k + 1 was never built). A subtask without a usable signal raises (the Isaac Lab
  generators whose environments publish per-step signals);
- ``stops_at_failure``: subtask k succeeded when the attempt went on to subtask k + 1 (CP-Gen stops an attempt at its
  first failed constraint). Here a mix record of subtask k itself shows that the attempt reached k, also when the
  motion plan to k then failed and k has no subtask record (``mix_record_proves_reach``);
- ``task_success``: every recorded subtask counts as reached, so the task-success label alone is used. For environments
  that publish no per-step subtask signal; declared per task (``task.fit.reached``).

Sanity report (``sanity_report``), per block, printed by the fit and written into every Isaac Lab chunk result so that a
problem shows at the first chunk: successful attempts that count as not having reached the block (an error above
``SanityLimits.max_unreached_success_rate``), the share of attempts in which the hand holds the carried object now and in
each source (a loud warning when nothing is ever held from the second subtask on), and the share of mix records that
left the segment unmixed, with the reasons.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tmg.core.chunks import CHUNKS_DIR, chunk_is_done
from tmg.core.layout import ATTEMPTS_JSONL
from tmg.core.records import RECORD_VERSION, RECORDS_DIR, read_attempts
from tmg.generation.method.fit_layout import FitLayout
from tmg.generation.method.interfaces import HeldRule
from tmg.generation.method.success_model import eef_subtasks

UNVERSIONED = 1          # the version of an attempt line without ``record_version``


# pools ----------------------------------------------------------------------------------------------------------------
@dataclass
class Attempts:
    """The attempt records of the pools of one kind, with the records folder and chunk index (0, 1, ... over the chunks
    read) of every attempt."""
    records: list[dict]
    folders: list[Path]
    chunks: list[int]

    def __len__(self) -> int:
        return len(self.records)

    def keep(self, indices: Sequence[int]) -> Attempts:
        return Attempts([self.records[i] for i in indices], [self.folders[i] for i in indices], [self.chunks[i] for i in indices])


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


def mix_records(attempt: Mapping, eef: str | None = None) -> dict[int, dict]:
    """The first mix record with weights of every subtask of an attempt (planned segments, then solves whose motion plan
    failed); with ``eef``, of that end effector's subtasks only."""
    out = {}
    for m in [s.get("mix") for s in attempt["subtasks"]] + list(attempt.get("unplanned_mixes") or []):
        if m and "weights" in m and (eef is None or m.get("eef") == eef) and int(m["subtask"]) not in out:
            out[int(m["subtask"])] = m
    return out


# record versions and checks ---------------------------------------------------------------------------------------
def record_version(attempt: Mapping) -> int:
    return int(attempt.get("record_version", UNVERSIONED))


def scene_errors(records: Sequence[Mapping]) -> list[str]:
    """``attempt eef/k: error`` of every subtask record whose scene could not be read (``scene_error``)."""
    return [f"attempt {r.get('attempt')} {s.get('eef')}/{s.get('subtask_ind')}: {s['scene_error']}"
            for r in records for s in r.get("subtasks") or [] if s.get("scene_error")]


def has_scene_error(attempt: Mapping) -> bool:
    return bool(attempt.get("invalid")) or any(s.get("scene_error") for s in attempt.get("subtasks") or [])


def check_arms(records: Sequence[Mapping], layout: FitLayout) -> None:
    """The records must name the end effectors of the layout (the generator config): the fit keys its blocks by the
    layout's arms, as the mixer does, and never guesses them from the records."""
    names = layout.arms.names
    if not names:
        return
    seen = {s.get("eef") for a in records for s in a.get("subtasks") or []}
    if not seen:
        return
    unknown = sorted(str(e) for e in seen - set(names))
    missing = sorted(set(names) - seen)
    if unknown or missing:
        raise ValueError(f"the records' end effectors {sorted(str(e) for e in seen)} disagree with the generator's arms "
                         f"{list(names)}" + (f": {unknown} are not arms of the generator" if unknown else
                                             f": no subtask of {missing} was recorded"))


UNVERSIONED_DISTANCE_STEP = 1e-4      # version-1 records tested the unrounded distance and recorded it rounded to 0.1 mm


def _disagrees(rule: HeldRule, distance: float, deformable: bool, held: bool) -> bool:
    """True when a recorded (rounded) hold distance contradicts the recorded verdict beyond the rounding step."""
    limit = rule.limit(deformable)
    return (float(distance) <= limit) != held and abs(float(distance) - limit) > UNVERSIONED_DISTANCE_STEP


def _upgrade_unversioned(attempt: Mapping, rule: HeldRule) -> None:
    """Give a version-1 attempt record the hold fields of version 2 (module docstring), in place."""
    for s in attempt.get("subtasks") or []:
        if s.get("scene_error"):
            continue
        s["held_rule"] = rule.as_dict()
        candidate = s.get("carried_candidate")
        distance = s.get("held_distance")
        if candidate is None or distance is None:
            continue
        deformable = s.get("object_frame") is not None and candidate == s.get("object_name")
        now = candidate in (s.get("objects") or {})
        source = candidate in (s.get("src_objects") or {})
        where = f"attempt {attempt.get('attempt')} {s.get('eef')}/{s.get('subtask_ind')}"
        if _disagrees(rule, distance["now"], deformable, now) or (now and _disagrees(rule, distance["source"], deformable, source)):
            raise ValueError(f"{where}: an unversioned record whose hold distances {distance} do not give its recorded "
                             f"in-hand poses (held now: {now}, in the source: {source}) under the hold rule declared for "
                             f"unversioned records {rule}: the records were made with another rule")
        s["held"] = {"now": now, "source": bool(now and source), "deformable": bool(deformable)}
        m = s.get("mix")
        if m and "weights" in m and "held" not in m:
            rows = m.get("inhand_features") if now else None
            m["held"] = {"now": bool(rows), "sources": [bool(rows) and any(float(v) != 0.0 for v in row) for row in rows]
                         if rows else [False] * len(m["weights"])}


def prepare(attempts: Attempts, layout: FitLayout) -> tuple[Attempts, dict]:
    """The attempts the fit may use, and what was done to get them: attempts whose scene could not be read are left out
    (counted), the end effectors are checked against the layout, and version-1 records get the hold fields of version 2
    when the layout has a hold rule for them (module docstring)."""
    versions = Counter(record_version(a) for a in attempts.records)
    newer = sorted(v for v in versions if v > RECORD_VERSION)
    if newer:
        raise ValueError(f"attempt records of version {newer}, this code reads up to {RECORD_VERSION}")
    usable = [i for i, a in enumerate(attempts.records) if not has_scene_error(a)]
    kept = attempts.keep(usable)
    check_arms(kept.records, layout)
    upgraded = 0
    if layout.unversioned_held_rule is not None:
        for a in kept.records:
            if record_version(a) == UNVERSIONED:
                _upgrade_unversioned(a, layout.unversioned_held_rule)
                upgraded += 1
    return kept, {"attempts": len(attempts), "without_scene": len(attempts) - len(usable),
                  "versions": {str(v): n for v, n in sorted(versions.items())}, "unversioned_upgraded": upgraded,
                  "scene_error_examples": scene_errors(attempts.records)[:3]}


def measured_with(records: Sequence[Mapping]) -> tuple[dict | None, HeldRule | None]:
    """``(object frame, hold rule)`` the records' scenes were measured with (None where they have none); records of
    different frames or rules are refused, since their features are not comparable."""
    frames, rules, with_rule, subtasks = set(), set(), 0, 0
    for a in records:
        for s in a.get("subtasks") or []:
            subtasks += 1
            if s.get("object_frame"):
                frames.add((s["object_frame"]["version"], s["object_frame"]["reference_sha256"]))
            if s.get("held_rule"):
                rules.add(HeldRule.from_mapping(s["held_rule"]))
                with_rule += 1
    if len(frames) > 1:
        raise ValueError(f"the records use different deformable-object frames {sorted(frames)}")
    if len(rules) > 1 or (rules and with_rule != subtasks):
        raise ValueError(f"the records use different hold rules ({sorted(rules, key=str)}; {subtasks - with_rule} of "
                         f"{subtasks} subtask records carry none)")
    frame = None
    if frames:
        version, sha = next(iter(frames))
        frame = {"version": version, "reference_sha256": sha}
    return frame, (next(iter(rules)) if rules else None)


# which subtasks an attempt reached ----------------------------------------------------------------------------------
SIGNAL, ASSUMED = "signal", "assumed"      # where an earlier subtask's outcome came from


class ReachedRule:
    """How the records say which subtasks an attempt reached (module docstring)."""
    name = ""
    mix_record_proves_reach = False        # a mix record of subtask k exists only in attempts that reached k

    def num_subtasks(self, records: Sequence[Mapping], eef: str | None, task: Mapping) -> int:
        """Subtasks of end effector ``eef``: the most a successful attempt recorded (of any attempt when none succeeded)."""
        successful = [a for a in records if a["success"]] or list(records)
        return max((len(eef_subtasks(a, eef)) for a in successful), default=0)

    def outcomes(self, attempts: Attempts, eef: str | None, num_subtasks: int, task: Mapping) -> tuple[list[list[bool]], dict]:
        """Success of every subtask of ``eef`` in every attempt, and per earlier subtask how many attempts' outcome came
        from a signal or was assumed: ``{k: {"signal": n, "assumed": n}}``."""
        raise NotImplementedError


def _used(num_subtasks: int) -> dict:
    return {k: {SIGNAL: 0, ASSUMED: 0} for k in range(max(num_subtasks - 1, 0))}


def step_signal_outcomes(attempt: Mapping, steps: Mapping, signals: Sequence) -> list[bool]:
    """Success of every subtask of one attempt under ``step_signals``. ``steps`` holds the attempt's per-step arrays
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


class StepSignals(ReachedRule):
    name = "step_signals"

    @staticmethod
    def signals(task: Mapping) -> list:
        signals = task.get("subtask_term_signals")
        if not signals:
            raise ValueError(f"reached rule step_signals needs task.subtask_term_signals (task {task.get('name')!r} has none)")
        return list(signals)

    def num_subtasks(self, records, eef, task) -> int:
        return len(self.signals(task))

    def outcomes(self, attempts, eef, num_subtasks, task):
        signals, out, used = self.signals(task), [], _used(num_subtasks)
        for a, folder in zip(attempts.records, attempts.folders):
            with np.load(Path(folder) / a["npz"]) as z:
                steps = {k: z[k] for k in z.files if k in ("seg_sub", "seg_of_step") or k.startswith("term__")}
            out.append(step_signal_outcomes(a, steps, signals))
        for k in used:
            used[k][SIGNAL] = len(out)
        return out, used


def stopped_outcomes(attempt: Mapping, num_subtasks: int, eef: str | None = None) -> list[bool]:
    """Subtask success of a generator that stops an attempt at its first failed subtask: subtask k succeeded when the
    attempt went on to subtask k + 1; the last subtask succeeded when the attempt succeeded."""
    n = len(attempt.get("subtasks") or []) if eef is None else len(eef_subtasks(attempt, eef))
    return [n > k + 1 if k < num_subtasks - 1 else bool(attempt["success"]) and n >= num_subtasks for k in range(num_subtasks)]


class StopsAtFailure(ReachedRule):
    name = "stops_at_failure"
    mix_record_proves_reach = True

    def outcomes(self, attempts, eef, num_subtasks, task):
        out, used = [stopped_outcomes(a, num_subtasks, eef) for a in attempts.records], _used(num_subtasks)
        for k in used:
            used[k][SIGNAL] = len(out)
        return out, used


class TermWindow(ReachedRule):
    name = "term_window"

    def num_subtasks(self, records, eef, task) -> int:
        successful = [a for a in records if a["success"]] or list(records)
        return max((int(s["subtask_ind"]) + 1 for a in successful for s in eef_subtasks(a, eef)), default=0)

    def outcomes(self, attempts, eef, num_subtasks, task):
        out, used = [], _used(num_subtasks)
        for a in attempts.records:
            subs = {int(s["subtask_ind"]): s for s in eef_subtasks(a, eef)}
            if any("start_step" not in s for s in subs.values()):
                raise ValueError(f"attempt {a.get('attempt')} {eef}: subtask records lack start_step: the pools were made "
                                 f"before the runtime recorded the subtask scene; regenerate them")
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
                if k not in subs:                           # never built: not reached, whatever the signals say
                    used[k][SIGNAL] += 1
                    row.append(False)
                    continue
                if signal is None or not (str(signal) in fired or n_steps == 0):
                    why = (f"its term signal {signal!r} is not among the attempt's per-step signals {sorted(fired)}"
                           if signal is not None else "it has no term signal")
                    raise ValueError(f"attempt {a.get('attempt')} {eef}/{k}: {why}. If the task's environment publishes no "
                                     f"per-step subtask signals, declare it in the task config (fit: {{reached: task_success}}): "
                                     f"every recorded subtask then counts as reached and the task-success label alone is used")
                used[k][SIGNAL] += 1
                end = int(subs[k + 1]["start_step"]) if k + 1 in subs else n_steps
                first = int(fired.get(str(signal), -1))
                row.append(0 <= first < end)
            out.append(row)
        return out, used


class TaskSuccess(ReachedRule):
    name = "task_success"

    num_subtasks = TermWindow.num_subtasks

    def outcomes(self, attempts, eef, num_subtasks, task):
        out, used = [], _used(num_subtasks)
        for a in attempts.records:
            recorded = {int(s["subtask_ind"]) for s in eef_subtasks(a, eef)}
            out.append([k in recorded if k < num_subtasks - 1 else bool(a["success"]) for k in range(num_subtasks)])
        for k in used:
            used[k][ASSUMED] = len(out)
        return out, used


REACHED: dict[str, ReachedRule] = {rule.name: rule for rule in (StepSignals(), TermWindow(), StopsAtFailure(), TaskSuccess())}


def reached_rule(name: str) -> ReachedRule:
    if name not in REACHED:
        raise ValueError(f"unknown reached rule {name!r} (fit layout); known: {sorted(REACHED)}")
    return REACHED[name]


def describe_outcomes(used: Mapping) -> str:
    """One line: where the outcome of each earlier subtask (which decides whether an attempt reached this one) came from."""
    parts = []
    for k, n in sorted(used.items()):
        kind = ("signal" if n[ASSUMED] == 0 else
                "ASSUMED: a recorded subtask counts as reached, the label is the task success" if n[SIGNAL] == 0 else
                "signal where recorded, else assumed")
        parts.append(f"earlier subtask {k}: {kind} ({n[SIGNAL]} attempts by signal, {n[ASSUMED]} assumed)")
    return "; ".join(parts) or "first subtask: every attempt reaches it"


# sanity ---------------------------------------------------------------------------------------------------------------
def _ratio(n: int, of: int) -> float | None:
    return round(n / of, 4) if of else None


def block_sanity(records: Sequence[Mapping], outcomes: Sequence[Sequence[bool]] | None, eef: str | None, k: int) -> dict:
    """The sanity numbers of block (eef, k) over ``records`` (module docstring). ``outcomes``: the reached rule's verdicts
    (None when a mix record itself proves the reach)."""
    out = {"attempts": 0, "successes": 0, "successes_unreached": 0}
    held_now, held_n = 0, 0
    source_held, source_n = Counter(), Counter()
    mixes, identity, reasons = 0, 0, Counter()
    for index, a in enumerate(records):
        subs = {int(s["subtask_ind"]): s for s in eef_subtasks(a, eef)}
        if a.get("success"):
            out["successes"] += 1
            if outcomes is not None and not all(outcomes[index][:k]):
                out["successes_unreached"] += 1
        s = subs.get(k)
        if s is None:
            continue
        out["attempts"] += 1
        m = s.get("mix")
        if m:
            mixes += 1
            if not m.get("mixed"):
                identity += 1
                reasons[str(m.get("identity_reason") or "the weights kept the selected source alone")] += 1
        held = s.get("held")
        if held is None:
            continue
        held_n += 1
        held_now += bool(held["now"])
        per_source = (m or {}).get("held")
        flags = dict(enumerate(per_source["sources"])) if per_source else {int(s["src_demo_ind"]): held["source"]}
        for j, flag in flags.items():
            source_n[j] += 1
            source_held[j] += bool(flag)
    out["successes_unreached_rate"] = _ratio(out["successes_unreached"], out["successes"])
    if held_n:
        out["held"] = {"tested": held_n, "now": _ratio(held_now, held_n),
                       "sources": {str(j): _ratio(source_held[j], source_n[j]) for j in sorted(source_n)}}
    if mixes:
        out["mix"] = {"records": mixes, "identity": identity, "identity_rate": _ratio(identity, mixes), "identity_reasons": dict(reasons)}
    return out


def sanity_report(attempts: Attempts, layout: FitLayout, task: Mapping) -> dict:
    """The sanity report of the attempts of one pool kind or one chunk: ``{"blocks": {key: block_sanity}, "errors": [...],
    "warnings": [...]}``. Attempts whose scene could not be read are left out, as in the fit."""
    attempts = attempts.keep([i for i, a in enumerate(attempts.records) if not has_scene_error(a)])
    rule, limits = reached_rule(layout.reached), layout.sanity
    report = {"reached": rule.name, "blocks": {}, "errors": [], "warnings": []}
    for eef in layout.arms.record_eefs:
        try:
            K = rule.num_subtasks(attempts.records, eef, task)
            outcomes, _used_by = rule.outcomes(attempts, eef, K, task)
        except (ValueError, KeyError) as error:
            report["errors"].append(f"{eef or 'arm'}: the reached rule {rule.name} cannot read the records: {error}")
            continue
        for k in range(K):
            key = layout.arms.subtask(k, eef).key
            block = block_sanity(attempts.records, outcomes, eef, k)
            report["blocks"][str(key)] = block
            rate = block["successes_unreached_rate"]
            if (limits.max_unreached_success_rate is not None and rate is not None and block["successes"] >= limits.min_successes
                    and rate > limits.max_unreached_success_rate):
                report["errors"].append(
                    f"block {key}: {block['successes_unreached']} of {block['successes']} successful attempts count as not "
                    f"having reached it ({100 * rate:.0f} % > {100 * limits.max_unreached_success_rate:.0f} %, reached rule "
                    f"{rule.name}): the rule misreads these records")
            held = block.get("held")
            if k >= 1 and held and held["now"] == 0:
                report["warnings"].append(f"block {key}: the hand holds the carried object in none of {held['tested']} "
                                          f"attempts, so its in-hand features are never used")
            if k >= 1 and held and held["sources"] and all(v == 0 for v in held["sources"].values()):
                report["warnings"].append(f"block {key}: no source holds the carried object ({held['sources']})")
    return report


def print_sanity(report: Mapping, label: str) -> None:
    """The sanity report on stdout, warnings and errors loud."""
    for key, b in report["blocks"].items():
        parts = [f"{b['attempts']} attempts", f"{b['successes_unreached']}/{b['successes']} successes not reached"]
        if "held" in b:
            parts.append(f"held now {b['held']['now']}, per source {b['held']['sources']}")
        if "mix" in b:
            parts.append(f"unmixed mix records {b['mix']['identity']}/{b['mix']['records']} {b['mix']['identity_reasons']}")
        print(f"[tmg] sanity {label} block {key}: {'; '.join(parts)}", flush=True)
    for text in report["warnings"]:
        print(f"[tmg] WARNING sanity {label}: {text}", flush=True)
    for text in report["errors"]:
        print(f"[tmg] ERROR sanity {label}: {text}", flush=True)
