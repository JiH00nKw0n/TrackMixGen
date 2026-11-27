"""The attempt records as ``tmg fit_coefficients`` (and a chunk's records check) reads them: the pools, the record versions,
which subtasks an attempt reached, and the sanity report.

Record versions (``record_version`` of an attempt line, ``tmg.core.records.RECORD_VERSION``). A line without the field is
version 1: written before 2026-11-27 (commit 2fcfc0d and earlier). Version 2 adds, for a runtime with a hold test, the
hold rule of every subtask record (``held_rule``), whether the tested object is deformable (``held_distance.deformable``)
and, in the mix record, who holds the carried object (``held``: now, every source) and, for a deformable subtask, the
settings its donors were registered with (``registration``). (This is version 3; version 2 was an interim format with
which no pool was made, and is refused.) ``prepare`` brings version-1 records of such a generator to that shape with the
hold rule and the donor registration the fit layout declares for them (``FitLayout.unversioned_held_rule``,
``unversioned_donor_registration``: the values the code of 2fcfc0d had as constants), says so loudly, and checks the rule
against what the records do carry: the hold
test of a subtask record is rebuilt from its recorded distances (``success_model.held_test_of_record``), exactly as for a
version-2 record, and must agree with the in-hand poses the record kept. A version-1 mix record has no per-source hold
flags. Where the hand held the object, a source whose two recorded in-hand features are exactly 0 next to in-hand
features of other sources did not hold it (the version-1 mixer wrote 0, 0 for exactly that case). Where the hand held
nothing, the version-1 mixer recorded nothing about the sources: ``complete_source_holds`` fills their flags from the
hold tests the same pools did record for that (end effector, subtask, source), when every such test agrees (whether a
source holds the object at its segment start is a property of the source); a mix record that stays unknown makes its
row unusable for a rule with the hold-mismatch feature, and the fit leaves it out and counts it.

Reached rules (``REACHED``, named by ``FitLayout.reached``; ``register_reached_rule`` adds one, so a generator whose
records need another rule registers it in its own adapter module). An attempt reaches subtask k when every earlier
subtask of the same end effector succeeded; the last subtask succeeds when the attempt does:

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

Sanity report. ``sanity_report`` counts, per block: successful attempts that count as not having reached the block, how
often the hand holds the carried object now and in each source, and how often a mix record left the segment unmixed
(with the reasons); and over all attempts, how many had a subtask scene that could not be read. The reports are counts,
so they add up (``merge_sanity``): the fit judges the whole pool, a chunk result the pool's finished chunks so far.
``judge_sanity`` turns counts into errors and warnings:

- a rate is an error only when the counts show with the configured confidence that the true rate is above its limit
  (``fit_layout.rate_exceeds``). One chunk's share is never compared with the limit: with a true rate of 5 % and a limit
  of 10 %, a pool of 40 chunks of 50 attempts would see some chunk above the limit with probability 0.79, and stop for
  nothing. A failure of every attempt still shows in the first chunk (50 of 50 is far beyond any limit);
- records the reached rule cannot read at all are an error at once (nothing statistical about it);
- the share of successful attempts that count as not reached is an error for the fit and for a calibration pool, a
  warning for a pool that is not read by the fit (``unreached_is_error``);
- a carried object that no hand ever holds from the second subtask on is a loud warning (it can be right: a hand that
  pushes a door holds nothing).

Measured on the Rope calibration pools of 2fcfc0d (2026-11-27, 490 unmix and 451 randmix attempts read): 0 of 526
successful attempts not reached, 0 of 941 attempts with a scene error, the rope held in 926 of 941 second subtasks (the
other 15: dropped, 0.35 to 0.40 m from the hand) and in every source, 14 of 902 mix records left unmixed (the weights
kept the selected source alone). The default limits of the Isaac Lab generators (0.05 for both rates, confidence 0.999)
are several times those measured rates' upper bounds (1.3 % and 0.8 % at the same confidence).
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np

from tmg.core.chunks import CHUNKS_DIR, chunk_is_done
from tmg.core.layout import ATTEMPTS_JSONL
from tmg.core.records import RECORD_VERSION, RECORDS_DIR, read_attempts
from tmg.generation.method.fit_layout import FitLayout, SanityLimits
from tmg.generation.method.interfaces import HELD_DISTANCE_DECIMALS, HeldRule
from tmg.generation.method.success_model import eef_subtasks, held_test_of_record

UNVERSIONED = 1          # the version of an attempt line without ``record_version``
READABLE_VERSIONS = (UNVERSIONED, RECORD_VERSION)


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


def _upgrade_unversioned(attempt: Mapping, rule: HeldRule, donor_registration: Mapping | None = None) -> None:
    """Give a version-1 attempt record the fields of the current version (module docstring), in place: the hold rule and
    test, and in the mix record of a deformable subtask the settings its donors were registered with."""
    step = 10.0 ** -HELD_DISTANCE_DECIMALS        # version 1 tested the unrounded distance and recorded it rounded
    for s in attempt.get("subtasks") or []:
        if s.get("scene_error"):
            continue
        s["held_rule"] = rule.as_dict()
        if donor_registration is not None and s.get("object_frame") is not None and (s.get("mix") or {}).get("weights"):
            s["mix"].setdefault("registration", dict(donor_registration))
        candidate = s.get("carried_candidate")
        distance = s.get("held_distance")
        if candidate is None or distance is None:
            continue
        distance["deformable"] = bool(s.get("object_frame") is not None and candidate == s.get("object_name"))
        test = held_test_of_record(s)                      # the verdicts every reader derives from the record
        limit = rule.limit(test.deformable)
        kept_now = candidate in (s.get("objects") or {})
        kept_source = candidate in (s.get("src_objects") or {})
        where = f"attempt {attempt.get('attempt')} {s.get('eef')}/{s.get('subtask_ind')}"
        # version 1 kept the in-hand pose now where the hand held the object, and the source's only next to it
        for kept, held, metres in ((kept_now, test.now, test.now_m), *([(kept_source, test.sources[0], test.sources_m[0])] if kept_now else [])):
            if kept != held and abs(metres - limit) > step:
                raise ValueError(f"{where}: an unversioned record whose hold distances {distance} do not give its recorded "
                                 f"in-hand poses (kept now: {kept_now}, of the source: {kept_source}) under the hold rule "
                                 f"declared for unversioned records {rule}: the records were made with another rule")
        m = s.get("mix")
        if m and "weights" in m and "held" not in m:
            rows = m.get("inhand_features") if test.now else None
            m["held"] = {"now": test.now, "sources": [any(float(v) != 0.0 for v in row) for row in rows] if rows else None}


def prepare(attempts: Attempts, layout: FitLayout) -> tuple[Attempts, dict]:
    """The attempts the fit may use, and what was done to get them: attempts whose scene could not be read are left out
    (counted), the end effectors are checked against the layout, and version-1 records get the fields of the current
    version when the layout has a hold rule for them (module docstring). Only versions 1 and the current one are read."""
    versions = Counter(record_version(a) for a in attempts.records)
    unreadable = sorted(v for v in versions if v not in READABLE_VERSIONS)
    if unreadable:
        raise ValueError(f"attempt records of version {unreadable}: this code reads version {UNVERSIONED} (lines without "
                         f"record_version) and version {RECORD_VERSION}; any other format, the interim version 2 included, "
                         f"has no reader")
    usable = [i for i, a in enumerate(attempts.records) if not has_scene_error(a)]
    kept = attempts.keep(usable)
    check_arms(kept.records, layout)
    upgraded = 0
    if layout.unversioned_held_rule is not None:
        for a in kept.records:
            if record_version(a) == UNVERSIONED:
                _upgrade_unversioned(a, layout.unversioned_held_rule, layout.unversioned_donor_registration)
                upgraded += 1
    return kept, {"attempts": len(attempts), "without_scene": len(attempts) - len(usable),
                  "versions": {str(v): n for v, n in sorted(versions.items())}, "unversioned_upgraded": upgraded,
                  "scene_error_examples": scene_errors(attempts.records)[:3]}


def complete_source_holds(records: Sequence[Mapping]) -> dict:
    """Fill the per-source hold flags of mix records that lack them (unversioned records where the hand held nothing,
    module docstring) from the hold tests ``records`` (every pool of the fit) did record for the same (end effector,
    subtask, source): the selected source of every subtask record and every mix record that has its flags. A source with
    no such test, or with tests that disagree, stays unknown, and so does its mix record. Returns
    ``{"filled": n, "unknown": n, "evidence": {"<eef>/<k>/<j>": [held, tested]}}``."""
    seen: dict[tuple, list[int]] = {}

    def note(eef, k, j, held):
        counts = seen.setdefault((eef, int(k), int(j)), [0, 0])
        counts[0] += bool(held)
        counts[1] += 1

    waiting = []
    for a in records:
        for s in a.get("subtasks") or []:
            test = None if s.get("scene_error") else held_test_of_record(s)
            if test is None:
                continue
            note(s.get("eef"), s["subtask_ind"], s["src_demo_ind"], test.sources[0])
            held = (s.get("mix") or {}).get("held")
            if held and held.get("sources") is not None:
                for j, flag in enumerate(held["sources"]):
                    note(s.get("eef"), s["subtask_ind"], j, flag)
            elif held:
                waiting.append((s.get("eef"), int(s["subtask_ind"]), s["mix"]))
    filled = unknown = 0
    for eef, k, mix in waiting:
        counts = [seen.get((eef, k, j)) for j in range(len(mix["weights"]))]
        if all(c is not None and c[0] in (0, c[1]) for c in counts):          # every source tested, every test agrees
            mix["held"]["sources"] = [c[0] == c[1] for c in counts]
            filled += 1
        else:
            unknown += 1
    return {"filled": filled, "unknown": unknown,
            "evidence": {f"{eef}/{k}/{j}": counts for (eef, k, j), counts in sorted(seen.items(), key=str)}}


class Measured(NamedTuple):
    """What the records' scenes were measured with, and their donors prepared with (None where they have none); the rule
    file records all three and the mixer refuses a scene that differs."""
    object_frame: dict | None
    held_rule: HeldRule | None
    donor_registration: dict | None


def measured_with(records: Sequence[Mapping]) -> Measured:
    """The ``Measured`` of ``records``; records of different frames, hold rules or donor registrations are refused, since
    their features and mixed segments are not comparable."""
    frames, rules, registrations, with_rule, subtasks = set(), set(), [], 0, 0
    for a in records:
        for s in a.get("subtasks") or []:
            subtasks += 1
            if s.get("object_frame"):
                frames.add((s["object_frame"]["version"], s["object_frame"]["reference_sha256"]))
            if s.get("held_rule"):
                rules.add(HeldRule.from_mapping(s["held_rule"]))
                with_rule += 1
            registration = (s.get("mix") or {}).get("registration")
            if registration is not None and registration not in registrations:
                registrations.append(dict(registration))
    if len(frames) > 1:
        raise ValueError(f"the records use different deformable-object frames {sorted(frames)}")
    if len(rules) > 1 or (rules and with_rule != subtasks):
        raise ValueError(f"the records use different hold rules ({sorted(rules, key=str)}; {subtasks - with_rule} of "
                         f"{subtasks} subtask records carry none)")
    if len(registrations) > 1:
        raise ValueError(f"the records' donors were registered with different settings {registrations}")
    frame = None
    if frames:
        version, sha = next(iter(frames))
        frame = {"version": version, "reference_sha256": sha}
    return Measured(frame, next(iter(rules)) if rules else None, registrations[0] if registrations else None)


# which subtasks an attempt reached ----------------------------------------------------------------------------------
SIGNAL, ASSUMED = "signal", "assumed"      # where an earlier subtask's outcome came from


class ReachedRule:
    """How the records say which subtasks an attempt reached (module docstring)."""
    name = ""
    mix_record_proves_reach = False        # a mix record of subtask k exists only in attempts that reached k

    def num_subtasks(self, records: Sequence[Mapping], eef: str | None, task: Mapping) -> int:
        """Subtasks of end effector ``eef``: one more than the largest subtask index a successful attempt recorded (of any
        attempt when none succeeded)."""
        successful = [a for a in records if a["success"]] or list(records)
        return max((int(s["subtask_ind"]) + 1 for a in successful for s in eef_subtasks(a, eef)), default=0)

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

    def num_subtasks(self, records, eef, task) -> int:
        """The most subtask records a successful attempt has (the count ``stopped_outcomes`` compares with)."""
        successful = [a for a in records if a["success"]] or list(records)
        return max((len(eef_subtasks(a, eef)) for a in successful), default=0)

    def outcomes(self, attempts, eef, num_subtasks, task):
        out, used = [stopped_outcomes(a, num_subtasks, eef) for a in attempts.records], _used(num_subtasks)
        for k in used:
            used[k][SIGNAL] = len(out)
        return out, used


class TermWindow(ReachedRule):
    name = "term_window"

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

    def outcomes(self, attempts, eef, num_subtasks, task):
        out, used = [], _used(num_subtasks)
        for a in attempts.records:
            recorded = {int(s["subtask_ind"]) for s in eef_subtasks(a, eef)}
            out.append([k in recorded if k < num_subtasks - 1 else bool(a["success"]) for k in range(num_subtasks)])
        for k in used:
            used[k][ASSUMED] = len(out)
        return out, used


REACHED: dict[str, ReachedRule] = {}


def register_reached_rule(rule: ReachedRule) -> ReachedRule:
    """Make a reached rule available to fit layouts under its ``name``."""
    if not rule.name or type(REACHED.get(rule.name, rule)) is not type(rule):
        raise ValueError(f"reached rule {rule.name!r}: no name, or the name is already registered")
    REACHED[rule.name] = rule
    return rule


for _rule in (StepSignals(), TermWindow(), StopsAtFailure(), TaskSuccess()):
    register_reached_rule(_rule)


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
def block_sanity(records: Sequence[Mapping], outcomes: Sequence[Sequence[bool]] | None, eef: str | None, k: int) -> dict:
    """The sanity counts of block (eef, k) over ``records`` (module docstring). ``outcomes``: the reached rule's verdicts
    (None when a mix record itself proves the reach)."""
    out = {"attempts": 0, "successes": 0, "successes_unreached": 0}
    held = {"tested": 0, "now": 0, "sources": {}}
    mix = {"records": 0, "identity": 0, "identity_reasons": {}}
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
            mix["records"] += 1
            if not m.get("mixed"):
                mix["identity"] += 1
                reason = str(m.get("identity_reason") or "the weights kept the selected source alone")
                mix["identity_reasons"][reason] = mix["identity_reasons"].get(reason, 0) + 1
        test = held_test_of_record(s)
        if test is None:
            continue
        held["tested"] += 1
        held["now"] += test.now
        per_source = (m or {}).get("held")
        flags = (dict(enumerate(per_source["sources"])) if per_source and per_source.get("sources") is not None
                 else {int(s["src_demo_ind"]): test.sources[0]})
        for j, flag in flags.items():
            counts = held["sources"].setdefault(str(j), [0, 0])            # [held, tested]
            counts[0] += bool(flag)
            counts[1] += 1
    if held["tested"]:
        out["held"] = held
    if mix["records"]:
        out["mix"] = mix
    return out


def sanity_report(attempts: Attempts, layout: FitLayout, task: Mapping) -> dict:
    """The sanity counts of the attempts of one pool kind or one chunk (module docstring): ``{"reached", "attempts",
    "without_scene", "blocks": {key: block_sanity}, "unreadable": [...]}``. The blocks count the attempts whose scenes
    were read, as the fit uses them; ``unreadable`` says why the reached rule could not read an end effector's records."""
    usable = attempts.keep([i for i, a in enumerate(attempts.records) if not has_scene_error(a)])
    rule = reached_rule(layout.reached)
    report = {"reached": rule.name, "attempts": len(attempts), "without_scene": len(attempts) - len(usable), "blocks": {},
              "unreadable": []}
    for eef in layout.arms.record_eefs:
        try:
            K = rule.num_subtasks(usable.records, eef, task)
            outcomes, _used_by = rule.outcomes(usable, eef, K, task)
        except (ValueError, KeyError) as error:
            report["unreadable"].append(f"{eef or 'arm'}: the reached rule {rule.name} cannot read the records: {error}")
            continue
        for k in range(K):
            report["blocks"][str(layout.arms.subtask(k, eef).key)] = block_sanity(usable.records, outcomes, eef, k)
    return report


def _add(total, part):
    """``part``'s counts added into ``total`` (nested dicts and count lists; a first-seen key is copied)."""
    for key, value in part.items():
        if isinstance(value, Mapping):
            _add(total.setdefault(key, {}), value)
        elif isinstance(value, list):
            total[key] = [a + b for a, b in zip(total.get(key) or [0] * len(value), value)]
        else:
            total[key] = total.get(key, 0) + value
    return total


def merge_sanity(reports: Sequence[Mapping]) -> dict:
    """The sum of sanity reports of the same layout (the chunks of a pool)."""
    reports = [r for r in reports if r]
    if not reports:
        return {"reached": None, "attempts": 0, "without_scene": 0, "blocks": {}, "unreadable": []}
    out = {"reached": reports[0]["reached"], "attempts": 0, "without_scene": 0, "blocks": {}, "unreadable": []}
    for r in reports:
        out["attempts"] += int(r["attempts"])
        out["without_scene"] += int(r["without_scene"])
        out["unreadable"] += [text for text in r["unreadable"] if text not in out["unreadable"]]
        _add(out["blocks"], r["blocks"])
    return out


def _percent(count: int, total: int) -> str:
    return f"{count} of {total} ({100 * count / total:.1f} %)" if total else f"{count} of 0"


def judge_sanity(report: Mapping, limits: SanityLimits, *, unreached_is_error: bool = True) -> tuple[list[str], list[str]]:
    """``(errors, warnings)`` of a sanity report under ``limits`` (module docstring)."""
    errors, warnings = list(report["unreadable"]), []
    if limits.scene_errors_exceeded(report["without_scene"], report["attempts"]):
        errors.append(f"{_percent(report['without_scene'], report['attempts'])} attempts without the scene of a subtask: with "
                      f"confidence {limits.confidence} more than record_checks.max_scene_error_rate = {limits.max_scene_error_rate}")
    for key, block in report["blocks"].items():
        k = subtask_of_key(key)
        if limits.unreached_exceeded(block["successes_unreached"], block["successes"]):
            text = (f"block {key}: {_percent(block['successes_unreached'], block['successes'])} successful attempts count as not "
                    f"having reached it: with confidence {limits.confidence} more than record_checks.max_unreached_success_rate = "
                    f"{limits.max_unreached_success_rate} (reached rule {report['reached']}): the rule misreads these records")
            (errors if unreached_is_error else warnings).append(text)
        held = block.get("held")
        if k >= 1 and held and held["now"] == 0:
            warnings.append(f"block {key}: the hand holds the carried object in none of {held['tested']} attempts, so its "
                            f"in-hand features are never used")
        if k >= 1 and held and held["sources"] and all(c[0] == 0 for c in held["sources"].values()):
            warnings.append(f"block {key}: no source holds the carried object ({held['sources']})")
    return errors, warnings


def subtask_of_key(key) -> int:
    """The subtask index of a block key (``k`` or ``<eef>/<k>``)."""
    return int(str(key).rpartition("/")[2])


def print_sanity(report: Mapping, label: str, errors: Sequence[str] = (), warnings: Sequence[str] = ()) -> None:
    """The sanity report on stdout, warnings and errors loud."""
    print(f"[tmg] sanity {label}: {_percent(report['without_scene'], report['attempts'])} attempts without the scene of a subtask",
          flush=True)
    for key, b in report["blocks"].items():
        parts = [f"{b['attempts']} attempts", f"{_percent(b['successes_unreached'], b['successes'])} successes not reached"]
        if "held" in b:
            parts.append(f"held now {_percent(b['held']['now'], b['held']['tested'])}, per source [held, tested] {b['held']['sources']}")
        if "mix" in b:
            parts.append(f"unmixed mix records {_percent(b['mix']['identity'], b['mix']['records'])} {b['mix']['identity_reasons']}")
        print(f"[tmg] sanity {label} block {key}: {'; '.join(parts)}", flush=True)
    for text in warnings:
        print(f"[tmg] WARNING sanity {label}: {text}", flush=True)
    for text in errors:
        print(f"[tmg] ERROR sanity {label}: {text}", flush=True)
