#!/usr/bin/env python3
"""Summarize what the mixing of a pool did, per subtask, from its attempt records: the check of a method's condition
before its full pool is generated (docs/SKILLMIX_FACTOR_EXPERIMENT.md; scripts/run_factor_experiment.sh CHECK=1 runs it on a
50-attempt pool).

  scripts/mix_report.py --pool <data root>/pools/<generator>/<task>/<scene>/<selection>/<method>/<generation> [--json]

Per subtask, over the attempts that executed it (``subtasks[k].mix`` of ``records/attempts.jsonl`` in every chunk):
the fitted or derived ``d`` values, the share of blended subtasks (more than one source), the effective number of sources
``1 / sum w^2`` (mean, minimum, maximum), the mean weight of every source, how often each source had the largest weight,
the sources dropped by ``mixing.max_source_distance`` (share of subtasks with a drop, mean number dropped) and the share
whose temperature ``mixing.min_effective_sources`` raised. A method without mixing has no mix records; its sources per
subtask are counted instead. Standard library only (the drivers run it with the host's python3).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tmg.core.chunks import CHUNKS_DIR  # noqa: E402
from tmg.core.io_json import read_jsonl  # noqa: E402
from tmg.core.layout import ATTEMPTS_JSONL, RECORDS_DIR  # noqa: E402


def read_pool(pool: Path) -> list[dict]:
    """The attempt lines of every chunk of ``pool`` that has records, in chunk-name order."""
    files = sorted((Path(pool) / CHUNKS_DIR).glob(f"*/{RECORDS_DIR}/{ATTEMPTS_JSONL}"))
    return [attempt for f in files for attempt in read_jsonl(f)]


def effective_sources(weights) -> float:
    """``1 / sum w^2`` of the weights."""
    return 1.0 / sum(w * w for w in weights)


def _mean(values) -> float | None:
    values = list(values)
    return round(statistics.fmean(values), 4) if values else None


def summarize_subtask(executed: list[dict]) -> dict:
    """The summary of one subtask from its executed subtask records (module docstring)."""
    mixes = [s["mix"] for s in executed if s.get("mix") and "weights" in s["mix"]]
    out = {"executed": len(executed), "with_weights": len(mixes),
           "sources_selected": dict(sorted(Counter(int(s["src_demo_ind"]) for s in executed if "src_demo_ind" in s).items()))}
    if not mixes:
        return out
    weights = [m["weights"] for m in mixes]
    effective = [effective_sources(w) for w in weights]
    caps = [m["distance_cap"] for m in mixes if "distance_cap" in m]
    out.update(
        d=sorted({m["d"] for m in mixes if "d" in m}),
        blended_share=_mean(sum(1 for v in w if v > 0) > 1 for w in weights),
        effective_sources={"mean": _mean(effective), "min": round(min(effective), 4), "max": round(max(effective), 4)},
        mean_weights=[_mean(w[j] for w in weights) for j in range(len(weights[0]))],
        largest_weight_source=dict(sorted(Counter(max(range(len(w)), key=w.__getitem__) for w in weights).items())),
        temperature_raised_share=_mean("temperature" in m for m in mixes),
        distance_cap=None if not caps else {"limits": {k: caps[0].get(k) for k in ("rot_deg", "pos_cm")},
                                            "with_drop_share": _mean(c["n_dropped"] > 0 for c in caps),
                                            "mean_dropped": _mean(c["n_dropped"] for c in caps)})
    return out


def summarize(attempts: list[dict]) -> dict:
    """``{attempts, successes, subtasks: {k: summary}}`` of a pool's attempt lines."""
    by_subtask: dict[int, list[dict]] = {}
    for attempt in attempts:
        for s in attempt.get("subtasks") or []:
            by_subtask.setdefault(int(s["subtask_ind"]), []).append(s)
    return {"attempts": len(attempts), "successes": sum(1 for a in attempts if a.get("success")),
            "subtasks": {k: summarize_subtask(by_subtask[k]) for k in sorted(by_subtask)}}


def format_report(summary: dict) -> str:
    """The summary as the text the check prints."""
    lines = [f"attempts {summary['attempts']}, successes {summary['successes']}"]
    for k, s in summary["subtasks"].items():
        lines.append(f"subtask {k}: executed {s['executed']}, with mixing weights {s['with_weights']}")
        if not s["with_weights"]:
            lines.append(f"  no mixing; selected sources (source: count) {s['sources_selected']}")
            continue
        e, cap = s["effective_sources"], s["distance_cap"]
        lines += [f"  d {s['d']}, blended {s['blended_share']:.0%} of the subtasks, temperature raised in {s['temperature_raised_share']:.0%}",
                  f"  effective sources: mean {e['mean']}, min {e['min']}, max {e['max']}",
                  f"  mean weight per source: {s['mean_weights']}",
                  f"  largest-weight source (source: count): {s['largest_weight_source']}",
                  "  distance cap: off" if cap is None else
                  f"  distance cap {cap['limits']}: a source dropped in {cap['with_drop_share']:.0%} of the subtasks, "
                  f"{cap['mean_dropped']} dropped on average"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, required=True, help="pool folder (the output of tmg generate)")
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args(argv)
    attempts = read_pool(args.pool)
    if not attempts:
        print(f"no attempt records under {args.pool}", file=sys.stderr)
        return 1
    summary = summarize(attempts)
    print(json.dumps(summary, indent=2) if args.json else format_report(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
