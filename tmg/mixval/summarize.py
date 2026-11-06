"""Success tables of the blend-ratio check from the shards' ``results.jsonl``.

    python -m tmg.mixval.summarize --run <dir> [--min-repeat 3] [--max-anchor-deg 40]

A scene instance is one (scene, perturbation). For each scene and condition, ``k`` is the number of its instances that
succeeded (0 to P + 1). A scene is "hard" when every original source fails on every instance (k = 0 for each orig_a).
A condition "rescues" a hard scene when k >= --min-repeat. A condition "loses" a scene when both of its sources succeed
on at least --min-repeat instances alone while the blend reaches fewer than --min-repeat. Attempts whose donor solve
failed (``mix_error``) are left out. Pairs whose largest anchor rotation exceeds --max-anchor-deg in any recorded
attempt are listed separately as incompatible.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def load(run: Path) -> list[dict]:
    rows = []
    for f in sorted(run.glob("shard_*/results.jsonl")):
        rows.extend(json.loads(line) for line in open(f))
    return rows


def summarize(rows: list[dict], min_repeat: int, max_anchor_deg: float) -> dict:
    plan_conds = sorted({r["cond"] for r in rows})
    k = defaultdict(int)                     # (scene, cond) -> successful instances
    n = defaultdict(int)                     # (scene, cond) -> counted instances
    angles = defaultdict(float)              # (primary, donor) -> largest anchor rotation seen
    errors = 0
    for r in rows:
        if r.get("mix_error"):
            errors += 1
            continue
        key = (r["scene"], r["cond"])
        n[key] += 1
        k[key] += int(r["success"])
        for m in r.get("mix") or []:
            a = m.get("anchor_rotation_degrees")
            if a:
                angles[(r["primary"], r["donor"])] = max(angles[(r["primary"], r["donor"])], max(a))
    scenes = sorted({s for s, _ in n})
    origs = [c for c in plan_conds if c.startswith("orig_")]
    complete = [s for s in scenes if all(n[(s, c)] > 0 for c in origs)]
    hard = [s for s in complete if all(k[(s, c)] == 0 for c in origs)]
    by_cond = {}
    for c in plan_conds:
        if c.startswith("orig_"):
            by_cond[c] = {"instance_success": sum(k[(s, c)] for s in scenes), "instances": sum(n[(s, c)] for s in scenes),
                          "scenes_k_ge_min": sum(k[(s, c)] >= min_repeat for s in scenes)}
            continue
        _, a, b, share = c.split("_")
        rescued = [s for s in hard if k[(s, c)] >= min_repeat]
        any_success = [s for s in hard if k[(s, c)] > 0]
        both_good = [s for s in complete if k[(s, f"orig_{a}")] >= min_repeat and k[(s, f"orig_{b}")] >= min_repeat and n[(s, c)] > 0]
        lost = [s for s in both_good if k[(s, c)] < min_repeat]
        by_cond[c] = {"primary": int(a), "donor": int(b), "donor_share": int(share),
                      "instance_success": sum(k[(s, c)] for s in scenes), "instances": sum(n[(s, c)] for s in scenes),
                      "hard_rescued": len(rescued), "hard_any_success": len(any_success), "rescued_scenes": rescued,
                      "both_sources_good": len(both_good), "lost": len(lost),
                      "max_anchor_deg": round(angles.get((int(a), int(b)), 0.0), 2),
                      "compatible": angles.get((int(a), int(b)), 0.0) <= max_anchor_deg}
    return {"attempts": len(rows), "mix_errors": errors, "scenes": len(scenes), "complete_scenes": len(complete),
            "hard_scenes": hard, "min_repeat": min_repeat, "conditions": by_cond}


def table(summary: dict) -> str:
    lines = [f"attempts {summary['attempts']}, mix errors {summary['mix_errors']}, scenes {summary['scenes']} "
             f"(all originals present {summary['complete_scenes']}), hard scenes {len(summary['hard_scenes'])}", "",
             "| condition | instance success | hard scenes rescued (k>=min) | hard scenes with any success | lost / both good | max anchor deg |",
             "|---|---|---|---|---|---|"]
    for c, v in summary["conditions"].items():
        rate = f"{v['instance_success']}/{v['instances']}"
        if c.startswith("orig_"):
            lines.append(f"| {c} | {rate} | | | | |")
        else:
            lines.append(f"| {c} | {rate} | {v['hard_rescued']} | {v['hard_any_success']} | {v['lost']}/{v['both_sources_good']} | {v['max_anchor_deg']} |")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--min-repeat", type=int, default=3)
    p.add_argument("--max-anchor-deg", type=float, default=40.0)
    args = p.parse_args(argv)
    run = Path(args.run)
    s = summarize(load(run), args.min_repeat, args.max_anchor_deg)
    (run / "summary.json").write_text(json.dumps(s, indent=1))
    text = table(s)
    (run / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
