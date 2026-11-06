"""The attempt list of the blend-ratio check.

    python -m tmg.mixval.plan --out <dir> --sources 1 6 7 --scenes 25-74 --perturbations 4 --shards 20

Conditions per (scene, perturbation), for sources S (the top-ranked CP-Gen sources of the task):

    orig_<a>              source a alone, as CP-Gen runs it (no mixing)
    mix_<a>_<b>_<s>       primary a blended with donor b, donor share s % (25, 50, 75) of the pose at every step
    mix_<a>_<b>_100       donor share 100 %: donor b's solved poses time-warped onto a's timeline (a's gripper commands
                          and symmetry transforms), the control that separates the warp from the blend

With three sources that is 3 + 6 x 4 = 27 conditions. Perturbation 0 is the stored scene; perturbations 1..P move
every free object by a fixed offset (``runtime.perturbation``). Attempts are sorted by scene, perturbation and condition
and dealt to the shards round-robin, so each shard advances through the scenes in order and a partial run already covers
the first scenes under every condition. ``--originals-only`` keeps only the orig_<a> conditions: the control run that plays
the remaining sources unmixed on the same scene instances.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

DONOR_SHARES = (25, 50, 75, 100)


def conditions(sources: list[int], shares=DONOR_SHARES) -> list[dict]:
    """Every condition of the module docstring; ``weight_a`` is the primary's share (1 - donor share)."""
    out = [{"cond": f"orig_{a}", "primary": a, "donor": None, "weight_a": 1.0, "donor_share": 0} for a in sources]
    for a, b in itertools.permutations(sources, 2):
        for s in shares:
            out.append({"cond": f"mix_{a}_{b}_{s}", "primary": a, "donor": b, "weight_a": round(1 - s / 100, 4), "donor_share": s})
    return out


def parse_range(text: str) -> list[int]:
    """``25-74`` or ``25,30,31`` (inclusive ranges)."""
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return out


def build(sources: list[int], scenes: list[int], perturbations: int, originals_only: bool = False) -> list[dict]:
    """The attempts in (scene, perturbation, condition) order, each with its id."""
    conds = [c for c in conditions(sources) if not originals_only or c["donor"] is None]
    attempts = []
    for scene in scenes:
        for pert in range(perturbations + 1):
            for c in conds:
                attempts.append({"id": f"s{scene:03d}_p{pert}_{c['cond']}", "scene": scene, "pert": pert, **c})
    return attempts


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--sources", type=int, nargs="+", required=True)
    p.add_argument("--scenes", required=True, help="reset indices of the scenes file, e.g. 25-74")
    p.add_argument("--perturbations", type=int, default=4)
    p.add_argument("--shards", type=int, required=True)
    p.add_argument("--originals-only", action="store_true",
                   help="only the orig_<a> conditions (control: the other sources unmixed on the same scene instances)")
    args = p.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    attempts = build(args.sources, parse_range(args.scenes), args.perturbations, args.originals_only)
    for k in range(args.shards):
        shard = out / f"shard_{k:02d}"
        shard.mkdir(exist_ok=True)
        with open(shard / "attempts.jsonl", "w") as f:
            for a in attempts[k::args.shards]:
                f.write(json.dumps(a) + "\n")
    meta = {"sources": args.sources, "scenes": args.scenes, "perturbations": args.perturbations, "shards": args.shards,
            "attempts": len(attempts), "originals_only": args.originals_only,
            "conditions": sorted({a["cond"] for a in attempts})}
    (out / "plan.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps({k: v for k, v in meta.items() if k != "conditions"}))


if __name__ == "__main__":
    main()
