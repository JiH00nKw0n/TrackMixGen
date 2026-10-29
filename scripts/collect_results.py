#!/usr/bin/env python3
"""Gather every result file of a baseline experiment into one folder (and a .tar.gz) for analysis elsewhere.

    scripts/collect_results.py --data-root <root> --out <folder> --policy diffusion_policy --seeds 202601 202602 202603 \
        --job softmimicgen/rope/ours/official/none/pool_isaac/image [--job ...] [--logs <driver log folder>]

A job is ``<generator>/<task>/<scene>/<selection>/<method>/<generation>/<observation>`` (the folder layout of pools, datasets and runs). For each job the bundle holds, without
any model weights or image data:

    <job>/generation/DONE.json, plan.json, provenance.json, resolved_config.yaml   the pool summary and its config
    <job>/generation/attempts.jsonl         every attempt of every chunk (one line each, with the chunk name added):
                                            success, source demonstration indices, subtask records, elapsed time
    <job>/generation/chunk_logs/            the chunk process logs
    <job>/dataset/*.json, *.yaml            which attempts went into the training dataset
    <job>/resets.json                       the fixed validation and test scene seeds
    <job>/seed<k>/train_log.jsonl, train_result.json, validation.jsonl, SELECTED.json, resolved_config.yaml, provenance.json
    <job>/seed<k>/checkpoints/<update>.json the checkpoint metadata (update, epoch, loss)
    <job>/seed<k>/eval/<split>/<checkpoint>/eval_summary.json, episodes.jsonl   every episode: success, steps, scene
                                            seed, initial object placement, policy seed
    <job>/seed<k>/*.log                     validation logs of the pipeline
    summary.json, summary.md                generation success rate, per-seed validation/test results, mean and
                                            standard deviation over the seeds
    driver_logs/                            the driver's logs when --logs is given

Only standard-library modules are used, so the script runs on the host without the images.
"""
from __future__ import annotations

import argparse
import json
import shutil
import statistics
import tarfile
from pathlib import Path


def copy(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def collect_generation(pool: Path, out: Path) -> dict:
    for name in ("DONE.json", "plan.json", "provenance.json", "resolved_config.yaml"):
        copy(pool / name, out / name)
    out.mkdir(parents=True, exist_ok=True)
    attempts = successes = 0
    per_source: dict[str, list[int]] = {}
    with open(out / "attempts.jsonl", "w") as f:
        for chunk in sorted((pool / "chunks").glob("seed*")) if (pool / "chunks").exists() else []:
            rec = chunk / "records" / "attempts.jsonl"
            if not rec.exists():
                continue
            for line in rec.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                row["chunk"] = chunk.name
                f.write(json.dumps(row) + "\n")
                attempts += 1
                successes += int(bool(row.get("success")))
                for i in row.get("src_demo_inds") or []:
                    per_source.setdefault(str(i), [0, 0])
                    per_source[str(i)][0] += 1
                    per_source[str(i)][1] += int(bool(row.get("success")))
            for logf in (chunk / "run.log", pool / "logs" / f"{chunk.name}.log"):
                copy(logf, out / "chunk_logs" / f"{chunk.name}_{logf.name}")
    done = read_json(pool / "DONE.json")
    return {"pool": str(pool), "done": done is not None, "attempts": attempts, "successes": successes,
            "success_rate": (successes / attempts if attempts else None),
            "per_source_demo": {k: {"attempts": v[0], "successes": v[1], "rate": v[1] / v[0]} for k, v in sorted(per_source.items())}}


def collect_run(run: Path, out: Path) -> dict:
    for name in ("train_log.jsonl", "train_result.json", "validation.jsonl", "SELECTED.json", "resolved_config.yaml", "provenance.json"):
        copy(run / name, out / name)
    for logf in run.glob("*.log"):
        copy(logf, out / logf.name)
    for ckpt in sorted((run / "checkpoints").glob("[0-9]*")) if (run / "checkpoints").exists() else []:
        copy(ckpt / "checkpoint.json", out / "checkpoints" / f"{ckpt.name}.json")
    tests = []
    for summary in sorted((run / "eval").glob("*/*/eval_summary.json")) if (run / "eval").exists() else []:
        rel = summary.relative_to(run)
        copy(summary, out / rel)
        copy(summary.parent / "episodes.jsonl", out / rel.parent / "episodes.jsonl")
        copy(summary.parent / "resolved_config.yaml", out / rel.parent / "resolved_config.yaml")
        s = read_json(summary)
        if s and str(s.get("split")) == "test":
            tests.append({"folder": str(rel.parent), **{k: s["all"].get(k) for k in ("n", "n_success", "success_rate", "ci_low", "ci_high")},
                          "horizon": s.get("horizon"), "checkpoint": s.get("checkpoint")})
    validation = []
    if (run / "validation.jsonl").exists():
        validation = [json.loads(l) for l in (run / "validation.jsonl").read_text().splitlines() if l.strip()]
    selected = read_json(run / "SELECTED.json")
    return {"run": str(run), "trained": (run / "train_result.json").exists(), "checkpoints_validated": len(validation),
            "best_validation": (max((v.get("success_rate") or 0.0) for v in validation) if validation else None),
            "selected": (selected or {}).get("checkpoint"), "test": (tests[-1] if tests else None)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", default="diffusion_policy")
    ap.add_argument("--seeds", nargs="+", type=int, default=[202601, 202602, 202603])
    ap.add_argument("--job", action="append", required=True, help="<generator>/<task>/<scene>/<selection>/<method>/<generation>/<observation>")
    ap.add_argument("--logs", default=None, help="driver log folder to include")
    ap.add_argument("--no-tar", action="store_true")
    a = ap.parse_args()
    root, out = Path(a.data_root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"data_root": str(root), "policy": a.policy, "seeds": a.seeds, "jobs": {}}
    lines = ["# Results", "", "| generator/task | generation success | seed | validated checkpoints | best validation | selected | test success (95% CI) |", "|---|---|---|---|---|---|---|"]
    for job in a.job:
        gen, task, scene, selection, method, generation, observation = job.split("/")
        jo = out / gen / task
        g = collect_generation(root / "pools" / gen / task / scene / selection / method / generation, jo / "generation")
        ds_dir = root / "datasets" / gen / task / scene / selection / method / generation / observation
        for f in list(ds_dir.glob("*.json")) + list(ds_dir.glob("*.yaml")) if ds_dir.exists() else []:
            copy(f, jo / "dataset" / f.name)
        for f in (root / "resets" / gen / task / scene).glob("*") if (root / "resets" / gen / task / scene).exists() else []:
            if f.suffix in (".json", ".yaml"):
                copy(f, jo / "resets" / f.name)
        seeds = {}
        for seed in a.seeds:
            seeds[str(seed)] = collect_run(root / "runs" / gen / task / scene / selection / method / generation / observation / a.policy / f"seed{seed}", jo / f"seed{seed}")
        rates = [s["test"]["success_rate"] for s in seeds.values() if s["test"] and s["test"]["success_rate"] is not None]
        agg = {"mean": (statistics.mean(rates) if rates else None), "std": (statistics.stdev(rates) if len(rates) > 1 else None), "n_seeds": len(rates)}
        summary["jobs"][job] = {"generation": g, "seeds": seeds, "test_over_seeds": agg}
        grate = f"{g['successes']}/{g['attempts']} = {g['success_rate']:.3f}" if g["attempts"] else "no records"
        for seed, s in seeds.items():
            t = s["test"]
            tcell = f"{t['n_success']}/{t['n']} = {t['success_rate']:.3f} ({t['ci_low']:.3f}-{t['ci_high']:.3f})" if t else "not evaluated"
            bv = f"{s['best_validation']:.2f}" if s["best_validation"] is not None else "-"
            lines.append(f"| {gen}/{task} | {grate} | {seed} | {s['checkpoints_validated']} | {bv} | {Path(s['selected']).name if s['selected'] else '-'} | {tcell} |")
        if agg["mean"] is not None:
            lines.append(f"| {gen}/{task} | | mean of {agg['n_seeds']} seeds | | | | {agg['mean']:.3f} ± {agg['std'] if agg['std'] is not None else 0:.3f} |")
    if a.logs and Path(a.logs).exists():
        for f in Path(a.logs).glob("*"):
            if f.is_file():
                copy(f, out / "driver_logs" / f.name)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if not a.no_tar:
        tar = out.with_suffix(".tar.gz")
        with tarfile.open(tar, "w:gz") as t:
            t.add(out, arcname=out.name)
        print(f"bundle: {tar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
