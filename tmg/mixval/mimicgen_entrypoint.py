"""One shard of the blend-ratio check with MimicGen, inside the robosuite image.

    python -m tmg.mixval.mimicgen_entrypoint --template <chunk_spec.json> --mg-config <mg_config.json> --shard <dir> --scenes <scenes.hdf5>

``--template`` and ``--mg-config`` are the chunk spec and MimicGen config of a pool chunk of the task (task, scene chain,
graft hashes, subtask settings). The shard's MimicGen config is that config with ``num_trials`` = the attempts left,
generation output under ``<dir>/mg`` and failed attempts not kept. Observations stay on, because MimicGen's writer of a
successful demonstration requires them.
``<dir>/attempts.jsonl`` is the shard's part of the plan; attempts already in ``<dir>/results.jsonl`` are skipped.
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

GENERATE_MODULE = "mimicgen.scripts.generate_dataset"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True)
    p.add_argument("--mg-config", required=True)
    p.add_argument("--shard", required=True)
    p.add_argument("--scenes", required=True)
    p.add_argument("--limit", type=int, default=None, help="run only the first N remaining attempts (smoke test)")
    args = p.parse_args(argv)
    shard = Path(args.shard).resolve()
    spec = json.loads(Path(args.template).read_text())
    attempts = [json.loads(line) for line in open(shard / "attempts.jsonl")]
    results = shard / "results.jsonl"
    done = {json.loads(line)["id"] for line in open(results)} if results.exists() else set()
    attempts = [a for a in attempts if a["id"] not in done]
    if args.limit is not None:
        attempts = attempts[:args.limit]
    if not attempts:
        print(f"[mixval] {shard}: nothing left", flush=True)
        return
    from tmg.generation.generators._entry import register_scene_env
    from tmg.mixval import mimicgen_runtime, scenes

    register_scene_env(spec)
    rt = mimicgen_runtime.apply(attempts, scenes.load_scenes(args.scenes, sorted({a["scene"] for a in attempts})),
                                results=results, records_dir=shard / "records",
                                graft_hashes=dict(spec["extra"].get("graft_hashes") or {}))
    cfg = json.loads(Path(args.mg_config).read_text())
    exp = cfg["experiment"]
    exp["name"] = f"mixval_{shard.name}"
    exp["generation"].update(path=str(shard / "mg"), num_trials=len(attempts), keep_failed=False, guarantee=False)
    exp["max_num_failures"] = None
    exp["log_every_n_attempts"] = 10 ** 9
    (shard / "mg").mkdir(exist_ok=True)
    config_path = shard / "mg_config.json"
    config_path.write_text(json.dumps(cfg, indent=1))
    print(f"[mixval] {shard}: {len(attempts)} attempts ({len(done)} already done), env {exp['task']['name']}", flush=True)
    sys.argv = ["generate_dataset.py", "--config", str(config_path), "--auto-remove-exp"]
    runpy.run_module(GENERATE_MODULE, run_name="__main__")
    rt.finish()


if __name__ == "__main__":
    main()
