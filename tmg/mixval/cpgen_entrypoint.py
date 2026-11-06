"""One shard of the blend-ratio check, inside the CP-Gen image.

    python -m tmg.mixval.cpgen_entrypoint --template <chunk_spec.json> --shard <dir> --scenes <scenes.hdf5>

``--template`` is the chunk spec of an all-sources CP-Gen pool chunk of the task (it carries the task, the scene chain,
CP-Gen's config and the graft hashes). ``<dir>/attempts.jsonl`` is the shard's part of the plan (``plan``); attempts
already in ``<dir>/results.jsonl`` are skipped, so a rerun resumes. Per-attempt hdf5 files of CP-Gen are not written
(``save_demo`` is replaced by a no-op); our records go to ``<dir>/records``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True)
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
    from tmg.core import egl

    egl.apply()
    from tmg.generation.generators._entry import register_scene_env
    from tmg.generation.generators.cpgen.checkout import cpgen_checkout

    extra = spec["extra"]
    with cpgen_checkout(extra["source_root"], workdir=shard / "cwd"):
        from tmg.mixval import cpgen_runtime as runtime

        register_scene_env(spec)
        scenes = runtime.load_scenes(args.scenes, sorted({a["scene"] for a in attempts}))
        runtime.apply(attempts, scenes, results=results, records_dir=shard / "records",
                      graft_hashes=dict(extra.get("graft_hashes") or {}))
        import demo_aug.generate as official

        official.save_demo = lambda *a, **k: None
        config = dict(extra["cpgen_config"])
        config.update(n_demos=len(attempts), require_n_demos=False, save_dir=shard / "generated",
                      merge_demo_save_path=str(shard / "generated.hdf5"))
        if isinstance(config.get("initialization"), dict):
            config["initialization"] = official.InitializationConfig(**config["initialization"])
        cfg = official.Config(**config)
        print(f"[mixval] {shard}: {len(attempts)} attempts ({len(done)} already done), env {cfg.env_name}, "
              f"sources [{cfg.load_demos_start_idx}, {cfg.load_demos_end_idx})", flush=True)
        try:
            official.main(cfg)
        except runtime.AllAttemptsDone:
            print(f"[mixval] {shard}: all attempts done", flush=True)
            return
        raise RuntimeError("CP-Gen main returned before the last attempt")


if __name__ == "__main__":
    main()
