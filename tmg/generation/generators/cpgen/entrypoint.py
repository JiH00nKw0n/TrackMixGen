"""Chunk process of the CP-Gen adapter.

    python -m tmg.generation.generators.cpgen.entrypoint --spec <chunk>/chunk_spec.json

Steps: work inside the CP-Gen checkout (``checkout.cpgen_checkout``: the checkout on ``sys.path`` for
``scripts.dataset.mp4_from_h5``, and a per-chunk folder that mirrors it as the working directory, because its code
opens ``demo_aug/configs/...`` relative to the working directory and writes scratch files there), register the scene's
environment class, install the grafts (``runtime.apply``) unless the spec disables them for the identity check, then
call ``demo_aug.generate.main`` with the chunk's ``Config``. CP-Gen's ``main`` raises after the attempts when no
attempt succeeded (it tries to render observations of a merged file that does not exist); that case ends the
chunk normally once every attempt is recorded.
"""
from __future__ import annotations

import traceback
from pathlib import Path

from tmg.generation.generators._entry import install_runtime, load_spec, register_scene_env
from tmg.generation.generators.cpgen.checkout import cpgen_checkout


def main(argv=None):
    spec = load_spec(argv)
    from tmg.core import egl

    egl.apply()
    extra = spec["extra"]
    with cpgen_checkout(extra["source_root"], workdir=Path(spec["folder"]) / "cwd"):
        from tmg.core.records import RECORDS_DIR, read_attempts
        from tmg.generation.generators.cpgen import runtime

        register_scene_env(spec)
        install_runtime(spec, runtime.apply, all_sources=bool(extra.get("all_sources")))
        import demo_aug.generate as official

        config = dict(extra["cpgen_config"])
        config["save_dir"] = Path(config["save_dir"])
        cfg = official.Config(**config)
        print(f"[tmg] CP-Gen main: seed {cfg.seed}, {cfg.n_demos} attempts, env {cfg.env_name}, source {cfg.demo_path} "
              f"[{cfg.load_demos_start_idx}, {cfg.load_demos_end_idx})", flush=True)
        try:
            official.main(cfg)
        except Exception:
            records = read_attempts(Path(spec["folder"]) / RECORDS_DIR)
            attempts = int(spec["chunk"]["attempts"])
            if len(records) >= attempts and not any(r.get("success") for r in records):
                print(f"[tmg] CP-Gen main failed after all {attempts} attempts with no success (expected: nothing to render):\n"
                      f"{traceback.format_exc()}", flush=True)
                return
            raise


if __name__ == "__main__":
    main()
