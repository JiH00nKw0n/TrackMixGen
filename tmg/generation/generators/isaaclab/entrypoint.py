"""Chunk process of the Isaac Lab adapters.

    python -m tmg.generation.generators.isaaclab.entrypoint --spec <chunk>/chunk_spec.json

The official ``generate_dataset.py`` parses its own arguments and starts Isaac Sim (``AppLauncher``) before
importing anything from Isaac Lab, so the grafts cannot be installed before the script runs. The entrypoint
therefore reads the script's source, inserts one statement, ``_tmg_install(globals())``, right before its
``if __name__ == "__main__":`` block, and executes the result with ``_tmg_install`` bound to :func:`install`.
That function runs after the script's imports and before its ``main()``:

1. imports the task's ``register_modules`` and camera module (environment registration, a camera-enabled environment;
   :func:`import_task_modules`);
2. wraps the script's ``setup_env_config`` (:func:`wrap_setup_env_config`) so that the scene ranges are written into the
   environment config (``simulators.isaaclab.apply_scene``), the task's low-dim observation terms are added to the policy
   group (``simulators.isaaclab_lowdim.add_lowdim_terms``), the chunk seed is written into ``datagen_config.seed`` (the
   official scripts take no ``--seed``; they seed random, numpy and torch with that config value), ``generation_guarantee``
   is set from ``generation.protocol.guarantee`` (true: ``--generation_num_trials`` counts successes, the official value;
   false: it counts attempts, for the test pilots and the ranking pools) and the overrides are logged;
3. installs the grafts (:func:`install_runtime`): the ``install_from_spec`` of ``spec["extra"]["runtime_module"]`` (AutoData's
   runtime for DexMimicGen) or, when the spec names none, of ``isaaclab.runtime``; both also install the shared-path
   ``generate`` when tracking is on.

The official script file is never modified; only the in-memory copy run by this process is.
"""
from __future__ import annotations

import argparse
import ast
import functools
import importlib
import json
import sys
from pathlib import Path

DEFAULT_RUNTIME = "tmg.generation.generators.isaaclab.runtime"


def install(namespace: dict, spec: dict) -> None:
    """Called from inside the official script after its imports (see the module docstring)."""
    import_task_modules(spec)
    wrap_setup_env_config(namespace, spec)
    install_runtime(spec, namespace)


def import_task_modules(spec: dict) -> None:
    """Import the task's environment-registration modules and its camera environment module."""
    for module in list(spec["extra"].get("register_modules") or []):      # e.g. isaaclab_mimic.envs registers the GR1 tasks
        __import__(str(module))
        print(f"[tmg] module imported (environment registration): {module}", flush=True)
    if spec["extra"].get("camera_module"):
        __import__(spec["extra"]["camera_module"])
        print(f"[tmg] camera task module imported: {spec['extra']['camera_module']}", flush=True)


def wrap_setup_env_config(namespace: dict, spec: dict) -> None:
    """Replace the script's ``setup_env_config`` with one that applies the chunk's scene, seed and protocol to the config."""
    from tmg.generation.simulators.isaaclab import apply_scene, isolate_controller_scratch
    from tmg.generation.simulators.isaaclab_lowdim import add_lowdim_terms

    scene, task_name = spec["scene"], str(spec["task"]["name"])
    chunk_seed = int(spec["chunk"]["seed"])
    guarantee = bool((spec.get("protocol") or {}).get("guarantee", True))
    original_setup = namespace["setup_env_config"]

    @functools.wraps(original_setup)
    def setup_env_config(*args, **kwargs):
        env_cfg, *rest = _as_tuple(original_setup(*args, **kwargs))
        applied = apply_scene(env_cfg, scene, task_name)
        print(f"[tmg] scene {scene.get('name')}: {'ranges applied ' + json.dumps(applied) if applied else 'official ranges kept'}", flush=True)
        added = add_lowdim_terms(env_cfg, spec["task"])       # low-dim object terms, recorded with the policy group
        print(f"[tmg] low-dim observation terms: {json.dumps(added) if added else 'none'}", flush=True)
        moved = isolate_controller_scratch(env_cfg, spec["folder"])   # per-chunk URDF folder (parallel chunks shared /tmp/urdf)
        if moved:
            print(f"[tmg] controller URDF output folders: {json.dumps(moved)}", flush=True)
        datagen = getattr(env_cfg, "datagen_config", None)
        if datagen is not None:        # the official script seeds random, numpy and torch with this value
            datagen.seed = chunk_seed
            print(f"[tmg] datagen_config.seed = {chunk_seed} (chunk seed)", flush=True)
            # generation.protocol.guarantee decides what --generation_num_trials counts: successes (true, the official
            # value of every Isaac Lab task; the full pools) or attempts (false; the test pilots and the ranking pools)
            if hasattr(datagen, "generation_guarantee"):
                datagen.generation_guarantee = guarantee
                print(f"[tmg] datagen_config.generation_guarantee = {guarantee} (trials count "
                      f"{'successes' if guarantee else 'attempts'})", flush=True)
        return (env_cfg, *rest) if rest else env_cfg

    namespace["setup_env_config"] = setup_env_config


def install_runtime(spec: dict, namespace: dict) -> None:
    """Install the grafts through the runtime module the spec names (default: the Isaac Lab runtime), unless disabled."""
    if spec["extra"].get("disable_graft"):
        print("[tmg] grafts DISABLED (identity check)", flush=True)
        return
    module = str(spec["extra"].get("runtime_module") or DEFAULT_RUNTIME)
    info = importlib.import_module(module).install_from_spec(spec, namespace)
    print(f"[tmg] grafts installed ({module}): {json.dumps(info, indent=1)}", flush=True)


def _as_tuple(value):
    return value if isinstance(value, tuple) else (value,)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    script = Path(spec["extra"]["script"])
    source = script.read_text()
    tree = ast.parse(source, str(script))
    index = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.If) and "__name__" in ast.unparse(node.test))
    tree.body.insert(index, ast.parse("_tmg_install(globals(), _tmg_spec)").body[0])
    ast.fix_missing_locations(tree)
    sys.argv = [str(script), *spec["extra"]["script_args"]]
    print(f"[tmg] running {script} with {sys.argv[1:]}", flush=True)
    exec(compile(tree, str(script), "exec"), {"__name__": "__main__", "__file__": str(script), "_tmg_install": install, "_tmg_spec": spec})


if __name__ == "__main__":
    main()
