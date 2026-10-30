"""Noisy-state replay of the selected CP-Gen successes: the authors' ``add-noise`` and ``render-rgb`` steps.

The authors' launcher (``scripts/pipeline/run_datagen_and_training.sh``) builds every training dataset in two steps
after generation, and all sixteen released datasets are their output
(``...-original-action-noisy-state-action-std-0.03-rgb-84-84.hdf5``):

- ``add-noise``: ``scripts/playback_dataset_w_noise_mp.py --num_retries 6 --use-actions --num_processes 12
  --action_noise_std 0.03`` replays each success open loop with Gaussian noise added to its actions and keeps the
  replayed states with the original (noise-free) actions.
- ``render-rgb``: ``dataset_states_to_obs --done_mode 2 --camera_names agentview robot0_eye_in_hand --camera_height 84
  --camera_width 84 --compress --exclude-next-obs`` renders the observations from those states.

Here both steps run on the dataset's selected candidates only (``size`` of them, not the whole pool), with the
official functions unchanged: ``playback_dataset`` of the CP-Gen checkout and robomimic's ``dataset_states_to_obs``
(the function CP-Gen's own generation renders with; the launcher's adaflow copy is not in the image, and reading the two
side by side found no difference that changes the observations). Files under
the work folder (``<dataset stem>.noisy_replay/`` next to the dataset)::

    selected.hdf5                    the chosen successes from the chunks' generated.hdf5 (states, actions,
                                     constraint_data, model_file), as data/demo_<j> in selection order
    replay/demo_demo_<j>_w_noise.hdf5   the official per-demo output (the official merged file follows completion
                                     order, so it is not used)
    noisy_state_<p>.hdf5             the per-demo outputs regrouped in selection order into contiguous parts
    noisy_state_<p>_obs.hdf5         the rendered parts (one process each); demo_<j> keeps its selection index
    REPLAY_DONE.json                 parameters, selection and the count of demos that kept their original states

Official behaviour kept as it is: the noise std decays by (6 - k) / 6 over the attempts and is scaled down to
0.05-0.6 within 10 steps of a constraint timestep; the success of a replay is the task check after its last step; a
demo whose six replays all fail keeps its original states and stays in the dataset; the recorded states lag the
actions by one step (the script records the start state twice: images 0 and 1 both show the start state), as in the
released datasets.

    python -m tmg.generation.generators.cpgen.noisy_replay --work <folder> --chunk-spec <chunk_spec.json> ...
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

from tmg.generation.generators.cpgen.adapter import GENERATED

SELECTED = "selected.hdf5"
REPLAY_DIR = "replay"
REPLAY_DONE = "REPLAY_DONE.json"
PART = "noisy_state_{part}.hdf5"
PART_OBS = "noisy_state_{part}_obs.hdf5"
DEFAULTS = {"num_retries": 6, "action_noise_std": 0.03, "num_processes": 12, "render_processes": 12,
            "done_mode": 2, "compress": True}


def work_folder(output: Path) -> Path:
    """``<dataset stem>.noisy_replay`` next to the dataset file."""
    output = Path(output)
    return output.parent / f"{output.stem}.noisy_replay"


def no_obs_file(c: dict) -> Path:
    """The chunk's merged success file without observations (``generated.hdf5``) beside the candidate's rendered file."""
    return Path(c["demo_file"]).parent / GENERATED


def write_selection(chosen: list[dict], path: Path) -> None:
    """The chosen successes as ``data/demo_<j>`` of one file, in selection order.

    Each demo is read from its chunk's ``generated.hdf5``, whose k-th demo is the k-th success like the rendered file
    the candidate points at; the actions of the two are compared so that a shifted file is an error.

    Raises:
        ValueError: a demo's actions differ between the chunk's two files.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    opened: dict[str, h5py.File] = {}
    try:
        with h5py.File(path, "w") as out:
            data = out.create_group("data")
            for j, c in enumerate(chosen):
                src_path = str(no_obs_file(c))
                if src_path not in opened:
                    opened[src_path] = h5py.File(src_path, "r")
                    if "env_args" not in data.attrs:
                        data.attrs["env_args"] = opened[src_path]["data"].attrs["env_args"]
                src = opened[src_path]["data"][c["demo_key"]]
                with h5py.File(c["demo_file"], "r") as rendered:
                    if not np.array_equal(np.asarray(rendered["data"][c["demo_key"]]["actions"]), np.asarray(src["actions"])):
                        raise ValueError(f"{c['chunk']}/{c['demo_key']}: actions of {GENERATED} and {Path(c['demo_file']).name} differ")
                g = data.create_group(f"demo_{j}")
                g.create_dataset("actions", data=np.asarray(src["actions"]))
                g.create_dataset("states", data=np.asarray(src["states"]))
                if "constraint_data" in src:
                    src.copy("constraint_data", g)
                for key in ("model_file", "failure_type"):
                    if key in src.attrs:
                        g.attrs[key] = src.attrs[key]
                g.attrs["chunk"], g.attrs["demo_key"] = c["chunk"], c["demo_key"]
    finally:
        for f in opened.values():
            f.close()


def parts_of(n: int, parts: int) -> list[range]:
    """``n`` selection indices split into at most ``parts`` contiguous ranges."""
    parts = max(1, min(int(parts), n))
    bounds = np.linspace(0, n, parts + 1).round().astype(int)
    return [range(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def regroup(work: Path, n: int, parts: int) -> tuple[list[Path], int]:
    """Write the official per-demo outputs into contiguous part files in selection order.

    Returns:
        The part files and the number of demos whose replays all failed (they keep the original states).

    Raises:
        FileNotFoundError: a demo has no official output.
    """
    fallback = 0
    files = []
    with h5py.File(work / SELECTED, "r") as sel:
        env_args = sel["data"].attrs["env_args"]
        for p, idx in enumerate(parts_of(n, parts)):
            part = work / PART.format(part=p)
            with h5py.File(part, "w") as out:
                data = out.create_group("data")
                data.attrs["env_args"] = env_args
                for j in idx:
                    src_path = work / REPLAY_DIR / f"demo_demo_{j}_w_noise.hdf5"
                    if not src_path.exists():
                        raise FileNotFoundError(f"no replay output for demo_{j}: {src_path}")
                    with h5py.File(src_path, "r") as f:
                        src = f["data/demo_0"]
                        g = data.create_group(f"demo_{j}")
                        states = np.asarray(src["states"])
                        g.create_dataset("actions", data=np.asarray(src["actions"]))
                        g.create_dataset("states", data=states)
                        g.attrs["model_file"] = src.attrs["model_file"]
                        if np.array_equal(states, np.asarray(sel["data"][f"demo_{j}"]["states"])):
                            fallback += 1
            files.append(part)
    return files, fallback


def render_part(part: str, cameras: list[str], size: list[int], done_mode: int, compress: bool) -> str:
    """Render one part with robomimic's ``dataset_states_to_obs`` (the authors' render-rgb flags); returns the output."""
    from robomimic.scripts.dataset_states_to_obs import dataset_states_to_obs

    out_name = Path(part).name.replace(".hdf5", "_obs.hdf5")
    dataset_states_to_obs(SimpleNamespace(
        dataset=part, output_name=out_name, n=None, shaped=False, camera_names=list(cameras),
        camera_height=int(size[0]), camera_width=int(size[1]), depth=False, done_mode=int(done_mode),
        copy_rewards=False, copy_dones=False, exclude_next_obs=True, compress=bool(compress), use_actions=False))
    return str(Path(part).parent / out_name)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--chunk-spec", required=True, help="a chunk_spec.json of the pool (scene chain and checkout)")
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--cameras", nargs="*", default=[])
    ap.add_argument("--size", type=int, nargs=2, default=[84, 84])
    for key, value in DEFAULTS.items():
        ap.add_argument(f"--{key.replace('_', '-')}", type=type(value) if not isinstance(value, bool) else int, default=value)
    args = ap.parse_args(argv)
    work = Path(args.work).resolve()
    spec = json.loads(Path(args.chunk_spec).read_text())

    from tmg.core import egl

    egl.apply()
    from tmg.generation.generators._entry import register_scene_env
    from tmg.generation.generators.cpgen.checkout import cpgen_checkout

    with cpgen_checkout(spec["extra"]["source_root"], workdir=work / "cwd"):
        register_scene_env(spec)              # the scene's environment class, for env_args naming ours (forked below)
        import scripts.playback_dataset_w_noise_mp as official

        official.playback_dataset(SimpleNamespace(
            dataset=str(work / SELECTED), filter_key=None, n=None, gen_obs_dataset=False, use_obs=False, use_actions=True,
            render=False, video_path=None, video_skip=5, render_image_names=None, render_depth_names=None, first=False,
            action_noise_std=float(args.action_noise_std), num_retries=int(args.num_retries),
            save_demo_dir=str(work / REPLAY_DIR), num_processes=int(args.num_processes), use_noisy_actions=False))
        parts, fallback = regroup(work, args.n, args.render_processes)
        import multiprocessing as mp

        with mp.Pool(processes=len(parts)) as pool:
            outputs = pool.starmap(render_part, [(str(p), args.cameras, args.size, args.done_mode, bool(args.compress))
                                                 for p in parts])
    print(json.dumps({"parts": outputs, "fallback_original_states": fallback}), flush=True)
    (work / "replay_result.json").write_text(json.dumps({"parts": outputs, "fallback_original_states": fallback}))


def replay_selection(chosen: list[dict], *, pool: Path, output: Path, cfg: dict, observation: dict) -> tuple[list[dict], dict]:
    """Run the noisy replay on the chosen candidates and point them at the rendered parts.

    Reuses a finished work folder whose ``REPLAY_DONE.json`` lists the same selection and parameters.

    Args:
        chosen: the selected candidates (``tmg.data.dataset.select``).
        pool: the pool folder (a chunk spec of it supplies the scene chain and the CP-Gen checkout).
        output: the dataset file; the work folder is ``work_folder(output)``.
        cfg: the generator's ``noisy_state_replay`` settings (keys of ``DEFAULTS`` override them).
        observation: the observation config (``cameras``, ``image_size``).

    Returns:
        The candidates with ``demo_file`` and ``demo_key`` of the rendered parts, and the replay summary.
    """
    params = {**DEFAULTS, **{k: v for k, v in dict(cfg).items() if k in DEFAULTS}}
    cameras = [str(c) for c in (observation.get("cameras") or [])]
    size = [int(v) for v in (observation.get("image_size") or [84, 84])]
    selection = [[c["chunk"], c["demo_key"]] for c in chosen]
    work = work_folder(output)
    done_path = work / REPLAY_DONE
    expected = {"params": params, "cameras": cameras, "image_size": size, "selection": selection}
    done = json.loads(done_path.read_text()) if done_path.exists() else None
    if not done or any(done.get(k) != v for k, v in expected.items()):
        if work.exists():
            import shutil

            shutil.rmtree(work)                  # a different selection or setting: start over
        write_selection(chosen, work / SELECTED)
        spec = Path(pool) / "chunks" / chosen[0]["chunk"] / "chunk_spec.json"
        cmd = [sys.executable, "-m", "tmg.generation.generators.cpgen.noisy_replay", "--work", str(work),
               "--chunk-spec", str(spec), "--n", str(len(chosen)), "--size", *map(str, size)]
        if cameras:
            cmd += ["--cameras", *cameras]
        for key, value in params.items():
            cmd += [f"--{key.replace('_', '-')}", str(int(value) if isinstance(value, bool) else value)]
        print(f"[tmg] noisy-state replay of {len(chosen)} demos: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
        result = json.loads((work / "replay_result.json").read_text())
        done = {**expected, **result}
        done_path.write_text(json.dumps(done))
    where = {}
    for part in done["parts"]:
        with h5py.File(part, "r") as f:
            for key in f["data"].keys():
                where[key] = part
    out = []
    for j, c in enumerate(chosen):
        key = f"demo_{j}"
        if key not in where:
            raise FileNotFoundError(f"rendered parts of {work} have no {key}")
        out.append({**c, "demo_file": where[key], "demo_key": key})
    summary = {"work": str(work), "fallback_original_states": int(done["fallback_original_states"]), **params}
    print(f"[tmg] noisy-state replay: {len(chosen)} demos, {summary['fallback_original_states']} kept their original states "
          f"(all {params['num_retries']} replays failed)", flush=True)
    return out, summary


if __name__ == "__main__":
    main()
