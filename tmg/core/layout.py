"""Folder layout and file names of every result, in one place (standard library only, so the host can import it).

The data root (``paths.root``) holds one subfolder per kind of result; a condition is the six choices that name a
pool, and datasets and runs extend that path::

    pools/<generator>/<task>/<scene>/<selection>/<method>/<generation>
    datasets/<same six>/<observation>/n<size>_seed<seed>.hdf5     with its sidecars <stem>.DATASET.json and
                                                                   <stem>.resolved_config.yaml next to it
    runs/<same six>/<observation>/<policy>/seed<seed>              the run folder (layout below)
    resets/<generator>/<task>/<scene>/val<V>_test<T>_seed<seed>.hdf5   (``.json`` for the Isaac Lab generators)

Run folder::

    checkpoints/<update, 7 digits>/pretrained_model/, checkpoint.json, training_state.pt, stats.json | processors/
    train_log.jsonl, train_result.json, validation.jsonl, SELECTED.json, resolved_config.yaml, provenance.json
    eval/<split>_h<horizon>/<checkpoint name>/episodes.jsonl, eval_summary.json

The Hydra templates in ``configs/command/*.yaml`` spell the same paths (``tests/unit/test_layout.py`` checks that
the builders here and the composed configs agree); shell drivers and servers rely on every name below.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# subfolders of the data root (configs/machine/paths/*.yaml)
POOLS_DIR = "pools"
DATASETS_DIR = "datasets"
RESETS_DIR = "resets"
RUNS_DIR = "runs"
REPORTS_DIR = "reports"

# per-chunk attempt records of a pool (``tmg.core.records`` re-exports these; defined here so the standard-library-only
# host scripts, scripts/collect_results.py, can name them without importing numpy)
RECORDS_DIR = "records"
ATTEMPTS_JSONL = "attempts.jsonl"

# dataset sidecars (next to the hdf5, named after its stem)
DATASET_SIDECAR = "DATASET.json"
RESOLVED_CONFIG = "resolved_config.yaml"

# run folder
CHECKPOINTS_DIR = "checkpoints"
PRETRAINED_DIR = "pretrained_model"
TRAINING_STATE = "training_state.pt"
CHECKPOINT_JSON = "checkpoint.json"
STATS_JSON = "stats.json"
PIPELINE_DIR = "processors"
TRAIN_LOG = "train_log.jsonl"
TRAIN_RESULT = "train_result.json"
VALIDATION_JSONL = "validation.jsonl"
SELECTED_JSON = "SELECTED.json"
EVAL_DIR = "eval"
EPISODES = "episodes.jsonl"
SUMMARY = "eval_summary.json"


@dataclass(frozen=True)
class Condition:
    """The six choices that name a pool: generator, task, scene, source-selection rule, method and generation config."""

    generator: str
    task: str
    scene: str
    selection: str
    method: str
    generation: str

    @property
    def parts(self) -> tuple[str, ...]:
        """The folder names under ``pools/`` (and under ``datasets/`` and ``runs/``), in order."""
        return (self.generator, self.task, self.scene, self.selection, self.method, self.generation)

    @classmethod
    def from_plain(cls, plain: dict) -> Condition:
        """From a resolved config that carries all six groups (``generate``, ``dataset``, ``train``)."""
        return cls(*(str(plain[group]["name"]) for group in ("generator", "task", "scene", "selection", "method", "generation")))


def pool_dir(pools_root: Path, condition: Condition) -> Path:
    """``<pools_root>/<generator>/<task>/<scene>/<selection>/<method>/<generation>`` (``output`` of ``tmg generate``)."""
    return Path(pools_root).joinpath(*condition.parts)


def dataset_dir(datasets_root: Path, condition: Condition, observation: str) -> Path:
    """The folder of a condition's datasets of one observation choice."""
    return Path(datasets_root).joinpath(*condition.parts, str(observation))


def dataset_name(size: int, seed: int) -> str:
    """``n<size>_seed<seed>.hdf5``."""
    return f"n{int(size)}_seed{int(seed)}.hdf5"


def dataset_path(datasets_root: Path, condition: Condition, observation: str, size: int, seed: int) -> Path:
    """``output`` of ``tmg dataset`` and ``dataset.path`` of ``tmg train``."""
    return dataset_dir(datasets_root, condition, observation) / dataset_name(size, seed)


def run_dir(runs_root: Path, condition: Condition, observation: str, policy: str, seed: int) -> Path:
    """``output`` of ``tmg train``: ``<runs_root>/<six parts>/<observation>/<policy>/seed<seed>``."""
    return Path(runs_root).joinpath(*condition.parts, str(observation), str(policy), f"seed{int(seed)}")


def resets_dir(resets_root: Path, generator: str, task: str, scene: str) -> Path:
    """The folder of a (generator, task, scene)'s resets files."""
    return Path(resets_root) / str(generator) / str(task) / str(scene)


def resets_name(n_val: int, n_test: int, seed: int, ext: str) -> str:
    """``val<V>_test<T>_seed<seed>.<ext>``."""
    return f"val{int(n_val)}_test{int(n_test)}_seed{int(seed)}.{ext}"


def resets_path(plain: dict, *, isaaclab: bool) -> Path:
    """The resets file of a composed ``resets`` or ``eval`` config: ``.json`` (a seed list) for Isaac Lab generators,
    ``.hdf5`` (stored states) otherwise. ``tmg.evaluation.envs.is_isaaclab`` decides ``isaaclab``."""
    ev = plain["evaluation"]
    folder = resets_dir(plain["paths"]["resets"], plain["generator"]["name"], plain["task"]["name"], plain["scene"]["name"])
    return folder / resets_name(ev["val_scenes"], ev["test_scenes"], ev["seed"], "json" if isaaclab else "hdf5")


def checkpoint_dir(run_folder: Path, update: int) -> Path:
    """``<run>/checkpoints/<update as 7 digits>``."""
    return Path(run_folder) / CHECKPOINTS_DIR / f"{int(update):07d}"


def run_dir_of_checkpoint(checkpoint: Path) -> Path:
    """The run folder of ``<run>/checkpoints/<name>`` (the parent of a folder that is not under ``checkpoints/``)."""
    checkpoint = Path(checkpoint)
    return checkpoint.parent.parent if checkpoint.parent.name == CHECKPOINTS_DIR else checkpoint.parent


def eval_output(checkpoint: Path, split: str, horizon: int | None = None) -> Path:
    """``<run folder>/eval/<split>_h<horizon>/<checkpoint name>`` for a checkpoint under ``<run folder>/checkpoints/``
    (``<split>`` alone when no horizon is given). Evaluations at different step limits never share a folder."""
    checkpoint = Path(checkpoint)
    folder = f"{split}_h{int(horizon)}" if horizon is not None else str(split)
    return run_dir_of_checkpoint(checkpoint) / EVAL_DIR / folder / checkpoint.name


def sidecar(path: Path, name: str) -> Path:
    """``<folder>/<stem>.<name>`` next to a result file: ``n1000_seed0.hdf5`` -> ``n1000_seed0.DATASET.json``."""
    path = Path(path)
    return path.with_name(path.stem + "." + name)


def part_file(path: Path) -> Path:
    """The temporary name a file is written under before the atomic rename: ``<name>.part``."""
    path = Path(path)
    return path.with_suffix(path.suffix + ".part")


def key_index(key: str) -> int:
    """The integer of an indexed hdf5 key: ``demo_12`` -> 12, ``reset_3`` -> 3."""
    return int(key.split("_")[1])


def indexed_keys(names: Iterable[str], prefix: str | None = None) -> list[str]:
    """Indexed keys sorted by their integer (``demo_0, demo_1, ..., demo_10``, not alphabetically).

    Args:
        names: the keys of a group (``f["data"]`` of a demonstration file, a resets file).
        prefix: keep only names starting with it (``"reset_"``); None sorts every name, so a name without an
            index is an error, as it is for the demonstration groups.
    """
    names = [k for k in names if prefix is None or k.startswith(prefix)]
    return sorted(names, key=key_index)
