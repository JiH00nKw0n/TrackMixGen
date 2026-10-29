"""Configuration loading: Hydra composition over ``configs/`` with strict keys.

``compose(command, overrides)`` composes ``configs/command/<command>.yaml`` with the group choices and ``key=value``
overrides given on the command line, resolves interpolations and returns a plain-dict-convertible
``DictConfig`` in struct mode (unknown keys are rejected). Hydra's run directory and logging are not used:
result folders are chosen by each command (``output``), never by Hydra.

``COMMANDS`` is the command table of the ``tmg`` command line: for every command the config file
``configs/command/<name>.yaml``, the function that runs it (``module:function``, imported only when the command
runs, since most of them need a simulator or torch), its one-line help, and the result key that decides the exit
code. ``tmg.cli`` builds its argument parser and its help text from this table.
"""
from __future__ import annotations

import importlib
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from hydra import compose as _hydra_compose
from hydra import initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from tmg import CONFIGS_ROOT

LAYERS = ("machine", "experiment")


@dataclass(frozen=True)
class CommandSpec:
    """One ``tmg`` subcommand."""

    target: str                     # "module:function"; the function takes the composed DictConfig
    help: str                       # one line for ``tmg --help``
    exit_code: str | None = None    # key of the result dict: exit 1 when its value is false, else 0; None: always 0

    def resolve(self) -> Callable:
        """Import the module and return the function (deferred: most targets import a simulator or torch)."""
        module, _, function = self.target.partition(":")
        return getattr(importlib.import_module(module), function)


COMMANDS: dict[str, CommandSpec] = {
    "generate": CommandSpec("tmg.generation.runner:run_pool",
                            "one pool of generation attempts (generator, task, scene, method, generation)"),
    "verify_baseline": CommandSpec("tmg.generation.verify:verify_baseline",
                                   "run one chunk officially and through the adapter (method=none) and compare the demos",
                                   exit_code="identical"),
    "sources": CommandSpec("tmg.cli:fetch_sources", "download and prepare the human source demonstrations of a task"),
    "check_scene": CommandSpec("tmg.generation.scene_check:check_scene",
                               "reset a (task, scene) environment many times and report edge overhang, contacts, falls, reach",
                               exit_code="ok"),
    "rank_sources": CommandSpec("tmg.generation.rank_sources:rank_sources",
                                "order a task's source demonstrations by their generation success in a baseline pool"),
    "dataset": CommandSpec("tmg.data.dataset:build_dataset",
                           "select successful attempts of a finished pool and write one training hdf5"),
    "train": CommandSpec("tmg.policy.trainer:train", "train one policy on one dataset (train image)"),
    "select_checkpoint": CommandSpec("tmg.policy.trainer:select_checkpoint_cmd",
                                     "choose a run's checkpoint from validation.jsonl (or the last one)"),
    "serve": CommandSpec("tmg.policy.server:serve", "policy server for one checkpoint (train image)"),
    "resets": CommandSpec("tmg.evaluation.resets:make_resets",
                          "draw the fixed evaluation scenes of (task, scene) (simulator image)"),
    "eval": CommandSpec("tmg.evaluation.rollout:evaluate", "play the fixed scenes against a served checkpoint (simulator image)"),
}


def compose(command: str, overrides: Iterable[str] = (), *, configs_root: Path | None = None) -> DictConfig:
    """Compose the config of ``command`` from ``configs/command/<command>.yaml`` plus command-line overrides."""
    if command not in COMMANDS:
        raise ValueError(f"unknown command {command!r}; known: {tuple(COMMANDS)}")
    root = Path(configs_root or CONFIGS_ROOT).resolve()
    overrides = [str(o) for o in overrides]
    for o in overrides:
        if "=" not in o:
            raise ValueError(f"override {o!r} is not key=value (Hydra '+key' additions are not allowed)")
        if o.startswith("+") or o.startswith("~"):
            raise ValueError(f"override {o!r}: adding or deleting keys on the command line is not allowed")
    # Layers: configs/command/<command>.yaml is the primary config; the groups it names live under configs/machine
    # (cluster, paths), configs/experiment (generator, method, generation, policy, ...) and, for the chosen generator,
    # configs/experiment/generator/<generator>/ (task, scene). Hydra finds them through the search path, so the command
    # line keeps the short group names (generator=mimicgen task=square scene=ours cluster=a100).
    generator = _generator_choice(root / "command" / f"{command}.yaml", overrides)
    layers = [root / layer for layer in LAYERS]
    if generator:
        layers.append(root / "experiment" / "generator" / generator)   # task, scene (and ranking data) of that generator
    searchpath = "hydra.searchpath=[" + ",".join(f"file://{p}" for p in layers) + "]"
    with initialize_config_dir(config_dir=str(root), version_base="1.3", job_name=f"tmg-{command}"):
        cfg = _hydra_compose(config_name=f"command/{command}", overrides=[searchpath, *overrides])
    OmegaConf.set_struct(cfg, True)
    cfg.command = command
    cfg.overrides = overrides
    return cfg


def _generator_choice(command_file: Path, overrides: list[str]) -> str | None:
    """The generator name of this composition: ``generator=<name>`` on the command line, else the command file's
    default ``- /generator: <name>`` entry, else None (commands without a generator group)."""
    for o in overrides:
        if o.startswith("generator="):
            return o.split("=", 1)[1]
    if command_file.exists():
        m = re.search(r"^\s*-\s*/generator:\s*(\S+)", command_file.read_text(), re.MULTILINE)
        if m:
            return m.group(1)
    return None


def resolved(cfg: DictConfig) -> dict:
    """Plain dict with every interpolation resolved (what gets written to a result folder)."""
    return OmegaConf.to_container(cfg, resolve=True)


def dump_yaml(cfg: DictConfig | dict, path: Path) -> Path:
    """Write a config (or plain dict) as YAML with every interpolation resolved.

    Args:
        cfg: a composed ``DictConfig`` or an already plain mapping.
        path: destination file; parent folders are created.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = resolved(cfg) if isinstance(cfg, DictConfig) else cfg
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(data)))
    return path
