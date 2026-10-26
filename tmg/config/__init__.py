"""Configuration loading: Hydra composition over ``configs/`` with strict keys.

``compose(command, overrides)`` composes ``configs/command/<command>.yaml`` with the group choices and ``key=value``
overrides given on the command line, resolves interpolations and returns a plain-dict-convertible
``DictConfig`` in struct mode (unknown keys are rejected). Hydra's run directory and logging are not used:
result folders are chosen by each command (``output``), never by Hydra.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from hydra import compose as _hydra_compose
from hydra import initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from tmg import CONFIGS_ROOT

LAYERS = ("machine", "experiment")
COMMANDS = ("generate", "verify_baseline", "sources", "check_scene", "rank_sources", "dataset", "train", "select_checkpoint",
          "serve", "resets", "eval", "report")


def compose(command: str, overrides: Iterable[str] = (), *, configs_root: Path | None = None) -> DictConfig:
    """Compose the config of ``command`` from ``configs/command/<command>.yaml`` plus command-line overrides."""
    if command not in COMMANDS:
        raise ValueError(f"unknown command {command!r}; known: {COMMANDS}")
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
