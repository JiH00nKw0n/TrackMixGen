"""The generator contract every adapter implements and the runner relies on.

A generator adapter knows one official package. It prepares a chunk (writes the generator's own config and a
chunk spec), says how to launch the chunk process, and recognises when a chunk finished. Everything else
(chunk plan, parallel launch, resume, provenance, records) is shared code in ``tmg.generation.runner`` and
``tmg.core``. Adapters may import their official package only inside the chunk process, never at module
import time, because the runner also runs where the package is absent.
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from tmg.core.chunks import Chunk
from tmg.core.records import RECORDS_DIR


@dataclass
class ChunkSpec:
    """Everything a chunk process needs, written to ``<chunk>/chunk_spec.json``."""
    generator: str
    chunk: dict                     # index, seed, attempts
    folder: str
    records_dir: str
    task: dict
    scene: dict                     # leaf scene with its parent chain inlined under 'parent'
    method: dict
    protocol: dict
    generator_config: str           # path of the official generator's own config file
    expected_outputs: list = field(default_factory=list)   # files that must exist when the chunk finished
    extra: dict = field(default_factory=dict)

    def identity(self) -> dict:
        """What must be equal for a finished chunk to be reused (machine-specific paths excluded)."""
        return {"generator": self.generator, "chunk": self.chunk, "task": _without_paths(self.task),
                "scene": self.scene, "method": _without_paths(self.method), "protocol": self.protocol,
                "extra": self.extra}


def _deep_merge(base: dict, extra: Mapping) -> dict:
    """Copy of ``base`` with ``extra`` merged in; nested mappings merge, other values replace."""
    out = dict(base)
    for k, v in extra.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, Mapping) and isinstance(out.get(k), Mapping) else v
    return out


def apply_method_overrides(method: dict, generator_cfg: Mapping) -> dict:
    """The method config with ``generator_cfg['method_overrides'][method.name]`` merged in.

    Method files (``configs/experiment/method``) hold generator-independent values; a generator lists the few
    values that must differ for it (for example DexMimicGen's 40-degree anchor bound and closure-based hand
    events) under ``method_overrides: {<method name>: {...}}`` in its own config file.
    """
    overrides = (generator_cfg.get("method_overrides") or {}).get(method.get("name"), None) if method else None
    if not overrides:
        return dict(method)
    merged = _deep_merge(method, overrides)
    merged["overrides_from_generator"] = str(generator_cfg.get("name"))
    return merged


def _without_paths(mapping: dict) -> dict:
    """Copy of ``mapping`` without string values that look like paths, so chunk identities compare across
    machines.
    """
    out = {}
    for k, v in mapping.items():
        if isinstance(v, str) and ("/" in v or v.startswith("$")):
            continue
        out[k] = _without_paths(v) if isinstance(v, dict) else v
    return out


class SyntheticDataGenerator(ABC):
    """Base of all generator adapters.

    Class attributes: ``name`` is the generator name (``generator.name`` of its config file); ``entrypoint_module`` is
    the module the default :meth:`chunk_command` runs with ``python -m <module> --spec <path>``.
    """

    name: str = "abstract"
    entrypoint_module: str = ""

    def __init__(self, cfg: dict):
        """``generator`` and ``task`` are always present; ``scene``, ``method`` and ``generation`` only in the
        commands that generate (``sources`` composes none of them). The method config is the generator-independent
        file with the generator's ``method_overrides`` for that method merged in (``apply_method_overrides``)."""
        self.cfg = cfg
        self.generator_cfg = dict(cfg["generator"])
        self.task = dict(cfg["task"])
        self.scene = dict(cfg.get("scene") or {})
        self.method = apply_method_overrides(dict(cfg.get("method") or {}), self.generator_cfg)
        self.generation = dict(cfg.get("generation") or {})
        self.protocol = dict(self.generation.get("protocol") or {})

    def require_generation_config(self) -> None:
        """Raise ``KeyError`` when the command's config lacks the ``scene``, ``method`` or ``generation`` group."""
        for key in ("scene", "method", "generation"):
            if not getattr(self, key):
                raise KeyError(f"this command needs the '{key}' config group (command {self.cfg.get('command')!r} has none)")

    # sources --------------------------------------------------------------------------------------
    def source_path(self) -> Path:
        """Path of the task's source demonstration file (``task.source``)."""
        return Path(str(self.task["source"])).expanduser()

    def check_sources(self) -> list[str]:
        """Problems that stop generation (missing source files), empty when ready."""
        path = self.source_path()
        return [] if path.exists() else [f"source demonstrations missing: {path} (run 'tmg sources')"]

    @abstractmethod
    def fetch_sources(self) -> dict:
        """Download and prepare the source demonstrations; returns a manifest dict."""

    # chunks ---------------------------------------------------------------------------------------
    @abstractmethod
    def prepare_chunk(self, chunk: Chunk, folder: Path) -> ChunkSpec:
        """Write the official config of the chunk into ``folder`` and return the spec."""

    def _spec(self, chunk: Chunk, folder: Path, *, scene: dict, generator_config: str, expected_outputs: list,
              extra: dict, protocol: dict | None = None, generator: str | None = None) -> ChunkSpec:
        """The ``ChunkSpec`` of ``chunk`` in ``folder`` with the fields every adapter fills the same way.

        ``chunk`` becomes ``{"index", "seed", "attempts"}``, ``records_dir`` is ``<folder>/records``, ``task`` and
        ``method`` are the adapter's, ``protocol`` defaults to the adapter's and ``generator`` to the config's
        ``generator.name``. ``extra`` is written as given: its keys and values are part of the chunk identity, so the
        adapter adds ``graft_hashes`` itself with :meth:`graft_hashes`.
        """
        folder = Path(folder)
        return ChunkSpec(
            generator=self.generator_cfg["name"] if generator is None else generator,
            chunk={"index": chunk.index, "seed": chunk.seed, "attempts": chunk.attempts},
            folder=str(folder), records_dir=str(folder / RECORDS_DIR), task=self.task, scene=scene, method=self.method,
            protocol=self.protocol if protocol is None else protocol, generator_config=generator_config,
            expected_outputs=list(expected_outputs), extra=extra)

    def graft_hashes(self) -> dict:
        """``generator.graft_hashes`` as a plain dict (the pinned source hashes the chunk process verifies)."""
        return dict(self.generator_cfg.get("graft_hashes") or {})

    def chunk_command(self, spec: ChunkSpec, spec_path: Path) -> list[str]:
        """Command line of the chunk process: ``python -m <entrypoint_module> --spec <spec_path>``."""
        if not self.entrypoint_module:
            raise NotImplementedError(f"{type(self).__name__} sets no entrypoint_module and overrides no chunk_command")
        return [sys.executable, "-m", self.entrypoint_module, "--spec", str(spec_path)]

    @abstractmethod
    def chunk_result(self, spec: ChunkSpec) -> dict | None:
        """Summary of a finished chunk (attempts, successes, ...) or None when it did not finish."""

    @abstractmethod
    def demo_layout(self) -> dict:
        """How ``tmg dataset`` reads this generator's chunk outputs (the ``tmg.data.dataset.DemoLayout`` fields)."""

    def chunk_environment(self) -> dict:
        """Extra environment variables of a chunk process."""
        return {}
