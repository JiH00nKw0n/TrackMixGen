"""Batch preparation and action post-processing per policy family (``PolicyIO``).

The training loop and the policy server hand every batch through one object with two methods:
``prepare(batch)`` turns a collated batch (uint8 images, raw state and action, optional ``task`` strings) into
what ``policy.forward`` / ``select_action`` expect on the device, and ``postprocess(action)`` turns a
predicted action back into the dataset's units. Two implementations of the ``PolicyIO`` protocol:

- ``NormalizerIO`` (BC-RNN, Diffusion Policy): images to float [0, 1], then ``tmg.policy.normalize.Normalizer``
  in the modes the policy config declares; saved as ``stats.json``.
- ``PipelineIO`` (π0): LeRobot's own pre/post processor pipelines built for the policy config with the dataset
  statistics (``make_pre_post_processors``): batch axis, device, task-string tokenization for the PaliGemma
  tokenizer, mean/std normalization of state and action, and the inverse on the output. Saved with
  ``save_pretrained`` as ``policy_preprocessor.json`` / ``policy_postprocessor.json`` plus safetensors, so a
  checkpoint reloads exactly the pipeline it trained with.

Which one a policy uses is the ``uses_pipeline`` flag of its ``PolicySpec`` (``tmg.policy.lerobot_policies``).
``make_io(policy_name, config, stats, device)`` builds the implementation for a fresh policy; ``load_io(checkpoint_dir,
...)`` restores it next to a saved checkpoint. Both classes share ``build(config, stats, device)`` and
``load(folder, config, device)`` class methods, so the callers need no branches.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from tmg.core.features import TASK
from tmg.core.layout import PIPELINE_DIR, STATS_JSON
from tmg.data.loader import images_to_float, to_device
from tmg.policy.lerobot_policies import POLICY_SPECS, normalization_modes
from tmg.policy.normalize import Normalizer

__all__ = ["STATS_JSON", "PIPELINE_DIR", "PolicyIO", "NormalizerIO", "PipelineIO", "uses_pipeline", "io_class", "make_io",
           "load_io"]


class PolicyIO(Protocol):
    """What the trainer and the server need from a policy family's input/output handling."""

    kind: str

    def prepare(self, batch: dict) -> dict:
        """A collated batch (uint8 images, raw state, optional ``task`` strings) as the policy expects it, on the device."""

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        """A predicted action back in the dataset's units."""

    def save(self, folder: Path) -> None:
        """Write what ``load`` needs into a checkpoint folder."""

    @classmethod
    def build(cls, config, stats: dict, device: str) -> PolicyIO:
        """For a freshly built policy, from the LeRobot config and the dataset statistics (``compute_stats``)."""

    @classmethod
    def load(cls, folder: Path, config, device: str) -> PolicyIO:
        """From a checkpoint folder written by ``save``; ``config`` is the policy's LeRobot config."""


class NormalizerIO:
    """Our normalizer around the batch (see the module docstring)."""

    kind = "normalizer"

    def __init__(self, normalizer: Normalizer, device: str):
        self.normalizer = normalizer
        self.device = device

    @classmethod
    def build(cls, config, stats: dict, device: str) -> NormalizerIO:
        return cls(Normalizer(stats, normalization_modes(config)), device)

    def prepare(self, batch: dict) -> dict:
        batch = {k: v for k, v in batch.items() if k != TASK}
        return self.normalizer(to_device(batch, self.device))

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return self.normalizer.unnormalize_action(action)

    def save(self, folder: Path) -> None:
        self.normalizer.save(Path(folder) / STATS_JSON)

    @classmethod
    def load(cls, folder: Path, config, device: str) -> NormalizerIO:
        """``config`` is unused: the modes were saved with the statistics."""
        return cls(Normalizer.load(Path(folder) / STATS_JSON), device)


class PipelineIO:
    """LeRobot processor pipelines (π0): tokenization, device placement and normalization in one pass."""

    kind = "pipeline"

    def __init__(self, preprocessor, postprocessor, device: str):
        self.pre, self.post, self.device = preprocessor, postprocessor, device

    @classmethod
    def build(cls, config, stats: dict, device: str) -> PipelineIO:
        from lerobot.processor.factory import make_pre_post_processors

        tensor_stats = {k: {s: torch.as_tensor(np.asarray(v), dtype=torch.float32) for s, v in d.items()} for k, d in stats.items()}
        pre, post = make_pre_post_processors(config, dataset_stats=tensor_stats,
                                             preprocessor_overrides={"device_processor": {"device": device}})
        return cls(pre, post, device)

    def prepare(self, batch: dict) -> dict:
        return self.pre(images_to_float(batch))

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return self.post(action)

    def save(self, folder: Path) -> None:
        folder = Path(folder) / PIPELINE_DIR
        folder.mkdir(parents=True, exist_ok=True)
        self.pre.save_pretrained(str(folder))
        self.post.save_pretrained(str(folder))

    @classmethod
    def load(cls, folder: Path, config, device: str) -> PipelineIO:
        from lerobot.processor.factory import make_pre_post_processors

        pre, post = make_pre_post_processors(config, pretrained_path=str(Path(folder) / PIPELINE_DIR),
                                             preprocessor_overrides={"device_processor": {"device": device}})
        return cls(pre, post, device)


def uses_pipeline(policy_name: str) -> bool:
    """Whether the policy's spec asks for LeRobot's pipelines (π0: the tokenizer); the others use our normalizer.
    An unknown name answers False, as before the specs existed."""
    spec = POLICY_SPECS.get(policy_name)
    return bool(spec is not None and spec.uses_pipeline)


def io_class(policy_name: str) -> type[PolicyIO]:
    """The ``PolicyIO`` implementation of a policy name."""
    return PipelineIO if uses_pipeline(policy_name) else NormalizerIO


def make_io(policy_name: str, config, stats: dict, device: str) -> PolicyIO:
    """The ``PolicyIO`` of a freshly built policy."""
    return io_class(policy_name).build(config, stats, device)


def load_io(checkpoint_dir: Path, policy_name: str, config, device: str) -> PolicyIO:
    """The ``PolicyIO`` saved next to a checkpoint."""
    return io_class(policy_name).load(Path(checkpoint_dir), config, device)
