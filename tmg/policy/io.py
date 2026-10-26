"""Batch preparation and action post-processing per policy family (``PolicyIO``).

The training loop and the policy server hand every batch through one object with two methods:
``prepare(batch)`` turns a collated batch (uint8 images, raw state and action, optional ``task`` strings) into
what ``policy.forward`` / ``select_action`` expect on the device, and ``postprocess(action)`` turns a
predicted action back into the dataset's units. Two implementations:

- ``NormalizerIO`` (BC-RNN, Diffusion Policy): images to float [0, 1], then ``tmg.policy.normalize.Normalizer``
  in the modes the policy config declares; saved as ``stats.json``.
- ``PipelineIO`` (π0): LeRobot's own pre/post processor pipelines built for the policy config with the dataset
  statistics (``make_pre_post_processors``): batch axis, device, task-string tokenization for the PaliGemma
  tokenizer, mean/std normalization of state and action, and the inverse on the output. Saved with
  ``save_pretrained`` as ``policy_preprocessor.json`` / ``policy_postprocessor.json`` plus safetensors, so a
  checkpoint reloads exactly the pipeline it trained with.

``make_io(policy_name, config, stats, device)`` picks the implementation; ``load_io(checkpoint_dir, ...)``
restores it next to a saved checkpoint.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tmg.data.loader import to_device
from tmg.policy.lerobot_policies import normalization_modes
from tmg.policy.normalize import Normalizer

STATS_JSON = "stats.json"
PIPELINE_DIR = "processors"


class NormalizerIO:
    """Our normalizer around the batch (see the module docstring)."""

    kind = "normalizer"

    def __init__(self, normalizer: Normalizer, device: str):
        self.normalizer = normalizer
        self.device = device

    def prepare(self, batch: dict) -> dict:
        batch = {k: v for k, v in batch.items() if k != "task"}
        return self.normalizer(to_device(batch, self.device))

    def postprocess(self, action: torch.Tensor) -> torch.Tensor:
        return self.normalizer.unnormalize_action(action)

    def save(self, folder: Path) -> None:
        self.normalizer.save(Path(folder) / STATS_JSON)

    @classmethod
    def load(cls, folder: Path, device: str) -> NormalizerIO:
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
        batch = dict(batch)
        for key, v in list(batch.items()):
            if key.startswith("observation.images.") and isinstance(v, torch.Tensor) and v.dtype == torch.uint8:
                batch[key] = v.float() / 255.0
        return self.pre(batch)

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
    """π0 needs LeRobot's pipelines (tokenizer); the others use our normalizer."""
    return policy_name == "pi0"


def make_io(policy_name: str, config, stats: dict, device: str):
    """The ``PolicyIO`` of a freshly built policy."""
    if uses_pipeline(policy_name):
        return PipelineIO.build(config, stats, device)
    return NormalizerIO(Normalizer(stats, normalization_modes(config)), device)


def load_io(checkpoint_dir: Path, policy_name: str, config, device: str):
    """The ``PolicyIO`` saved next to a checkpoint."""
    if uses_pipeline(policy_name):
        return PipelineIO.load(Path(checkpoint_dir), config, device)
    return NormalizerIO.load(Path(checkpoint_dir), device)
