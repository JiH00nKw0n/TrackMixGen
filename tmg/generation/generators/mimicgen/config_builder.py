"""MimicGen generation config (json) from the official task template and the generation protocol."""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path

PROTOCOL_KEYS = ("keep_failed", "max_num_failures", "guarantee", "select_src_per_subtask", "selection_strategy",
                 "selection_strategy_kwargs", "action_noise", "interpolation_steps", "num_fixed_steps", "collect_obs",
                 "camera_names", "camera_size", "render_video", "num_source_demos")
SUBTASK_PREFIX = "subtask_"


def experiment_name(env_name: str, attempts: int, seed: int) -> str:
    """MimicGen experiment folder name of a chunk: ``<env>_n<attempts>_seed<seed>``."""
    return f"{env_name}_n{int(attempts)}_seed{int(seed)}"


def validate_protocol(protocol: Mapping) -> dict:
    """Check that a ``generation.protocol`` mapping has every key and a usable ``camera_size``.

    Raises:
        KeyError: a protocol key is missing.
        ValueError: ``guarantee`` is true (chunks count attempts) or ``camera_size`` is not ``[height, width]``.
    """
    protocol = dict(protocol)
    missing = [k for k in PROTOCOL_KEYS if k not in protocol]
    if missing:
        raise KeyError(f"generation.protocol lacks {missing}")
    if protocol["guarantee"]:
        raise ValueError("generation.protocol.guarantee=true makes MimicGen count successes instead of attempts; "
                         "chunks are planned by attempt count, so guarantee must be false")
    size = protocol["camera_size"]
    if size is not None and len(list(size)) != 2:
        raise ValueError(f"generation.protocol.camera_size must be [height, width], got {size}")
    return protocol


def build_generation_config(template: Mapping, protocol: Mapping, *, env_name: str, source_dataset: str,
                            output_folder: str, attempts: int, seed: int) -> dict:
    """Apply a protocol to a MimicGen task template.

    Args:
        template: the official template json (``exps/templates/robosuite/<task>.json``), left unmodified.
        protocol: the ``generation.protocol`` mapping (attempt count policy, source selection, noise, stitch
            steps, observations, rendering).
        env_name: registered environment class to generate in.
        source_dataset: path of the prepared source hdf5.
        output_folder: folder MimicGen writes its experiment folder into.
        attempts: ``num_trials`` (attempts, since ``guarantee`` is false).
        seed: MimicGen's seed for this chunk.

    Returns:
        A new config dict ready to be written as ``mg_config.json``.
    """
    if attempts < 1:
        raise ValueError("attempts must be >= 1")
    protocol = validate_protocol(protocol)
    config = copy.deepcopy(dict(template))
    experiment = config["experiment"]
    experiment["name"] = experiment_name(env_name, attempts, seed)
    experiment["render_video"] = bool(protocol["render_video"])
    if not protocol["render_video"]:
        experiment["num_demo_to_render"] = 0
        experiment["num_fail_demo_to_render"] = 0
    experiment["seed"] = int(seed)
    experiment["max_num_failures"] = protocol["max_num_failures"]
    experiment["source"]["dataset_path"] = str(source_dataset)
    experiment["source"]["n"] = protocol["num_source_demos"]
    generation = experiment["generation"]
    generation["path"] = str(output_folder)
    generation["num_trials"] = int(attempts)
    generation["guarantee"] = bool(protocol["guarantee"])
    generation["keep_failed"] = bool(protocol["keep_failed"])
    generation["select_src_per_subtask"] = bool(protocol["select_src_per_subtask"])
    experiment["task"]["name"] = env_name
    config["obs"]["collect_obs"] = bool(protocol["collect_obs"])
    config["obs"]["camera_names"] = list(protocol["camera_names"] or [])
    if protocol["camera_size"] is not None:
        height, width = (int(v) for v in protocol["camera_size"])
        config["obs"]["camera_height"] = height
        config["obs"]["camera_width"] = width
    kwargs = protocol["selection_strategy_kwargs"]
    for key, subtask in config["task"]["task_spec"].items():
        if not key.startswith(SUBTASK_PREFIX):
            continue
        subtask["selection_strategy"] = str(protocol["selection_strategy"])
        subtask["selection_strategy_kwargs"] = None if kwargs is None else dict(kwargs)
        if protocol["action_noise"] is not None:
            subtask["action_noise"] = float(protocol["action_noise"])
        if protocol["interpolation_steps"] is not None:
            subtask["num_interpolation_steps"] = int(protocol["interpolation_steps"])
        if protocol["num_fixed_steps"] is not None:
            subtask["num_fixed_steps"] = int(protocol["num_fixed_steps"])
        if subtask.get("object_ref") is None and subtask["selection_strategy"] != "random":
            raise ValueError(f"{key} has no reference object; MimicGen supports only 'random' selection there")
    return config


def save_config(config: dict, path: Path) -> Path:
    """Write a config dict as ``mg_config.json``-style JSON (indent 4) and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=4))
    return path
