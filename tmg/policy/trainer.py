"""``tmg train``: the one training loop for every policy backend.

What the loop does, in order:

1. Load the dataset into memory (``tmg.data.loader``), build the policy from the ``policy`` config and the
   dataset features (``tmg.policy.lerobot_policies``), and derive the delta indices for temporal stacking
   from the policy config, so the batches always match the model.
2. Prepare batches through the policy family's ``PolicyIO`` (``tmg.policy.io``): our normalizer with this
   dataset's statistics for BC-RNN and Diffusion Policy, LeRobot's processor pipelines (tokenizer, normalization)
   for π0; what it needs is saved with every checkpoint.
3. Optimize for ``updates`` optimizer steps (the policy's budget from ``training.by_policy``, or the
   ``updates`` override). An epoch is ``updates_per_epoch`` (500) updates and is only a logging unit: at
   the end of each epoch ``train_log.jsonl`` gets the mean loss, the gradient norm before clipping (mean and
   max), the share of clipped batches, the learning rate and the throughput.
4. Clip the gradient norm (L2) at ``grad_clip_norm`` (1000) before every step; keep an EMA copy of the
   weights when the policy config asks for it (Diffusion Policy), and evaluate that copy.
5. Every ``checkpoint_every_epochs`` epochs write ``checkpoints/<update>/`` with ``pretrained_model/``
   (LeRobot layout: ``config.json`` + ``model.safetensors``, EMA weights when enabled), the ``PolicyIO`` files
   (``stats.json`` or ``processors/``) and ``training_state.pt`` (raw weights, optimizer, scheduler, EMA, RNG, update count) for resume.
   If ``training.validation.command`` is set, run it with ``{run_dir}`` and ``{checkpoint}`` substituted
   (a host-side command that launches the policy server and the rollout evaluation); otherwise the
   checkpoints are validated afterwards (``make validate``) and selected with ``tmg select_checkpoint``.
6. Resume from the newest checkpoint of the run folder when one exists; stop when the budget is reached and
   write ``train_result.json``.

The loop never imports a simulator; validation rollouts run in the simulator image through the policy server.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from tmg.config import dump_yaml, resolved
from tmg.core.chunks import read_json, write_json
from tmg.core.provenance import write_provenance
from tmg.data.loader import FrameDataset, TrajectoryStore, collate, compute_stats
from tmg.policy.ema import EMA
from tmg.policy.storage import prune_training_states
from tmg.policy.io import make_io
from tmg.policy.lerobot_policies import build_policy, task_policy_overrides

CHECKPOINTS_DIR = "checkpoints"
PRETRAINED_DIR = "pretrained_model"
TRAINING_STATE = "training_state.pt"
CHECKPOINT_JSON = "checkpoint.json"
TRAIN_LOG = "train_log.jsonl"
TRAIN_RESULT = "train_result.json"
VALIDATION_JSONL = "validation.jsonl"
SELECTED_JSON = "SELECTED.json"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dataset_features(store: TrajectoryStore, observation: str = "image") -> dict:
    """``{"cameras", "image_shape": (3, H, W) or None, "state_dim", "env_state_dim", "action_dim"}`` of a loaded dataset.

    Raises:
        ValueError: the dataset does not hold what the observation choice needs (no images for ``image``, no object
            state for ``low_dim``), which means it was written with another ``observation``.
    """
    if observation == "image" and store.image_shape is None:
        raise ValueError("the dataset has no camera images; observation=image needs a dataset written with observation=image")
    if observation == "low_dim" and store.env_state_dim == 0:
        raise ValueError("the dataset has no object state; observation=low_dim needs a dataset written with observation=low_dim")
    shape = None
    if store.image_shape is not None:
        h, w, c = store.image_shape
        shape = (c, h, w)
    return {"cameras": list(store.cameras), "image_shape": shape, "state_dim": store.state_dim,
            "env_state_dim": store.env_state_dim, "action_dim": store.action_dim}


def policy_budget(training: dict, policy_name: str, observation: str, override) -> tuple[int, int]:
    """``(batch_size, updates)`` from ``training.by_policy[policy][observation]``, with the ``updates`` override."""
    block = ((training.get("by_policy") or {}).get(policy_name) or {}).get(observation)
    if block is None:
        raise KeyError(f"training.by_policy has no entry for policy {policy_name!r} and observation {observation!r}")
    updates = int(override) if override is not None else int(block["updates"])
    return int(block["batch_size"]), updates


def list_checkpoints(run_dir: Path) -> list[Path]:
    """Checkpoint folders of a run, oldest first (folder name = zero-padded update count)."""
    root = Path(run_dir) / CHECKPOINTS_DIR
    if not root.exists():
        return []
    return sorted((p for p in root.iterdir() if p.is_dir() and (p / CHECKPOINT_JSON).exists()), key=lambda p: int(p.name))


def save_checkpoint(run_dir: Path, *, update: int, epoch: int, policy, ema: EMA | None, io,
                    optimizer, scheduler, loss: float) -> Path:
    """Write ``checkpoints/<update>/`` (layout in the module docstring) and return its folder."""
    folder = Path(run_dir) / CHECKPOINTS_DIR / f"{update:07d}"
    tmp = folder.with_name(folder.name + ".part")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    to_save = ema.model if ema is not None else policy
    to_save.save_pretrained(str(tmp / PRETRAINED_DIR))
    io.save(tmp)
    torch.save({"update": update, "epoch": epoch, "model": policy.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "ema": ema.state_dict() if ema is not None else None,
                "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}},
               tmp / TRAINING_STATE)
    write_json(tmp / CHECKPOINT_JSON, {"update": update, "epoch": epoch, "loss": loss, "ema": ema is not None})
    if folder.exists():
        shutil.rmtree(folder)
    tmp.replace(folder)
    return folder


def load_training_state(folder: Path, *, policy, ema: EMA | None, optimizer, scheduler) -> int:
    """Restore the loop state saved by ``save_checkpoint``; returns the update count to continue from."""
    state = torch.load(folder / TRAINING_STATE, map_location="cpu", weights_only=False)
    policy.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler") is not None:
        scheduler.load_state_dict(state["scheduler"])
    if ema is not None and state.get("ema") is not None:
        ema.load_state_dict(state["ema"])
    torch.set_rng_state(state["rng"]["torch"])
    np.random.set_state(state["rng"]["numpy"])
    random.setstate(state["rng"]["python"])
    return int(state["update"])


def run_validation_command(template: str, *, run_dir: Path, checkpoint: Path) -> int:
    """Run the host-side validation command for one checkpoint; returns its exit code (logged, not fatal)."""
    cmd = template.format(run_dir=str(run_dir), checkpoint=str(checkpoint))
    print(f"[tmg] validation: {cmd}", flush=True)
    return subprocess.run(cmd, shell=True).returncode


def train(cfg) -> dict:
    """Run the loop described in the module docstring.

    Args:
        cfg: the composed ``train`` config.

    Returns:
        The summary written to ``train_result.json``.
    """
    plain = resolved(cfg)
    run_dir = Path(plain["output"])
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(plain, run_dir / "resolved_config.yaml")
    write_provenance(run_dir, extra={"command": "train", "output": str(run_dir)})
    training, policy_cfg = plain["training"], plain["policy"]
    device = str(plain.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device=cuda but no GPU is visible")
    torch.set_num_threads(int(training.get("num_threads", 8)))
    seed_everything(int(plain["seed"]))

    observation = str(plain["observation"]["name"])
    store = TrajectoryStore.load(Path(plain["dataset"]["path"]), action_key=str(training.get("action_key", "actions")),
                                 image_cache=str(training.get("cache", "ram")))
    features = dataset_features(store, observation)
    overrides = task_policy_overrides(plain["task"], str(policy_cfg["name"]))
    if overrides:
        print(f"[tmg] task policy overrides ({plain['task']['name']}): {overrides}", flush=True)
    policy = build_policy(policy_cfg, features, device, observation, overrides)
    policy.train()
    config = policy.config
    io = make_io(str(policy_cfg["name"]), config, compute_stats(store), device)
    dataset = FrameDataset(store, obs_delta=config.observation_delta_indices, action_delta=config.action_delta_indices or [0],
                           drop_n_last_frames=int(getattr(config, "drop_n_last_frames", 0) or 0),
                           task=str(plain["task"].get("instruction")) if plain["task"].get("instruction") else None)
    batch_size, updates = policy_budget(training, str(policy_cfg["name"]), observation, plain.get("updates"))
    optimizer = config.get_optimizer_preset().build(policy.get_optim_params())
    scheduler_preset = config.get_scheduler_preset()
    scheduler = scheduler_preset.build(optimizer, updates) if scheduler_preset is not None else None
    ema = EMA(policy) if policy_cfg.get("ema", False) else None
    clip = float(training.get("grad_clip_norm", 0) or 0)
    per_epoch = int(training["updates_per_epoch"])
    ckpt_every = int(training["checkpoint_every_epochs"]) * per_epoch
    use_amp = bool(training.get("amp", False)) and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    validation_cmd = (training.get("validation") or {}).get("command")

    start = 0
    existing = list_checkpoints(run_dir)
    if existing:
        start = load_training_state(existing[-1], policy=policy, ema=ema, optimizer=optimizer, scheduler=scheduler)
        print(f"[tmg] resumed from {existing[-1]} at update {start}", flush=True)
    print(f"[tmg] train {policy_cfg['name']} on {store.num_demos} demos / {store.num_frames} frames, batch {batch_size}, "
          f"updates {start}->{updates}, device {device}, ema {ema is not None}, clip {clip}", flush=True)

    rng = np.random.default_rng(int(plain["seed"]) + start)
    batches = dataset.batches(batch_size, rng)
    log_path = run_dir / TRAIN_LOG
    epoch_losses, epoch_norms, epoch_clipped, t_epoch, n_examples = [], [], 0, time.time(), 0
    checkpoints = [str(p) for p in existing]
    update = start
    while update < updates:
        idx = next(batches)
        batch = io.prepare(collate([dataset[int(i)] for i in idx]))
        with torch.autocast(device_type="cuda", enabled=use_amp):
            loss, _ = policy.forward(batch)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), clip if clip > 0 else float("inf")))
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        if ema is not None:
            ema.update(policy)
        update += 1
        n_examples += len(idx)
        epoch_losses.append(float(loss.detach()))
        epoch_norms.append(norm)
        epoch_clipped += int(clip > 0 and norm > clip)
        if update % per_epoch == 0 or update == updates:
            epoch = update // per_epoch
            seconds = time.time() - t_epoch
            row = {"epoch": epoch, "update": update, "loss": float(np.mean(epoch_losses)),
                   "grad_norm_mean": float(np.mean(epoch_norms)), "grad_norm_max": float(np.max(epoch_norms)),
                   "clipped_fraction": epoch_clipped / max(len(epoch_norms), 1),
                   "lr": float(optimizer.param_groups[0]["lr"]), "seconds": round(seconds, 2),
                   "examples_per_s": round(n_examples / max(seconds, 1e-9), 1), "ema_decay": ema.decay() if ema else None}
            with open(log_path, "a") as f:
                f.write(json.dumps(row) + "\n")
            print(f"[tmg] epoch {epoch} update {update} loss {row['loss']:.4f} grad {row['grad_norm_mean']:.1f}/{row['grad_norm_max']:.1f} "
                  f"clipped {row['clipped_fraction']:.2f} lr {row['lr']:.2e} {row['examples_per_s']} ex/s", flush=True)
            epoch_losses, epoch_norms, epoch_clipped, t_epoch, n_examples = [], [], 0, time.time(), 0
            if update % ckpt_every == 0 or update == updates:
                folder = save_checkpoint(run_dir, update=update, epoch=epoch, policy=policy, ema=ema, io=io,
                                         optimizer=optimizer, scheduler=scheduler, loss=row["loss"])
                checkpoints.append(str(folder))
                if training.get("keep_training_states", "all") == "last":
                    prune_training_states(run_dir, folder)
                print(f"[tmg] checkpoint {folder}", flush=True)
                if validation_cmd:
                    run_validation_command(str(validation_cmd), run_dir=run_dir, checkpoint=folder)
    result = {"policy": policy_cfg["name"], "observation": observation, "updates": updates, "batch_size": batch_size, "checkpoints": checkpoints,
              "dataset": plain["dataset"]["path"], "demos": store.num_demos, "frames": store.num_frames,
              "features": {k: (list(v) if isinstance(v, tuple) else v) for k, v in features.items()},
              "selection": training.get("selection", "best_validation")}
    write_json(run_dir / TRAIN_RESULT, result)
    print(f"[tmg] training done: {len(checkpoints)} checkpoints in {run_dir}", flush=True)
    return result


def select_checkpoint(run_dir: Path, rule: str = "best_validation") -> dict:
    """Choose the checkpoint of a run and write ``SELECTED.json``.

    Args:
        run_dir: the run folder.
        rule: ``best_validation`` (highest validation success rate in ``validation.jsonl``, earliest update on
            ties; every checkpoint must have a row) or ``last`` (the newest checkpoint).

    Returns:
        ``{"checkpoint": path, "rule": ..., "success_rate": ... or None}``.
    """
    run_dir = Path(run_dir)
    folders = list_checkpoints(run_dir)
    if not folders:
        raise FileNotFoundError(f"no checkpoints under {run_dir}")
    if rule == "last":
        chosen, rate = folders[-1], None
    elif rule == "best_validation":
        path = run_dir / VALIDATION_JSONL
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: validate the checkpoints first (make validate)")
        rows = {}
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                rows[Path(r["checkpoint"]).name] = float(r["success_rate"])
        missing = [f.name for f in folders if f.name not in rows]
        if missing:
            raise ValueError(f"checkpoints without a validation row: {missing}")
        chosen = max(folders, key=lambda f: (rows[f.name], -int(f.name)))
        rate = rows[chosen.name]
    else:
        raise ValueError(f"unknown selection rule {rule!r}")
    out = {"checkpoint": str(chosen), "rule": rule, "success_rate": rate}
    write_json(run_dir / SELECTED_JSON, out)
    print(f"[tmg] selected {chosen.name} ({rule}, success {rate})")
    return out


def validation_rows(run_dir: Path) -> list[dict]:
    """Rows of ``validation.jsonl`` (``make validate`` appends one per evaluated checkpoint)."""
    path = Path(run_dir) / VALIDATION_JSONL
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checkpoint_meta(folder: Path) -> dict:
    return read_json(Path(folder) / CHECKPOINT_JSON)
