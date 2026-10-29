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
The run folder's file names are ``tmg.core.layout`` (re-exported here for the callers that import them from the trainer).
"""
from __future__ import annotations

import random
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from tmg.config import dump_yaml, resolved
from tmg.core.io_json import append_jsonl, read_jsonl, write_json
from tmg.core.layout import (  # noqa: F401  re-exported: server, tests and scripts import the names from here
    CHECKPOINT_JSON,
    CHECKPOINTS_DIR,
    PRETRAINED_DIR,
    RESOLVED_CONFIG,
    SELECTED_JSON,
    TRAIN_LOG,
    TRAIN_RESULT,
    TRAINING_STATE,
    VALIDATION_JSONL,
    checkpoint_dir,
)
from tmg.core.provenance import write_provenance
from tmg.core.seeding import seed_all
from tmg.data.loader import FrameDataset, TrajectoryStore, collate, compute_stats
from tmg.policy.ema import EMA
from tmg.policy.io import PolicyIO, make_io
from tmg.policy.lerobot_policies import build_policy, task_policy_overrides


def seed_everything(seed: int) -> None:
    """``random``, numpy and torch from one seed (``tmg.core.seeding.seed_all``)."""
    seed_all(seed)


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


def save_checkpoint(run_dir: Path, *, update: int, epoch: int, policy, ema: EMA | None, io: PolicyIO,
                    optimizer, scheduler, loss: float) -> Path:
    """Write ``checkpoints/<update>/`` (layout in the module docstring) and return its folder."""
    folder = checkpoint_dir(run_dir, update)
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


@dataclass
class TrainSetup:
    """Everything ``train`` needs once ``_setup`` has built it (in the order the module docstring lists)."""

    plain: dict
    run_dir: Path
    device: str
    training: dict
    policy_cfg: dict
    observation: str
    store: TrajectoryStore
    features: dict
    policy: object
    io: PolicyIO
    dataset: FrameDataset
    batch_size: int
    updates: int
    optimizer: object
    scheduler: object
    ema: EMA | None
    clip: float                    # gradient-norm clip; 0 disables
    per_epoch: int                 # updates per logging epoch
    ckpt_every: int                # updates between checkpoints
    use_amp: bool
    scaler: object                 # torch.amp.GradScaler
    validation_cmd: str | None
    start: int                     # update count resumed from (0 for a fresh run)
    checkpoints: list[str] = field(default_factory=list)   # checkpoint folders so far (resumed ones included)


def _setup(cfg) -> TrainSetup:
    """Steps 1, 2 and the preparation of 3 to 6: run folder, seeds, dataset, policy, ``PolicyIO``, optimizer, EMA, resume."""
    plain = resolved(cfg)
    run_dir = Path(plain["output"])
    run_dir.mkdir(parents=True, exist_ok=True)
    dump_yaml(plain, run_dir / RESOLVED_CONFIG)
    write_provenance(run_dir, extra={"command": "train", "output": str(run_dir)})
    training, policy_cfg = plain["training"], plain["policy"]
    device = str(plain.get("device", "cuda"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device=cuda but no GPU is visible")
    torch.set_num_threads(int(training.get("num_threads", 8)))
    seed_all(int(plain["seed"]))

    observation = str(plain["observation"]["name"])
    store = TrajectoryStore.load(Path(plain["dataset"]["path"]), action_key=str(training.get("action_key", "actions")))
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
    return TrainSetup(plain=plain, run_dir=run_dir, device=device, training=training, policy_cfg=policy_cfg,
                      observation=observation, store=store, features=features, policy=policy, io=io, dataset=dataset,
                      batch_size=batch_size, updates=updates, optimizer=optimizer, scheduler=scheduler, ema=ema, clip=clip,
                      per_epoch=per_epoch, ckpt_every=ckpt_every, use_amp=use_amp, scaler=scaler, validation_cmd=validation_cmd,
                      start=start, checkpoints=[str(p) for p in existing])


def _train_step(s: TrainSetup, batch: dict) -> tuple[float, float]:
    """One optimizer update on a prepared batch: forward (mixed precision when enabled), backward, gradient-norm clip,
    step, scheduler and EMA. Returns ``(loss, gradient norm before clipping)``."""
    with torch.autocast(device_type="cuda", enabled=s.use_amp):
        loss, _ = s.policy.forward(batch)
    s.optimizer.zero_grad(set_to_none=True)
    s.scaler.scale(loss).backward()
    s.scaler.unscale_(s.optimizer)
    norm = float(torch.nn.utils.clip_grad_norm_(s.policy.parameters(), s.clip if s.clip > 0 else float("inf")))
    s.scaler.step(s.optimizer)
    s.scaler.update()
    if s.scheduler is not None:
        s.scheduler.step()
    if s.ema is not None:
        s.ema.update(s.policy)
    return float(loss.detach()), norm


class EpochLog:
    """Running statistics of the current logging epoch, written as one ``train_log.jsonl`` row at its end."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.losses, self.norms, self.clipped, self.t0, self.examples = [], [], 0, time.time(), 0

    def add(self, loss: float, norm: float, *, clipped: bool, examples: int) -> None:
        self.losses.append(loss)
        self.norms.append(norm)
        self.clipped += int(clipped)
        self.examples += examples

    def row(self, *, epoch: int, update: int, lr: float, ema_decay: float | None) -> dict:
        seconds = time.time() - self.t0
        return {"epoch": epoch, "update": update, "loss": float(np.mean(self.losses)),
                "grad_norm_mean": float(np.mean(self.norms)), "grad_norm_max": float(np.max(self.norms)),
                "clipped_fraction": self.clipped / max(len(self.norms), 1),
                "lr": lr, "seconds": round(seconds, 2),
                "examples_per_s": round(self.examples / max(seconds, 1e-9), 1), "ema_decay": ema_decay}


def train(cfg) -> dict:
    """Run the loop described in the module docstring.

    Args:
        cfg: the composed ``train`` config.

    Returns:
        The summary written to ``train_result.json``.
    """
    s = _setup(cfg)
    rng = np.random.default_rng(int(s.plain["seed"]) + s.start)
    batches = s.dataset.batches(s.batch_size, rng)
    log_path = s.run_dir / TRAIN_LOG
    epoch_log = EpochLog()
    update = s.start
    while update < s.updates:
        idx = next(batches)
        batch = s.io.prepare(collate([s.dataset[int(i)] for i in idx]))
        loss, norm = _train_step(s, batch)
        update += 1
        epoch_log.add(loss, norm, clipped=s.clip > 0 and norm > s.clip, examples=len(idx))
        if update % s.per_epoch == 0 or update == s.updates:
            epoch = update // s.per_epoch
            row = epoch_log.row(epoch=epoch, update=update, lr=float(s.optimizer.param_groups[0]["lr"]),
                                ema_decay=s.ema.decay() if s.ema else None)
            append_jsonl(log_path, row)
            print(f"[tmg] epoch {epoch} update {update} loss {row['loss']:.4f} grad {row['grad_norm_mean']:.1f}/{row['grad_norm_max']:.1f} "
                  f"clipped {row['clipped_fraction']:.2f} lr {row['lr']:.2e} {row['examples_per_s']} ex/s", flush=True)
            epoch_log.reset()
            if update % s.ckpt_every == 0 or update == s.updates:
                folder = save_checkpoint(s.run_dir, update=update, epoch=epoch, policy=s.policy, ema=s.ema, io=s.io,
                                         optimizer=s.optimizer, scheduler=s.scheduler, loss=row["loss"])
                s.checkpoints.append(str(folder))
                print(f"[tmg] checkpoint {folder}", flush=True)
                if s.validation_cmd:
                    run_validation_command(str(s.validation_cmd), run_dir=s.run_dir, checkpoint=folder)
    result = {"policy": s.policy_cfg["name"], "observation": s.observation, "updates": s.updates, "batch_size": s.batch_size,
              "checkpoints": s.checkpoints, "dataset": s.plain["dataset"]["path"], "demos": s.store.num_demos,
              "frames": s.store.num_frames,
              "features": {k: (list(v) if isinstance(v, tuple) else v) for k, v in s.features.items()},
              "selection": s.training.get("selection", "best_validation")}
    write_json(s.run_dir / TRAIN_RESULT, result)
    print(f"[tmg] training done: {len(s.checkpoints)} checkpoints in {s.run_dir}", flush=True)
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
        rows = {Path(r["checkpoint"]).name: float(r["success_rate"]) for r in validation_rows(run_dir)}
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


def select_checkpoint_cmd(cfg) -> dict:
    """``tmg select_checkpoint``: ``select_checkpoint`` of the composed config's run folder (``output``) with its
    ``training.selection`` rule."""
    plain = resolved(cfg)
    return select_checkpoint(plain["output"], str(plain["training"].get("selection", "best_validation")))


def validation_rows(run_dir: Path) -> list[dict]:
    """Rows of ``validation.jsonl`` (``make validate`` appends one per evaluated checkpoint; empty when absent)."""
    return read_jsonl(Path(run_dir) / VALIDATION_JSONL)
