"""Trainer units that need torch (skipped without it): the epoch log row, the command wrapper of ``select_checkpoint``,
the layout names re-exported from the trainer, and the seeding helper."""
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from tmg.config import compose  # noqa: E402  (after the torch skip)
from tmg.core import layout  # noqa: E402
from tmg.policy import trainer  # noqa: E402


def test_epoch_log_row_has_the_train_log_fields():
    log = trainer.EpochLog()
    log.add(1.0, 10.0, clipped=False, examples=4)
    log.add(3.0, 30.0, clipped=True, examples=4)
    row = log.row(epoch=1, update=500, lr=1e-4, ema_decay=None)
    assert list(row) == ["epoch", "update", "loss", "grad_norm_mean", "grad_norm_max", "clipped_fraction", "lr", "seconds",
                         "examples_per_s", "ema_decay"]
    assert row["loss"] == 2.0 and row["grad_norm_mean"] == 20.0 and row["grad_norm_max"] == 30.0
    assert row["clipped_fraction"] == 0.5
    log.reset()
    assert log.losses == [] and log.clipped == 0 and log.examples == 0


def test_select_checkpoint_cmd_uses_output_and_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    cfg = compose("select_checkpoint", ["task=square", "training.selection=last"])
    run_dir = tmp_path / "runs/mimicgen/square/ours/official/none/pool_robosuite_image/image/bc_rnn/seed1"
    assert Path(cfg.output) == run_dir
    for u in (500, 1000):
        d = layout.checkpoint_dir(run_dir, u)
        d.mkdir(parents=True)
        (d / layout.CHECKPOINT_JSON).write_text(json.dumps({"update": u}))
    out = trainer.select_checkpoint_cmd(cfg)
    assert out["checkpoint"].endswith("0001000") and out["rule"] == "last"
    assert json.loads((run_dir / layout.SELECTED_JSON).read_text()) == out
    assert trainer.validation_rows(run_dir) == []
    # without the override the rule is the config's default, best_validation, which needs validation.jsonl
    with pytest.raises(FileNotFoundError, match="validation.jsonl"):
        trainer.select_checkpoint_cmd(compose("select_checkpoint", ["task=square"]))
    rows = [{"checkpoint": str(layout.checkpoint_dir(run_dir, u)), "success_rate": r} for u, r in ((500, 0.5), (1000, 0.25))]
    (run_dir / layout.VALIDATION_JSONL).write_text("".join(json.dumps(r) + "\n" for r in rows))
    best = trainer.select_checkpoint_cmd(compose("select_checkpoint", ["task=square"]))
    assert best == {"checkpoint": str(layout.checkpoint_dir(run_dir, 500)), "rule": "best_validation", "success_rate": 0.5}


def test_trainer_reexports_the_layout_names():
    for name in ("CHECKPOINTS_DIR", "PRETRAINED_DIR", "TRAINING_STATE", "CHECKPOINT_JSON", "TRAIN_LOG", "TRAIN_RESULT",
                 "VALIDATION_JSONL", "SELECTED_JSON"):
        assert getattr(trainer, name) == getattr(layout, name)
    # the committed trainer's literal names (server, scripts and shell drivers rely on them)
    assert (trainer.CHECKPOINTS_DIR, trainer.PRETRAINED_DIR, trainer.TRAINING_STATE, trainer.CHECKPOINT_JSON) == \
        ("checkpoints", "pretrained_model", "training_state.pt", "checkpoint.json")
    assert (trainer.TRAIN_LOG, trainer.TRAIN_RESULT, trainer.VALIDATION_JSONL, trainer.SELECTED_JSON) == \
        ("train_log.jsonl", "train_result.json", "validation.jsonl", "SELECTED.json")


def test_seed_all_seeds_every_source_and_the_env():
    import random

    import numpy as np

    from tmg.core.seeding import seed_all

    class Env:
        def seed(self, s):
            self.s = s

    env = Env()
    seed_all(7, env)
    a = (random.random(), np.random.random(), torch.rand(1).item())
    # the three seeders the trainer, the Isaac Lab evaluator and the server used before ``seed_all`` existed
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    assert env.s == 7 and a == (random.random(), np.random.random(), torch.rand(1).item())
    trainer.seed_everything(7)               # the trainer's wrapper still seeds the same three sources
    assert a == (random.random(), np.random.random(), torch.rand(1).item())
    # a seed beyond 32 bits: numpy takes it modulo 2**32 (as the Isaac Lab evaluator and the server always did),
    # torch and random take it whole
    big = 2**32 + 2**31 + 7
    seed_all(big)
    b = (random.random(), np.random.random(), torch.rand(1).item())
    random.seed(big)
    np.random.seed(2**31 + 7)
    torch.manual_seed(big)
    assert b == (random.random(), np.random.random(), torch.rand(1).item())
