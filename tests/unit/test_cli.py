"""``tmg`` command dispatch: every command name maps to a config file and a target function, ``show`` prints the composed
config, and the exit codes of ``verify_baseline`` and ``check_scene`` follow the result flag."""
import ast
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tmg import CONFIGS_ROOT
from tmg.cli import HELP, main
from tmg.config import COMMANDS, compose

# The dispatch of the committed ``tmg.cli.main`` (one ``if args.command == ...`` branch per command): the function each
# command ran and the result key that set its exit code. Frozen so a table entry cannot silently point elsewhere.
DISPATCH = {
    "generate": ("tmg.generation.runner:run_pool", None),
    "verify_baseline": ("tmg.generation.verify:verify_baseline", "identical"),
    "sources": ("tmg.cli:fetch_sources", None),
    "check_scene": ("tmg.generation.scene_check:check_scene", "ok"),
    "rank_sources": ("tmg.generation.rank_sources:rank_sources", None),
    "dataset": ("tmg.data.dataset:build_dataset", None),
    "train": ("tmg.policy.trainer:train", None),
    "select_checkpoint": ("tmg.policy.trainer:select_checkpoint_cmd", None),
    "serve": ("tmg.policy.server:serve", None),
    "resets": ("tmg.evaluation.resets:make_resets", None),
    "eval": ("tmg.evaluation.rollout:evaluate", None),
}

# The help line of every command in the committed ``HELP`` text of ``tmg --help`` (only the layout and order changed).
HELP_LINES = {
    "generate": "one pool of generation attempts (generator, task, scene, method, generation)",
    "verify_baseline": "run one chunk officially and through the adapter (method=none) and compare the demos",
    "rank_sources": "order a task's source demonstrations by their generation success in a baseline pool",
    "train": "train one policy on one dataset (train image)",
    "select_checkpoint": "choose a run's checkpoint from validation.jsonl (or the last one)",
    "serve": "policy server for one checkpoint (train image)",
    "resets": "draw the fixed evaluation scenes of (task, scene) (simulator image)",
    "eval": "play the fixed scenes against a served checkpoint (simulator image)",
    "dataset": "select successful attempts of a finished pool and write one training hdf5",
    "check_scene": "reset a (task, scene) environment many times and report edge overhang, contacts, falls, reach",
    "sources": "download and prepare the human source demonstrations of a task",
}


def test_every_command_has_a_config_file_and_every_config_file_a_command():
    files = {p.stem for p in (CONFIGS_ROOT / "command").glob("*.yaml")}
    assert files == set(COMMANDS) == set(DISPATCH) == set(HELP_LINES)
    assert {name: (spec.target, spec.exit_code) for name, spec in COMMANDS.items()} == DISPATCH
    assert {name: spec.help for name, spec in COMMANDS.items()} == HELP_LINES
    lines = HELP.splitlines()
    for name, text in HELP_LINES.items():
        assert sum(1 for line in lines if line.split() and line.split()[0] == name and line.endswith(text)) == 1, name
    assert any(line.split()[:1] == ["show"] and line.endswith("print the composed configuration of a command without "
                                                                 "running it") for line in lines)


def test_every_target_names_a_function_defined_in_its_module():
    """Checked on the source text: most target modules import a simulator or torch, which the unit tests do not have."""
    for name, spec in COMMANDS.items():
        module, _, function = spec.target.partition(":")
        origin = importlib.util.find_spec(module).origin
        tree = ast.parse(Path(origin).read_text())
        defined = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        assert function in defined, (name, spec.target)
        assert spec.exit_code in (None, "identical", "ok")


def test_show_prints_the_resolved_config(capsys, tmp_path, monkeypatch):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    assert main(["show", "generate", "task=square"]) == 0
    shown = OmegaConf.create(capsys.readouterr().out)
    assert shown.task.name == "square" and shown.output == f"{tmp_path}/pools/mimicgen/square/official_d1/official/none/pilot20"
    with pytest.raises(SystemExit) as e:
        main(["show"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        main(["frobnicate", "task=square"])
    assert e.value.code == 2
    with pytest.raises(SystemExit) as e:
        # listed in the old COMMANDS tuple but never implemented: the committed CLI accepted it and then crashed in Hydra
        # (no configs/command/report.yaml, exit 1 with a traceback); it is now refused by argparse (exit 2)
        main(["report", "task=square"])
    assert e.value.code == 2


def test_plus_overrides_are_refused_before_dispatch():
    with pytest.raises(ValueError):
        main(["generate", "task=square", "+extra=1"])


def _fake_module(monkeypatch, name: str, **functions):
    """Register ``name`` in ``sys.modules`` as a module holding ``functions`` (the real one may need a simulator)."""
    module = types.ModuleType(name)
    for key, fn in functions.items():
        setattr(module, key, fn)
    monkeypatch.setitem(sys.modules, name, module)
    parent, _, child = name.rpartition(".")
    if parent:
        monkeypatch.setattr(importlib.import_module(parent), child, module, raising=False)
    return module


def test_dataset_and_eval_dispatch_pass_the_composed_config(monkeypatch, tmp_path):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    seen = {}
    _fake_module(monkeypatch, "tmg.data.dataset", build_dataset=lambda cfg: seen.setdefault("dataset", cfg))
    _fake_module(monkeypatch, "tmg.evaluation.rollout", evaluate=lambda cfg: seen.setdefault("eval", cfg))
    assert main(["dataset", "task=square", "size=3"]) == 0
    assert seen["dataset"].command == "dataset" and seen["dataset"].size == 3
    assert seen["dataset"].overrides == ["task=square", "size=3"]
    assert main(["eval", "task=square", "checkpoint=/x/checkpoints/0000500", "split=test"]) == 0
    assert seen["eval"].command == "eval" and seen["eval"].split == "test" and seen["eval"].checkpoint == "/x/checkpoints/0000500"


def test_result_flags_set_the_exit_code(monkeypatch, tmp_path):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    _fake_module(monkeypatch, "tmg.generation.verify", verify_baseline=lambda cfg: {"identical": cfg.task.name == "square"})
    _fake_module(monkeypatch, "tmg.generation.scene_check", check_scene=lambda cfg: {"ok": cfg.resets == 200})
    assert main(["verify_baseline", "task=square"]) == 0
    assert main(["verify_baseline", "task=coffee"]) == 1
    assert main(["check_scene", "task=square"]) == 0
    assert main(["check_scene", "task=square", "resets=5"]) == 1


def test_select_checkpoint_dispatch_passes_the_composed_config(monkeypatch, tmp_path):
    """The command's function receives the composed config whose ``output`` is the run folder and whose
    ``training.selection`` is the rule (the function itself, ``trainer.select_checkpoint_cmd``, needs torch and is
    tested in test_trainer.py)."""
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    seen = {}
    _fake_module(monkeypatch, "tmg.policy.trainer", select_checkpoint_cmd=lambda cfg: seen.setdefault("cfg", cfg))
    assert main(["select_checkpoint", "task=square", "seed=3", "training.selection=last"]) == 0
    cfg = seen["cfg"]
    assert cfg.command == "select_checkpoint" and cfg.training.selection == "last"
    assert cfg.output == f"{tmp_path}/runs/mimicgen/square/ours/official/none/pool_robosuite_image/image/bc_rnn/seed3"
    assert cfg.output == compose("train", ["task=square", "seed=3"]).output


def test_sources_dispatch_fetches_through_the_generator(monkeypatch, tmp_path):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    seen = []

    class Generator:
        def __init__(self, plain):
            assert type(plain) is dict          # the generators take the resolved plain dict, not the DictConfig
            self.plain = plain

        def fetch_sources(self):
            seen.append(self.plain["task"]["name"])

    _fake_module(monkeypatch, "tmg.generation.generators", make_generator=Generator)
    assert main(["sources", "task=square"]) == 0
    assert seen == ["square"]
