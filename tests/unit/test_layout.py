"""The one place for paths and names (``tmg.core.layout``) must spell exactly what the Hydra templates, the old
hand-built strings and the shell drivers spell; plus the small core helpers that moved next to it."""
import json
from pathlib import Path

import pytest

from tmg.config import compose, resolved
from tmg.core import layout
from tmg.core.io_json import append_jsonl, read_json, read_jsonl, write_json
from tmg.core.layout import Condition, dataset_path, eval_output, indexed_keys, pool_dir, run_dir, sidecar
from tmg.core.records import IMAGES_FORMAT, LABELS_FORMAT, NPZ_FORMAT, record_file

# every one of the six names differs from the others, so a builder that swaps two folders cannot pass
COND = Condition("mimicgen", "square", "ours", "official", "trackmix", "pool_robosuite_image")
GROUPS = ["generator=mimicgen", "task=square", "scene=ours", "selection=official", "method=trackmix",
          "generation=pool_robosuite_image"]


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    return tmp_path


def test_builders_match_the_hydra_templates(root):
    gen = resolved(compose("generate", GROUPS))
    assert pool_dir(root / "pools", COND) == Path(gen["output"]) and Condition.from_plain(gen) == COND
    ds = resolved(compose("dataset", [*GROUPS, "observation=low_dim", "size=200", "seed=3"]))
    assert dataset_path(root / "datasets", COND, "low_dim", 200, 3) == Path(ds["output"])
    assert pool_dir(root / "pools", COND) == Path(ds["pool"])
    assert Condition.from_plain(ds) == COND
    tr = resolved(compose("train", [*GROUPS, "observation=image", "policy=diffusion_policy", "seed=202601", "dataset.size=500"]))
    assert run_dir(root / "runs", COND, "image", "diffusion_policy", 202601) == Path(tr["output"])
    assert dataset_path(root / "datasets", COND, "image", 500, 0) == Path(tr["dataset"]["path"])
    assert Condition.from_plain(tr) == COND
    assert layout.resets_dir(root / "resets", "mimicgen", "square", "ours") == Path(root / "resets/mimicgen/square/ours")
    # the same paths spelt out (what the yaml templates render for these choices)
    assert pool_dir(root / "pools", COND) == root / "pools/mimicgen/square/ours/official/trackmix/pool_robosuite_image"
    assert dataset_path(root / "datasets", COND, "low_dim", 200, 3) == \
        root / "datasets/mimicgen/square/ours/official/trackmix/pool_robosuite_image/low_dim/n200_seed3.hdf5"
    assert run_dir(root / "runs", COND, "image", "diffusion_policy", 202601) == \
        root / "runs/mimicgen/square/ours/official/trackmix/pool_robosuite_image/image/diffusion_policy/seed202601"
    ev = resolved(compose("eval", ["task=square", "scene=ours", "checkpoint=/c"]))
    assert layout.resets_path(ev, isaaclab=False) == root / "resets/mimicgen/square/ours/val25_test300_seed202600.hdf5"
    # the paths yaml names the subfolders the builders assume
    paths = gen["paths"]
    for key, name in (("pools", layout.POOLS_DIR), ("datasets", layout.DATASETS_DIR), ("resets", layout.RESETS_DIR),
                      ("runs", layout.RUNS_DIR), ("reports", layout.REPORTS_DIR)):
        assert Path(paths[key]) == root / name


def test_resets_path_reproduces_the_old_formula(root):
    for overrides, isaaclab, ext in ((["task=square", "scene=ours"], False, "hdf5"),
                                     (["generator=softmimicgen", "task=rope", "scene=ours"], True, "json")):
        plain = resolved(compose("resets", overrides))
        ev = plain["evaluation"]
        old = Path(plain["paths"]["resets"]) / plain["generator"]["name"] / plain["task"]["name"] / plain["scene"]["name"] / \
            f"val{int(ev['val_scenes'])}_test{int(ev['test_scenes'])}_seed{int(ev['seed'])}.{ext}"
        assert layout.resets_path(plain, isaaclab=isaaclab) == old
    from tmg.evaluation.resets import resets_path
    plain = resolved(compose("eval", ["generator=softmimicgen", "task=rope", "scene=ours", "checkpoint=/c"]))
    assert resets_path(plain).suffix == ".json"


def test_sidecar_reproduces_the_three_old_spellings():
    ds = Path("/d/n1000_seed0.hdf5")
    assert sidecar(ds, "DATASET.json") == ds.with_name(ds.stem + "." + "DATASET.json") == Path("/d/n1000_seed0.DATASET.json")
    assert sidecar(ds, "resolved_config.yaml") == ds.with_name(ds.stem + ".resolved_config.yaml")
    for resets in (Path("/r/val25_test300_seed202600.hdf5"), Path("/r/val25_test300_seed202600.json")):
        assert sidecar(resets, "resolved_config.yaml") == resets.with_suffix(".resolved_config.yaml")
        assert layout.part_file(resets) == resets.with_suffix(resets.suffix + ".part")
    assert layout.part_file(ds) == Path("/d/n1000_seed0.hdf5.part")


def test_eval_output_and_checkpoint_dirs():
    ckpt = Path("/data/runs/r/checkpoints/0040000")
    assert eval_output(ckpt, "test", 600) == Path("/data/runs/r/eval/test_h600/0040000")
    assert eval_output(ckpt, "val") == Path("/data/runs/r/eval/val/0040000")
    assert eval_output(Path("/elsewhere/0040000"), "val", 10) == Path("/elsewhere/eval/val_h10/0040000")
    assert layout.checkpoint_dir("/data/runs/r", 500) == ckpt.with_name("0000500")
    assert layout.run_dir_of_checkpoint(ckpt) == Path("/data/runs/r")


def test_indexed_keys_sort_numerically_and_filter_by_prefix():
    assert indexed_keys(["demo_10", "demo_2", "demo_0"]) == ["demo_0", "demo_2", "demo_10"]
    assert indexed_keys(["reset_3", "other", "reset_1"], "reset_") == ["reset_1", "reset_3"]
    assert layout.key_index("reset_12") == 12
    with pytest.raises((ValueError, IndexError)):
        indexed_keys(["demo_1", "meta"])          # a group of demonstrations must hold nothing else (as the old sort key did)


def test_record_file_names_and_jsonl_helpers(tmp_path):
    assert record_file("/r", NPZ_FORMAT, 12) == Path("/r/attempt_000012.npz")
    assert record_file(tmp_path, LABELS_FORMAT, 3).name == "labels_000003.npz" and (IMAGES_FORMAT % 7) == "images_000007.npz"
    path = tmp_path / "rows.jsonl"
    assert read_jsonl(path) == []                 # missing file
    path.write_text("")
    assert read_jsonl(path) == []                 # existing, empty file
    append_jsonl(path, {"a": 1})
    path.write_text(path.read_text() + "\n")      # a blank line is skipped
    append_jsonl(path, {"a": 2})
    assert read_jsonl(path) == [{"a": 1}, {"a": 2}]
    write_json(tmp_path / "x.json", {"b": 3, "p": Path("/x")})          # default=str: non-JSON values become strings
    assert read_json(tmp_path / "x.json") == {"b": 3, "p": "/x"} and not (tmp_path / "x.json.tmp").exists()


def test_collect_results_builds_the_same_paths_as_the_configs(root):
    from scripts import collect_results

    cond, observation = collect_results.parse_job("mimicgen/square/ours/official/trackmix/pool_robosuite_image/low_dim")
    assert cond == COND and observation == "low_dim"
    assert pool_dir(root / layout.POOLS_DIR, cond) == root / "pools/mimicgen/square/ours/official/trackmix/pool_robosuite_image"
    assert layout.dataset_dir(root / layout.DATASETS_DIR, cond, observation) == \
        root / "datasets/mimicgen/square/ours/official/trackmix/pool_robosuite_image/low_dim"
    assert run_dir(root / layout.RUNS_DIR, cond, observation, "bc_rnn", 202602) == \
        root / "runs/mimicgen/square/ours/official/trackmix/pool_robosuite_image/low_dim/bc_rnn/seed202602"
    # collect_run reads the run folder by the layout names
    run = root / "run"
    (run / layout.CHECKPOINTS_DIR / "0000500").mkdir(parents=True)
    (run / layout.CHECKPOINTS_DIR / "0000500" / layout.CHECKPOINT_JSON).write_text(json.dumps({"update": 500}))
    (run / layout.TRAIN_RESULT).write_text("{}")
    append_jsonl(run / layout.VALIDATION_JSONL, {"checkpoint": str(run / "checkpoints/0000500"), "success_rate": 0.4})
    (run / layout.SELECTED_JSON).write_text(json.dumps({"checkpoint": str(run / "checkpoints/0000500")}))
    ev = run / layout.EVAL_DIR / "test_h600" / "0000500"
    ev.mkdir(parents=True)
    block = {"n": 2, "n_success": 1, "success_rate": 0.5, "ci_low": 0.1, "ci_high": 0.9}
    (ev / layout.SUMMARY).write_text(json.dumps({"split": "test", "horizon": 600, "checkpoint": "c", "all": block}))
    out = collect_results.collect_run(run, root / "out")
    assert out["trained"] and out["checkpoints_validated"] == 1 and out["best_validation"] == 0.4
    assert out["test"]["success_rate"] == 0.5 and out["test"]["folder"] == "eval/test_h600/0000500"
    assert (root / "out" / layout.CHECKPOINTS_DIR / "0000500.json").exists()
    assert (root / "out" / "eval/test_h600/0000500" / layout.SUMMARY).exists()


def test_collect_results_runs_without_the_image_packages():
    """scripts/run_*_baseline.sh call ``python3 scripts/collect_results.py`` on the host, whose interpreter may have no numpy:
    the committed script used the standard library only, so its imports (tmg.core.layout, chunks, io_json) must too."""
    import subprocess
    import sys

    from tmg.core import records

    assert (records.RECORDS_DIR, records.ATTEMPTS_JSONL) == ("records", "attempts.jsonl")
    script = Path(__file__).resolve().parents[2] / "scripts/collect_results.py"
    blocker = (
        "import runpy, sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in ('numpy', 'h5py', 'yaml', 'scipy', 'omegaconf', 'hydra', 'torch'):\n"
        "            raise ImportError('not on the host: ' + name)\n"
        "sys.meta_path.insert(0, Block())\n"
        f"sys.argv = [{str(script)!r}, '--help']\n"
        f"runpy.run_path({str(script)!r}, run_name='__main__')\n")
    result = subprocess.run([sys.executable, "-I", "-c", blocker], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--data-root" in result.stdout


def test_bootstrap_apt_command_and_image_table():
    from scripts import bootstrap_native as b

    root = Path("/ROOT")
    assert b.apt_cmd(root) == ["apt-get", "-o", "Dir::State::lists=/ROOT/tools/apt/lists", "-o",
                               "Dir::Cache=/ROOT/tools/apt/cache"]
    assert b.uv_path(root) == Path("/ROOT/bin/uv") and b.venv_python(root, "train") == Path("/ROOT/envs/train/bin/python")
    assert b.IMAGES["train"].python_tuple == [3, 12] and b.IMAGES["isaaclab-soft"].python_tuple == [3, 11]
    assert b.IMAGES["train"].expected == {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "lerobot": "0.6.1"}
    dex, soft = b.IMAGES["isaaclab-dex"].expected, b.IMAGES["isaaclab-soft"].expected
    assert list(dex) == ["isaacsim", "torch", "torchvision", "torchaudio", "qpsolvers", "packaging", "click", "idna", "psutil",
                         "pillow"]
    assert dex["qpsolvers"] == "4.3.1" and soft["qpsolvers"] == "4.8.1"
    assert soft["torch"] == "2.7.0+cu128" and soft["pillow"] == "11.3.0"


def test_migration_plan_moves_every_non_selection_child(tmp_path):
    from scripts import migrate_selection_layout as m

    for kind in ("pools", "datasets", "runs"):
        (tmp_path / kind / "mimicgen/square/ours/none").mkdir(parents=True)
        (tmp_path / kind / "mimicgen/square/ours/official").mkdir(parents=True)      # already migrated: left alone
    (tmp_path / "pools/_verify_baseline/mimicgen").mkdir(parents=True)
    (tmp_path / "pools/softmimicgen/rope/ours/trackmix").mkdir(parents=True)
    (tmp_path / "reports/rank_sources/mimicgen/square/ours/rank_human_demos").mkdir(parents=True)
    (tmp_path / "pools/mimicgen/square/ours/README.md").write_text("file, not a folder")
    moves = m.plan(tmp_path)
    assert moves == [
        (tmp_path / "pools/mimicgen/square/ours/none", tmp_path / "pools/mimicgen/square/ours/ours/none"),
        (tmp_path / "pools/softmimicgen/rope/ours/trackmix", tmp_path / "pools/softmimicgen/rope/ours/official/trackmix"),
        (tmp_path / "datasets/mimicgen/square/ours/none", tmp_path / "datasets/mimicgen/square/ours/ours/none"),
        (tmp_path / "runs/mimicgen/square/ours/none", tmp_path / "runs/mimicgen/square/ours/ours/none"),
        (tmp_path / "reports/rank_sources/mimicgen/square/ours/rank_human_demos",
         tmp_path / "reports/rank_sources/mimicgen/square/ours/ours/rank_human_demos"),
    ]
    assert m.plan(tmp_path / "nowhere") == []
