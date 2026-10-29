"""Baseline identity check (``tmg.generation.verify``): the chunk launch shared with the runner, the hdf5 comparison
and the two-run report."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from omegaconf import OmegaConf

from tests.unit.test_runner import FakeGenerator, FakeLaunch
from tmg.core.chunks import CHUNK_SPEC, Chunk, read_json
from tmg.generation import runner, verify

CFG = {"generator": {"name": "fake"}, "task": {"name": "square"}, "scene": {"name": "ours"}, "method": {"name": "none"},
       "generation": {"name": "quick", "attempts": 2, "seed_offset": 4}}


def _writing_stats(launch, stats_of=lambda spec: {"num_success": 1, "num_attempts": 2}, hdf5_of=None):
    """A ``FakeLaunch.chunk`` replacement that also writes ``important_stats.json`` (and ``demo.hdf5``) into the
    experiment folder, as the official generators do."""
    def chunk(cmd, *, stdout, stderr, stdin, env, cwd):
        result = FakeLaunch.chunk(launch, cmd, stdout=stdout, stderr=stderr, stdin=stdin, env=env, cwd=cwd)
        spec = json.loads(Path(cmd[1]).read_text())
        experiment = Path(spec["folder"]) / spec["extra"]["experiment"]
        (experiment / verify.IMPORTANT_STATS).write_text(json.dumps(stats_of(spec)))
        if hdf5_of is not None:
            with h5py.File(experiment / "demo.hdf5", "w") as f:
                f.create_dataset("data/demo_0/actions", data=hdf5_of(spec))
        return result
    return chunk


def test_run_starts_from_an_empty_folder_and_records_disable_graft(monkeypatch, tmp_path):
    generator = FakeGenerator(CFG)
    launch = FakeLaunch()
    launch.chunk = _writing_stats(launch)
    monkeypatch.setattr(runner.subprocess, "run", launch)          # verify launches through runner.launch_chunk
    monkeypatch.setenv("OPENBLAS_NUM_THREADS", "7")
    monkeypatch.setenv("OMP_NUM_THREADS", "7")
    folder = tmp_path / "official"
    (folder / "stale").mkdir(parents=True)
    (folder / "stale" / "old.txt").write_text("x")
    out = verify._run(generator, Chunk(index=0, seed=5, attempts=2), folder, disable_graft=True, log=tmp_path / "official.log")
    assert not (folder / "stale").exists()                                 # the folder is rebuilt from nothing
    assert out["stats"] == {"num_success": 1, "num_attempts": 2} and out["experiment"] == folder / "out"
    assert out["spec"].extra == {"experiment": "out", "disable_graft": True}
    assert out["spec"].chunk == {"index": 0, "seed": 5, "attempts": 2} and out["spec"].folder == str(folder)
    assert read_json(folder / CHUNK_SPEC)["extra"] == {"experiment": "out", "disable_graft": True}
    assert (tmp_path / "official.log").read_text() == "fake chunk seed 5\n"
    call = launch.calls[0]
    assert call["cwd"] == str(folder) and call["cmd"] == ["fake-chunk", str(folder / CHUNK_SPEC)]
    # single-threaded OpenMP/MKL regardless of the host; other variables pass through unchanged (no cluster config here)
    assert call["env"]["OMP_NUM_THREADS"] == "1" and call["env"]["MKL_NUM_THREADS"] == "1"
    assert call["env"]["PYTHONUNBUFFERED"] == "1" and call["env"]["OPENBLAS_NUM_THREADS"] == "7"
    launch.chunk = lambda cmd, **kw: FakeLaunch.chunk(launch, cmd, **kw)     # no important_stats.json: the chunk did not finish
    with pytest.raises(RuntimeError, match=r"chunk did not finish \(exit 0\); see .*adapter.log"):
        verify._run(generator, Chunk(index=0, seed=5, attempts=2), folder, disable_graft=False, log=tmp_path / "adapter.log")
    assert read_json(folder / CHUNK_SPEC)["extra"] == {"experiment": "out", "disable_graft": False}


def test_compare_hdf5_lists_missing_shape_and_value_differences(tmp_path):
    a, b = tmp_path / "a.hdf5", tmp_path / "b.hdf5"
    with h5py.File(a, "w") as f:
        f.create_dataset("data/demo_0/actions", data=np.zeros((3, 7)))
        f.create_dataset("data/demo_0/states", data=np.arange(4.0))
        f.create_dataset("data/demo_1/actions", data=np.ones(2))
    with h5py.File(b, "w") as f:
        f.create_dataset("data/demo_0/actions", data=np.zeros((3, 7)))
        f.create_dataset("data/demo_0/states", data=np.array([0.0, 1.0, 2.0, 3.5]))
        f.create_dataset("data/demo_2/actions", data=np.ones((2, 1)))
    diffs = verify.compare_hdf5(a, b)
    assert diffs == ["/data/demo_0/states: values differ (max abs 0.5)", "/data/demo_1/actions: only in official",
                     "/data/demo_2/actions: only in adapter"]
    with h5py.File(b, "a") as f:
        del f["data/demo_0/actions"]
        f.create_dataset("data/demo_0/actions", data=np.zeros((3, 8)))
    assert verify.compare_hdf5(a, b)[0] == "/data/demo_0/actions: shape (3, 7) vs (3, 8)"
    assert verify.compare_hdf5(a, a) == []


def _baseline_cfg(tmp_path):
    return OmegaConf.create({**CFG, "output": str(tmp_path / "verify")})


@pytest.fixture
def baseline(monkeypatch, tmp_path):
    """``run(stats_of, hdf5_of)`` runs ``verify_baseline`` with the fake generator; both chunk runs go through ``launch``."""
    monkeypatch.setattr(verify, "make_generator", lambda plain: FakeGenerator(plain))

    def run(stats_of, hdf5_of):
        launch = FakeLaunch()
        launch.chunk = _writing_stats(launch, stats_of, hdf5_of)
        monkeypatch.setattr(runner.subprocess, "run", launch)
        return verify.verify_baseline(_baseline_cfg(tmp_path)), launch
    return run


def test_verify_baseline_runs_the_same_chunk_officially_and_through_the_adapter(baseline, tmp_path):
    report, launch = baseline(lambda spec: {"num_success": 1, "num_attempts": 2, "num_failures": 1, "num_problematic": 0},
                              lambda spec: np.arange(6.0).reshape(2, 3))
    output = tmp_path / "verify"
    assert report["chunk"] == {"index": 0, "seed": 5, "attempts": 2}             # seed_offset + 1, generation.attempts
    assert [c["seed"] for c in launch.calls] == [5, 5]
    assert [c["cwd"] for c in launch.calls] == [str(output / "official"), str(output / "adapter")]
    assert read_json(output / "official" / CHUNK_SPEC)["extra"]["disable_graft"] is True
    assert read_json(output / "adapter" / CHUNK_SPEC)["extra"]["disable_graft"] is False
    assert report["official_stats"] == report["adapter_stats"] == {"num_success": 1, "num_attempts": 2, "num_failures": 1,
                                                                   "num_problematic": 0}
    assert report["diffs"] == {"demo.hdf5": []} and report["identical"] is True
    assert read_json(output / "VERIFY.json") == report
    assert (output / "official.log").read_text() == (output / "adapter.log").read_text() == "fake chunk seed 5\n"


def test_verify_baseline_reports_differing_stats_and_datasets(baseline, tmp_path):
    def stats_of(spec):
        official = spec["extra"]["disable_graft"]
        return {"num_success": 2 if official else 1, "num_attempts": 2, "num_failures": 0 if official else 1}

    def hdf5_of(spec):
        return np.zeros((2, 3)) if spec["extra"]["disable_graft"] else np.full((2, 3), 0.25)

    report, _ = baseline(stats_of, hdf5_of)
    assert report["identical"] is False
    assert report["diffs"] == {"demo.hdf5": ["/data/demo_0/actions: values differ (max abs 0.25)"],
                               "important_stats": ["num_success: 2 vs 1", "num_failures: 0 vs 1"]}
    assert read_json(tmp_path / "verify" / "VERIFY.json")["identical"] is False
