"""Pool runner (``tmg.generation.runner.run_pool``) with a fake generator and a faked chunk process.

The chunk process is replaced by a function that writes the generator's completion file, so the tests cover the
runner's own logic only: the chunk plan, resume of finished and partial chunks, change detection of finished
chunks, failure reporting and the pool summary. Every expected value is written out by hand from the pool config
(5 attempts in chunks of 2 after seed offset 100 give seeds 101, 102, 103 with 2, 2, 1 attempts; the fake chunk
reports ``attempts - 1`` successes).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from tmg.core.chunks import (
    CHUNK_DONE,
    CHUNK_SPEC,
    CHUNKS_DIR,
    LOGS_DIR,
    POOL_DONE,
    POOL_PLAN,
    RESOLVED_CONFIG,
    Chunk,
    read_json,
    write_json,
)
from tmg.generation import runner
from tmg.generation.base import ChunkSpec, SyntheticDataGenerator

STATS = "out/stats.json"


class FakeGenerator(SyntheticDataGenerator):
    """A real ``SyntheticDataGenerator`` (so the fake cannot drift from the contract the runner relies on) whose chunk
    process is ``["fake-chunk", <spec path>]`` and whose completion file is ``out/stats.json``."""
    name = "fake"

    def __init__(self, cfg, *, sources_ok=True):
        super().__init__(cfg)
        self.sources_ok = sources_ok
        self.prepared: list[str] = []

    def check_sources(self):
        return [] if self.sources_ok else ["source demonstrations missing: nowhere"]

    def fetch_sources(self):
        return {}

    def demo_layout(self):
        return {}

    def prepare_chunk(self, chunk: Chunk, folder: Path) -> ChunkSpec:
        folder.mkdir(parents=True, exist_ok=True)
        config = folder / "gen_config.json"
        config.write_text(json.dumps({"seed": chunk.seed, "attempts": chunk.attempts}))
        self.prepared.append(str(folder))
        return self._spec(chunk, folder, scene=self.scene, generator_config=str(config),
                          expected_outputs=[str(folder / STATS)], extra={"experiment": "out"})

    def chunk_command(self, spec: ChunkSpec, spec_path: Path) -> list[str]:
        return ["fake-chunk", str(spec_path)]

    def chunk_result(self, spec: ChunkSpec):
        stats = Path(spec.folder) / STATS
        if not stats.exists():
            return None
        return json.loads(stats.read_text())


class FakeLaunch:
    """Stands in for ``subprocess.run``: reads the chunk spec, writes ``out/stats.json`` unless the seed must fail.

    ``failing_seeds`` exit with code 3 and write nothing; ``erroring_seeds`` exit 0 but their stats carry an
    ``error`` entry (the generator ran and reported a problem). Other commands (the provenance's ``git`` calls) go to
    the real ``subprocess.run``, since the runner imports the stdlib module and patching its ``run`` patches it for
    every caller.
    """

    def __init__(self, failing_seeds=(), erroring_seeds=()):
        self.failing = set(failing_seeds)
        self.erroring = set(erroring_seeds)
        self.calls: list[dict] = []
        self.real_run = subprocess.run

    def __call__(self, cmd, *args, **kwargs):
        if cmd[0] != "fake-chunk":
            return self.real_run(cmd, *args, **kwargs)
        return self.chunk(cmd, **kwargs)

    def chunk(self, cmd, *, stdout, stderr, stdin, env, cwd):
        spec = json.loads(Path(cmd[1]).read_text())
        seed = spec["chunk"]["seed"]
        self.calls.append({"cmd": list(cmd), "env": dict(env), "cwd": cwd, "seed": seed})
        stdout.write(f"fake chunk seed {seed}\n")
        if seed in self.failing:
            return _Completed(3)
        attempts = int(spec["chunk"]["attempts"])
        stats = Path(spec["folder"]) / STATS
        stats.parent.mkdir(parents=True, exist_ok=True)
        result = {"attempts": attempts, "successes": attempts - 1}
        if seed in self.erroring:
            result["error"] = f"generator gave up on seed {seed}"
        stats.write_text(json.dumps(result))
        return _Completed(0)


class _Completed:
    def __init__(self, code):
        self.returncode = code


def _cfg(tmp_path, **generation):
    gen = {"name": "quick", "attempts": 5, "chunk": 2, "seed_offset": 100, "on_config_change": "error", **generation}
    return OmegaConf.create({
        "output": str(tmp_path / "pool"), "generator": {"name": "fake"}, "task": {"name": "square"},
        "scene": {"name": "ours"}, "method": {"name": "none"}, "generation": gen,
        "cluster": {"workers": 2, "max_workers": 2, "threads_per_worker": 3, "min_free_disk_gib": 0},
    })


PLAN_5_BY_2 = [{"index": 0, "seed": 101, "attempts": 2}, {"index": 1, "seed": 102, "attempts": 2},
               {"index": 2, "seed": 103, "attempts": 1}]


@pytest.fixture
def pool(monkeypatch, tmp_path):
    """``run(cfg, failing_seeds=(), erroring_seeds=())`` runs the pool with a fake generator and returns
    ``(summary, launch)``; ``run.state["generator"]`` is the generator of the last run."""
    state = {"generator": None}

    def make(plain):
        state["generator"] = FakeGenerator(plain, sources_ok=state.get("sources_ok", True))
        return state["generator"]

    monkeypatch.setattr(runner, "make_generator", make)

    def run(cfg, failing_seeds=(), erroring_seeds=()):
        launch = FakeLaunch(failing_seeds, erroring_seeds)
        monkeypatch.setattr(runner.subprocess, "run", launch)
        return runner.run_pool(cfg), launch

    run.state = state
    return run


def test_run_pool_plans_runs_and_summarizes(pool, tmp_path):
    cfg = _cfg(tmp_path)
    summary, launch = pool(cfg)
    output = tmp_path / "pool"
    assert summary["plan"] == PLAN_5_BY_2
    assert sorted(c["seed"] for c in launch.calls) == [101, 102, 103]
    # successes per chunk: 1, 1, 0
    assert summary["attempts"] == 5 and summary["successes"] == 2 and summary["success_rate"] == 2 / 5
    assert summary["generator"] == "fake" and summary["task"] == "square" and summary["scene"] == "ours"
    assert summary["method"] == "none" and summary["generation"] == "quick"
    assert [(c["chunk"], c["attempts"], c["successes"]) for c in summary["chunks"]] == [("seed101", 2, 1), ("seed102", 2, 1),
                                                                                       ("seed103", 1, 0)]
    for c in summary["chunks"]:
        assert c["ok"] is True and c["exit_code"] == 0 and c["elapsed_s"] >= 0
        assert c["log"] == str(output / LOGS_DIR / f"{c['chunk']}.log")
    # on-disk layout: pool markers, per-chunk spec and completion marker, one log per chunk
    assert read_json(output / POOL_DONE) == summary
    assert read_json(output / POOL_PLAN) == {"chunks": PLAN_5_BY_2, "generator": "fake"}
    assert (output / RESOLVED_CONFIG).exists() and (output / "provenance.json").exists()
    for seed, attempts in ((101, 2), (102, 2), (103, 1)):
        folder = output / CHUNKS_DIR / f"seed{seed}"
        assert read_json(folder / CHUNK_DONE) == next(c for c in summary["chunks"] if c["chunk"] == f"seed{seed}")
        spec = read_json(folder / CHUNK_SPEC)
        assert spec["chunk"] == {"index": seed - 101, "seed": seed, "attempts": attempts}
        assert spec["folder"] == str(folder) and spec["records_dir"] == str(folder / "records")
        assert spec["generator_config"] == str(folder / "gen_config.json") and spec["expected_outputs"] == [str(folder / STATS)]
        assert spec["extra"] == {"experiment": "out"} and spec["generator"] == "fake"
        assert (output / LOGS_DIR / f"seed{seed}.log").read_text() == f"fake chunk seed {seed}\n"
    # the chunk process runs in its folder with the thread caps of the cluster config
    call = next(c for c in launch.calls if c["seed"] == 102)
    assert call["cwd"] == str(output / CHUNKS_DIR / "seed102")
    assert call["cmd"] == ["fake-chunk", str(output / CHUNKS_DIR / "seed102" / CHUNK_SPEC)]
    assert all(call["env"][k] == "3" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    assert call["env"]["PYTHONUNBUFFERED"] == "1"


def test_run_pool_returns_the_stored_summary_of_a_complete_pool_without_inspecting_chunks(pool, tmp_path):
    cfg = _cfg(tmp_path)
    first, _ = pool(cfg)
    output = tmp_path / "pool"
    stored = dict(first, note="written by the first run")           # a marker only the stored file carries
    write_json(output / POOL_DONE, stored)
    again, launch = pool(cfg)
    assert again == stored                                          # read back, not recomputed
    assert launch.calls == [] and pool.state["generator"].prepared == []
    assert read_json(output / POOL_DONE) == stored


def test_run_pool_with_a_changed_plan_keeps_matching_chunks_and_runs_the_new_ones(pool, tmp_path):
    output = tmp_path / "pool"
    pool(_cfg(tmp_path, attempts=4))                                # seeds 101, 102 with 2 attempts each
    assert read_json(output / POOL_DONE)["plan"] == PLAN_5_BY_2[:2]
    summary, launch = pool(_cfg(tmp_path, attempts=6))              # DONE.json exists but its plan differs
    assert [c["seed"] for c in launch.calls] == [103]
    assert summary["plan"] == [{"index": 0, "seed": 101, "attempts": 2}, {"index": 1, "seed": 102, "attempts": 2},
                               {"index": 2, "seed": 103, "attempts": 2}]
    assert summary["attempts"] == 6 and summary["successes"] == 3
    assert read_json(output / POOL_DONE) == summary


def test_run_pool_resumes_partial_and_missing_chunks_only(pool, tmp_path):
    cfg = _cfg(tmp_path)
    pool(cfg)
    output = tmp_path / "pool"
    (output / POOL_DONE).unlink()
    (output / CHUNKS_DIR / "seed102" / CHUNK_DONE).unlink()                 # partial: restarted from scratch
    (output / CHUNKS_DIR / "seed102" / "leftover.txt").write_text("x")
    summary, launch = pool(cfg)
    assert [c["seed"] for c in launch.calls] == [102]
    assert not (output / CHUNKS_DIR / "seed102" / "leftover.txt").exists()
    assert summary["attempts"] == 5 and [c["chunk"] for c in summary["chunks"]] == ["seed101", "seed102", "seed103"]
    assert read_json(output / POOL_DONE) == summary


def test_run_pool_refuses_finished_chunks_of_another_configuration_unless_rebuild(pool, tmp_path):
    pool(_cfg(tmp_path))
    output = tmp_path / "pool"
    (output / POOL_DONE).unlink()                 # a complete pool with the same plan is returned as is; resume inspects
    changed = _cfg(tmp_path)
    changed.method.name = "trackmix"
    with pytest.raises(runner.PoolError) as info:
        pool(changed)
    message = str(info.value)
    assert message.startswith("finished chunks differ from the requested configuration (set generation.on_config_change=rebuild")
    for seed in (101, 102, 103):
        assert f"seed{seed}: finished chunk was made under another configuration: ['method']" in message
    assert not (output / POOL_DONE).exists()
    assert read_json(output / CHUNKS_DIR / "seed101" / CHUNK_SPEC)["method"] == {"name": "none"}      # untouched
    rebuild = _cfg(tmp_path, on_config_change="rebuild")
    rebuild.method.name = "trackmix"
    summary, launch = pool(rebuild)
    assert sorted(c["seed"] for c in launch.calls) == [101, 102, 103]
    assert summary["method"] == "trackmix"
    assert read_json(output / CHUNKS_DIR / "seed101" / CHUNK_SPEC)["method"] == {"name": "trackmix"}


def test_run_pool_treats_a_finished_chunk_without_a_spec_as_changed(pool, tmp_path):
    cfg = _cfg(tmp_path)
    pool(cfg)
    output = tmp_path / "pool"
    (output / POOL_DONE).unlink()
    (output / CHUNKS_DIR / "seed103" / CHUNK_SPEC).unlink()
    with pytest.raises(runner.PoolError, match=f"seed103: finished chunk has no {CHUNK_SPEC}"):
        pool(cfg)
    summary, launch = pool(_cfg(tmp_path, on_config_change="rebuild"))
    assert [c["seed"] for c in launch.calls] == [103] and summary["attempts"] == 5


def test_run_pool_reports_failed_chunks_and_keeps_the_finished_ones(pool, tmp_path):
    cfg = _cfg(tmp_path)
    output = tmp_path / "pool"
    with pytest.raises(runner.PoolError, match="1 chunk\\(s\\) failed; see .*seed102.log"):
        pool(cfg, failing_seeds=[102])
    assert not (output / POOL_DONE).exists()
    assert (output / CHUNKS_DIR / "seed101" / CHUNK_DONE).exists() and (output / CHUNKS_DIR / "seed103" / CHUNK_DONE).exists()
    assert not (output / CHUNKS_DIR / "seed102" / CHUNK_DONE).exists()
    summary, launch = pool(cfg)                                              # the rerun only runs the failed chunk
    assert [c["seed"] for c in launch.calls] == [102] and summary["attempts"] == 5


def test_run_pool_treats_a_result_with_an_error_entry_as_a_failed_chunk(pool, tmp_path):
    cfg = _cfg(tmp_path)
    output = tmp_path / "pool"
    with pytest.raises(runner.PoolError, match="1 chunk\\(s\\) failed; see .*seed101.log"):
        pool(cfg, erroring_seeds=[101])
    assert (output / CHUNKS_DIR / "seed101" / STATS).exists()               # the generator did finish ...
    assert not (output / CHUNKS_DIR / "seed101" / CHUNK_DONE).exists()      # ... but the chunk is not complete
    assert (output / CHUNKS_DIR / "seed102" / CHUNK_DONE).exists()


def test_run_pool_stops_on_missing_sources_or_too_little_disk(pool, tmp_path):
    pool.state["sources_ok"] = False
    with pytest.raises(runner.PoolError, match="source demonstrations missing"):
        pool(_cfg(tmp_path))
    pool.state["sources_ok"] = True
    cfg = _cfg(tmp_path)
    cfg.cluster.min_free_disk_gib = 10 ** 9
    with pytest.raises(runner.PoolError, match="GiB free under"):
        pool(cfg)
    assert not (tmp_path / "pool").exists()                                 # nothing is written before the checks pass


def test_relocate_spec_rewrites_every_path_field_and_nothing_else():
    scratch, folder = Path("/scratch/tmp123/seed7"), Path("/pool/chunks/seed7")
    spec = ChunkSpec(generator="fake", chunk={"index": 0, "seed": 7, "attempts": 2}, folder=str(scratch),
                     records_dir=str(scratch / "records"), task={"name": "square", "source": "/data/src.hdf5"},
                     scene={"name": "ours"}, method={"name": "none"}, protocol={"p": 1},
                     generator_config=str(scratch / "gen_config.json"),
                     expected_outputs=[str(scratch / "out" / "demo.hdf5"), str(scratch / "out" / "stats.json")],
                     extra={"experiment": "out"})
    before = spec.identity()
    out = runner._relocate_spec(spec, scratch, folder)
    assert out is spec
    assert out.folder == "/pool/chunks/seed7" and out.records_dir == "/pool/chunks/seed7/records"
    assert out.generator_config == "/pool/chunks/seed7/gen_config.json"
    assert out.expected_outputs == ["/pool/chunks/seed7/out/demo.hdf5", "/pool/chunks/seed7/out/stats.json"]
    assert out.identity() == before                                        # non-path fields untouched
    assert out.task == {"name": "square", "source": "/data/src.hdf5"} and out.extra == {"experiment": "out"}


def test_run_pool_shards_split_the_chunks_and_the_last_one_writes_done(pool, tmp_path):
    # two runners on one pool (one container per GPU): each runs the chunks of its index modulo 2, does not prepare the
    # other shard's unfinished chunks, and only the runner that finds every chunk done writes DONE.json
    output = tmp_path / "pool"
    first, launch = pool(_cfg(tmp_path, shard=[1, 2]))
    assert [c["seed"] for c in launch.calls] == [102]
    assert first["partial"] is True and sorted(first["waiting"]) == ["seed101", "seed103"]
    assert not (output / POOL_DONE).exists()
    assert {Path(f).name for f in pool.state["generator"].prepared} == {"seed102"}
    second, launch = pool(_cfg(tmp_path, shard=[0, 2]))
    assert sorted(c["seed"] for c in launch.calls) == [101, 103]
    assert second["attempts"] == 5 and [c["chunk"] for c in second["chunks"]] == ["seed101", "seed102", "seed103"]
    assert read_json(output / POOL_DONE) == second


def test_run_pool_refuses_a_bad_shard(pool, tmp_path):
    with pytest.raises(runner.PoolError, match="shard"):
        pool(_cfg(tmp_path, shard=[2, 2]))
