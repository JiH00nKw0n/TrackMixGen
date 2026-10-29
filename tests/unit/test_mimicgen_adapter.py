"""MimicGen adapter units without mimicgen installed: the chunk spec, ``chunk_result`` on fake outputs, and
``sources.fetch`` with a fake download and a fake ``mimicgen`` module."""
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tmg.config import compose, resolved
from tmg.core.chunks import plan_chunks
from tmg.core.records import ATTEMPTS_JSONL, RECORD_ERROR, RECORDS_DIR
from tmg.generation.base import ChunkSpec
from tmg.generation.generators import _sources, make_generator
from tmg.generation.generators.mimicgen import sources
from tmg.generation.generators.mimicgen.adapter import IMPORTANT_STATS, MimicGenGenerator

TEMPLATE = json.loads((Path(__file__).parent / "fixtures" / "mimicgen_square_template.json").read_text())


def square_generator(tmp_path, monkeypatch, *overrides):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(MimicGenGenerator, "template", lambda self: json.loads(json.dumps(TEMPLATE)))
    return make_generator(resolved(compose("generate", ["task=square", "generation=test/pilot20", *overrides])))


def test_prepare_chunk_spec(tmp_path, monkeypatch):
    gen = square_generator(tmp_path, monkeypatch, "scene=ours", "method=trackmix")
    folder = tmp_path / "chunk"
    spec = gen.prepare_chunk(plan_chunks(20, 20, 9000)[0], folder)
    assert isinstance(spec, ChunkSpec)
    assert spec.generator == "mimicgen" and spec.chunk == {"index": 0, "seed": 9001, "attempts": 20}
    assert spec.folder == str(folder) and spec.records_dir == str(folder / RECORDS_DIR)
    assert spec.generator_config == str(folder / "mg_config.json") and (folder / "mg_config.json").exists()
    assert spec.expected_outputs == [str(folder / "Square_OURS_n20_seed9001" / IMPORTANT_STATS),
                                     str(folder / RECORDS_DIR / ATTEMPTS_JSONL)]
    assert list(spec.extra) == ["env_name", "experiment", "graft_hashes", "commit"]       # key order is part of the identity
    assert spec.extra == {"env_name": "Square_OURS", "experiment": "Square_OURS_n20_seed9001",
                          "graft_hashes": {"generate": "5259272430921d419d37661d168dea8f6d95c8d79258f040a9a35dee606e7be1",
                                           "execute": "5bdf546e1ddd515b1b7fc95624d5c9428b0a1f21d8be16c2ea853e171415f643"},
                          "commit": "72bd767"}
    assert spec.protocol["camera_size"] == [84, 84]            # filled from task.image_size when the protocol leaves it null
    assert spec.task is gen.task and spec.method is gen.method and spec.scene == gen.scene_chain()
    assert set(spec.identity()) == {"generator", "chunk", "task", "scene", "method", "protocol", "extra"}
    assert json.loads(json.dumps(asdict(spec))) == asdict(spec)          # the spec is plain JSON
    assert gen.chunk_command(spec, folder / "chunk_spec.json") == [
        sys.executable, "-m", "tmg.generation.generators.mimicgen.entrypoint", "--spec", str(folder / "chunk_spec.json")]
    layout = gen.demo_layout()
    assert layout["demo_glob"] == "*/demo.hdf5" and layout["labels_from_records"] is False


def write_records(records_dir: Path, successes: list[bool], steps: int = 2) -> None:
    records_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, ok in enumerate(successes):
        npz = f"attempt_{i:06d}.npz"
        np.savez(records_dir / npz, w_pos=np.zeros((steps, 1, 3), dtype=np.float32))
        lines.append(json.dumps({"attempt": i, "success": ok, "n_steps": steps, "npz": npz}))
    (records_dir / ATTEMPTS_JSONL).write_text("\n".join(lines) + "\n")


def write_demos(path: Path, n: int) -> None:
    with h5py.File(path, "w") as f:
        for i in range(n):
            f.create_group(f"data/demo_{i}")


def finished_chunk(tmp_path, monkeypatch, *, keep_failed=True, successes=(True, False, True)):
    gen = square_generator(tmp_path, monkeypatch, "scene=ours", f"generation.protocol.keep_failed={str(keep_failed).lower()}")
    folder = tmp_path / "chunk"
    spec = gen.prepare_chunk(plan_chunks(len(successes), len(successes), 9000)[0], folder)
    experiment = folder / spec.extra["experiment"]
    experiment.mkdir(parents=True)
    n_ok = sum(successes)
    (experiment / IMPORTANT_STATS).write_text(json.dumps({"num_attempts": len(successes), "num_success": n_ok,
                                                          "num_failures": len(successes) - n_ok, "num_problematic": 0,
                                                          "time spent (hrs)": 0.01}))
    write_records(folder / RECORDS_DIR, list(successes))
    write_demos(experiment / "demo.hdf5", n_ok)
    if keep_failed:
        write_demos(experiment / "demo_failed.hdf5", len(successes) - n_ok)
    return gen, spec, experiment


def test_chunk_result_summary(tmp_path, monkeypatch):
    gen, spec, experiment = finished_chunk(tmp_path, monkeypatch)
    assert gen.chunk_result(spec) == {
        "attempts": 3, "successes": 2, "failures": 1, "num_problematic": 0, "time_hours": 0.01,
        "demo_hdf5": str(experiment / "demo.hdf5"), "failed_hdf5": str(experiment / "demo_failed.hdf5"),
        "hdf5_counts": {"demo.hdf5": 2, "demo_failed.hdf5": 1}}


def test_chunk_result_without_kept_failures(tmp_path, monkeypatch):
    gen, spec, _ = finished_chunk(tmp_path, monkeypatch, keep_failed=False)
    result = gen.chunk_result(spec)
    assert result["failed_hdf5"] is None and result["hdf5_counts"] == {"demo.hdf5": 2}


def test_chunk_result_unfinished_and_errors(tmp_path, monkeypatch):
    gen, spec, experiment = finished_chunk(tmp_path, monkeypatch)
    records = Path(spec.records_dir)
    stats = experiment / IMPORTANT_STATS
    text = stats.read_text()
    stats.unlink()
    assert gen.chunk_result(spec) is None                     # MimicGen has not written its stats yet
    stats.write_text(text)
    (records / ATTEMPTS_JSONL).rename(records / "gone")
    assert gen.chunk_result(spec) is None                     # no records yet
    (records / "gone").rename(records / ATTEMPTS_JSONL)
    write_records(records, [True, False])                     # fewer lines than planned: an error, not "unfinished"
    assert gen.chunk_result(spec) == {"error": "2 attempt records, 3 planned"}
    write_records(records, [True, False, True])
    (records / RECORD_ERROR).write_text("{}")
    assert gen.chunk_result(spec) == {"error": f"record writing failed, see {records / RECORD_ERROR}"}
    (records / RECORD_ERROR).unlink()
    stats.write_text(json.dumps({"num_attempts": 3, "num_success": 1, "num_failures": 2}))
    assert gen.chunk_result(spec) == {"error": "records (3 attempts, 2 successes, 1 failures) differ from important_stats.json (3, 1, 2)"}
    stats.write_text(text)
    (experiment / "demo.hdf5").unlink()
    assert gen.chunk_result(spec) == {"error": f"{experiment / 'demo.hdf5'} missing although 2 demonstrations were expected"}
    write_demos(experiment / "demo.hdf5", 1)
    assert gen.chunk_result(spec) == {"error": f"{experiment / 'demo.hdf5'} holds 1 demonstrations, expected 2"}


def test_fetch_downloads_prepares_and_writes_the_manifest(tmp_path, monkeypatch, capsys):
    dest = tmp_path / "src" / "square" / "source.hdf5"
    fake_mimicgen = SimpleNamespace(DATASET_REGISTRY={"source": {"square": {"url": "core/square.hdf5"}}}, HF_REPO_ID="repo/default")
    monkeypatch.setitem(sys.modules, "mimicgen", fake_mimicgen)
    calls = []

    def fake_download(repo_id, filename, path, revision=None):
        calls.append(("download", repo_id, filename, str(path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            for i in range(3):
                f.create_group(f"data/demo_{i}")
        return path

    def fake_prepare(path, interface):
        calls.append(("prepare", str(path), interface))
        with h5py.File(path, "a") as f:
            for i in range(3):
                f.create_group(f"data/demo_{i}/datagen_info")

    monkeypatch.setattr(_sources, "download", fake_download)      # the module ensure_downloaded resolves it from
    monkeypatch.setattr(sources, "prepare", fake_prepare)
    task = {"name": "square", "source": str(dest), "env_prefix": "Square"}
    manifest = sources.fetch(task, {"source_repo": "amandlek/mimicgen_datasets"})
    assert calls == [("download", "amandlek/mimicgen_datasets", "core/square.hdf5", str(dest)), ("prepare", str(dest), "MG_Square")]
    assert list(manifest) == ["task", "file", "bytes", "md5", "demos", "prepared_here", "registry_url"]
    assert manifest["task"] == "square" and manifest["file"] == str(dest) and manifest["demos"] == 3
    assert manifest["prepared_here"] is True and manifest["registry_url"] == "core/square.hdf5"
    assert manifest["md5"] == sources.file_md5(dest) and manifest["bytes"] == dest.stat().st_size
    on_disk = dest.parent / "manifest.json"
    assert on_disk.read_text() == json.dumps(manifest, indent=2) + "\n"
    # second call: nothing downloaded or prepared, the manifest is rewritten from the existing file
    again = sources.fetch(task, {})
    assert len(calls) == 2 and again == dict(manifest, prepared_here=False)
    assert "using existing" in capsys.readouterr().out


def test_fetch_refuses_unknown_registry_name(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mimicgen", SimpleNamespace(DATASET_REGISTRY={"source": {"square": {"url": "u"}}}, HF_REPO_ID="r"))
    with pytest.raises(KeyError, match="mimicgen has no source dataset 'nope'; known: \\['square'\\]"):
        sources.fetch({"name": "nope", "source": str(tmp_path / "s.hdf5")}, {})


def test_fetch_fails_when_prepare_leaves_demos_without_datagen_info(tmp_path, monkeypatch):
    dest = tmp_path / "source.hdf5"
    write_demos(dest, 2)                                                        # exists: nothing is downloaded
    monkeypatch.setitem(sys.modules, "mimicgen", SimpleNamespace(DATASET_REGISTRY={"source": {"square": {"url": "u"}}}, HF_REPO_ID="r"))
    monkeypatch.setattr(_sources, "download", lambda *a, **k: pytest.fail("the file exists, nothing to download"))
    prepared = []
    monkeypatch.setattr(sources, "prepare", lambda path, interface: prepared.append((str(path), interface)))     # adds nothing
    with pytest.raises(RuntimeError, match="still lacks datagen_info after prepare_src_dataset"):
        sources.fetch({"name": "square", "source": str(dest), "env_prefix": "Square", "datagen_interface": "MG_Custom"}, {})
    assert prepared == [(str(dest), "MG_Custom")]
    assert not (tmp_path / "manifest.json").exists()


def test_download_moves_the_file_atomically_and_makes_it_writable(tmp_path, monkeypatch):
    """``download`` moves the Hugging Face cache file (read-only there) to ``dest`` with mode 0o644, leaving no scratch."""
    seen = {}

    def fake_hf_hub_download(*, repo_id, filename, repo_type, cache_dir, revision):
        seen.update(repo_id=repo_id, filename=filename, repo_type=repo_type, revision=revision)
        assert Path(cache_dir).parent == tmp_path / "out"                         # the scratch folder sits next to dest
        got = Path(cache_dir) / "blobs" / "abc"
        got.parent.mkdir(parents=True)
        got.write_bytes(b"payload")
        got.chmod(0o444)
        return str(got)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=fake_hf_hub_download))
    dest = tmp_path / "out" / "source.hdf5"
    assert _sources.download("repo/x", "core/square.hdf5", dest, revision="v1") == dest
    assert seen == {"repo_id": "repo/x", "filename": "core/square.hdf5", "repo_type": "dataset", "revision": "v1"}
    assert dest.read_bytes() == b"payload" and (dest.stat().st_mode & 0o777) == 0o644
    assert sorted(p.name for p in dest.parent.iterdir()) == ["source.hdf5"]       # no .partial, no temp dir left behind


def test_runtime_private_noise_stream_is_seeded_with_chunk_seed_plus_7919(tmp_path):
    """Finished pools were produced with this stream; its seed must stay ``chunk_seed + 7919``."""
    from tmg.generation.generators.mimicgen.runtime import MimicGenRuntime
    from tmg.generation.method import build_method

    method = build_method({"name": "none", "mixing": {"enabled": False}, "tracking": {"enabled": False}}, chunk_seed=9001)
    runtime = MimicGenRuntime(method, chunk_seed=9001, records_dir=tmp_path / "records")
    expected = np.random.RandomState(9001 + 7919)
    assert np.array_equal(runtime.exec.private_draw(6), expected.randn(6))
    assert np.array_equal(runtime.exec.private_draw(3), expected.randn(3))
    assert runtime.rec.dir == tmp_path / "records" and runtime.rec.enabled and (tmp_path / "records").is_dir()
    assert MimicGenRuntime(method, chunk_seed=9001, records_dir=None).rec.enabled is False
