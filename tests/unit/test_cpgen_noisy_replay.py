"""CP-Gen noisy-state replay without CP-Gen or a simulator: the selection file, the regrouping of the official per-demo
outputs in selection order, the fallback count, the hook in ``tmg dataset`` and the reuse of a finished work folder."""
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tmg.config import compose, resolved
from tmg.generation.generators import make_generator
from tmg.generation.generators.cpgen import noisy_replay as nr


def write_chunk(folder: Path, n: int, offset: float) -> list[dict]:
    """A chunk with ``generated.hdf5`` (states, actions, constraint_data, model_file) and a rendered file of the same demos."""
    folder.mkdir(parents=True)
    cands = []
    with h5py.File(folder / "generated.hdf5", "w") as g, h5py.File(folder / "generated_obs.hdf5", "w") as o:
        g.create_group("data").attrs["env_args"] = json.dumps({"env_name": "Coffee_OURS"})
        o.create_group("data")
        for k in range(n):
            T = 5 + k
            actions = np.full((T, 7), offset + k, np.float32)
            d = g["data"].create_group(f"demo_{k}")
            d.create_dataset("actions", data=actions)
            d.create_dataset("states", data=np.full((T, 3), offset + k))
            d.create_group("constraint_data").create_dataset("constraint_0", data=json.dumps({"timesteps": [1, 2]}))
            d.attrs["model_file"] = f"<mujoco {offset + k}/>"
            o["data"].create_group(f"demo_{k}").create_dataset("actions", data=actions)
            cands.append({"chunk": folder.name, "demo_file": str(folder / "generated_obs.hdf5"), "demo_key": f"demo_{k}",
                          "attempt": k, "n_steps": T, "src_demo_inds": [0]})
    return cands


def fake_official_outputs(work: Path, n: int, keep_original: set[int]) -> None:
    """The official per-demo files: replayed (shifted) states, or the original ones for the demos in ``keep_original``."""
    (work / nr.REPLAY_DIR).mkdir(parents=True, exist_ok=True)
    with h5py.File(work / nr.SELECTED, "r") as sel:
        for j in range(n):
            src = sel["data"][f"demo_{j}"]
            states = np.asarray(src["states"]) + (0 if j in keep_original else 0.5)
            with h5py.File(work / nr.REPLAY_DIR / f"demo_demo_{j}_w_noise.hdf5", "w") as f:
                d = f.create_group("data/demo_0")
                d.create_dataset("actions", data=np.asarray(src["actions"]))
                d.create_dataset("states", data=states)
                d.attrs["model_file"] = src.attrs["model_file"]


def test_selection_keeps_order_constraints_and_model_file(tmp_path):
    a = write_chunk(tmp_path / "chunks" / "seed1", 3, 0.0)
    b = write_chunk(tmp_path / "chunks" / "seed2", 2, 100.0)
    chosen = [b[1], a[0], a[2]]
    nr.write_selection(chosen, tmp_path / "w" / nr.SELECTED)
    with h5py.File(tmp_path / "w" / nr.SELECTED, "r") as f:
        assert sorted(f["data"].keys()) == ["demo_0", "demo_1", "demo_2"]
        assert json.loads(f["data"].attrs["env_args"])["env_name"] == "Coffee_OURS"
        assert f["data/demo_0/actions"][0, 0] == 101.0 and f["data/demo_1/actions"][0, 0] == 0.0
        assert f["data/demo_2"].attrs["model_file"] == "<mujoco 2.0/>"
        assert json.loads(f["data/demo_2/constraint_data/constraint_0"][()])["timesteps"] == [1, 2]


def test_selection_refuses_a_shifted_rendered_file(tmp_path):
    cands = write_chunk(tmp_path / "chunks" / "seed1", 2, 0.0)
    with h5py.File(cands[0]["demo_file"], "a") as o:
        o["data/demo_0/actions"][...] = 9.0
    with pytest.raises(ValueError, match="differ"):
        nr.write_selection(cands, tmp_path / "w" / nr.SELECTED)


def test_regroup_writes_selection_order_and_counts_fallbacks(tmp_path):
    cands = write_chunk(tmp_path / "chunks" / "seed1", 5, 0.0)
    work = tmp_path / "w"
    nr.write_selection(cands, work / nr.SELECTED)
    fake_official_outputs(work, 5, keep_original={1, 4})
    parts, fallback = nr.regroup(work, 5, 2)
    assert fallback == 2 and len(parts) == 2
    keys = []
    for p in parts:
        with h5py.File(p, "r") as f:
            keys += sorted(f["data"].keys(), key=lambda k: int(k[5:]))
            assert "env_args" in f["data"].attrs
    assert keys == [f"demo_{j}" for j in range(5)]
    with h5py.File(parts[0], "r") as f:
        assert f["data/demo_0/states"][0, 0] == 0.5        # replayed states, original actions
        assert f["data/demo_1/states"][0, 0] == 1.0
        assert f["data/demo_0/actions"][0, 0] == 0.0


def test_parts_cover_every_index_once():
    for n, parts in [(1, 12), (5, 2), (1000, 12), (7, 7)]:
        idx = [j for r in nr.parts_of(n, parts) for j in r]
        assert idx == list(range(n))


def test_replay_selection_points_candidates_at_parts_and_reuses_the_work(tmp_path, monkeypatch):
    cands = write_chunk(tmp_path / "pool" / "chunks" / "seed1", 4, 0.0)
    (tmp_path / "pool" / "chunks" / "seed1" / "chunk_spec.json").write_text("{}")
    output = tmp_path / "ds" / "data.hdf5"
    calls = []

    def fake_run(cmd, check):
        calls.append(cmd)
        work = Path(cmd[cmd.index("--work") + 1])
        n = int(cmd[cmd.index("--n") + 1])
        fake_official_outputs(work, n, keep_original={2})
        parts, fallback = nr.regroup(work, n, 2)
        rendered = []
        for p in parts:                                   # stand-in for dataset_states_to_obs: copy with an obs group
            out = Path(str(p).replace(".hdf5", "_obs.hdf5"))
            with h5py.File(p, "r") as src, h5py.File(out, "w") as dst:
                src.copy("data", dst)
            rendered.append(str(out))
        (work / "replay_result.json").write_text(json.dumps({"parts": rendered, "fallback_original_states": fallback}))

    monkeypatch.setattr(nr.subprocess, "run", fake_run)
    obs = {"cameras": ["agentview", "robot0_eye_in_hand"], "image_size": [84, 84]}
    out, summary = nr.replay_selection(cands, pool=tmp_path / "pool", output=output, cfg={"render_processes": 2}, observation=obs)
    assert [c["demo_key"] for c in out] == ["demo_0", "demo_1", "demo_2", "demo_3"]
    assert all(c["demo_file"].endswith("_obs.hdf5") for c in out) and out[3]["chunk"] == "seed1" and out[3]["attempt"] == 3
    assert summary["fallback_original_states"] == 1 and summary["action_noise_std"] == 0.03 and summary["num_retries"] == 6
    cmd = calls[0]
    assert cmd[cmd.index("--cameras") + 1:cmd.index("--cameras") + 3] == ["agentview", "robot0_eye_in_hand"]
    assert cmd[cmd.index("--done-mode") + 1] == "2" and cmd[cmd.index("--compress") + 1] == "1"
    nr.replay_selection(cands, pool=tmp_path / "pool", output=output, cfg={"render_processes": 2}, observation=obs)
    assert len(calls) == 1                                # same selection and settings: the finished work is reused
    nr.replay_selection(cands[:3], pool=tmp_path / "pool", output=output, cfg={"render_processes": 2}, observation=obs)
    assert len(calls) == 2                                # a different selection starts over


def test_cpgen_config_enables_the_replay_and_other_generators_pass_through(tmp_path, monkeypatch):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    cfg = resolved(compose("generate", ["generator=cpgen", "task=coffee", "scene=official_wide", "method=none",
                                        "generation=test/pilot20"]))
    assert cfg["generator"]["noisy_state_replay"] == {"enabled": True, "num_retries": 6, "action_noise_std": 0.03,
                                                      "num_processes": 12, "render_processes": 12, "done_mode": 2,
                                                      "compress": True}
    seen = {}
    monkeypatch.setattr(nr, "replay_selection", lambda chosen, **kw: seen.update(kw) or (chosen, {"ok": 1}))
    _, info = make_generator(cfg).prepare_selection([{"chunk": "seed1"}], pool=tmp_path, output=tmp_path / "d.hdf5",
                                                    observation={"cameras": []})
    assert info == {"ok": 1} and seen["cfg"]["enabled"] is True
    mg = resolved(compose("generate", ["generator=mimicgen", "task=coffee", "scene=ours", "method=none", "generation=test/pilot20"]))
    chosen = [{"chunk": "seed1"}]
    assert make_generator(mg).prepare_selection(chosen, pool=tmp_path, output=tmp_path / "d.hdf5", observation={}) == (chosen, None)
