"""The pieces shared by the generator adapters: records summary, RECORD_ERROR marker, entrypoint helpers, graft
installation, the adapter registry and the CP-Gen checkout context."""
import inspect
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

from tmg.core.records import ATTEMPTS_JSONL, RECORD_ERROR, AttemptRecorder
from tmg.generation.base import ChunkSpec, SyntheticDataGenerator
from tmg.generation.generators import ADAPTERS, _entry, _shared, adapter_class, make_generator
from tmg.generation.generators.cpgen import runtime as cpgen_rt
from tmg.generation.generators.cpgen.checkout import cpgen_checkout
from tmg.generation.generators.mimicgen import runtime as mimicgen_rt
from tmg.generation.graft import Graft, GraftError, install_grafts, source_hash


def write_records(records_dir: Path, successes: list[bool], steps: int = 2) -> None:
    records_dir.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, ok in enumerate(successes):
        npz = f"attempt_{i:06d}.npz"
        np.savez(records_dir / npz, w_pos=np.zeros((steps, 1, 3), dtype=np.float32))
        lines.append(json.dumps({"attempt": i, "success": ok, "n_steps": steps, "npz": npz}))
    (records_dir / ATTEMPTS_JSONL).write_text("\n".join(lines) + "\n")


# _shared ------------------------------------------------------------------------------------------------------
def test_records_summary_checking_the_verdict(tmp_path):
    records = tmp_path / "records"
    assert _shared.records_summary(records, 3, check_ok=True) is None                 # no attempts.jsonl: not finished
    write_records(records, [True, False])
    assert _shared.records_summary(records, 3, check_ok=True) == {"error": "2 attempt records, 3 planned"}
    write_records(records, [True, False, True])
    summary = _shared.records_summary(records, 3, check_ok=True)
    assert (summary["successes"], summary["failures"], len(summary["lines"])) == (2, 1, 3)
    (records / RECORD_ERROR).write_text("{}")
    assert _shared.records_summary(records, 3, check_ok=True) == {"error": f"record writing failed, see {records / RECORD_ERROR}"}


def test_records_summary_ignoring_the_verdict(tmp_path):
    records = tmp_path / "records"
    assert _shared.records_summary(records, 3, check_ok=False) is None
    write_records(records, [True, False])
    assert _shared.records_summary(records, 3, check_ok=False) is None               # short: still running
    write_records(records, [True, False, True])
    (records / "attempt_000001.npz").unlink()                                          # bad records, but not checked
    summary = _shared.records_summary(records, 3, check_ok=False)
    assert (summary["successes"], summary["failures"], len(summary["lines"])) == (2, 1, 3)


def test_count_hdf5_demos(tmp_path):
    path = tmp_path / "d.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("other")
    assert _shared.count_hdf5_demos(path) == 0
    with h5py.File(path, "a") as f:
        f.create_group("data/demo_0")
        f.create_group("data/demo_1")
    assert _shared.count_hdf5_demos(path) == 2


def test_record_error_guard_writes_the_marker_and_reraises(tmp_path):
    rec = AttemptRecorder()
    rec.begin_attempt()
    rec.begin_attempt()
    with pytest.raises(ValueError), _shared.record_error_guard(tmp_path / "records", rec):
        raise ValueError("boom")
    payload = json.loads((tmp_path / "records" / RECORD_ERROR).read_text())
    assert payload["attempt"] == 1 and payload["error"] == "ValueError('boom')" and "boom" in payload["traceback"]
    assert isinstance(payload["time"], float) and list(payload) == ["attempt", "error", "traceback", "time"]
    with pytest.raises(ValueError), _shared.record_error_guard(None, rec):          # records disabled: nothing written
        raise ValueError("boom")


# _entry -------------------------------------------------------------------------------------------------------
def spec_dict(tmp_path, **extra) -> dict:
    return {"generator": "x", "chunk": {"index": 0, "seed": 11, "attempts": 2}, "folder": str(tmp_path), "records_dir": str(tmp_path / "records"),
            "task": {}, "scene": {}, "method": {"name": "none", "mixing": {"enabled": False}, "tracking": {"enabled": False}},
            "protocol": {}, "generator_config": "", "expected_outputs": [], "extra": {"graft_hashes": {"a": "b"}, **extra}}


def test_load_spec_reads_the_json(tmp_path):
    path = tmp_path / "chunk_spec.json"
    path.write_text(json.dumps(spec_dict(tmp_path)))
    assert _entry.load_spec(["--spec", str(path)]) == spec_dict(tmp_path)
    with pytest.raises(SystemExit):
        _entry.load_spec([])


def test_install_runtime_calls_apply_or_skips(tmp_path, capsys):
    seen = {}

    def fake_apply(method, *, chunk_seed, records_dir, graft_hashes, **extra):
        seen.update(method=method.name, chunk_seed=chunk_seed, records_dir=records_dir, graft_hashes=graft_hashes, extra=extra)
        return {"installed": True}

    assert _entry.install_runtime(spec_dict(tmp_path), fake_apply, all_sources=True) == {"installed": True}
    assert seen == {"method": "none", "chunk_seed": 11, "records_dir": tmp_path / "records", "graft_hashes": {"a": "b"},
                    "extra": {"all_sources": True}}
    assert '"installed": true' in capsys.readouterr().out
    seen.clear()
    assert _entry.install_runtime(spec_dict(tmp_path, disable_graft=True), fake_apply) is None
    assert not seen and "DISABLED" in capsys.readouterr().out


# graft installation ---------------------------------------------------------------------------------------------
class Official:
    def run(self, items):
        total = 0
        for value in items:
            total = total + value
        return total

    def other(self):
        return 1


def test_install_grafts_swaps_and_restore_puts_back():
    original_run, original_other = Official.run, Official.other
    run = Graft(Official.run, expected_hash=None, label="Official.run").insert_before_final_return("_h.append(total)")
    other = Graft(Official.other, expected_hash=None, label="Official.other").insert_before_final_return("_h.append(-1)")
    seen = []
    installed = install_grafts([(Official, "run", run), (Official, "other", other)], {"_h": seen})
    assert Official().run([1, 2]) == 3 and Official().other() == 1 and seen == [3, -1]
    assert set(installed.info) == {"run", "other"} and installed.info["run"]["label"] == "Official.run"
    assert installed.info["run"]["hash"] == source_hash(original_run) and len(installed.info["run"]["edits"]) == 1
    installed.restore()
    assert Official.run is original_run and Official.other is original_other
    installed.restore()                                                               # idempotent
    assert Official.run is original_run


def test_find_returns_covers_nested_functions_in_source_order():
    def outer(x):
        def inner(y):
            return y
        if x:
            return inner(x)
        return 0

    g = Graft(outer, expected_hash=None, label="outer")
    hits = g._find_returns()
    assert [stmts[i].value.__class__.__name__ for stmts, i in hits] == ["Name", "Call", "Constant"]
    with pytest.raises(GraftError):
        g.insert_before_return("pass", occurrence=0, expected_returns=2)              # three, the nested one counts


def grafted_call_names(graft: Graft) -> list[tuple[str, int]]:
    """``(callback name, argument count)`` of every ``_tmg.<name>(...)`` call the graft's edits insert."""
    import ast

    calls = []
    for node in ast.walk(graft.node):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) \
                and node.func.value.id == "_tmg":
            calls.append((node.func.attr, len(node.args)))
    return calls


def test_grafted_calls_match_the_runtime_callbacks():
    """Every ``_tmg.xxx(...)`` statement the grafts insert names a runtime method taking exactly those arguments."""
    def generate(self, env, env_interface):                     # the anchor names only; the values do not matter
        new_initial_state = env.get_state()                      # noqa: F841
        src_subtask_object_pose = 1                              # noqa: F841
        src_subtask_eef_poses, src_subtask_target_poses, src_subtask_gripper_actions = 1, 2, 3   # noqa: F841
        transformed_seq = 4                                      # noqa: F841
        results = {}
        return results

    def execute(self, env, env_interface):
        for seq in self.waypoint_sequences:
            for j in range(len(seq)):
                waypoint = seq[j]
                action_pose = env_interface.target_pose_to_action(waypoint)
                play_action = action_pose
                datagen_info = env_interface.get_datagen_info(play_action)      # noqa: F841
                cur_success_metrics = env.is_success()                          # noqa: F841

    gen, exe = mimicgen_rt.build_grafts(generate, execute, {})
    cp = cpgen_rt.build_graft(_cpgen_like, None)
    for graft, runtime_cls in ((gen, mimicgen_rt.MimicGenRuntime), (exe, mimicgen_rt.MimicGenRuntime), (cp, cpgen_rt.CPGenRuntime)):
        for name, n_args in grafted_call_names(graft):
            method = getattr(runtime_cls, name)
            params = [p for p in inspect.signature(method).parameters.values() if p.name != "self"]
            assert len(params) == n_args, (graft.label, name, n_args, [p.name for p in params])


def _cpgen_like(self, system_noise_cfg=None):
    obs = self.env.reset()
    state = self.env.get_state()
    while True:
        action_dct = self.policy.get_action(obs)
        action = action_dct["action"]
        action_to_take = action
        in_constraint_segment = False
        reward = 0
        state = self.env.get_state()
        break
    episode_data = {"actions": [action], "state": state, "action_to_take": action_to_take, "r": reward, "c": in_constraint_segment}
    return episode_data


# registry and base defaults -------------------------------------------------------------------------------------
def test_adapter_registry_names_and_unknown_generator():
    assert list(ADAPTERS) == ["mimicgen", "softmimicgen", "dexmimicgen", "cpgen"]
    assert ADAPTERS == {"mimicgen": "tmg.generation.generators.mimicgen.adapter:MimicGenGenerator",
                        "softmimicgen": "tmg.generation.generators.isaaclab.adapter:SoftMimicGenGenerator",
                        "dexmimicgen": "tmg.generation.generators.autodata.adapter:DexMimicGenGenerator",
                        "cpgen": "tmg.generation.generators.cpgen.adapter:CPGenGenerator"}
    assert adapter_class("cpgen").__name__ == "CPGenGenerator" and adapter_class("mimicgen").name == "mimicgen"
    with pytest.raises(KeyError, match="no generator adapter named 'nope' \\(available: mimicgen, softmimicgen, dexmimicgen, cpgen\\)"):
        make_generator({"generator": {"name": "nope"}})


def test_registry_imports_adapters_only_on_use():
    """The runner imports the registry where the official packages are absent: no adapter module may load with it."""
    import subprocess

    code = ("import sys; from tmg.generation.generators import ADAPTERS, adapter_class; "
            "loaded = sorted(m for m in sys.modules if m.startswith('tmg.generation.generators.') and m.endswith('.adapter')); "
            "adapter_class('cpgen'); "
            "after = sorted(m for m in sys.modules if m.startswith('tmg.generation.generators.') and m.endswith('.adapter')); "
            "print(loaded, after)")
    out = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True).stdout.strip()
    assert out == "[] ['tmg.generation.generators.cpgen.adapter']"


class Bare(SyntheticDataGenerator):
    name = "bare"

    def fetch_sources(self):
        return {}

    def prepare_chunk(self, chunk, folder):
        return self._spec(chunk, folder, scene={"s": 1}, generator_config="", expected_outputs=[], extra={"k": "v"})

    def chunk_result(self, spec):
        return None

    def demo_layout(self):
        return {}


def test_base_spec_helper_and_default_command(tmp_path):
    from tmg.core.chunks import Chunk

    gen = Bare({"generator": {"name": "bare_cfg"}, "task": {"name": "t"}, "generation": {"protocol": {"p": 1}}})
    spec = gen.prepare_chunk(Chunk(index=2, seed=9, attempts=4), tmp_path / "c")
    assert spec == ChunkSpec(generator="bare_cfg", chunk={"index": 2, "seed": 9, "attempts": 4}, folder=str(tmp_path / "c"),
                             records_dir=str(tmp_path / "c" / "records"), task={"name": "t"}, scene={"s": 1}, method={},
                             protocol={"p": 1}, generator_config="", expected_outputs=[], extra={"k": "v"})
    with pytest.raises(NotImplementedError):
        gen.chunk_command(spec, tmp_path / "chunk_spec.json")
    gen.entrypoint_module = "some.module"
    assert gen.chunk_command(spec, tmp_path / "chunk_spec.json") == [sys.executable, "-m", "some.module", "--spec",
                                                                     str(tmp_path / "chunk_spec.json")]


# robosuite base ---------------------------------------------------------------------------------------------------
def test_robosuite_env_name_resolves_a_selector_only_scene_from_the_scene_files():
    """``scene`` may name the scene only; ``scene_chain`` loads configs/experiment/generator/cpgen/scene/<name>.yaml."""
    from tmg.generation.generators.cpgen.adapter import CPGenGenerator

    gen = CPGenGenerator({"generator": {"name": "cpgen"}, "task": {"name": "threading", "env_prefix": "Threading"},
                          "scene": {"name": "official_wide"}})
    chain = gen.scene_chain()
    assert chain["env_class"] == "{env_prefix}Wide" and chain["env_suffix"] == "Wide" and chain["override_bounds"] is False
    assert gen.env_name() == "ThreadingWide"
    gen.scene = {"name": "official_d0"}
    assert gen.env_name() == "Threading_D0"
    gen.scene = {"name": "ours", "generator": "cpgen", "env_suffix": "OURS", "bounds": {}, "override_bounds": False}   # details inline
    assert gen.env_name() == "Threading_OURS"


# cpgen checkout -------------------------------------------------------------------------------------------------
def test_cpgen_checkout_mirrors_and_restores_the_working_directory(tmp_path):
    root = tmp_path / "cpgen"
    (root / "demo_aug").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "README").write_text("x")
    before = os.getcwd()
    with cpgen_checkout(root, workdir=tmp_path / "chunk" / "cwd") as cwd:
        assert cwd == tmp_path / "chunk" / "cwd" and Path(os.getcwd()).resolve() == cwd.resolve()
        assert sorted(p.name for p in cwd.iterdir()) == ["README", "demo_aug", "scripts"]
        assert all(p.is_symlink() for p in cwd.iterdir()) and sys.path[0] == str(root)
    assert os.getcwd() == before
    with cpgen_checkout(root) as cwd:                                                 # no mirror: the checkout itself
        assert cwd == root
    (tmp_path / "not_a_checkout").mkdir()                                            # exists, but is no CP-Gen checkout
    with pytest.raises(FileNotFoundError, match="has no demo_aug/"), cpgen_checkout(tmp_path / "not_a_checkout"):
        pass
    assert os.getcwd() == before
    with pytest.raises(RuntimeError, match="inside"), cpgen_checkout(root, workdir=tmp_path / "chunk2" / "cwd"):
        assert Path(os.getcwd()).resolve() == (tmp_path / "chunk2" / "cwd").resolve()
        raise RuntimeError("inside")
    assert os.getcwd() == before                                                      # restored on the error path too
