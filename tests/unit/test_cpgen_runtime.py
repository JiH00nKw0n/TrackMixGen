"""CP-Gen adapter units beyond the grafted fixture: ``chunk_result`` on fake outputs, ``sources.fetch`` with a fake
download and a fake windows step, and the per-attempt source pick of selection=ours."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from tmg.config import compose, resolved
from tmg.core.chunks import plan_chunks
from tmg.core.records import ATTEMPTS_JSONL, RECORDS_DIR
from tmg.generation.generators import _sources, make_generator
from tmg.generation.generators.cpgen import runtime as rt
from tmg.generation.generators.cpgen import sources, windows
from tmg.generation.method import build_method


def coffee_generator(tmp_path, monkeypatch, *overrides):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    return make_generator(resolved(compose("generate", ["generator=cpgen", "task=coffee", "scene=ours", "method=none",
                                                        "generation=test/pilot20", *overrides])))


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


def test_chunk_spec_and_command(tmp_path, monkeypatch):
    monkeypatch.setenv("TMG_CPGEN_SRC", str(tmp_path / "cpgen_src"))
    gen = coffee_generator(tmp_path, monkeypatch)
    folder = tmp_path / "chunk"
    spec = gen.prepare_chunk(plan_chunks(3, 3, 9000)[0], folder)
    assert spec.generator == "cpgen" and spec.chunk == {"index": 0, "seed": 9001, "attempts": 3} and spec.generator_config == ""
    assert spec.folder == str(folder) and spec.records_dir == str(folder / RECORDS_DIR)
    assert spec.expected_outputs == [str(folder / RECORDS_DIR / ATTEMPTS_JSONL)]
    assert list(spec.extra) == ["cpgen_config", "source_root", "graft_hashes", "all_sources", "per_subtask_sources",
                                "resolve_after_bridge"]
    assert spec.extra["resolve_after_bridge"] is True                       # cpgen.yaml: on by default
    assert spec.extra["all_sources"] is False and spec.extra["per_subtask_sources"] is False and spec.extra["graft_hashes"] == {
        "generate_demo": "a497e9da6ef4f84624c9646fd49ee55fae1dedaa93e428c90f8fa312972b40e2",
        "solve_constraint": "efb3f236ae9e51ae2e38f98819ff2f0f5d40b5bdeda75c43df7bb6d984582711"}
    assert spec.extra["source_root"] == str(tmp_path / "cpgen_src")
    assert spec.extra["cpgen_config"] == json.loads((folder / "cpgen_config.json").read_text())
    assert gen.chunk_command(spec, folder / "chunk_spec.json") == [
        sys.executable, "-m", "tmg.generation.generators.cpgen.entrypoint", "--spec", str(folder / "chunk_spec.json")]
    with pytest.raises(ValueError):
        coffee_generator(tmp_path, monkeypatch, "generation.protocol.camera_names=[agentview]").prepare_chunk(
            plan_chunks(3, 3, 9000)[0], tmp_path / "bad")


def test_chunk_result(tmp_path, monkeypatch):
    gen = coffee_generator(tmp_path, monkeypatch)
    folder = tmp_path / "chunk"
    spec = gen.prepare_chunk(plan_chunks(3, 3, 9000)[0], folder)
    assert gen.chunk_result(spec) is None                       # no records at all
    write_records(folder / RECORDS_DIR, [True, False])
    assert gen.chunk_result(spec) is None                       # fewer attempts than planned: still running
    write_records(folder / RECORDS_DIR, [True, False, True])
    assert gen.chunk_result(spec) is None                       # successes but no rendered file yet
    write_demos(folder / "generated_obs.hdf5", 1)
    assert gen.chunk_result(spec) is None                       # rendering not finished
    write_demos(folder / "generated_obs.hdf5", 2)
    assert gen.chunk_result(spec) == {"attempts": 3, "successes": 2, "exported_episodes": 2,
                                      "generated_hdf5": str(folder / "generated_obs.hdf5")}
    write_records(folder / RECORDS_DIR, [False, False, False])
    (folder / "generated_obs.hdf5").unlink()
    assert gen.chunk_result(spec) == {"attempts": 3, "successes": 0, "exported_episodes": 0, "generated_hdf5": None}


def fake_download_factory(calls, demos=10):
    def fake_download(repo_id, filename, path, revision=None):
        calls.append((repo_id, filename, str(path)))
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(json.dumps([{"name": "gripper0_right-coffee_pod", "timesteps": [1, 2]},
                                        {"name": "coffee_pod-coffee_machine", "timesteps": [5, 6]}]))
        else:
            with h5py.File(path, "w") as f:
                for i in range(demos):
                    f.create_group(f"data/demo_{i}")
        return path
    return fake_download


def test_fetch_downloads_both_files_and_derives_windows(tmp_path, monkeypatch):
    source, annotation = tmp_path / "cpgen" / "coffee" / "source.hdf5", tmp_path / "cpgen" / "coffee" / "constraints.json"
    calls, annotated = [], []
    monkeypatch.setattr(_sources, "download", fake_download_factory(calls))
    monkeypatch.setattr(windows, "has_constraint_data", lambda path: bool(annotated))

    md5_before_windows = []

    def fake_annotate(demo_path, constraints_path, *, source_root, report_path=None):
        annotated.append((demo_path, constraints_path, source_root, report_path))
        md5_before_windows.append(sources.file_md5(Path(demo_path)))
        with h5py.File(demo_path, "a") as f:                                      # the real step appends constraint_data groups
            f.require_group("data/demo_0/constraint_data")
        Path(report_path).write_text("{}")
        return {"thresholds": [0.01, 0.03], "calibration": [{"found": True}], "written": {0: 2, 1: 2}}

    monkeypatch.setattr(windows, "annotate_source", fake_annotate)
    task = {"name": "coffee", "source": str(source), "annotation": str(annotation), "source_file": "datasets/source/coffee.hdf5",
            "annotation_file": "datasets/source/coffee/constraints.json", "num_sources": 10}
    generator = {"source_repo": "cpgen/datasets-src", "source_root": "/opt/src/cpgen"}
    out = sources.fetch(task, generator)
    assert calls == [("cpgen/datasets-src", "datasets/source/coffee.hdf5", str(source)),
                     ("cpgen/datasets-src", "datasets/source/coffee/constraints.json", str(annotation))]
    assert annotated == [(str(source), str(annotation), "/opt/src/cpgen", str(source.with_suffix(".windows.json")))]
    assert list(out) == ["path", "bytes", "md5", "demos", "annotation", "annotation_md5", "constraints", "expected_sources", "windows"]
    assert out["path"] == str(source) and out["demos"] == 10 and out["constraints"] == 2 and out["expected_sources"] == 10
    assert out["md5"] == sources.file_md5(source) and out["annotation_md5"] == sources.file_md5(annotation)
    assert out["md5"] != md5_before_windows[0]                                    # recomputed after the windows were appended
    assert out["windows"] == {"thresholds_m": [0.01, 0.03], "calibration_demo0": [{"found": True}], "written": {0: 2, 1: 2},
                              "report": str(source.with_suffix(".windows.json"))}
    manifest = source.with_suffix(".manifest.json")
    assert manifest.read_text() == json.dumps(out, indent=1)
    # everything present: the manifest is returned as is, nothing downloaded or derived again
    assert sources.fetch(task, generator) == json.loads(manifest.read_text()) and len(calls) == 2 and len(annotated) == 1
    # files and manifest present but the windows gone (has_constraint_data False): derived again, nothing downloaded
    annotated.clear()
    again = sources.fetch(task, generator)
    assert len(calls) == 2 and len(annotated) == 1 and again["windows"]["report"] == str(source.with_suffix(".windows.json"))
    assert manifest.read_text() == json.dumps(again, indent=1)


def test_fetch_without_windows_and_count_mismatch(tmp_path, monkeypatch):
    source, annotation = tmp_path / "s.hdf5", tmp_path / "c.json"
    calls = []
    monkeypatch.setattr(_sources, "download", fake_download_factory(calls, demos=3))
    monkeypatch.setattr(windows, "annotate_source", lambda *a, **k: pytest.fail("windows must not be derived"))
    task = {"name": "coffee", "source": str(source), "annotation": str(annotation), "source_file": "a", "annotation_file": "b",
            "per_demo_windows": False}
    out = sources.fetch(task, {"source_repo": "r", "source_root": "/x"})
    assert "windows" not in out and out["demos"] == 3 and out["expected_sources"] is None
    source.with_suffix(".manifest.json").unlink()                  # otherwise the complete set is returned as is
    with pytest.raises(RuntimeError, match="3 demos, task config expects 10"):
        sources.fetch(dict(task, num_sources=10), {"source_repo": "r", "source_root": "/x"})
    assert not source.with_suffix(".manifest.json").exists()                     # nothing written on the mismatch
    with pytest.raises(RuntimeError, match="3 demos, task config expects 2"):      # too many demos is a mismatch too
        sources.fetch(dict(task, num_sources=2), {"source_repo": "r", "source_root": "/x"})


class PickPolicy:
    def __init__(self, n):
        self.constraint_sequences = [object()] * n
        self.current_constraint_sequence_idx = 0
        self.current_constraint_idx = 7


def test_all_sources_picks_one_source_per_attempt_from_the_chunk_stream(tmp_path):
    method = build_method({"name": "none", "mixing": {"enabled": False}, "tracking": {"enabled": False}}, chunk_seed=3)
    runtime = rt.CPGenRuntime(method, chunk_seed=3, records_dir=None, all_sources=True)
    policy = PickPolicy(10)
    gen = SimpleNamespace(policy=policy, env=SimpleNamespace(env=None))
    picks = []
    for _ in range(6):
        runtime.begin_attempt(gen)
        picks.append(policy.current_constraint_sequence_idx)
        assert policy.current_constraint_idx == 0
    expected = np.random.default_rng(np.random.SeedSequence([3, 515151]))
    assert picks == [int(expected.integers(10)) for _ in range(6)]
    official = rt.CPGenRuntime(method, chunk_seed=3, records_dir=None)
    many = PickPolicy(10)
    official.begin_attempt(SimpleNamespace(policy=many, env=SimpleNamespace(env=None)))
    assert many.current_constraint_sequence_idx == 0 and many.current_constraint_idx == 7        # official: untouched
    only_one = PickPolicy(1)
    runtime.begin_attempt(SimpleNamespace(policy=only_one, env=SimpleNamespace(env=None)))
    assert only_one.current_constraint_idx == 7                                   # one source: nothing to pick, policy untouched
    assert official.all_sources is False and runtime.all_sources is True


def test_end_attempt_records_a_record_error_file(tmp_path):
    method = build_method({"name": "none", "mixing": {"enabled": False}, "tracking": {"enabled": False}}, chunk_seed=3)
    runtime = rt.CPGenRuntime(method, chunk_seed=3, records_dir=tmp_path / "records")
    runtime.begin_attempt(SimpleNamespace(policy=PickPolicy(1), env=SimpleNamespace(env=None)))
    runtime.rec.flush = lambda **kw: (_ for _ in ()).throw(OSError("disk full"))
    with pytest.raises(OSError):
        runtime.end_attempt(SimpleNamespace(policy=PickPolicy(1)), {"success": True})
    payload = json.loads((tmp_path / "records" / "RECORD_ERROR.json").read_text())
    assert list(payload) == ["attempt", "error", "traceback", "time"]
    assert payload["attempt"] == 0 and payload["error"] == "OSError('disk full')" and "disk full" in payload["traceback"]
    assert isinstance(payload["time"], float)


def test_source_picker_draws_a_source_per_subtask_with_selection_ours():
    """selection=ours with select_src_per_subtask: every advance to the next constraint draws a new source; without the
    flag the attempt keeps its first source."""
    from tmg.generation.generators.cpgen.runtime import SourcePicker

    class Policy:
        def __init__(self):
            self.constraint_sequences = [[f"s{j}c{k}" for k in range(3)] for j in range(10)]
            self.current_constraint_idx = 0
            self.current_constraint_sequence_idx = 0

        def advance_constraint(self):
            self.current_constraint_idx += 1

    seen = set()
    for seed in range(20):
        pol = Policy()
        SourcePicker(True, seed, per_subtask=True).pick(pol)
        chosen = [pol.current_constraint_sequence_idx]
        for _ in range(2):
            pol.advance_constraint()
            chosen.append(pol.current_constraint_sequence_idx)
        assert pol.current_constraint_idx == 2 and all(0 <= j < 10 for j in chosen)
        seen.add(len(set(chosen)) > 1)
    assert True in seen                                              # sources differ across subtasks in some attempts
    pol = Policy()
    SourcePicker(True, 0, per_subtask=False).pick(pol)
    first = pol.current_constraint_sequence_idx
    pol.advance_constraint(); pol.advance_constraint()
    assert pol.current_constraint_sequence_idx == first


# re-solve after the bridge ---------------------------------------------------------------------------------------------
def _pose(x=0.0, z=0.8):
    out = np.eye(4)
    out[:3, 3] = [x, 0.0, z]
    return out


def _constraint(name, kind, objects, carries, x, steps=3):
    """One annotated CP-Gen constraint as far as the runtime and the mixing read it."""
    return SimpleNamespace(name=name, constraint_type=kind, obj_names=list(objects), attached_object_names=list(carries),
                           duplicate_actions=False, timesteps=list(range(steps)), src_gripper_action=np.full(steps, 1.0),
                           src_obj_pose={o: np.stack([_pose(x)] * steps) for o in objects},
                           src_obs={"robot0_eef_pos": np.tile([x, 0.0, 1.0], (steps, 1)),
                                    "robot0_eef_quat_site": np.tile([0, 0, 0, 1.0], (steps, 1))})


class GraspThenPlacePolicy:
    """CP-Gen's policy for a task of two constraints (grasp the pod, place it on the machine) as the grafted
    ``generate_demo`` drives it: ``get_action`` solves the current constraint when the queue is empty and queues a bridge
    of ``BRIDGE`` steps plus the solved steps; the solve calls the runtime's ``prepare_mix`` as the grafted
    ``solve_constraint`` does. ``resolve`` sets what a solve after the first of a constraint returns."""
    BRIDGE, STEPS = 2, 3
    robot_configuration = eef_configuration = None
    ignore_scaling = True

    def __init__(self, runtime, resolve=None):
        self.runtime, self.resolve = runtime, {"steps": self.STEPS, "success": True, **(resolve or {})}
        self.constraint_sequences = [[_constraint("grasp", "robot-object", ["pod"], [], x),
                                      _constraint("place", "object-object", ["pod", "machine"], ["pod"], x)] for x in (0.0, 0.3)]
        self.current_constraint_sequence_idx, self.current_constraint_idx = 0, 0
        self.action_queue, self.new_constraint, self.solves = [], False, {0: 0, 1: 0}
        self.curr_motion_plan_steps = self.curr_constraint_steps = 0

    @property
    def current_constraint(self):
        return self.constraint_sequences[self.current_constraint_sequence_idx][self.current_constraint_idx]

    def solve_constraint(self, constraint, obs):
        eye = np.eye(4)
        self.runtime.prepare_mix(self, constraint, obs, eye, eye, eye)
        k = self.current_constraint_idx
        self.solves[k] += 1
        again = self.solves[k] > 1
        steps = self.resolve["steps"] if again else self.STEPS
        return SimpleNamespace(success=self.resolve["success"] if again else True,
                               actions=[np.full(7, 10.0 * k + self.solves[k]) for _ in range(steps)])

    def get_action(self, obs):
        self.new_constraint = False
        if not self.action_queue:
            if self.solves[self.current_constraint_idx]:
                self.current_constraint_idx += 1
            solved = self.solve_constraint(self.current_constraint, obs)
            self.curr_motion_plan_steps, self.curr_constraint_steps = self.BRIDGE, len(solved.actions)
            self.action_queue = [np.zeros(7) for _ in range(self.BRIDGE)] + list(solved.actions)
            self.new_constraint = True
        return {"action": self.action_queue.pop(0)}


def _grasp_then_place(tmp_path, method="none", resolve=None, resolve_after_bridge=True):
    """Run one attempt of the two-constraint task through the grafted callbacks: ``(executed actions, attempt record,
    policy)``. The pod sits 2 cm from where the hand left it when the placement starts."""
    cfg = {"name": method, "mixing": {"enabled": False}}
    if method == "skillmix":                     # source 0 alone in both constraints (d <= 0), so no other source is solved
        from tmg.generation.fit_coefficients import write_rule

        block = {"a": [5.0, 0.0], "g": dict.fromkeys(("dx", "dy", "rotation", "inhand_position", "inhand_rotation"), 0.0),
                 "d": -1.0, "c0": 0.0}
        path = write_rule(tmp_path / "rule.yaml", generator="cpgen", task="t", basis="test", num_sources=2, blocks={0: block, 1: block})
        cfg["mixing"] = {"enabled": True, "type": "skillmix", "rule": "softmax", "coefficients": str(path)}
    runtime = rt.CPGenRuntime(build_method(cfg, chunk_seed=3), chunk_seed=3, records_dir=tmp_path / "records", all_sources=True,
                              resolve_after_bridge=resolve_after_bridge)
    runtime.official = SimpleNamespace()                                      # no other source is solved: never called
    policy = GraspThenPlacePolicy(runtime, resolve)
    generator = SimpleNamespace(policy=policy, env=SimpleNamespace(env=None))
    runtime.begin_attempt(generator)
    policy.current_constraint_sequence_idx = 0                                # the attempt's source, whatever was drawn
    obs = {"robot0_eef_pos": np.array([0.0, 0.0, 1.0]), "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
           "robot0_eef_quat_site": np.array([0, 0, 0, 1.0]), "pod_pose": _pose(0.02), "machine_pose": _pose(0.5)}
    noise = SimpleNamespace(constraint_segment_noise_magnitude=0.0, motion_segment_noise_magnitude=0.0)
    executed = []
    for _ in range(2 * (policy.BRIDGE + policy.STEPS)):
        action = runtime.get_action(generator, obs)["action"]
        runtime.record_step(action, action, 0.0, False, noise)
        executed.append(float(action[0]))
    runtime.end_attempt(generator, {"success": True})
    record = json.loads((tmp_path / "records" / ATTEMPTS_JSONL).read_text().splitlines()[-1])
    return executed, record, policy


def test_resolve_after_bridge_executes_the_second_solve_of_a_carrying_constraint(tmp_path):
    executed, record, policy = _grasp_then_place(tmp_path)
    # the grasp carries nothing and keeps its plan (solve 1); the placement is solved again after its bridge (solve 2)
    assert executed == [0, 0, 1, 1, 1, 0, 0, 12, 12, 12] and policy.solves == {0: 1, 1: 2}
    grasp, place = record["subtasks"]
    assert "resolve_after_bridge" not in grasp and place["resolve_after_bridge"] == "replaced"
    off, record, policy = _grasp_then_place(tmp_path / "off", resolve_after_bridge=False)
    assert off == [0, 0, 1, 1, 1, 0, 0, 11, 11, 11] and policy.solves == {0: 1, 1: 1}
    assert "resolve_after_bridge" not in record["subtasks"][1]


def test_resolve_after_bridge_keeps_the_plan_when_the_second_solve_does_not_fit(tmp_path):
    for i, (resolve, outcome) in enumerate((({"steps": 4}, "kept: the re-solve has 4 steps, the plan 3"),
                                            ({"success": False}, "kept: the re-solve failed"))):
        executed, record, policy = _grasp_then_place(tmp_path / str(i), resolve=resolve)
        assert executed[-3:] == [11, 11, 11] and policy.solves[1] == 2
        assert record["subtasks"][1]["resolve_after_bridge"] == outcome


def test_resolve_after_bridge_records_the_mix_that_was_executed(tmp_path):
    _, record, _ = _grasp_then_place(tmp_path, method="skillmix")
    place = record["subtasks"][1]
    first, executed = place["mix_before_resolve"], place["mix"]
    assert place["resolve_after_bridge"] == "replaced" and executed["weights"] == [1.0, 0.0]
    # both solves describe the same hand: the in-hand features are those of the pod carried into the placement, 2 cm from
    # where source 0 held it, not of the machine it is placed on
    assert first["carried_object"] == executed["carried_object"] == "pod"
    assert executed["inhand_features"] == first["inhand_features"] and executed["inhand_features"][0][0] == pytest.approx(0.02)
    assert "unplanned_mixes" not in record                                    # the second solve is not a failed plan
    _, record, _ = _grasp_then_place(tmp_path / "kept", method="skillmix", resolve={"success": False})
    assert "mix_before_resolve" not in record["subtasks"][1] and "unplanned_mixes" not in record
