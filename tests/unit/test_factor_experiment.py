"""The SkillMix factor experiment (docs/SKILLMIX_FACTOR_EXPERIMENT.md): derived rule files, the condition methods, the
generation configs, the mixing report of a check pool and the driver."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import derive_coefficients as derive_cli
from scripts import mix_report
from tmg.config import compose, resolved
from tmg.generation import derive_coefficients as dc
from tmg.generation.base import apply_method_overrides
from tmg.generation.fit_coefficients import write_rule
from tmg.generation.method import build_method
from tmg.generation.method.skillmix import effective_sources, load_rule, rule_path

REPO = Path(__file__).resolve().parents[2]
# condition of the experiment -> (method config, what its mixing block adds to the method it inherits)
CONDITIONS = {
    "C2": ("skillmix_select", "skillmix_sample", {"derived": "select_only"}),
    "C3": ("skillmix_top1", "skillmix", {"derived": "select_only"}),
    "C4": ("skillmix_uniform", "skillmix", {"derived": "uniform"}),
    "C7": ("skillmix_distcap", "skillmix", {"max_source_distance": {"rot_deg": 30, "pos_cm": 3}}),
    "C8": ("skillmix_neff3", "skillmix", {"min_effective_sources": 3}),
    "G": ("skillmix_mix_grasp", "skillmix_sample", {"derived": "mix_grasp"}),
    "P": ("skillmix_mix_place", "skillmix_sample", {"derived": "mix_place"}),
}


# derived rule files ---------------------------------------------------------------------------------------------------
def _configs(tmp_path, num_subtasks=3, kinds=("open", "grasp", "place")):
    """A configs root with one fitted rule file (generator g, task t; d = 0.5 everywhere) and its stages file."""
    block = {"a": [0.1, -0.2, 0.3], "g": dict.fromkeys(("dx", "dy", "rotation", "inhand_position", "inhand_rotation"), 0.0),
             "d": 0.5, "c0": -1.0}
    root = tmp_path / "configs"
    write_rule(rule_path("g/t", "softmax", root), rule="softmax", generator="g", task="t", basis="test", num_sources=3,
               blocks={k: {**block, "a": list(block["a"])} for k in range(num_subtasks)})
    if kinds is not None:
        stages = root / "experiment/generator/g/stages/t.yaml"
        stages.parent.mkdir(parents=True)
        stages.write_text(yaml.safe_dump({"subtask_kinds": list(kinds)}))
    return root


def test_select_subtasks_by_index_and_by_kind():
    kinds = ["open", "grasp", "place", "grasp"]
    assert dc.select_subtasks("all", [0, 1, 2, 3], kinds) == [0, 1, 2, 3]
    assert dc.select_subtasks([2, 0], [0, 1, 2, 3], None) == [2, 0]
    assert dc.select_subtasks({"kind": "grasp"}, [0, 1, 2, 3], kinds) == [1, 3]
    assert dc.select_subtasks({"except_kind": "grasp"}, [0, 1, 2, 3], kinds) == [0, 2]
    for selector, kinds_given, message in (([7], kinds, "not subtasks"), ({"kind": "grasp"}, None, "one subtask kind per subtask"),
                                           ({"kind": "grasp"}, kinds[:2], "one subtask kind per subtask"),
                                           ({"kind": "pour"}, kinds, "no subtask of kind"), ("grasp", kinds, "subtasks must be")):
        with pytest.raises(ValueError, match=message):
            dc.select_subtasks(selector, [0, 1, 2, 3], kinds_given)


def test_derive_applies_the_transforms_to_a_copy():
    data = {"subtasks": {0: {"a": [0.0, 1.0], "d": 0.5}, 1: {"a": [2.0, 3.0], "d": 0.7}}}
    out = dc.derive(data, [{"type": "set_d", "subtasks": [1], "value": -1},
                           {"type": "boost_a", "subtasks": "all", "source": 0, "by": 100}])
    assert out["subtasks"] == {0: {"a": [100.0, 1.0], "d": 0.5}, 1: {"a": [102.0, 3.0], "d": -1.0}}
    assert data["subtasks"][1] == {"a": [2.0, 3.0], "d": 0.7}                  # the fitted mapping is untouched
    with pytest.raises(ValueError, match="unknown transform"):
        dc.derive(data, [{"type": "scale_g", "subtasks": "all"}])
    with pytest.raises(ValueError, match="source 5"):
        dc.derive(data, [{"type": "boost_a", "subtasks": "all", "source": 5, "by": 1}])


def test_write_derived_writes_a_readable_rule_file_that_says_how_it_was_made(tmp_path):
    root = _configs(tmp_path)
    transforms = dc.load_conditions()["mix_grasp"]                             # the committed condition, on the test task
    path = dc.write_derived("g", "t", "mix_grasp", transforms, root=root)
    assert path == root / "experiment/generator/g/coefficients/derived/mix_grasp/t.yaml"
    text = path.read_text()
    assert text.startswith("# DERIVED rule file, condition 'mix_grasp'") and "coefficients/softmax/t.yaml" in text.splitlines()[1]
    held = yaml.safe_load(text)
    fitted = rule_path("g/t", "softmax", root)
    assert held["derived"] == {"condition": "mix_grasp", "from": "experiment/generator/g/coefficients/softmax/t.yaml",
                               "from_sha256": dc.sha256(fitted), "transforms": transforms}
    rule = load_rule("g/t", "softmax", root, derived="mix_grasp")              # what a method with mixing.derived reads
    assert [rule.for_subtask(k).d for k in range(3)] == [-1.0, 0.5, -1.0]      # only the grasp subtask keeps its fitted d
    assert np.array_equal(rule.for_subtask(0).a, load_rule("g/t", "softmax", root).for_subtask(0).a)
    assert dc.is_current(path, root) and dc.derived_files(root) == [path]
    fitted.write_text(fitted.read_text().replace("d: 0.5", "d: 0.6"))          # a refit makes the derived file stale
    assert not dc.is_current(path, root)
    with pytest.raises(FileNotFoundError, match="fitted rule file"):
        dc.write_derived("g", "missing", "mix_grasp", transforms, root=root)


def test_derive_cli_writes_configured_and_inline_conditions(tmp_path, capsys):
    root = _configs(tmp_path, kinds=None)                                       # a task without subtask kinds
    shutil.copy(REPO / "configs" / dc.CONDITIONS_FILE, root / dc.CONDITIONS_FILE)
    assert derive_cli.main(["--all", "--configs", str(root)]) == 0
    out = capsys.readouterr().out
    assert sorted(p.parent.name for p in dc.derived_files(root)) == ["select_only", "uniform"]
    assert "skipped g/t mix_grasp" in out and "skipped g/t mix_place" in out   # --all skips what a task cannot have
    with pytest.raises(ValueError, match="one subtask kind per subtask"):      # asked for by name, it is an error
        derive_cli.main(["--generator", "g", "--task", "t", "--condition", "mix_grasp", "--configs", str(root)])
    assert derive_cli.main(["--generator", "g", "--task", "t", "--condition", "force_source0_last", "--configs", str(root),
                            "--transform", "{type: set_d, subtasks: all, value: -1}",
                            "--transform", "{type: boost_a, subtasks: [2], source: 0, by: 100}"]) == 0
    forced = load_rule("g/t", "softmax", root, derived="force_source0_last")
    assert forced.for_subtask(2).a.tolist() == [100.1, -0.2, 0.3] and forced.for_subtask(1).a.tolist() == [0.1, -0.2, 0.3]
    assert derive_cli.main(["--check", "--configs", str(root)]) == 0
    with pytest.raises(SystemExit):
        derive_cli.main(["--generator", "g", "--task", "t", "--condition", "no_such", "--configs", str(root)])


def test_committed_derived_rule_files_are_current_and_complete():
    files = dc.derived_files()
    assert files and all(dc.is_current(f) for f in files)                       # none stale after a refit or a hand edit
    conditions = dc.load_conditions()
    assert set(conditions) == {"select_only", "uniform", "mix_grasp", "mix_place"}
    for fitted in sorted((REPO / "configs/experiment/generator").glob("*/coefficients/softmax/*.yaml")):
        generator, task = fitted.parts[-4], fitted.stem
        subtasks = range(len(yaml.safe_load(fitted.read_text())["subtasks"]))
        base = load_rule(f"{generator}/{task}", "softmax")
        assert all(load_rule(f"{generator}/{task}", "softmax", derived="select_only").for_subtask(k).d == -1 for k in subtasks)
        uniform = load_rule(f"{generator}/{task}", "softmax", derived="uniform")
        assert all(uniform.for_subtask(k).d == 1000 for k in subtasks)
        for k in subtasks:                                                      # d = 1000: nearly equal weights, a unchanged
            scores = uniform.for_subtask(k).a
            assert np.array_equal(scores, base.for_subtask(k).a)
            w = np.exp(scores / 1000) / np.exp(scores / 1000).sum()
            assert effective_sources(w) > len(scores) - 0.01
    grasp, place = (load_rule("mimicgen/three_piece_assembly", "softmax", derived=c) for c in ("mix_grasp", "mix_place"))
    base = load_rule("mimicgen/three_piece_assembly", "softmax")
    assert [grasp.for_subtask(k).d for k in range(4)] == [base.for_subtask(0).d, -1.0, base.for_subtask(2).d, -1.0]
    assert [place.for_subtask(k).d for k in range(4)] == [-1.0, base.for_subtask(1).d, -1.0, base.for_subtask(3).d]


# methods and generation configs ----------------------------------------------------------------------------------------
def test_condition_methods_have_their_own_names_and_differ_only_in_the_factor(monkeypatch, tmp_path):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))

    def method(name, generator="mimicgen"):
        return resolved(compose("generate", [f"generator={generator}", "task=threading", "scene=ours", "selection=ours",
                                             f"method={name}", "generation=pool_factor_lowdim"]))

    names = ["none", "skillmix", *[m for m, _, _ in CONDITIONS.values()]]
    assert len(set(names)) == 9
    outputs = {method(name)["output"] for name in names}
    assert len(outputs) == 9 and all("/pool_factor_lowdim" in o for o in outputs)       # no two conditions share a folder
    for name, parent, added in CONDITIONS.values():
        plain, base = method(name)["method"], method(parent)["method"]
        assert plain["name"] == name
        assert plain["mixing"] == {**base["mixing"], **added} and plain["labels"] == base["labels"]
        mixer = build_method(plain, chunk_seed=0).mixer                          # the rule file it names exists and loads
        assert mixer.rule == "softmax"
        # CP-Gen's labels are pre-noise for every method, the conditions included
        cpgen = method(name, "cpgen")
        assert apply_method_overrides(cpgen["method"], cpgen["generator"])["labels"] == "pre_noise"
    # the existing methods are untouched: no new key enters the method identity of their finished chunks
    assert set(method("skillmix")["method"]["mixing"]) == {"enabled", "type", "rule", "coefficients", "single_source", "symmetries"}


def test_factor_generation_configs_keep_the_lowdim_protocol_under_their_own_names(monkeypatch, tmp_path):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))

    def generation(name):
        return resolved(compose("generate", ["task=threading", "scene=ours", f"generation={name}"]))["generation"]

    old, new, check = generation("pool_robosuite_lowdim"), generation("pool_factor_lowdim"), generation("test/factor_check50")
    assert new["name"] == "pool_factor_lowdim" and {**new, "name": old["name"]} == old
    assert check["name"] == "factor_check50" and (check["attempts"], check["chunk"]) == (50, 50)
    assert check["protocol"] == new["protocol"] and new["protocol"]["camera_names"] == []


# the mixing report of a check pool -------------------------------------------------------------------------------------
def _write_pool(pool: Path, attempts: list[dict]) -> None:
    records = pool / "chunks" / "chunk_0000" / "records"
    records.mkdir(parents=True)
    (records / "attempts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in attempts))


def test_mix_report_summarizes_weights_effective_sources_and_drops(tmp_path, capsys):
    def attempt(success, w0, cap=None, w1=None):
        mix0 = {"subtask": 0, "d": 0.5, "weights": w0, **({"distance_cap": cap} if cap else {})}
        subtasks = [{"subtask_ind": 0, "src_demo_ind": 0, "mix": mix0}]
        if w1 is not None:
            subtasks.append({"subtask_ind": 1, "src_demo_ind": 1, "mix": {"subtask": 1, "d": -1.0, "weights": w1,
                                                                        "min_effective_sources": 2, "temperature": 0.3}})
        return {"success": success, "subtasks": subtasks}

    cap = {"rot_deg": 30.0, "pos_cm": 3.0, "dropped": [2], "n_dropped": 1, "weights_before": [0.4, 0.4, 0.2]}
    none = {"rot_deg": 30.0, "pos_cm": 3.0, "dropped": [], "n_dropped": 0, "weights_before": [1.0, 0.0, 0.0]}
    _write_pool(tmp_path / "pool", [attempt(True, [0.5, 0.5, 0.0], cap, [0.5, 0.5, 0.0]), attempt(False, [1.0, 0.0, 0.0], none)])
    summary = mix_report.summarize(mix_report.read_pool(tmp_path / "pool"))
    assert summary["attempts"] == 2 and summary["successes"] == 1 and list(summary["subtasks"]) == [0, 1]
    first, second = summary["subtasks"][0], summary["subtasks"][1]
    assert first["executed"] == 2 and first["d"] == [0.5] and first["blended_share"] == 0.5
    assert first["effective_sources"] == {"mean": 1.5, "min": 1.0, "max": 2.0} and first["mean_weights"] == [0.75, 0.25, 0.0]
    assert first["largest_weight_source"] == {0: 2} and first["temperature_raised_share"] == 0.0
    assert first["distance_cap"] == {"limits": {"rot_deg": 30.0, "pos_cm": 3.0}, "with_drop_share": 0.5, "mean_dropped": 0.5}
    assert second["executed"] == 1 and second["temperature_raised_share"] == 1.0 and second["distance_cap"] is None
    assert mix_report.main(["--pool", str(tmp_path / "pool")]) == 0
    text = capsys.readouterr().out
    assert "subtask 0: executed 2" in text and "effective sources: mean 1.5, min 1.0, max 2.0" in text
    assert "a source dropped in 50% of the subtasks" in text and "temperature raised in 100%" in text
    # a pool made without mixing: the selected sources are counted instead
    _write_pool(tmp_path / "unmixed", [{"success": True, "subtasks": [{"subtask_ind": 0, "src_demo_ind": 4}]}])
    unmixed = mix_report.summarize(mix_report.read_pool(tmp_path / "unmixed"))["subtasks"][0]
    assert unmixed == {"executed": 1, "with_weights": 0, "sources_selected": {4: 1}}
    assert mix_report.main(["--pool", str(tmp_path / "empty")]) == 1


# the driver ------------------------------------------------------------------------------------------------------------
def _driver(tmp_path, **variables):
    """Run scripts/run_factor_experiment.sh from a copy of the scripts folder whose baseline driver and ``make`` are
    stand-ins that write down how they were called: ``(result, the calls)``."""
    scripts = tmp_path / "repo" / "scripts"
    scripts.mkdir(parents=True)
    (tmp_path / "repo" / "configs").symlink_to(REPO / "configs")
    (tmp_path / "repo" / "tmg").symlink_to(REPO / "tmg")
    for name in ("run_factor_experiment.sh", "mix_report.py"):
        shutil.copy(REPO / "scripts" / name, scripts / name)
    calls = tmp_path / "calls"
    baseline = scripts / "run_mimicgen_baseline.sh"
    baseline.write_text(f"#!/bin/sh\necho \"baseline METHOD=$METHOD GENERATION=$GENERATION OBSERVATION=$OBSERVATION "
                        f"SELECTION=$SELECTION TASKS=$TASKS POLICIES=$POLICIES SEEDS=$SEEDS SIZE=$SIZE "
                        f"POLICY_PARALLEL=$POLICY_PARALLEL PORT_BASE=$PORT_BASE RUN_NAME=$RUN_NAME GPU=$GPU\" >> '{calls}'\n"
                        f"[ \"$METHOD\" != \"$FAIL_METHOD\" ]\n")
    baseline.chmod(0o755)
    binary = tmp_path / "bin"
    binary.mkdir()
    (binary / "make").write_text(f"#!/bin/sh\ncase \"$*\" in *print-data-root*) echo '{tmp_path}/data' ;; "
                                 f"*) echo \"make $*\" >> '{calls}' ;; esac\n")
    (binary / "make").chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", CLUSTER="a100", GPU="6", **variables)
    env.pop("TASKS", None) if "TASKS" not in variables else None
    result = subprocess.run(["bash", str(scripts / "run_factor_experiment.sh")], env=env, capture_output=True, text=True, check=False)
    return result, (calls.read_text().splitlines() if calls.exists() else [])


def _bash_has_wait_n() -> bool:
    version = subprocess.run(["bash", "-c", "echo ${BASH_VERSINFO[0]}.${BASH_VERSINFO[1]}"], capture_output=True, text=True, check=False)
    major, minor = (int(v) for v in version.stdout.strip().split("."))
    return (major, minor) >= (4, 3)


@pytest.mark.skipif(not _bash_has_wait_n(), reason="the drivers need bash 4.3 (wait -n)")
def test_driver_runs_every_condition_through_the_baseline_driver_with_its_own_ports(tmp_path):
    result, calls = _driver(tmp_path, TASKS="threading coffee", CONDITIONS="none skillmix_distcap skillmix_neff3",
                            PORT_BASE="9000", FAIL_METHOD="skillmix_neff3")
    assert result.returncode == 1                                               # one condition failed: reported, the others ran
    common = ("GENERATION=pool_factor_lowdim OBSERVATION=low_dim SELECTION=ours TASKS=threading coffee "
              "POLICIES=diffusion_policy SEEDS=202601 202602 202603 SIZE=1000 POLICY_PARALLEL=3")
    # 2 tasks x 1 policy x 3 seeds x 40 ports = 240 ports per condition
    assert calls == [f"baseline METHOD=none {common} PORT_BASE=9000 RUN_NAME=factor_mimicgen_none GPU=6",
                     f"baseline METHOD=skillmix_distcap {common} PORT_BASE=9240 RUN_NAME=factor_mimicgen_skillmix_distcap GPU=6",
                     f"baseline METHOD=skillmix_neff3 {common} PORT_BASE=9480 RUN_NAME=factor_mimicgen_skillmix_neff3 GPU=6"]
    log = (tmp_path / "data/logs/factor_mimicgen/driver.log").read_text()
    assert "[skillmix_neff3] FAILED" in log and "[none] done" in log and "ports 9000..9719" in log
    assert (tmp_path / "data/logs/factor_mimicgen/FAILURES").read_text().split() == ["skillmix_neff3"]


def test_driver_refuses_an_unknown_condition_and_a_missing_task_list(tmp_path):
    result, calls = _driver(tmp_path, TASKS="threading", CONDITIONS="skillmix skillmix_typo")
    assert result.returncode == 2 and "unknown condition skillmix_typo" in result.stderr and calls == []
    shutil.rmtree(tmp_path / "repo"), shutil.rmtree(tmp_path / "bin")
    result, calls = _driver(tmp_path)
    assert result.returncode != 0 and "set TASKS" in result.stderr and calls == []


def test_driver_check_mode_generates_fifty_attempts_and_prints_the_mixing_report(tmp_path):
    pool = tmp_path / "data/pools/cpgen/threading/ours/ours/skillmix_distcap/factor_check50"
    cap = {"rot_deg": 30.0, "pos_cm": 3.0, "dropped": [1], "n_dropped": 1, "weights_before": [0.5, 0.5]}
    _write_pool(pool, [{"success": True, "subtasks": [{"subtask_ind": 0, "src_demo_ind": 0,
                                                       "mix": {"subtask": 0, "d": 0.5, "weights": [1.0, 0.0], "distance_cap": cap}}]}])
    result, calls = _driver(tmp_path, GENERATOR="cpgen", TASKS="threading", CONDITIONS="skillmix_distcap", CHECK="1")
    assert result.returncode == 0, result.stderr
    assert calls == [f"make -s -C {tmp_path}/repo sources GENERATOR=cpgen TASK=threading CLUSTER=a100",
                     (f"make -s -C {tmp_path}/repo generate GENERATOR=cpgen TASK=threading CLUSTER=a100 SCENE=ours "
                      "SELECTION=ours METHOD=skillmix_distcap GENERATION=test/factor_check50")]     # nothing is trained
    assert "a source dropped in 100% of the subtasks" in result.stdout and "effective sources: mean 1.0" in result.stdout
    assert (tmp_path / "data/logs/factor_cpgen/check_skillmix_distcap_threading.report").exists()
