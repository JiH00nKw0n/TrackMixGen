"""``tmg fit_coefficients``: one fit path read through each generator's fit layout; the reached rules; the record versions
(with real records of the Rope calibration pools made by commit 2fcfc0d); the sanity report."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from tmg.config import compose, resolved
from tmg.generation.fit_coefficients import entropy, fit_blocks, fit_coefficients, fit_subtask, randmix_rows, unmix_rows
from tmg.generation.method.fit_layout import FitLayout, SanityLimits, binomial_tail, rate_exceeds
from tmg.generation.method.fit_records import (
    REACHED,
    Attempts,
    ReachedRule,
    complete_source_holds,
    judge_sanity,
    measured_with,
    merge_sanity,
    prepare,
    reached_rule,
    read_pools,
    record_version,
    sanity_report,
    step_signal_outcomes,
)
from tmg.generation.method.interfaces import Arms, HeldRule, SubtaskId
from tmg.generation.method.skillmix import SkillMixer, load_rule
from tmg.generation.method.success_model import (
    FEATURES,
    HELD_FEATURES,
    FeatureUnknown,
    design_rows,
    held_test_of_record,
    mix_record_features,
    scene_of_record,
    source_features,
)

S = SubtaskId
REPO = Path(__file__).resolve().parents[2]
GENERATORS = REPO / "configs" / "experiment" / "generator"
RULE = HeldRule(version=1, rigid_m=0.2, nodes_m=0.1)
MIMICGEN = FitLayout(arms=Arms(), reached="step_signals", source_poses="datagen_info")
CPGEN = FitLayout(arms=Arms(), reached="stops_at_failure", source_poses="records")


def isaac_layout(arms, reached="term_window", **more):
    return FitLayout(arms=Arms.of(arms), reached=reached, source_poses="records", features=HELD_FEATURES,
                     unversioned_held_rule=RULE, **more)


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    p = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    p[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    p[:3, 3] = [x, y, z]
    return p


def flat(p):
    return [*p[:3, 3], *p[:3, :3].reshape(-1)]


def write_pool(folder, chunks, attempts):
    """A finished pool with ``attempts(chunk)`` as the attempt records of every chunk."""
    for c in range(chunks):
        records = folder / "chunks" / f"seed{c:05d}" / "records"
        records.mkdir(parents=True)
        (records.parent / "CHUNK_DONE.json").write_text("{}")
        (records / "attempts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in attempts(c)))


def attempts_of(records):
    return Attempts(list(records), [Path(".")] * len(records), [0] * len(records))


def generator_config(name, task):
    """The generator and task config of the composed fit_coefficients command (the real files, with their defaults)."""
    cfg = resolved(compose("fit_coefficients", [f"generator={name}", f"task={task}"]))
    return cfg["generator"], cfg["task"]


# the model ------------------------------------------------------------------------------------------------------------
def test_fit_recovers_the_sign_of_the_spread_effect():
    rng = np.random.default_rng(0)
    J, n = 3, 6000
    a = np.array([0.5, 0.0, -0.5])
    for d_true in (1.5, -1.5):
        W = rng.dirichlet(np.full(J, 0.7), size=n)
        H = np.array([entropy(w) for w in W])
        y = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + W @ a + d_true * H)))).astype(float)
        rows = [np.concatenate([w, np.zeros(5), [h]]) for w, h in zip(W, H)]
        groups = list(np.arange(n) // 50)
        block = fit_subtask((rows, list(y), groups), ([], [], []), J, with_cv=False)
        assert np.sign(block["d"]) == np.sign(d_true) and abs(block["d"] - d_true) < 0.5
        assert block["a"][0] > block["a"][2]


# rows of unmixed attempts (MimicGen records, datagen_info) --------------------------------------------------------
def _attempt(src0, src1, cur0, cur1, srcpose0, srcpose1, eef, objects):
    return {"success": True, "subtasks": [
        {"subtask_ind": 0, "object_name": "a", "src_demo_ind": src0, "src_subtask_start": 0,
         "cur_obj_pose": flat(cur0), "src_obj_pose": flat(srcpose0), "eef_pose": flat(eef), "objects": {"a": flat(objects["a"])}},
        {"subtask_ind": 1, "object_name": "b", "src_demo_ind": src1, "src_subtask_start": 2,
         "cur_obj_pose": flat(cur1), "src_obj_pose": flat(srcpose1), "eef_pose": flat(eef), "objects": {k: flat(v) for k, v in objects.items()}}]}


def test_design_rows_keep_only_attempts_that_reached_the_subtask():
    infos = [{"eef_pose": np.stack([pose()] * 5), "object_poses": {"a": np.stack([pose(z=-0.1)] * 5)}} for _ in range(2)]
    objects = {"a": pose(z=-0.12), "b": pose(0.5)}
    a = _attempt(0, 1, pose(0.1), pose(0.5), pose(), pose(0.4), pose(), objects)
    X, y, kept = design_rows([a, a], [[True, True], [False, False]], 1, 2, source_poses="datagen_info", source_infos=infos)
    assert kept == [0] and X.shape == (1, 2 + 3 + 2) and y.tolist() == [1.0]
    assert np.allclose(X[0, :2], [0, 1]) and X[0, 2] == pytest.approx(0.1) and X[0, 5] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="datagen_info"):
        design_rows([a], [[True, True]], 1, 2, source_poses="datagen_info")
    with pytest.raises(ValueError, match="src_eef_pose"):              # a layout that reads the record needs its poses
        design_rows([a], [[True, True]], 1, 2, source_poses="records")


def test_design_rows_label_is_task_success_not_the_subtask_success():
    objects = {"a": pose(z=-0.12), "b": pose(0.5)}
    ok = _attempt(0, 1, pose(0.1), pose(0.5), pose(), pose(0.4), pose(), objects)
    failed = dict(ok, success=False)                       # the grasp (subtask 0) succeeded, the task did not
    X, y, kept = design_rows([ok, failed], [[True, True], [True, False]], 0, 2)
    assert X.shape == (2, 2 + 3 + 2) and y.tolist() == [1.0, 0.0] and kept == [0, 1]
    assert np.all(X[:, 5:] == 0)                           # no in-hand features in the first subtask
    assert design_rows([ok], [[False, False]], 1, 2)[0].shape == (0, 2 + 3 + 2)      # nobody reached subtask 1
    assert design_rows([ok], [[False, False]], 1, 2, features=HELD_FEATURES)[0].shape == (0, 2 + 6)


# reached rules --------------------------------------------------------------------------------------------------------
def test_step_signal_outcomes_from_term_signals():
    steps = {"seg_sub": np.array([0, 0, 1, 1]), "seg_of_step": np.array([0, 0, 1, 2, 2, 3, 3]),
             "term__grasp": np.array([0, 0, 1, 1, 0, 0, 0])}
    assert step_signal_outcomes({"success": True}, steps, ["grasp", None]) == [True, True]
    late = dict(steps, term__grasp=np.array([0, 0, 0, 0, 0, 1, 1]))      # fires only during the second subtask
    assert step_signal_outcomes({"success": False}, late, ["grasp", None]) == [False, False]


def test_every_reached_rule_is_registered_by_its_name_and_an_unknown_one_is_refused():
    assert sorted(REACHED) == ["step_signals", "stops_at_failure", "task_success", "term_window"]
    assert reached_rule("stops_at_failure").mix_record_proves_reach and not reached_rule("term_window").mix_record_proves_reach
    with pytest.raises(ValueError, match="unknown reached rule"):
        reached_rule("isaaclab")
    with pytest.raises(ValueError, match="subtask_term_signals"):       # step_signals needs the task's signals
        reached_rule("step_signals").num_subtasks([], None, {"name": "square"})
    with pytest.raises(ValueError, match="source_poses"):
        FitLayout(arms=Arms(), reached="term_window", source_poses="hdf5")


def test_fit_reads_the_pools_of_a_generator_that_stops_at_the_first_failed_subtask(tmp_path):
    """CP-Gen records: a unmix attempt reached subtask k when it has k + 1 subtask records; a randmix attempt when it
    has a mix record of k, also one whose motion plan to k failed (``unplanned_mixes``)."""
    rng = np.random.default_rng(1)

    def sub(k, mix):
        s = {"subtask_ind": k, "object_name": "ab"[k], "src_demo_ind": int(rng.integers(2)), "src_subtask_start": 0,
             "cur_obj_pose": flat(pose(rng.normal(0, 0.05))), "src_obj_pose": flat(pose()), "eef_pose": flat(pose()),
             "objects": {}}
        return dict(s, mix=mix(k)) if mix else s

    def mix(k):
        w = rng.dirichlet([0.5, 0.5]).round(4)
        return {"subtask": k, "weights": [float(w[0]), float(1 - w[0])], "scene_features": rng.random((2, 3)).tolist()}

    def randmix(chunk):
        out = [{"success": bool(rng.random() < 0.5), "subtasks": [sub(0, mix), sub(1, mix)]} for _ in range(20)]
        return [*out, {"success": False, "subtasks": [sub(0, mix)], "unplanned_mixes": [mix(1)]}]      # plan to 1 failed

    def unmix(chunk):
        return [{"success": bool(n == 2 and rng.random() < 0.5), "subtasks": [sub(k, None) for k in range(n)]}
                for n in (1, 2, 2, 2, 2) * 4]

    write_pool(tmp_path / "randmix", 4, randmix)
    write_pool(tmp_path / "unmix", 3, unmix)
    blocks, reached = fit_blocks(read_pools([tmp_path / "randmix"]), read_pools([tmp_path / "unmix"]), layout=CPGEN, task={},
                                 num_sources=2, with_cv=False)
    assert sorted(blocks, key=lambda s: s.order) == [S(0), S(1)] and blocks[S(0)]["rows"] == {"randmix": 84, "unmix": 60}
    assert blocks[S(1)]["rows"] == {"randmix": 84, "unmix": 48}    # the failed plans count, the 12 that stopped do not
    assert len(blocks[S(1)]["a"]) == 2 and sorted(blocks[S(1)]["g"]) == sorted(FEATURES)
    assert "signal" in reached[S(1)]["reached"] and "first subtask" in reached[S(0)]["reached"]
    assert reached[S(1)]["unknown_rows"] == 0 and reached[S(0)]["constant_features"] == {"inhand_position": 0.0, "inhand_rotation": 0.0}


# Isaac Lab records: arms, term window, declared task_success -------------------------------------------------------
def isaac_attempt(rng, *, success, grasped, mix=None, eefs=("right", "left"), held=True, version=3):
    """One DexMimicGen-like attempt record: right subtasks 0 (grasp the bottle, term signal grasp_bottle) and 1 (place it in
    the fridge, carried: the bottle, held or not), left subtask 0 (hold the fridge); the scene fields the Isaac Lab runtime
    records (the current record version, 3, or the unversioned shape of commit 2fcfc0d)."""
    def sub(eef, k, obj, carried=None, signal=None):
        j = int(rng.integers(2))
        eef_now = pose(rng.normal(0, 0.02), 0.1)
        s = {"eef": eef, "subtask_ind": k, "object_name": obj, "src_demo_ind": j, "src_subtask_start": 4 * k, "start_step": 10 * k,
             "cur_obj_pose": flat(pose(rng.normal(0, 0.05), yaw=rng.normal(0, 0.2))), "src_obj_pose": flat(pose(0.01 * j)),
             "eef_pose": flat(eef_now), "src_eef_pose": flat(pose(0.0, 0.1)), "objects": {}, "src_objects": {},
             "carried_object": carried if held else None, "carried_candidate": carried, "term_signal": signal}
        if version >= 2:
            s["held_rule"] = RULE.as_dict()
        if carried is not None:
            src_held = j == 0                                   # source 0 holds the bottle at its segment start, source 1 does not
            s["held_distance"] = {"now": 0.05 if held else 0.7, "source": 0.05 if src_held else 0.7}
            if held:
                s["objects"][carried] = flat(eef_now @ pose(0.0, 0.0, rng.normal(0.05, 0.01)))
            if src_held and (held or version >= 2):             # version 1 kept the source's pose only next to the current one
                s["src_objects"][carried] = flat(pose(0.0, 0.1, 0.05))
            if version >= 2:
                s["held_distance"]["deformable"] = False
        if mix is not None:
            w = rng.dirichlet([0.5, 0.5]).round(4)
            s["mix"] = {"subtask": k, "eef": eef, "mixed": True, "weights": [float(w[0]), float(1 - w[0])],
                        "scene_features": rng.random((2, 3)).tolist()}
            if k and (held or version >= 2):
                s["mix"]["inhand_features"] = [[0.03, 0.2], [0.0, 0.0]] if held else [[0.0, 0.0], [0.0, 0.0]]
            if k and version >= 2:
                s["mix"]["held"] = {"now": held, "sources": [True, False]}
        return s

    subs = []
    if "right" in eefs:
        subs += [sub("right", 0, "bottle", signal="grasp_bottle"), sub("right", 1, "fridge", carried="bottle")]
    if "left" in eefs:
        subs.insert(1, sub("left", 0, "fridge"))
    out = {"success": bool(success), "subtasks": subs, "n_steps": 30, "term_first": {"grasp_bottle": 3 if grasped else -1}}
    return {**out, "record_version": version} if version >= 2 else out


def test_fit_reads_isaaclab_records_per_end_effector_and_subtask(tmp_path):
    """Two arms, subtasks numbered per arm: blocks right/0, right/1, left/0 (no collision on k = 0); right/1's rows are the
    attempts whose grasp signal fired before right/1 started; the in-hand features come from the records (no source file)."""
    from tmg.generation.fit_coefficients import write_rule

    rng = np.random.default_rng(3)
    write_pool(tmp_path / "randmix", 3, lambda c: [isaac_attempt(rng, success=rng.random() < 0.5, grasped=rng.random() < 0.8, mix=True)
                                                   for _ in range(20)])
    write_pool(tmp_path / "unmix", 3, lambda c: [isaac_attempt(rng, success=rng.random() < 0.4, grasped=n % 5 != 0) for n in range(20)])
    layout = isaac_layout(["left", "right"])
    randmix, unmix = (prepare(read_pools([tmp_path / kind]), layout)[0] for kind in ("randmix", "unmix"))
    blocks, reached = fit_blocks(randmix, unmix, layout=layout, task={}, num_sources=2, with_cv=False)
    assert sorted(key.key for key in blocks) == ["left/0", "right/0", "right/1"]
    assert blocks[S(0, "right")]["rows"]["unmix"] == 60 and blocks[S(0, "left")]["rows"]["unmix"] == 60
    assert blocks[S(1, "right")]["rows"]["unmix"] == 48               # the 12 attempts whose grasp never fired did not reach it
    assert "by signal" in reached[S(1, "right")]["reached"] and "first subtask" in reached[S(0, "right")]["reached"]
    assert sorted(blocks[S(1, "right")]["g"]) == sorted(HELD_FEATURES)
    frame, held_rule, _registration = measured_with(randmix.records + unmix.records)
    path = write_rule(tmp_path / "rule.yaml", generator="dexmimicgen", task="bottle_door", basis="test", num_sources=2, blocks=blocks,
                      features=layout.features, object_frame=frame, held_rule=held_rule)
    assert "<eef>/<k>" in path.read_text()
    rule = load_rule(str(path))
    assert rule.block(S(0, "left")).a.tolist() == pytest.approx(blocks[S(0, "left")]["a"]) and rule.held_rule == RULE
    assert rule.block(S(1, "right")).d == pytest.approx(blocks[S(1, "right")]["d"]) and rule.features == HELD_FEATURES
    # the unmix rows of right/1: non-zero in-hand features from the recorded poses where the selected source holds the
    # bottle, and the mismatch flag (with zero in-hand features) where it does not
    outcomes, used = reached_rule("term_window").outcomes(unmix, "right", 2, {})
    x = np.array(unmix_rows(unmix, outcomes, 1, 2, layout=layout, eef="right").x)
    holding, other = x[x[:, 0] == 1], x[x[:, 1] == 1]
    assert np.abs(holding[:, 2 + 3]).min() > 0 and np.all(holding[:, 2 + 5] == 0)
    assert np.all(other[:, 2 + 3:2 + 5] == 0) and np.all(other[:, 2 + 5] == 1) and used == {0: {"signal": 60, "assumed": 0}}


def test_the_fit_keys_its_blocks_by_the_layouts_arms_and_refuses_records_that_disagree():
    """The arms come from the generator's layout (config), as the mixer's do, never from the records."""
    rng = np.random.default_rng(13)
    two = [isaac_attempt(rng, success=True, grasped=True) for _ in range(4)]
    with pytest.raises(ValueError, match="'left' are not arms|disagree with the generator's arms"):
        prepare(attempts_of(copy.deepcopy(two)), isaac_layout(["right"]))                # a one-arm layout, two-arm records
    one = [isaac_attempt(rng, success=True, grasped=True, eefs=("right",)) for _ in range(4)]
    with pytest.raises(ValueError, match=r"no subtask of \['left'\] was recorded"):
        prepare(attempts_of(copy.deepcopy(one)), isaac_layout(["left", "right"]))        # a two-arm layout, one-arm records
    # one arm that the records name: plain keys k, as the one-armed mixer looks them up
    kept, _ = prepare(attempts_of(copy.deepcopy(one)), isaac_layout(["right"]))
    blocks, _ = fit_blocks(kept, kept, layout=isaac_layout(["right"]), task={}, num_sources=2, with_cv=False)
    assert sorted(key.key for key in blocks) == [0, 1]


def test_reached_counts_the_signal_only_before_the_next_subtask_of_the_arm_starts():
    rng = np.random.default_rng(6)
    rule = reached_rule("term_window")
    early, late, never = (isaac_attempt(rng, success=False, grasped=True) for _ in range(3))
    late["term_first"] = {"grasp_bottle": 12}                         # right/1 started at step 10: too late
    never["term_first"] = {"grasp_bottle": -1}
    no_steps = isaac_attempt(rng, success=False, grasped=True)
    no_steps.update(n_steps=0, term_first={})                          # the plan failed before the first step
    unbuilt = isaac_attempt(rng, success=False, grasped=True, eefs=("left",))     # the right arm built nothing
    out, used = rule.outcomes(attempts_of([early, late, never, no_steps, unbuilt]), "right", 2, {})
    assert [row[0] for row in out] == [True, False, False, False, False] and used[0] == {"signal": 5, "assumed": 0}
    renamed = isaac_attempt(rng, success=False, grasped=True)
    renamed["term_first"] = {"grasp": 3}                               # the recorded name is not among the signals
    with pytest.raises(ValueError, match="grasp_bottle"):
        rule.outcomes(attempts_of([renamed]), "right", 2, {})
    eager = isaac_attempt(rng, success=False, grasped=True)
    for s in eager["subtasks"]:
        s["start_step"] = 0                                           # every subtask built before the first step
    with pytest.raises(ValueError, match="not built when they started"):
        rule.outcomes(attempts_of([eager]), "right", 2, {})
    old = isaac_attempt(rng, success=False, grasped=True)
    del old["subtasks"][0]["start_step"]
    with pytest.raises(ValueError, match="regenerate"):
        rule.outcomes(attempts_of([old]), "right", 2, {})


def test_an_environment_without_per_step_signals_is_declared_per_task_and_fitted_on_the_task_label(tmp_path):
    """Records without any per-step signal: the term window refuses them and names the declaration; a task declared
    ``fit: {reached: task_success}`` counts every recorded subtask as reached, and the fit says it was assumed."""
    rng = np.random.default_rng(4)

    def pool(mix):
        def make(chunk):
            out = []
            for _ in range(20):
                a = isaac_attempt(rng, success=rng.random() < 0.5, grasped=True, mix=mix, eefs=("right",))
                a["term_first"] = {}
                out.append(a)
            return out
        return make

    write_pool(tmp_path / "randmix", 2, pool(True))
    write_pool(tmp_path / "unmix", 2, pool(None))
    randmix, unmix = read_pools([tmp_path / "randmix"]), read_pools([tmp_path / "unmix"])
    with pytest.raises(ValueError, match="reached: task_success"):          # a missing signal is loud by default
        fit_blocks(randmix, unmix, layout=isaac_layout(["right"]), task={}, num_sources=2, with_cv=False)
    blocks, reached = fit_blocks(randmix, unmix, layout=isaac_layout(["right"], reached="task_success"), task={}, num_sources=2,
                                 with_cv=False)
    assert sorted(key.key for key in blocks) == [0, 1] and blocks[S(1)]["rows"] == {"randmix": 40, "unmix": 40}
    assert "ASSUMED" in reached[S(1)]["reached"]


def test_the_adapters_declare_their_fit_layouts_and_tasks_declare_missing_signals_in_one_key():
    """One declaration per task (``task.fit.reached``) replaces the two generator-level ones; the arms are the generator
    config's ``eefs``, the same the runtime builds the mixer with."""
    from tmg.generation.generators import make_generator

    def layout(name, task):
        generator, t = generator_config(name, task)
        return make_generator({"generator": generator, "task": t}).fit_layout()

    assert layout("mimicgen", "square") == MIMICGEN and layout("cpgen", "threading") == CPGEN
    rope, towel = layout("softmimicgen", "rope"), layout("softmimicgen", "towel")
    assert rope.reached == "term_window" and towel.reached == "task_success"            # rope publishes its grasp signal
    assert rope.arms == Arms(("robot0",)) and rope.source_poses == "records" and rope.features == HELD_FEATURES
    assert rope.unversioned_held_rule == RULE
    assert rope.sanity == SanityLimits(max_scene_error_rate=0.05, max_unreached_success_rate=0.05, confidence=0.999)
    for task in ("bottle_door", "microwave", "pick_place"):
        dex = layout("dexmimicgen", task)
        assert dex.reached == "task_success" and dex.arms == Arms(("left", "right")) and dex.features == HELD_FEATURES
    for name in ("softmimicgen", "dexmimicgen"):                       # the old declarations are gone; the shared block is one file
        g = yaml.safe_load((GENERATORS / f"{name}.yaml").read_text())
        assert "per_step_subtask_signals" not in g and "tasks_without_step_signals" not in g
        assert g["defaults"] == ["isaaclab_common", "_self_"] and "scene_reading" not in g and "record_checks" not in g
    common = yaml.safe_load((GENERATORS / "isaaclab_common.yaml").read_text())
    assert sorted(common) == ["default_max_workers", "record_checks", "scene_reading"]
    for path in ("softmimicgen/task/towel", "dexmimicgen/task/bottle_door", "dexmimicgen/task/microwave", "dexmimicgen/task/pick_place"):
        lines = [line for line in (GENERATORS / f"{path}.yaml").read_text().splitlines() if "task_success" in line]
        assert len(lines) == 1 and lines[0].startswith("fit: {reached: task_success}")     # one line, explained once per generator
    assert "fit" not in yaml.safe_load((REPO / "configs/command/fit_coefficients.yaml").read_text())


def test_the_task_fit_block_and_the_scene_reading_are_not_part_of_a_chunks_identity():
    from tmg.generation.base import ChunkSpec

    def spec(task, **more):
        return ChunkSpec(generator="softmimicgen", chunk={"index": 0, "seed": 1, "attempts": 5}, folder="f", records_dir="r",
                         task=task, scene={}, method={}, protocol={}, generator_config="", **more)

    old = spec({"name": "towel"})                                       # a chunk made before the task declared its fit
    new = spec({"name": "towel", "fit": {"reached": "task_success"}}, scene_reading={"held_rule": RULE.as_dict()})
    assert old.identity() == new.identity() and spec({"name": "rope"}).identity() != old.identity()
    written = json.loads(json.dumps({k: v for k, v in vars(old).items() if k != "scene_reading"}))
    assert ChunkSpec(**written).scene_reading == {}                     # a spec written before the field existed still loads


# scenes that could not be read, frames, hold rules ----------------------------------------------------------------
def test_attempts_without_their_scene_are_left_out_and_counted():
    rng = np.random.default_rng(5)
    good = [isaac_attempt(rng, success=True, grasped=True) for _ in range(3)]
    broken = isaac_attempt(rng, success=True, grasped=True)
    broken["subtasks"][0] = {"eef": "right", "subtask_ind": 0, "object_name": "bottle", "scene_error": "ValueError: no yaw"}
    invalid = dict(isaac_attempt(rng, success=True, grasped=True), invalid="scene_error")
    kept, about = prepare(attempts_of([*good, broken, invalid]), isaac_layout(["left", "right"]))
    assert len(kept) == 3 and about["without_scene"] == 2 and about["attempts"] == 5
    assert "no yaw" in about["scene_error_examples"][0] and about["versions"] == {"3": 5}


def test_records_of_different_frames_or_hold_rules_are_refused():
    rng = np.random.default_rng(7)
    a, b = (isaac_attempt(rng, success=True, grasped=True) for _ in range(2))
    assert measured_with([a, b]) == (None, RULE, None)
    a["subtasks"][0]["object_frame"] = {"version": "nodal_yaw_v1", "reference_sha256": "x"}
    assert measured_with([a, b])[0] == {"version": "nodal_yaw_v1", "reference_sha256": "x"}
    b["subtasks"][0]["object_frame"] = {"version": "nodal_yaw_v1", "reference_sha256": "y"}
    with pytest.raises(ValueError, match="frames"):
        measured_with([a, b])
    c, d = (isaac_attempt(rng, success=True, grasped=True) for _ in range(2))
    d["subtasks"][0]["held_rule"] = {"version": 2, "rigid_m": 0.3, "nodes_m": 0.1}
    with pytest.raises(ValueError, match="hold rules"):
        measured_with([c, d])
    del d["subtasks"][0]["held_rule"]
    with pytest.raises(ValueError, match="carry none"):
        measured_with([c, d])
    assert measured_with([{"subtasks": [{"subtask_ind": 0}]}]) == (None, None, None)     # MimicGen, CP-Gen: none of them
    e, f = (isaac_attempt(rng, success=True, grasped=True, mix=True) for _ in range(2))
    e["subtasks"][0]["mix"]["registration"] = {"use_rotation_transform": True, "bend_coef": 0.1, "rot_coef": 1e-3}
    assert measured_with([e, f]).donor_registration == {"use_rotation_transform": True, "bend_coef": 0.1, "rot_coef": 0.001}
    f["subtasks"][0]["mix"]["registration"] = {"use_rotation_transform": True, "bend_coef": 0.2, "rot_coef": 1e-3}
    with pytest.raises(ValueError, match="registered with different settings"):
        measured_with([e, f])
    old = isaac_attempt(rng, success=True, grasped=True)
    del old["subtasks"][0]["src_obj_pose"]
    with pytest.raises(ValueError, match="regenerate"):
        scene_of_record(old["subtasks"][0], None)


# record versions ------------------------------------------------------------------------------------------------------
def test_the_recorder_writes_the_record_version_and_newer_records_are_refused(tmp_path):
    from tmg.core.records import RECORD_VERSION, AttemptRecorder, read_attempts

    rec = AttemptRecorder()
    rec.configure(tmp_path)
    rec.begin_attempt()
    rec.flush(success=True, src_demo_inds=[0])
    line = read_attempts(tmp_path)[0]
    assert RECORD_VERSION == 3 and line["record_version"] == 3 and record_version(line) == 3
    assert record_version({"attempt": 0}) == 1                        # no field: written before 2026-11-27
    for version in (2, 4):                # the interim format of this branch (no pool was made with it), and a later one
        with pytest.raises(ValueError, match=rf"version \[{version}\]: this code reads version 1 .* and version 3"):
            prepare(attempts_of([{"record_version": version, "subtasks": []}]), CPGEN)
    kept, about = prepare(attempts_of([{"subtasks": [], "success": True}, {"record_version": 3, "subtasks": [], "success": True}]), CPGEN)
    assert len(kept) == 2 and about["versions"] == {"1": 1, "3": 1}


def test_unversioned_records_get_the_declared_hold_rule_checked_against_their_distances():
    """Version-1 records carry the hold distances and the in-hand poses kept where held, not the rule: the rule declared
    for them is attached, the hold test is rebuilt from the distances as for any record, and it must agree with the poses
    the record kept."""
    rng = np.random.default_rng(9)
    layout = isaac_layout(["left", "right"])
    held = isaac_attempt(rng, success=True, grasped=True, mix=True, version=1)
    loose = isaac_attempt(rng, success=True, grasped=True, mix=True, held=False, version=1)
    assert all("held_rule" not in s and "deformable" not in (s.get("held_distance") or {}) for a in (held, loose) for s in a["subtasks"])
    kept, about = prepare(attempts_of([held, loose]), layout)
    assert about["unversioned_upgraded"] == 2 and about["versions"] == {"1": 2}
    s_held, s_loose = held["subtasks"][2], loose["subtasks"][2]
    assert all(s["held_rule"] == RULE.as_dict() for a in kept.records for s in a["subtasks"])
    assert held_test_of_record(s_held).now is True and held_test_of_record(s_loose).now is False
    assert "held" not in s_held and "held" not in s_loose              # the verdicts are derived, not stored a second time
    # the mix record where the hand held the object: who held, from the recorded in-hand features (0, 0 next to others)
    assert s_held["mix"]["held"] == {"now": True, "sources": [True, False]}
    assert mix_record_features(s_held["mix"], HELD_FEATURES, 2)[:, 5].tolist() == [0.0, 1.0]
    # where the hand held nothing the version-1 mixer recorded nothing about the sources: unknown, not "nobody holds"
    assert s_loose["mix"]["held"] == {"now": False, "sources": None}
    with pytest.raises(FeatureUnknown):
        mix_record_features(s_loose["mix"], HELD_FEATURES, 2)
    assert mix_record_features(s_loose["mix"], FEATURES, 2).shape == (2, 5)        # the five features do not need it
    assert measured_with(kept.records) == (None, RULE, None)
    # a second pass changes nothing; current-version records are left as they are
    again = copy.deepcopy(kept.records)
    prepare(attempts_of(again), layout)
    assert again == kept.records
    # records made with other distances contradict the declared rule: refused, not silently reinterpreted
    other = isaac_attempt(rng, success=True, grasped=True, version=1)
    other["subtasks"][2]["held_distance"]["now"] = 0.35                # kept as held at 0.35 m: not the 0.20 m rule
    with pytest.raises(ValueError, match="another rule"):
        prepare(attempts_of([other]), layout)
    borderline = isaac_attempt(rng, success=True, grasped=True, version=1)
    borderline["subtasks"][2]["held_distance"]["now"] = 0.20004         # within the 0.1 mm rounding of the record
    prepare(attempts_of([borderline]), layout)
    # a generator without a hold test (MimicGen, CP-Gen) has nothing to upgrade
    plain = {"success": True, "subtasks": [{"subtask_ind": 0, "object_name": "a"}]}
    kept, about = prepare(attempts_of([plain]), CPGEN)
    assert about["unversioned_upgraded"] == 0 and "held_rule" not in kept.records[0]["subtasks"][0]


def test_a_version_1_record_gives_the_hold_mismatch_the_new_mixer_gives_in_every_hold_situation():
    """The four situations (hand holds / does not, selected source holds / does not), each as a version-1 record and as
    the scene the new runtime would hand the mixer for the same distances: the fit's features of the record equal the
    mixer's features of the scene. In particular the hand holding nothing while the source holds the object is a
    mismatch (1), although version 1 kept no source pose there."""
    from tmg.generation.method.interfaces import HeldTest, SubtaskScene

    layout = isaac_layout(["right"])
    eef, src_eef = pose(0.0, 0.1), pose(0.0, 0.1)
    carried_now, carried_src = eef @ pose(0.0, 0.0, 0.05), pose(0.0, 0.1, 0.07)
    for now_m, source_m, mismatch in ((0.0048, 0.0024, 0.0), (0.0048, 0.6, 1.0), (0.35, 0.0024, 1.0), (0.35, 0.6, 0.0)):
        now, source = now_m <= 0.2, source_m <= 0.2
        record = {"eef": "right", "subtask_ind": 1, "object_name": "fridge", "src_demo_ind": 0, "src_subtask_start": 4, "start_step": 10,
                  "cur_obj_pose": flat(pose(0.3)), "src_obj_pose": flat(pose(0.1)), "eef_pose": flat(eef), "src_eef_pose": flat(src_eef),
                  "objects": {"bottle": flat(carried_now)} if now else {},
                  "src_objects": {"bottle": flat(carried_src)} if now and source else {},         # version 1: only next to the current pose
                  "carried_object": "bottle" if now else None, "carried_candidate": "bottle",
                  "held_distance": {"now": now_m, "source": source_m}, "term_signal": None}
        first = {"eef": "right", "subtask_ind": 0, "object_name": "bottle", "src_demo_ind": 0, "src_subtask_start": 0, "start_step": 0,
                 "cur_obj_pose": flat(pose()), "src_obj_pose": flat(pose()), "eef_pose": flat(eef), "src_eef_pose": flat(src_eef),
                 "objects": {}, "src_objects": {}, "carried_object": None, "carried_candidate": None, "term_signal": "grasp"}
        attempt = {"success": True, "n_steps": 30, "term_first": {"grasp": 3}, "subtasks": [first, record]}
        kept, _ = prepare(attempts_of([attempt]), layout)
        X, _y, _kept = design_rows(kept.records, [[True, True]], 1, 1, eef="right", features=HELD_FEATURES)
        test = HeldTest(rule=RULE, deformable=False, now_m=now_m, sources_m=[source_m])
        scene = SubtaskScene(subtask=1, eef="right", reference_object="fridge", current_object_pose=pose(0.3), source_object_poses=[pose(0.1)],
                             eef_pose=eef, source_eef_poses=[src_eef], carried_object="bottle", held=test, held_rule=RULE,
                             carried_pose=carried_now if test.now else None, source_carried_poses=[carried_src if test.sources[0] else None])
        mixer_features = source_features(scene, HELD_FEATURES)[0]
        np.testing.assert_allclose(X[0, 1:], mixer_features, atol=1e-6)
        assert X[0, 1 + 5] == mismatch and (X[0, 1 + 3] > 0) == (now and source)


def test_unknown_source_holds_of_version_1_mix_records_are_filled_from_the_pools_own_hold_tests_or_left_out():
    """Version-1 randmix rows where the hand held nothing: the per-source flags come from the hold tests the pools did
    record for the same (end effector, subtask, source) when they all agree; otherwise the row is left out and counted."""
    rng = np.random.default_rng(10)
    layout = isaac_layout(["left", "right"])
    unmix = [isaac_attempt(rng, success=True, grasped=True, version=1) for _ in range(40)]       # both sources get selected
    held = [isaac_attempt(rng, success=True, grasped=True, mix=True, version=1) for _ in range(3)]
    loose = [isaac_attempt(rng, success=True, grasped=True, mix=True, held=False, version=1) for _ in range(4)]
    randmix, _ = prepare(attempts_of(held + loose), layout)
    single, _ = prepare(attempts_of(unmix), layout)
    about = complete_source_holds(randmix.records + single.records)
    assert about["filled"] == 4 and about["unknown"] == 0
    assert about["evidence"]["right/1/0"][0] == about["evidence"]["right/1/0"][1] > 0 and about["evidence"]["right/1/1"][0] == 0
    for a in loose:
        mix = a["subtasks"][2]["mix"]
        assert mix["held"] == {"now": False, "sources": [True, False]}                      # source 0 holds: a mismatch for it
        assert mix_record_features(mix, HELD_FEATURES, 2)[:, 5].tolist() == [1.0, 0.0]
    outcomes, _ = reached_rule("term_window").outcomes(randmix, "right", 2, {})
    assert len(randmix_rows(randmix, outcomes, 1, 2, features=HELD_FEATURES, eef="right").x) == 7
    # without any test of source 1 in the pools (only source 0 was ever selected) the flags stay unknown: rows left out
    alone, _ = prepare(attempts_of([isaac_attempt(np.random.default_rng(3), success=True, grasped=True, mix=True, held=False, version=1)
                                    for _ in range(4)]), layout)
    for a in alone.records:
        a["subtasks"][2]["src_demo_ind"] = 0
    about = complete_source_holds(alone.records)
    assert about == {"filled": 0, "unknown": 4, "evidence": about["evidence"]} and "right/1/1" not in about["evidence"]
    unknown = []
    outcomes, _ = reached_rule("term_window").outcomes(alone, "right", 2, {})
    assert randmix_rows(alone, outcomes, 1, 2, features=HELD_FEATURES, eef="right", unknown=unknown).x == [] and unknown == [0, 1, 2, 3]
    # tests that disagree about a source settle nothing either
    mixed, _ = prepare(attempts_of([isaac_attempt(rng, success=True, grasped=True, mix=True, held=False, version=1)]), layout)
    conflicting = copy.deepcopy(single.records)
    for a in conflicting[:20]:
        if a["subtasks"][2]["src_demo_ind"] == 0:
            a["subtasks"][2]["held_distance"]["source"] = 0.7
    assert complete_source_holds(mixed.records + conflicting)["unknown"] == 1


# the Rope calibration pools of commit 2fcfc0d (real records) ------------------------------------------------------
ROPE = json.loads((Path(__file__).parent / "fixtures" / "rope_records_2fcfc0d.json").read_text())


def rope_layout():
    from tmg.generation.generators import make_generator

    generator, task = generator_config("softmimicgen", "rope")
    return make_generator({"generator": generator, "task": task}).fit_layout(), generator, task


def test_real_rope_records_of_2fcfc0d_are_read_as_version_1_with_the_declared_hold_rule():
    layout, _generator, task = rope_layout()
    for kind in ("randmix", "unmix"):
        records = copy.deepcopy(ROPE[kind])
        assert all(record_version(a) == 1 and "record_version" not in a for a in records)
        kept, about = prepare(attempts_of(records), layout)
        assert len(kept) == 2 and about["unversioned_upgraded"] == 2 and about["without_scene"] == 0
        frame, held_rule, registration = measured_with(kept.records)
        assert held_rule == HeldRule(1, 0.20, 0.10) and frame["version"] == "nodal_yaw_v1" and len(frame["reference_sha256"]) == 64
        # version 1 did not record how it registered the donors: the constants of 2fcfc0d, declared in the config, are attached
        # to its mix records (the unmix pool mixes nothing and has none)
        assert registration == ({"use_rotation_transform": True, "bend_coef": 0.1, "rot_coef": 0.001} if kind == "randmix" else None)
        for a in kept.records:
            grasp, shape = a["subtasks"]
            assert held_test_of_record(grasp) is None and grasp["held_rule"] == held_rule.as_dict()
            # the rope is held at one end: 2 to 7 mm to the nearest node, far inside the 0.10 m of the declared rule
            test = held_test_of_record(shape)
            assert (test.now, test.sources, test.deformable) == (True, [True], True) and test.now_m < 0.01
        # reached: the grasp signal fired (step 62 to 67) before the shaping subtask started (step 76 to 80)
        outcomes, used = reached_rule(layout.reached).outcomes(kept, "robot0", 2, task)
        assert [row[0] for row in outcomes] == [True, True] and used == {0: {"signal": 2, "assumed": 0}}
        report = sanity_report(kept, layout, task)
        assert judge_sanity(report, layout.sanity) == ([], []) and report["blocks"]["1"]["held"]["now"] == 2 == report["blocks"]["1"]["held"]["tested"]


def test_the_features_of_real_rope_records_are_the_ones_the_new_mixer_computes():
    """The version-1 mixer recorded every source's features; the selected source's row must be what the new code computes
    from the same record's poses (``scene_of_record`` + ``source_features``, the function the new mixer calls), so a rule
    fitted on these pools scores a scene as the new SkillMix generation measures it."""
    layout, _generator, _task = rope_layout()
    kept, _ = prepare(attempts_of(copy.deepcopy(ROPE["randmix"])), layout)
    for a in kept.records:
        grasp, shape = a["subtasks"]
        for s, carried in ((grasp, None), (shape, "object")):
            j, m = int(s["src_demo_ind"]), s["mix"]
            computed = source_features(scene_of_record(s, carried), HELD_FEATURES)[0]
            recorded = mix_record_features(m, HELD_FEATURES, 3)[j]
            assert np.allclose(computed, recorded, atol=1.5e-4), (computed, recorded)     # records round to 1e-4
            assert computed[5] == 0.0
        assert shape["mix"]["held"] == {"now": True, "sources": [True, True, True]}
        assert np.all(mix_record_features(grasp["mix"], HELD_FEATURES, 3)[:, 3:] == 0)       # first subtask: no in-hand terms


def test_real_records_of_a_dropped_rope_get_the_features_the_new_mixer_computes():
    """Two real attempts of the 2fcfc0d pools in which the rope was dropped before the second subtask (0.25 and 0.35 m from
    the hand; the selected source holds it at 2 mm). Version 1 kept no in-hand pose and, in the mix record, no in-hand
    features. Read with the declared hold rule, the hand holds nothing and the source does, so the fit's row of the
    selected source has in-hand features 0 and mismatch 1: what the new mixer computes for a scene with those hold
    distances. The randmix record's per-source flags come from the hold tests of the same pools."""
    from tmg.generation.method.interfaces import HeldTest

    layout, _generator, task = rope_layout()
    dropped = {kind: copy.deepcopy(ROPE["dropped"][kind]) for kind in ("unmix", "randmix")}
    assert all(record_version(a) == 1 and a["subtasks"][1]["carried_object"] is None and a["subtasks"][1]["objects"] == {}
               for a in dropped.values())
    assert [a["subtasks"][1]["held_distance"] for a in dropped.values()] == [{"now": 0.2454, "source": 0.0025}, {"now": 0.3475, "source": 0.0022}]
    assert "inhand_features" not in dropped["randmix"]["subtasks"][1]["mix"]
    held, _ = prepare(attempts_of(copy.deepcopy(ROPE["unmix"] + ROPE["randmix"])), layout)      # the pools' attempts that hold the rope
    kept, about = prepare(attempts_of(list(dropped.values())), layout)
    assert len(kept) == 2 and about["unversioned_upgraded"] == 2
    for a in kept.records:
        shape = a["subtasks"][1]
        test = held_test_of_record(shape)
        assert (test.now, test.sources, test.deformable) == (False, [True], True)
        row = source_features(scene_of_record(shape, "object"), HELD_FEATURES)[0]
        assert row[3:].tolist() == [0.0, 0.0, 1.0]
        # the scene the new runtime builds for the same distances: the mixer's features of the selected source are the same
        scene = scene_of_record(shape, "object")
        scene.held = HeldTest(rule=test.rule, deformable=True, now_m=shape["held_distance"]["now"], sources_m=[shape["held_distance"]["source"]])
        scene.carried_pose, scene.source_carried_poses = None, [scene.source_object_poses[0]]       # the runtime keeps the holder's pose
        np.testing.assert_array_equal(source_features(scene, HELD_FEATURES)[0], row)
    # the mix record of the dropped randmix attempt: hand empty, sources unknown until the pools' own hold tests fill them
    mix = kept.records[1]["subtasks"][1]["mix"]
    assert mix["held"] == {"now": False, "sources": None}
    assert mix["registration"] == {"use_rotation_transform": True, "bend_coef": 0.1, "rot_coef": 0.001}    # the declared v1 value
    filled = complete_source_holds(held.records + kept.records)
    assert filled["filled"] == 1 and filled["unknown"] == 0 and mix["held"] == {"now": False, "sources": [True, True, True]}
    assert mix_record_features(mix, HELD_FEATURES, 3)[:, 3:].tolist() == [[0.0, 0.0, 1.0]] * 3
    # the grasp signal never fired in these two attempts (term_first -1): they do not reach the second subtask's rows, and
    # count as a hand that holds nothing in the sanity report
    outcomes, _ = reached_rule(layout.reached).outcomes(kept, "robot0", 2, task)
    assert [row[0] for row in outcomes] == [False, False]
    assert sanity_report(kept, layout, task)["blocks"]["1"]["held"] == {"tested": 2, "now": 0, "sources": {"0": [2, 2], "1": [1, 1], "2": [1, 1]}}


def rope_pool(rng, kind, n, dropped=0.0):
    """``n`` attempts shaped like the real 2fcfc0d Rope records of ``kind`` (same fields, varied scene, source and label). A
    share ``dropped`` of them has lost the rope before the second subtask, as 15 of the 941 real attempts read on
    2026-11-27 had (0.35 to 0.40 m from the hand): version 1 then kept no in-hand pose and, in the mix record, no in-hand
    features."""
    out = []
    for i in range(n):
        a = copy.deepcopy(ROPE[kind][i % 2])
        a["attempt"] = i
        a["success"] = bool(rng.random() < 0.6)
        if rng.random() < dropped:
            shape = a["subtasks"][1]
            shape.update(objects={}, src_objects={}, carried_object=None)
            shape["held_distance"]["now"] = 0.3475
            a["success"] = False
            if shape["mix"]:
                shape["mix"].pop("inhand_features")
        for s in a["subtasks"]:
            s["cur_obj_pose"][0] += float(rng.normal(0, 0.03))
            s["cur_obj_pose"][1] += float(rng.normal(0, 0.03))
            if s["mix"]:
                w = rng.dirichlet(np.full(3, 0.6)).round(4)
                s["mix"]["weights"] = [float(w[0]), float(w[1]), float(round(1 - w[0] - w[1], 4))]
                s["mix"]["scene_features"] = (np.array(s["mix"]["scene_features"]) + rng.random((3, 3)) * 0.05).round(4).tolist()
            else:
                s["src_demo_ind"] = int(rng.integers(3))
        if rng.random() < 0.15:
            a["term_first"] = {"grasp": -1}                             # the grasp failed: the shaping subtask is not reached
            a["success"] = False
        out.append(a)
    return out


def rope_fit(tmp_path, monkeypatch, randmix, unmix, **more):
    """``fit_coefficients`` on Rope pools with the real softmimicgen and rope configs: ``(summary, rule file data)``."""
    from omegaconf import OmegaConf

    import tmg.generation.fit_coefficients as fc

    write_pool(tmp_path / "randmix", 4, randmix)
    write_pool(tmp_path / "unmix", 4, unmix)
    monkeypatch.setattr(fc, "write_provenance", lambda *a, **k: None)
    generator, task = generator_config("softmimicgen", "rope")
    cfg = OmegaConf.create({"randmix_pools": [str(tmp_path / "randmix")], "unmix_pools": [str(tmp_path / "unmix")],
                            "output": str(tmp_path / "out"), "min_attempts": 100, "task": task, "source": None,
                            "write_config": False, "generator": generator, **more})
    summary = fit_coefficients(cfg)
    return summary, yaml.safe_load((tmp_path / "out" / "coefficients.yaml").read_text()), generator


def test_a_rope_rule_fitted_on_2fcfc0d_shaped_pools_is_accepted_by_the_new_mixer(tmp_path, monkeypatch, capsys):
    """End to end: unversioned Rope pools -> fit_coefficients -> rule file (six features, the version-1 hold rule, the
    nodal frame) -> a mixer of the new code with the scene the new runtime builds from the same config."""
    rng = np.random.default_rng(21)
    summary, data, generator = rope_fit(tmp_path, monkeypatch, lambda c: rope_pool(rng, "randmix", 40), lambda c: rope_pool(rng, "unmix", 40))
    said = capsys.readouterr().out
    assert "carry no record_version" in said and "read as version 1" in said and "HeldRule(version=1, rigid_m=0.2, nodes_m=0.1)" in said
    assert summary["layout"] == {"arms": ["robot0"], "reached": "term_window", "source_poses": "records", "features": list(HELD_FEATURES)}
    assert summary["records"]["randmix"]["versions"] == {"1": 160} and summary["records"]["unmix"]["unversioned_upgraded"] == 160
    assert sorted(summary["subtasks"]) == [0, 1] and summary["subtasks"][1]["rows"]["unmix"] < summary["subtasks"][0]["rows"]["unmix"]
    assert "by signal" in summary["reached_from"]["1"] and summary["sanity"]["unmix"]["errors"] == []
    assert data["held_rule"] == {"version": 1, "rigid_m": 0.2, "nodes_m": 0.1} and data["features"] == list(HELD_FEATURES)
    assert data["donor_registration"] == {"use_rotation_transform": True, "bend_coef": 0.1, "rot_coef": 0.001}
    assert "with the donor registration declared for them, {'use_rotation_transform': True, 'bend_coef': 0.1, 'rot_coef': 0.001}" in said
    assert data["object_frame"] == ROPE["unmix"][0]["subtasks"][0]["object_frame"] and sorted(data["subtasks"]) == [0, 1]
    # the rope is held in every attempt of these pools: the mismatch column is constant 0, its slope 0, and the fit says so
    assert data["subtasks"][1]["g"]["inhand_mismatch"] == 0.0 and summary["constant_features"]["1"]["inhand_mismatch"] == 0.0
    assert "feature inhand_mismatch is 0.0 in every row (slope 0" in said and summary["unknown_rows"] == {"0": 0, "1": 0}
    assert summary["source_holds"]["filled"] == 0 and summary["source_holds"]["unknown"] == 0
    # the new runtime's scene: the hold rule of the same generator config, the frame of the same reference cloud
    rule = load_rule(str(tmp_path / "out" / "coefficients.yaml"))
    held_rule = HeldRule.from_mapping(generator["scene_reading"]["held_rule"])
    kept, _ = prepare(attempts_of(copy.deepcopy(ROPE["unmix"])), rope_layout()[0])
    s = kept.records[0]["subtasks"][1]
    scene = scene_of_record(s, "object")
    scene.source_object_poses, scene.source_eef_poses, scene.source_carried_poses = ([p[0]] * 3 for p in (
        scene.source_object_poses, scene.source_eef_poses, scene.source_carried_poses))
    scene.held.sources_m = scene.held.sources_m * 3
    scene.held_rule = held_rule
    # the donors of the new runtime: registered with what it reads from the official builder's call (SoftMimicGen c9d146ba)
    scene.donor_registration = {"use_rotation_transform": True, "bend_coef": 0.1, "rot_coef": 1e-3}
    mixer = SkillMixer(coefficients=rule, arms=Arms.of(generator["eefs"]))
    assert rule.donor_registration == scene.donor_registration
    w, record = mixer.decide(scene, 0, [np.full(4, -1.0)] * 3)
    assert w.sum() == pytest.approx(1.0) and len(record["scores"]) == 3 and record["held"] == {"now": True, "sources": [True] * 3}
    assert record["registration"] == scene.donor_registration
    scene.donor_registration = {"use_rotation_transform": True, "bend_coef": 0.2, "rot_coef": 1e-3}      # the official call changed
    with pytest.raises(ValueError, match="donors of the scene were registered with"):
        mixer.decide(scene, 0, [np.full(4, -1.0)] * 3)
    scene.donor_registration = dict(rule.donor_registration)
    scene.held_rule = HeldRule(2, 0.3, 0.1)                             # a later change of the rule: the rule file is refused
    with pytest.raises(ValueError, match="hold rule"):
        mixer.decide(scene, 0, [np.full(4, -1.0)] * 3)


def test_rope_pools_with_dropped_ropes_fit_the_mismatch_from_the_pools_own_hold_tests(tmp_path, monkeypatch, capsys):
    """As the real pools: in a few attempts the rope was dropped before the second subtask. Their unmix rows have the
    mismatch from the recorded distances (hand empty, source holds: 1), their randmix rows get every source's flag from
    the pools' own hold tests (every source holds), and the mismatch slope is fitted, not left at a silent 0."""
    rng = np.random.default_rng(22)
    summary, data, _generator = rope_fit(tmp_path, monkeypatch, lambda c: rope_pool(rng, "randmix", 40, dropped=0.15),
                                         lambda c: rope_pool(rng, "unmix", 40, dropped=0.15))
    said = capsys.readouterr().out
    holds = summary["source_holds"]
    assert holds["filled"] > 0 and holds["unknown"] == 0 and "filled from the hold tests" in said
    assert all(counts[0] == counts[1] > 0 for key, counts in holds["evidence"].items() if key.startswith("robot0/1/"))
    assert summary["unknown_rows"] == {"0": 0, "1": 0} and "inhand_mismatch" not in summary["constant_features"]["1"]
    assert data["subtasks"][1]["g"]["inhand_mismatch"] != 0.0
    held = summary["sanity"]["randmix"]["blocks"]["1"]["held"]
    assert 0 < held["now"] < held["tested"] and all(c[0] == c[1] for c in held["sources"].values())


def test_a_constant_feature_with_a_fitted_slope_is_a_loud_warning(capsys):
    from tmg.generation.fit_coefficients import constant_features, print_block

    rows = [np.array([1.0, 0.0, 0.1 * i, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0]) for i in range(4)]        # J = 2, six features, H
    constant = constant_features((rows, [1.0] * 4, [0] * 4), ([], [], []), 2, HELD_FEATURES)
    assert constant == {"dy": 0.2, "rotation": 0.0, "inhand_position": 0.0, "inhand_rotation": 0.0, "inhand_mismatch": 1.0}
    block = {"rows": {"randmix": 4, "unmix": 0}, "d": 0.1, "g": {**dict.fromkeys(HELD_FEATURES, 0.0), "inhand_mismatch": 0.7}}
    print_block("1", block, {"reached": "x", "unknown_rows": 3, "constant_features": constant})
    out = capsys.readouterr().out
    assert "WARNING subtask 1: feature inhand_mismatch is 1.0 in every row, yet its fitted slope is +0.7" in out
    assert "WARNING subtask 1: 3 randmix attempts left out" in out and "feature dy is 0.2 in every row (slope 0" in out


# sanity report --------------------------------------------------------------------------------------------------------
LIMITS = SanityLimits(max_scene_error_rate=0.05, max_unreached_success_rate=0.05, confidence=0.999)


def test_a_rate_is_judged_by_what_the_counts_show_not_by_a_share():
    """``rate_exceeds``: the counts must show with the confidence that the true rate is above the limit."""
    from scipy.stats import binom

    for count, total, rate in ((3, 20, 0.05), (50, 50, 0.05), (12, 400, 0.05), (40, 400, 0.05), (1, 941, 0.05), (0, 10, 0.3)):
        assert binomial_tail(count, total, rate) == pytest.approx(float(binom.sf(count - 1, total, rate)), rel=1e-9)
    assert not rate_exceeds(3, 20, 0.05, 0.999)                  # 15 % of one small chunk: a 5 % rate does that 7.5 % of the time
    assert not rate_exceeds(8, 50, 0.05, 0.999) and rate_exceeds(10, 50, 0.05, 0.999)
    assert rate_exceeds(5, 5, 0.05, 0.999) and rate_exceeds(50, 50, 0.05, 0.999)     # every attempt: at once
    assert not rate_exceeds(25, 400, 0.05, 0.999) and rate_exceeds(40, 400, 0.05, 0.999)   # 6 % of a pool is not shown, 10 % is
    assert not rate_exceeds(0, 50, 0.05, 0.999) and not rate_exceeds(5, 5, None, 0.999) and not rate_exceeds(1, 0, 0.05, 0.999)
    assert SanityLimits.from_mapping({"max_scene_error_rate": 0.05}) == SanityLimits(max_scene_error_rate=0.05)
    assert SanityLimits.from_mapping(None) == SanityLimits() and not SanityLimits().unreached_exceeded(50, 50)   # no limit: reported only
    with pytest.raises(ValueError, match="unknown keys"):
        SanityLimits.from_mapping({"min_successes": 10})


def test_a_healthy_pool_is_not_stopped_by_noise_and_a_broken_one_is_stopped_at_once():
    """The verdict on the pool's cumulative counts, chunk after chunk (as a chunk result judges them). A true 2 % scene
    error rate against the 5 % limit never trips over 80 chunks of 50 attempts, although single chunks exceed 5 % (the
    old per-chunk comparison would have stopped these pools); a 20 % rate trips within a few chunks; every attempt
    failing trips at the first."""
    rng = np.random.default_rng(31)

    def first_stop(rate, chunks=80, attempts=50):
        total = bad = 0
        for index in range(chunks):
            bad += int(rng.binomial(attempts, rate))
            total += attempts
            if LIMITS.scene_errors_exceeded(bad, total):
                return index
        return None

    per_chunk = rng.binomial(50, 0.02, size=(200, 80))
    assert (per_chunk.max(axis=1) > 0.05 * 50).mean() > 0.9           # a per-chunk share over the limit: nearly every pool
    assert all(first_stop(0.02) is None for _ in range(200))
    stops = [first_stop(0.20) for _ in range(50)]
    assert all(stop is not None and stop <= 3 for stop in stops)
    assert first_stop(1.0) == 0


def test_sanity_counts_add_up_over_chunks_and_are_judged_together():
    rng = np.random.default_rng(15)
    layout = isaac_layout(["left", "right"], sanity=LIMITS)
    ok = [isaac_attempt(rng, success=True, grasped=True) for _ in range(18)]
    misread = [isaac_attempt(rng, success=True, grasped=False) for _ in range(2)]      # succeeded, yet the grasp "never fired"
    chunk = sanity_report(attempts_of(ok + misread), layout, {})
    block = chunk["blocks"]["right/1"]
    assert (block["attempts"], block["successes"], block["successes_unreached"]) == (20, 20, 2)
    assert chunk["blocks"]["right/0"]["successes_unreached"] == 0 and chunk["attempts"] == 20 and chunk["without_scene"] == 0
    assert judge_sanity(chunk, LIMITS) == ([], [])                                # 2 of 20: nothing shown yet
    assert judge_sanity(merge_sanity([chunk] * 6), LIMITS) == ([], [])            # 12 of 120: a 5 % rate still does that
    pool = merge_sanity([chunk] * 15)                                             # the same share over fifteen chunks: 30 of 300
    assert pool["blocks"]["right/1"]["successes_unreached"] == 30 and pool["attempts"] == 300
    assert pool["blocks"]["right/1"]["held"]["sources"] == {k: [15 * v[0], 15 * v[1]] for k, v in block["held"]["sources"].items()}
    errors, warnings = judge_sanity(pool, LIMITS)
    assert len(errors) == 1 and "block right/1: 30 of 300 (10.0 %) successful attempts count as not having reached it" in errors[0]
    assert judge_sanity(pool, LIMITS, unreached_is_error=False) == ([], errors)       # a pool the fit does not read: a warning
    assert judge_sanity(pool, SanityLimits()) == ([], [])                             # no limit: reported only
    assert merge_sanity([]) == {"reached": None, "attempts": 0, "without_scene": 0, "blocks": {}, "unreadable": []}
    assert merge_sanity([None, chunk]) == merge_sanity([chunk])
    broken = sanity_report(attempts_of([isaac_attempt(rng, success=True, grasped=False) for _ in range(6)]), layout, {})
    assert "6 of 6" in judge_sanity(broken, LIMITS)[0][0]                             # every success misread: at the first chunk
    silent = [dict(isaac_attempt(rng, success=True, grasped=True), term_first={}) for _ in range(3)]
    report = sanity_report(attempts_of(silent), layout, {})                            # the rule cannot read the records at all
    assert "cannot read the records" in report["unreadable"][0] and "reached: task_success" in report["unreadable"][0]
    assert judge_sanity(report, LIMITS)[0] == report["unreadable"]                     # an error at once, whatever the counts
    assert judge_sanity(sanity_report(attempts_of(silent), isaac_layout(["left", "right"], reached="task_success"), {}), LIMITS) == ([], [])


def test_scene_errors_are_counted_over_all_attempts_and_judged_on_the_pool():
    rng = np.random.default_rng(19)
    layout = isaac_layout(["left", "right"], sanity=LIMITS)
    good = [isaac_attempt(rng, success=True, grasped=True) for _ in range(47)]
    bad = [dict(isaac_attempt(rng, success=True, grasped=True), invalid="scene_error") for _ in range(3)]
    chunk = sanity_report(attempts_of(good + bad), layout, {})
    assert (chunk["attempts"], chunk["without_scene"]) == (50, 3) and chunk["blocks"]["right/0"]["attempts"] == 47
    assert judge_sanity(chunk, LIMITS) == ([], [])                                  # 6 % of one chunk: not shown
    errors, _ = judge_sanity(merge_sanity([chunk] * 10), LIMITS)                    # 6 % of 500 attempts is still not shown
    assert errors == []
    many = sanity_report(attempts_of(good[:40] + bad * 4), layout, {})              # 12 of 52
    errors, _ = judge_sanity(many, LIMITS)
    assert len(errors) == 1 and "12 of 52 (23.1 %) attempts without the scene of a subtask" in errors[0]


def test_sanity_reports_the_held_counts_and_warns_when_nothing_is_ever_held():
    rng = np.random.default_rng(16)
    layout = isaac_layout(["left", "right"])
    held = [isaac_attempt(rng, success=True, grasped=True, mix=True) for _ in range(6)]
    loose = [isaac_attempt(rng, success=True, grasped=True, mix=True, held=False) for _ in range(2)]
    report = sanity_report(attempts_of(held + loose), layout, {})
    assert report["blocks"]["right/1"]["held"] == {"tested": 8, "now": 6, "sources": {"0": [8, 8], "1": [0, 8]}}
    assert "held" not in report["blocks"]["right/0"] and judge_sanity(report, LIMITS) == ([], [])
    report = sanity_report(attempts_of(loose), layout, {})
    assert report["blocks"]["right/1"]["held"]["now"] == 0
    assert any("holds the carried object in none of 2" in text for text in judge_sanity(report, LIMITS)[1])
    unmixed = [isaac_attempt(rng, success=True, grasped=True) for _ in range(40)]       # per source from the selected one
    sources = sanity_report(attempts_of(unmixed), layout, {})["blocks"]["right/1"]["held"]["sources"]
    assert sources["0"][0] == sources["0"][1] > 0 and sources["1"][0] == 0 < sources["1"][1]


def test_sanity_reports_the_share_of_unmixed_mix_records_with_their_reasons():
    rng = np.random.default_rng(17)
    attempts = [isaac_attempt(rng, success=True, grasped=True, mix=True) for _ in range(4)]
    attempts[0]["subtasks"][0]["mix"] = {"subtask": 0, "eef": "right", "mixed": False, "weights": [1.0, 0.0]}
    attempts[1]["subtasks"][0]["mix"] = {"subtask": 0, "eef": "right", "mixed": False, "identity_reason": "subtask has no reference object"}
    mix = sanity_report(attempts_of(attempts), isaac_layout(["left", "right"]), {})["blocks"]["right/0"]["mix"]
    assert mix == {"records": 4, "identity": 2,
                   "identity_reasons": {"the weights kept the selected source alone": 1, "subtask has no reference object": 1}}


def test_fit_coefficients_stops_on_a_sanity_error_before_it_fits(tmp_path, monkeypatch):
    rng = np.random.default_rng(18)

    def unmix(chunk):
        out = rope_pool(rng, "unmix", 30)
        for a in out:
            a.update(success=True, term_first={"grasp": 150})             # every success "grasped" long after shaping began
        return out

    with pytest.raises(ValueError, match=r"sanity check.*unmix pools: block 1: 120 of 120 \(100.0 %\) successful"):
        rope_fit(tmp_path, monkeypatch, lambda c: rope_pool(rng, "randmix", 30), unmix)
    assert not (tmp_path / "out" / "coefficients.yaml").exists()


# registering what a new generator needs, without editing shared code --------------------------------------------------
def test_a_new_generator_registers_its_own_reached_rule_feature_and_mixer(monkeypatch):
    """A reached rule, a feature group and a mixer are added by registering them (from the generator's own modules); the
    fit, the rule files and ``build_method`` then use them by name."""
    import sys
    import types

    import tmg.generation.method.fit_records as fit_records
    import tmg.generation.method.interfaces as interfaces
    import tmg.generation.method.success_model as success_model

    # copies of the registries, so what this test registers is gone afterwards
    monkeypatch.setattr(fit_records, "REACHED", dict(fit_records.REACHED))
    monkeypatch.setattr(interfaces, "_MIXERS", dict(interfaces._MIXERS))
    monkeypatch.setattr(success_model, "UNITS", dict(success_model.UNITS))
    monkeypatch.setattr(success_model, "_GROUP_OF", dict(success_model._GROUP_OF))

    class EveryOther(ReachedRule):
        name = "every_other"

        def outcomes(self, attempts, eef, num_subtasks, task):
            return [[index % 2 == 0] * num_subtasks for index in range(len(attempts))], {k: {"signal": len(attempts), "assumed": 0}
                                                                                        for k in range(num_subtasks - 1)}

    fit_records.register_reached_rule(EveryOther())
    assert fit_records.reached_rule("every_other").name == "every_other"
    with pytest.raises(ValueError, match="already registered"):
        fit_records.register_reached_rule(type("Other", (ReachedRule,), {"name": "every_other"})())
    rng = np.random.default_rng(23)
    attempts = attempts_of([isaac_attempt(rng, success=True, grasped=True, mix=True) for _ in range(40)])
    blocks, _ = fit_blocks(attempts, attempts, layout=isaac_layout(["left", "right"], reached="every_other"), task={}, num_sources=2,
                           with_cv=False)
    assert blocks[S(1, "right")]["rows"] == {"randmix": 20, "unmix": 20} and blocks[S(0, "right")]["rows"]["unmix"] == 40

    height = success_model.FeatureGroup(("height",), ("m",), lambda scene: np.full((scene.num_sources, 1), 0.3),
                                        lambda mix, num_sources: np.asarray(mix["height"], dtype=float).reshape(num_sources, 1))
    success_model.register_features(height)
    assert success_model.check_features((*FEATURES, "height"))[-1] == "height" and success_model.UNITS["height"] == "m"
    scene = scene_of_record(attempts.records[0]["subtasks"][0], None)
    assert source_features(scene, ("height", "dx")).shape == (1, 2) and source_features(scene, ("height", "dx"))[0, 0] == 0.3
    assert mix_record_features({"scene_features": [[0.1, 0.2, 0.3]] * 2, "height": [0.5, 0.6]}, ("dx", "height"), 2).tolist() == [[0.1, 0.5], [0.1, 0.6]]
    with pytest.raises(ValueError, match="already registered"):
        success_model.register_features(height)

    module = types.ModuleType("fake_generator_mixers")

    class Passthrough:
        @classmethod
        def from_config(cls, cfg, *, chunk_seed, method_name, arms):
            return cls()

    module.Passthrough = Passthrough
    monkeypatch.setitem(sys.modules, "fake_generator_mixers", module)
    interfaces.register_mixer("passthrough", Passthrough)
    method = interfaces.build_method({"name": "x", "mixing": {"enabled": True, "type": "passthrough", "module": "fake_generator_mixers"}},
                                     chunk_seed=0)
    assert isinstance(method.mixer, Passthrough)
    with pytest.raises(ValueError, match="unknown"):
        interfaces.build_method({"name": "x", "mixing": {"enabled": True, "type": "nowhere"}}, chunk_seed=0)
