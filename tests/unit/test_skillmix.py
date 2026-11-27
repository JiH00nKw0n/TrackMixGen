"""SkillMix: the features, the softmax and randmix weights, the rule files, the subtask ids and the mixer (the fit:
test_fit.py)."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from tmg.generation.method.interfaces import Arms, HeldRule, HeldTest, MixContext, SubtaskId, SubtaskScene, build_method
from tmg.generation.method.skillmix import (
    MIN_WEIGHT,
    RANDMIX_SEED,
    RandmixDraws,
    SkillMixer,
    SkillMixRule,
    SourceDraws,
    SubtaskRule,
    load_rule,
    randmix_weights,
    single_source,
    softmax_weights,
)
from tmg.generation.method.success_model import (
    FEATURES,
    HELD_FEATURES,
    in_hand,
    inhand_features,
    scene_features,
    source_features,
)

S = SubtaskId


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    p = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    p[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    p[:3, 3] = [x, y, z]
    return p


def flat(p):
    return [*p[:3, 3], *p[:3, :3].reshape(-1)]


def _rule(n, a=None, g=None, d=1.0, subtasks=(0,)):
    a = np.zeros(n) if a is None else np.asarray(a, dtype=float)
    g = np.zeros(5) if g is None else np.asarray(g, dtype=float)
    return SkillMixRule(name="g/t", num_sources=n, subtasks={S(k): SubtaskRule(a=a, g=g, d=float(d), c0=0.0) for k in subtasks})


# features and rows ----------------------------------------------------------------------------------------------------
def test_scene_and_inhand_features():
    f = scene_features(pose(0.1, -0.2, yaw=0.3), pose(0.0, 0.1, yaw=0.1))
    assert np.allclose(f, [0.1, 0.3, 0.2])
    h1 = in_hand(pose(1, 0, 0), pose(1, 0, -0.1))
    h2 = in_hand(pose(0, 0, 0, yaw=0.5), pose(0.02, 0, -0.1, yaw=0.5))
    d = inhand_features(h1, h2)
    assert d[0] == pytest.approx(0.02) and d[1] == pytest.approx(0.0, abs=1e-9)


# weights --------------------------------------------------------------------------------------------------------------
def test_softmax_weights_are_the_softmax_of_the_scores_over_the_spread_effect():
    s = np.array([1.0, 0.0, -1.0, 3.0])
    w = softmax_weights(s, 0.5, [0, 1, 2])                 # source 3 incompatible: weight 0 whatever its score
    e = np.exp(np.array([1.0, 0.0, -1.0]) / 0.5)
    assert np.allclose(w, [*(e / e.sum()), 0.0]) and w.sum() == pytest.approx(1.0)
    hot = softmax_weights(s, 0.05, [0, 1, 2, 3])           # small d: nearly the best source alone
    assert hot[3] == pytest.approx(1.0) and hot[:3].sum() < 1e-6
    flat_w = softmax_weights(s, 1e6, [0, 1, 2, 3])          # large d: nearly uniform
    assert np.allclose(flat_w, 0.25, atol=1e-5)


def test_softmax_weights_take_the_best_compatible_source_when_spreading_does_not_help():
    s = np.array([0.2, 0.9, 0.5])
    assert softmax_weights(s, 0.0, [0, 1, 2]).tolist() == [0.0, 1.0, 0.0]
    assert softmax_weights(s, -0.7, [0, 2]).tolist() == [0.0, 0.0, 1.0]      # source 1 incompatible
    with pytest.raises(ValueError):
        softmax_weights(s, 1.0, [])


def test_softmax_weights_drop_weights_below_the_minimum_and_renormalize():
    w = softmax_weights(np.array([0.0, -10.0, 0.0]), 1.0, [0, 1, 2])
    assert np.exp(-10) / (2 + np.exp(-10)) < MIN_WEIGHT
    assert w.tolist() == [0.5, 0.0, 0.5]


def test_rule_scores_add_the_feature_slopes_to_the_source_effects():
    block = SubtaskRule(a=np.array([0.5, -0.5]), g=np.array([-1.0, -2.0, -0.5, -10.0, -1.0]), d=0.3, c0=-1.0)
    scene = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    assert np.allclose(block.scores(np.hstack([scene, np.zeros((2, 2))])), [0.5 - 0.1 - 0.4 - 0.15, -0.5])
    assert np.allclose(block.scores(np.hstack([scene, [[0.01, 0.1], [0.0, 0.0]]])), [0.5 - 0.65 - 0.1 - 0.1, -0.5])


def test_randmix_draws_are_dirichlet_weights_from_their_own_stream():
    draws = RandmixDraws(chunk_seed=30001)
    first = draws.draw(S(0), 4)
    assert draws.draw(S(0), 4) is first                       # one draw per subtask and attempt (CP-Gen may re-solve)
    rng = np.random.default_rng(np.random.SeedSequence([30001, RANDMIX_SEED]))
    c = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
    gamma = rng.gamma(c, size=4)
    fallback = int(rng.integers(4))
    assert first[0] == c and np.array_equal(first[1], gamma) and first[2] == fallback
    w = randmix_weights(first, [0, 1, 3])
    expected = np.r_[gamma[[0, 1]], 0.0, gamma[3]] / gamma[[0, 1, 3]].sum()
    expected[expected < MIN_WEIGHT] = 0.0
    assert np.allclose(w, expected / expected.sum()) and w[2] == 0.0
    second = draws.draw(S(1), 4)
    draws.begin_attempt()
    assert draws.draw(S(0), 4)[0] != first[0] and second[0] != first[0]     # the stream moves on; a new attempt draws anew
    assert randmix_weights((0.1, np.zeros(4), 5), [1, 2]).tolist() == [0.0, 0.0, 1.0, 0.0]   # all gammas 0: fallback


# rule files -----------------------------------------------------------------------------------------------------------
COMMITTED = {"mimicgen": {"coffee": 2, "square": 2, "threading": 2, "stack_three": 4, "hammer_cleanup": 3, "mug_cleanup": 3,
                          "three_piece_assembly": 4},
             "cpgen": {"coffee": 3, "threading": 2, "hammer_cleanup": 4, "mug_cleanup": 4, "three_piece_assembly": 4}}


def test_committed_rule_files_load_and_give_weights():
    for generator, tasks in COMMITTED.items():
        for task, n in tasks.items():
            rule = load_rule(f"{generator}/{task}")
            assert rule.num_sources == 10 and sorted(rule.subtasks, key=lambda s: s.order) == [S(k) for k in range(n)]
            assert rule.features == FEATURES and rule.held_rule is None and rule.object_frame is None
            for k in range(n):
                block = rule.block(S(k))
                assert block.a.shape == (10,) and block.g.shape == (5,) and np.isfinite(block.d)
                if k == 0:
                    assert np.allclose(block.g[3:], 0.0)          # no in-hand terms in the first subtask
                w = softmax_weights(block.scores(np.hstack([np.full((10, 3), 0.05), np.full((10, 2), 0.01)])), block.d, list(range(10)))
                assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
    # the MimicGen fits favour spreading the weight (d > 0) in every subtask
    assert all(b.d > 0 for t in COMMITTED["mimicgen"] for b in load_rule(f"mimicgen/{t}").subtasks.values())


def test_every_committed_rule_file_loads_unchanged_with_single_arm_keys():
    """Every file under configs/experiment/generator/*/coefficients/ is read as before: integer keys, the five features, and
    a one-armed generator (ids k, whatever its records call the arm) finds its blocks while a two-armed one never does."""
    root = Path(__file__).resolve().parents[2] / "configs" / "experiment" / "generator"
    files = sorted(root.glob("*/coefficients/*.yaml"))
    assert len(files) >= sum(len(t) for t in COMMITTED.values())
    for path in files:
        data = yaml.safe_load(path.read_text())
        rule = load_rule(str(path))
        assert sorted(key.key for key in rule.subtasks) == sorted(int(k) for k in data["subtasks"])
        for k, block in data["subtasks"].items():
            b = rule.block(Arms().subtask(int(k)))
            assert b is rule.block(Arms(("robot0",)).subtask(int(k), "robot0"))      # a single-arm generator (SoftMimicGen)
            with pytest.raises(KeyError, match="more than one end effector"):
                rule.block(Arms(("left", "right")).subtask(int(k), "right"))       # never for a two-armed generator
            assert b.a.tolist() == [float(block["a"][j]) for j in range(rule.num_sources)] and b.d == float(block["d"])
            assert b.g.tolist() == [float(block["g"][f]) for f in FEATURES]
        assert rule.describe()["subtasks"] == sorted(int(k) for k in data["subtasks"])


def scene(n=2, k=0, eef=None, current=None, sources=None, **more):
    return SubtaskScene(subtask=k, eef=eef, reference_object="obj", current_object_pose=pose() if current is None else current,
                        source_object_poses=sources or [pose()] * n, **more)


def test_two_arm_rule_files_key_their_blocks_by_end_effector_and_subtask(tmp_path):
    g = dict.fromkeys(("dx", "dy", "rotation", "inhand_position", "inhand_rotation"), 0.0)
    path = tmp_path / "dex.yaml"
    path.write_text(yaml.safe_dump({"rule": "softmax", "num_sources": 2, "subtasks": {
        "right/0": {"a": [1.0, 0.0], "g": g, "d": 0.5, "c0": 0.0}, "right/1": {"a": [0.0, 0.0], "g": g, "d": 0.5, "c0": 0.0},
        "left/0": {"a": [0.0, 1.0], "g": g, "d": 0.5, "c0": 0.0}}}))
    rule = load_rule(str(path))
    assert sorted(rule.subtasks, key=lambda s: s.order) == [S(0, "left"), S(0, "right"), S(1, "right")]
    assert rule.describe()["subtasks"] == ["left/0", "right/0", "right/1"]
    assert rule.block(S(0, "right")).a.tolist() == [1.0, 0.0] and rule.block(S(0, "left")).a.tolist() == [0.0, 1.0]
    with pytest.raises(KeyError, match="left/1"):
        rule.block(S(1, "left"))
    with pytest.raises(KeyError, match="the generator has one arm"):
        rule.block(S(0))                                             # a two-arm file is not for a one-armed generator
    # the mixer picks the arm's block: right prefers source 0, left source 1 (no collision on k = 0)
    mixer = SkillMixer(coefficients=rule, arms=Arms(("left", "right")))
    w_right, rec_right = mixer.decide(scene(eef="right"), 0, [np.full(4, -1.0)] * 2)
    w_left, rec_left = mixer.decide(scene(eef="left"), 0, [np.full(4, -1.0)] * 2)
    assert w_right[0] > w_left[0] and rec_right["eef"] == "right" and rec_left["eef"] == "left"


def test_subtask_ids_and_arms():
    assert S(2).key == 2 and S(2, "right").key == "right/2" and str(S(2, "right")) == "right/2"
    assert S.parse(3) == S(3) and S.parse("3") == S(3) and S.parse("left/1") == S(1, "left")
    with pytest.raises(ValueError, match="<eef>/<k>"):
        S.parse("/1")
    assert sorted([S(1, "right"), S(0, "right"), S(0, "left")], key=lambda s: s.order) == [S(0, "left"), S(0, "right"), S(1, "right")]
    one, named, two = Arms(), Arms.of(["robot0"]), Arms.of(("left", "right"))
    assert Arms.of(None) == one and Arms.of(two) is two
    assert not one.per_arm and not named.per_arm and two.per_arm
    assert one.subtask(1) == S(1) and named.subtask(1, "robot0") == S(1) and two.subtask(1, "left") == S(1, "left")
    for eef in (None, "robot0"):
        with pytest.raises(KeyError, match="arms"):
            two.subtask(0, eef)                                       # a two-armed generator must name one of its arms
    assert one.record_eefs == (None,) and named.record_eefs == ("robot0",) and two.record_eefs == ("left", "right")


def test_randmix_and_source_draws_of_two_arms_do_not_collide_and_one_arm_streams_are_unchanged():
    one, keyed = RandmixDraws(3), RandmixDraws(3, arms=Arms(("robot0",)))
    assert one.draw(S(0), 4)[0] == keyed.draw(S(0), 4)[0]        # single arm: same key, same stream as before
    two = RandmixDraws(3, arms=["left", "right"])
    right, left = two.draw(S(0, "right"), 4), two.draw(S(0, "left"), 4)
    assert right[0] != left[0] and two.draw(S(0, "right"), 4) is right      # kept within the attempt per arm
    src = SourceDraws(5, arms=["left", "right"])
    assert src.draw(S(0, "right")) != src.draw(S(0, "left")) and src.draw(S(0, "right")) == src.draw(S(0, "right"))


def test_skillmix_mixer_scores_with_feature_poses_and_zero_inhand_for_unknown_sources():
    """SoftMimicGen blends a deformable object's segments in the world frame (identity blending frames) but scores the
    sources with their nodal frames (the scene's source_object_poses); a source without an in-hand pose gets in-hand 0."""
    segs = [_segment(0.0), _segment(0.0), _segment(0.0)]
    world = [np.eye(4)] * 3
    nodal = [pose(0.0), pose(0.05), pose(0.3)]
    rule = _rule(3, g=[-10.0, 0.0, 0.0, -10.0, 0.0], d=0.5, subtasks=(0, 1))
    mixer = SkillMixer(coefficients=rule, arms=Arms(("robot0",)))
    out = mixer.mix(_context(segs, world, 0, pose(0.0), features=nodal, eef="robot0"))
    rec = out.record
    assert rec["scores"] == [0.0, -0.5, -3.0] and rec["eef"] == "robot0" and rec["mixed"] is True
    assert len(set(rec["weights"])) == 3                              # non-identity weights from the nodal frames
    hand = dict(carried_object="obj", eef_pose=pose(), carried_pose=pose(0.0), source_eef_poses=[pose()] * 3,
                source_carried_poses=[pose(0.1), None, pose(0.0)])
    rec = SkillMixer(coefficients=rule).mix(_context(segs, nodal, 0, pose(0.0), subtask=1, **hand)).record
    assert rec["inhand_features"] == [[0.1, 0.0], [0.0, 0.0], [0.0, 0.0]] and "held" not in rec


def test_skillmix_mixer_with_a_current_object_pose_gives_non_identity_weights():
    """A scene with a current object pose: the mixer scores and blends (no identity record); without one it declines."""
    segs = [_segment(0.0), _segment(0.0), _segment(0.0)]
    objs = [pose(0.0), pose(0.02), pose(0.04)]
    rule = _rule(3, g=[-5.0, 0.0, 0.0, 0.0, 0.0], d=1.0)
    rec = SkillMixer(coefficients=rule, arms=Arms(("right",))).mix(_context(segs, objs, 1, pose(0.01), eef="right")).record
    assert "identity_reason" not in rec and rec["mixed"] is True and max(rec["weights"]) < 1.0
    rec = SkillMixer(coefficients=rule).mix(_context(segs, objs, 1, None)).record
    assert rec["mixed"] is False and rec["identity_reason"] == "no current object pose"
    # a subtask with a reference object and no scene at all is a runtime's omission: raised, never a silent identity
    with pytest.raises(ValueError, match="gave the mixer no scene"):
        SkillMixer(coefficients=rule).mix(MixContext(**{**vars(_context(segs, objs, 1, pose())), "scene": None}))


# the hold test: features, record and the rule file's check --------------------------------------------------------
RULE = HeldRule(version=1, rigid_m=0.2, nodes_m=0.1)


def held_scene(now_m, sources_m, k=1):
    """Three sources whose in-hand pose is 5 cm from the hand; held where the hold distance is within the rule."""
    test = HeldTest(rule=RULE, deformable=False, now_m=now_m, sources_m=list(sources_m))
    return scene(3, k=k, carried_object="obj", eef_pose=pose(), source_eef_poses=[pose()] * 3, held=test, held_rule=RULE,
                 carried_pose=pose(0.05) if test.now else None,
                 source_carried_poses=[pose(0.05 + 0.01 * j) if h else None for j, h in enumerate(test.sources)])


def test_the_hold_mismatch_feature_tells_a_source_that_does_not_hold_from_one_that_holds_alike():
    """Source 0 holds the object exactly as now, source 1 a centimetre away, source 2 does not hold it: the in-hand
    features of 0 and 2 are both (0, 0); only the mismatch tells them apart."""
    held = held_scene(0.01, [0.01, 0.02, 0.6])
    f = source_features(held, HELD_FEATURES)
    assert f.shape == (3, 6) and np.allclose(f[:, 3], [0.0, 0.01, 0.0]) and f[:, 5].tolist() == [0.0, 0.0, 1.0]
    assert np.array_equal(source_features(held, FEATURES), f[:, :5])            # the five features are unchanged by it
    not_now = held_scene(0.9, [0.01, 0.02, 0.6])                                # the hand holds nothing now
    f = source_features(not_now, HELD_FEATURES)
    assert np.all(f[:, 3:5] == 0) and f[:, 5].tolist() == [1.0, 1.0, 0.0]        # the two holders mismatch; source 2 is alike
    assert np.all(source_features(held_scene(0.01, [0.01] * 3, k=0), HELD_FEATURES)[:, 5] == 0)
    first = scene(3)                                                             # first subtask: nothing carried
    assert np.all(source_features(first, HELD_FEATURES)[:, 3:] == 0)
    with pytest.raises(ValueError, match="unknown or repeated"):
        source_features(first, ("dx", "height"))


def test_the_mixer_records_who_holds_and_scores_the_mismatch_with_the_rule_files_features():
    g = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -2.0])
    rule = SkillMixRule(name="g/t", num_sources=3, subtasks={S(1): SubtaskRule(a=np.zeros(3), g=g, d=1.0, c0=0.0)},
                        features=HELD_FEATURES, held_rule=RULE)
    w, rec = SkillMixer(coefficients=rule).decide(held_scene(0.01, [0.01, 0.02, 0.6]), 0, [np.full(4, -1.0)] * 3)
    assert rec["scores"] == [0.0, 0.0, -2.0] and rec["held"] == {"now": True, "sources": [True, True, False]}
    assert rec["inhand_features"][2] == [0.0, 0.0] and w[2] < w[0]
    randmix = SkillMixer(rule="randmix").decide(held_scene(0.9, [0.01, 0.02, 0.6]), 0, [np.full(4, -1.0)] * 3)[1]
    assert randmix["held"] == {"now": False, "sources": [True, True, False]} and np.all(np.array(randmix["inhand_features"]) == 0)


def test_the_mixer_checks_the_hold_rule_against_the_rule_file():
    def rule(held_rule, features=HELD_FEATURES):
        return SkillMixRule(name="g/t", num_sources=3, subtasks={S(1): SubtaskRule(a=np.zeros(3), g=np.zeros(len(features)), d=1.0, c0=0.0)},
                            features=features, held_rule=held_rule)

    series = [np.full(4, -1.0)] * 3
    assert "weights" in SkillMixer(coefficients=rule(RULE)).decide(held_scene(0.01, [0.01] * 3), 0, series)[1]
    with pytest.raises(ValueError, match="hold rule"):                # fitted with other distances
        SkillMixer(coefficients=rule(HeldRule(2, 0.3, 0.1))).decide(held_scene(0.01, [0.01] * 3), 0, series)
    with pytest.raises(ValueError, match="hold rule"):                # fitted without a hold test
        SkillMixer(coefficients=rule(None, FEATURES)).decide(held_scene(0.01, [0.01] * 3), 0, series)
    with pytest.raises(ValueError, match="hold rule"):                # a runtime without a hold test, a rule fitted with one
        SkillMixer(coefficients=rule(RULE)).decide(scene(3, k=1), 0, series)
    assert "weights" in SkillMixer(rule="randmix").decide(held_scene(0.01, [0.01] * 3), 0, series)[1]   # nothing fitted yet


def test_the_mixer_checks_how_the_donors_were_registered_against_the_rule_file(tmp_path):
    """A deformable subtask's donors are registered into the selected source's geometry before they are mixed. The rule
    file records the settings its records were made with (``donor_registration``), the scene carries the runtime's, and
    the mixer refuses a difference, as for the object frame and the hold rule."""
    from tmg.generation.fit_coefficients import write_rule

    official = {"use_rotation_transform": True, "bend_coef": 0.1, "rot_coef": 1e-3}
    block = {"a": [0.0, 0.0], "g": dict.fromkeys(FEATURES, 0.0), "d": 1.0, "c0": 0.0}
    path = write_rule(tmp_path / "rope.yaml", generator="softmimicgen", task="rope", basis="t", num_sources=2,
                      blocks={S(0): block}, donor_registration=official)
    assert yaml.safe_load(path.read_text())["donor_registration"] == {"use_rotation_transform": True, "bend_coef": 0.1, "rot_coef": 0.001}
    rule = load_rule(str(path))
    assert rule.donor_registration == official
    series = [np.full(4, -1.0)] * 2
    w, record = SkillMixer(coefficients=rule).decide(scene(2, donor_registration=dict(official)), 0, series)
    assert record["registration"] == official and w.sum() == pytest.approx(1.0)
    for other in ({**official, "bend_coef": 0.2}, {**official, "use_rotation_transform": False}, {"bend_coef": 0.1}):
        with pytest.raises(ValueError, match="donors of the scene were registered with"):
            SkillMixer(coefficients=rule).decide(scene(2, donor_registration=other), 0, series)
    with pytest.raises(ValueError, match="donors of the scene were registered with"):            # a rule fitted without any
        SkillMixer(coefficients=_rule(2)).decide(scene(2, donor_registration=dict(official)), 0, series)
    plain = SkillMixer(coefficients=rule).decide(scene(2), 0, series)[1]                       # a rigid subtask registers nothing
    assert "registration" not in plain
    randmix = SkillMixer(rule="randmix").decide(scene(2, donor_registration=dict(official)), 0, series)[1]
    assert randmix["registration"] == official                           # recorded for the fit, nothing to check yet
    assert _rule(2).donor_registration is None and load_rule("mimicgen/square").donor_registration is None


def test_a_rule_file_lists_its_features_and_hold_rule(tmp_path):
    from tmg.generation.fit_coefficients import write_rule

    block = {"a": [0.0, 0.0], "g": dict.fromkeys(HELD_FEATURES, 0.5), "d": 1.0, "c0": 0.0}
    path = write_rule(tmp_path / "rope.yaml", generator="softmimicgen", task="rope", basis="t", num_sources=2,
                      blocks={S(0): block}, features=HELD_FEATURES, held_rule=RULE)
    data = yaml.safe_load(path.read_text())
    assert data["features"] == list(HELD_FEATURES) and data["held_rule"] == {"version": 1, "rigid_m": 0.2, "nodes_m": 0.1}
    assert data["units"]["inhand_mismatch"] == "0 or 1" and "inhand_mismatch" in path.read_text().split("generator:")[0]
    rule = load_rule(str(path))
    assert rule.features == HELD_FEATURES and rule.held_rule == RULE and rule.block(S(0)).g.shape == (6,)
    data["features"] = [*FEATURES, "height"]
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match="unknown or repeated"):
        load_rule(str(path))


def test_a_coefficient_file_of_the_removed_above_mean_rule_is_refused(tmp_path):
    old = tmp_path / "old.yaml"
    old.write_text(yaml.safe_dump({"generator": "g", "task": "t", "num_sources": 2, "subtasks": {
        0: {"alpha": {0: 0.1, 1: -0.2}, "slopes": {"dx": -1.0, "dy": -1.0, "rotation": -0.5}}}}))
    with pytest.raises(ValueError, match="not a SkillMix softmax rule"):
        load_rule(str(old))
    with pytest.raises(FileNotFoundError, match="make coefficients"):
        load_rule(str(tmp_path / "missing.yaml"))


def test_build_method_reads_the_rule_file_and_keeps_randmix_out_of_the_skillmix_folders(tmp_path):
    from tmg.generation.fit_coefficients import write_rule

    path = write_rule(tmp_path / "rule.yaml", generator="g", task="t", basis="test", num_sources=2,
                      blocks={0: {"a": [0.1, -0.2], "g": dict.fromkeys(("dx", "dy", "rotation", "inhand_position", "inhand_rotation"), 0.0),
                                  "d": 0.4, "c0": -1.0}})
    m = build_method({"name": "skillmix", "mixing": {"enabled": True, "type": "skillmix", "rule": "softmax",
                                                     "coefficients": str(path)}}, chunk_seed=0)
    assert isinstance(m.mixer, SkillMixer) and m.mixer.rule == "softmax" and m.mixer.coefficients.block(S(0)).d == 0.4
    with pytest.raises(ValueError, match="skillmix_randmix"):
        build_method({"name": "skillmix", "mixing": {"enabled": True, "type": "skillmix", "rule": "randmix"}}, chunk_seed=0)
    with pytest.raises(ValueError, match="skillmix_randmix"):          # and no softmax pool in the randmix folder
        build_method({"name": "skillmix_randmix", "mixing": {"enabled": True, "type": "skillmix", "rule": "softmax",
                                                                 "coefficients": str(path)}}, chunk_seed=0)
    m = build_method({"name": "skillmix_randmix", "mixing": {"enabled": True, "type": "skillmix", "rule": "randmix",
                                                                 "coefficients": str(tmp_path / "missing.yaml")}}, chunk_seed=0)
    assert m.mixer.rule == "randmix" and m.mixer.coefficients is None          # randmix needs no rule file
    with pytest.raises(ValueError, match="mixing.rule"):
        SkillMixer(rule="above_mean")
    with pytest.raises(ValueError, match="rule file"):
        SkillMixer(rule="softmax")


# the mixer ------------------------------------------------------------------------------------------------------------
def test_single_source_is_the_best_or_drawn_from_the_softmax_of_the_scores():
    s = np.array([0.3, 0.0, -0.2, 5.0])
    assert single_source(s, [0, 1, 2]) == 0                          # best: largest score among the compatible ones
    rng = np.random.default_rng(0)
    draws = np.array([single_source(s, [0, 1, 2], u) for u in rng.random(20000)])
    p = np.exp(s[:3]) / np.exp(s[:3]).sum()
    assert np.allclose([np.mean(draws == j) for j in range(3)], p, atol=0.015)     # sample: softmax(s), T = 1
    assert not np.any(draws == 3)                                    # an incompatible source is never drawn
    w = softmax_weights(s, -0.5, [0, 1, 2], u=0.999)
    assert w.tolist() == [0.0, 0.0, 1.0, 0.0]                         # the top of the cumulative distribution
    assert softmax_weights(s, -0.5, [0, 1, 2], u=0.0).tolist() == [1.0, 0.0, 0.0, 0.0]


def test_source_draws_are_reproducible_and_kept_within_an_attempt():
    a, b = SourceDraws(30001), SourceDraws(30001)
    first = a.draw(S(0))
    assert a.draw(S(0)) == first and b.draw(S(0)) == first            # one draw per subtask; same chunk seed, same draws
    a.begin_attempt()
    assert a.draw(S(0)) != first


def test_skillmix_sample_draws_one_source_only_where_the_fit_gives_no_spread():
    segs = [_segment(0.0), _segment(0.0), _segment(0.0)]
    objs = [pose(0.0), pose(0.0), pose(0.0)]
    rule = SkillMixRule(name="g/t", num_sources=3, subtasks={
        S(0): SubtaskRule(a=np.array([0.2, 0.1, 0.0]), g=np.zeros(5), d=-0.3, c0=0.0),
        S(1): SubtaskRule(a=np.array([0.2, 0.1, 0.0]), g=np.zeros(5), d=0.5, c0=0.0)})
    picked = set()
    for seed in range(40):
        m = SkillMixer(coefficients=rule, chunk_seed=seed, single_source="sample")
        m.begin_attempt()
        w, rec = m.weights(S(0), scene(3), [0, 1, 2])
        assert sorted(w.tolist()) == [0.0, 0.0, 1.0] and "source_draw" in rec
        picked.add(int(np.argmax(w)))
        w1, rec1 = m.weights(S(1), scene(3, k=1), [0, 1, 2])
        assert (w1 > 0).sum() == 3 and "source_draw" not in rec1      # d > 0: blended as before
    assert picked == {0, 1, 2}
    best = SkillMixer(coefficients=rule, chunk_seed=0)
    assert best.weights(S(0), scene(3), [0, 1, 2])[0].tolist() == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="single_source"):
        SkillMixer(coefficients=rule, single_source="random")


def _segment(x, T=6):
    targets = np.stack([pose(x + 0.01 * t) for t in range(T)])
    return (pose(x), targets, np.full((T, 1), -1.0))


def _context(segs, objs, primary, current, subtask=0, features=None, eef=None, reference_object="obj", **more):
    """A mix context: ``objs`` are the blending frames, ``features`` the poses the sources are scored with (the same by
    default); ``more`` are further ``SubtaskScene`` fields."""
    scene = SubtaskScene(subtask=subtask, eef=eef, reference_object=reference_object, current_object_pose=current,
                         source_object_poses=list(features or objs), **more)
    return MixContext(subtask_ind=subtask, primary=primary, primary_eef_start=segs[primary][0], primary_targets=segs[primary][1],
                      primary_gripper=segs[primary][2], src_object_poses=objs, src_segments=segs, num_sources=len(segs),
                      has_reference_object=True, scene=scene)


def test_skillmix_mixer_weighs_the_sources_by_the_softmax_of_their_scores():
    segs = [_segment(0.0), _segment(0.0), _segment(0.0)]
    objs = [pose(0.0), pose(0.05), pose(0.3)]
    rule = _rule(3, a=[0.0, 0.0, 0.0], g=[-10.0, 0.0, 0.0, 0.0, 0.0], d=0.5)
    out = SkillMixer(coefficients=rule).mix(_context(segs, objs, 2, pose(0.0)))
    rec = out.record
    s = np.array([0.0, -0.5, -3.0])
    expected = np.exp(s / 0.5) / np.exp(s / 0.5).sum()
    assert rec["rule"] == "softmax" and rec["d"] == 0.5 and rec["scores"] == [0.0, -0.5, -3.0]
    assert np.allclose(rec["weights"], np.round(expected, 4)) and rec["mixed"] is True and rec["incompatible"] == []
    assert out.targets.shape[1:] == (4, 4)
    best = SkillMixer(coefficients=_rule(3, a=[0.0, 0.0, 0.0], g=[-10.0, 0.0, 0.0, 0.0, 0.0], d=-0.2)).mix(_context(segs, objs, 2, pose(0.0)))
    assert best.record["weights"] == [1.0, 0.0, 0.0] and best.record["mixed"] is True     # another source alone


def test_skillmix_mixer_identity_without_reference_object():
    segs = [_segment(0.0)] * 2
    ctx = MixContext(subtask_ind=0, primary=0, primary_eef_start=segs[0][0], primary_targets=segs[0][1],
                     primary_gripper=segs[0][2], src_object_poses=[None, None], src_segments=segs, num_sources=2,
                     has_reference_object=False, scene=SubtaskScene(subtask=0))
    rec = SkillMixer(coefficients=_rule(2)).mix(ctx).record
    assert rec["mixed"] is False and "identity_reason" in rec


def test_skillmix_randmix_mixer_draws_new_weights_per_attempt():
    segs = [_segment(0.0)] * 3
    objs = [pose(0.0)] * 3
    mixer = SkillMixer(rule="randmix", chunk_seed=7)
    first = mixer.mix(_context(segs, objs, 0, pose(0.0))).record
    assert first["rule"] == "randmix" and "dirichlet_c" in first and sum(first["weights"]) == pytest.approx(1.0, abs=1e-3)
    mixer.begin_attempt()
    second = mixer.mix(_context(segs, objs, 0, pose(0.0))).record
    assert second["dirichlet_c"] != first["dirichlet_c"]


def test_skillmix_mixer_blends_many_joint_hand_commands_with_the_closure_rule():
    """GR1 hands: events from the closure of four finger joints; the hand commands are time-warped and blended."""
    def hand(n, close_at, closed):
        targets = np.stack([pose(0.01 * t) for t in range(n)])
        g = np.zeros((n, 11))
        g[close_at:, :4] = closed
        return (targets[0], targets, g)

    segs = [hand(10, 6, -1.2), hand(14, 9, -1.0)]
    mixer = SkillMixer(coefficients=_rule(2), gripper_events={"type": "closure", "columns": [0, 1, 2, 3]})
    out = mixer.mix(_context(segs, [pose(), pose()], 0, pose()))
    assert out.record["weights"] == [0.5, 0.5] and out.gripper.shape == (len(out.targets), 11)
    closed = out.gripper[:, 0] < -0.5
    k = int(np.argmax(closed))
    assert k == round(0.5 * 6 + 0.5 * 9) and closed[k:].all()          # the event at the mean event step
    assert np.allclose(out.gripper[-1, :4], 0.5 * -1.2 + 0.5 * -1.0)    # the joint commands blended with the weights


# symmetry alignment ----------------------------------------------------------------------------------------------------
HAMMER_SYM = {"hammer": [{"rot_axis": "z", "rot_param": np.pi, "translation": [0, 0, 0]}]}


def _grasp_segment(obj, y, yaw, T=6, close_at=3):
    """A segment that grasps at (0, y, 0.03) in the object frame (gripper pointing down, turned ``yaw``), closing at
    step ``close_at``; world poses for the object pose ``obj``."""
    g = np.eye(4)
    g[:3, :3] = np.diag([1.0, -1.0, -1.0]) @ pose(yaw=yaw)[:3, :3]
    g[:3, 3] = [0.0, y, 0.03]
    targets = np.stack([obj @ g for _ in range(T)])
    targets[:, 2, 3] += np.linspace(0.05, 0.0, T)      # descends onto the grasp
    targets[close_at:, 2, 3] = targets[close_at, 2, 3]
    grip = np.where(np.arange(T) >= close_at, 1.0, -1.0).reshape(-1, 1)
    return (targets[0], targets, grip)


def _hammer_context(segs, objs, primary=0):
    return _context(segs, objs, primary, objs[primary], reference_object="hammer")


def test_skillmix_mixer_aligns_sources_held_from_the_other_side():
    # sources 0 and 1 grasp the hammer at y=-0.06, source 2 from the other side (the hammer turned about its z axis),
    # source 3 the same but with the gripper also turned about its approach axis
    objs = [pose(0.0), pose(0.0, 0.01), pose(0.0, -0.01), pose(0.01)]
    segs = [_grasp_segment(objs[0], -0.06, 0.0), _grasp_segment(objs[1], -0.055, 0.05),
            _grasp_segment(objs[2], 0.06, np.pi), _grasp_segment(objs[3], 0.06, 0.0)]
    rule = _rule(4)                                       # equal scores: equal weights
    out = SkillMixer(coefficients=rule, symmetries=HAMMER_SYM).mix(_hammer_context(segs, objs))
    rec = out.record
    assert rec["mixed"] and all(w == pytest.approx(0.25) for w in rec["weights"])
    assert rec["symmetry"]["1"]["symmetry"] == "identity"
    assert rec["symmetry"]["2"]["symmetry"] == "obj:z180" and rec["symmetry"]["2"]["pos_cm"] < 0.1
    assert rec["symmetry"]["3"]["symmetry"] == "obj:z180+eef:z180" and rec["symmetry"]["3"]["rot_deg"] < 0.1
    assert rec["symmetry"]["2"]["identity_pos_cm"] == pytest.approx(12.0, abs=0.01)
    assert rec["symmetry"]["2"]["identity_rot_deg"] == pytest.approx(180.0, abs=0.1)
    grasp = segs[0][1][3]                                 # the primary's grasp (object 0 is the primary's frame)
    k = int(np.argmax(np.asarray(out.gripper).reshape(-1) > 0))
    dp = np.linalg.norm(out.targets[k][:3, 3] - grasp[:3, 3])
    dr = np.degrees(np.arccos(np.clip((np.trace(out.targets[k][:3, :3].T @ grasp[:3, :3]) - 1) / 2, -1, 1)))
    assert dp < 0.005 and dr < 2.0
    # without the object symmetry only the gripper flip is available: source 2's position stays on the other side
    rec = SkillMixer(coefficients=rule).mix(_hammer_context(segs, objs)).record
    assert rec["symmetry"]["2"]["symmetry"] == "eef:z180" and rec["symmetry"]["2"]["pos_cm"] == pytest.approx(12.0, abs=0.01)


def test_skillmix_mixer_records_the_distance_after_alignment():
    objs = [pose(0.0), pose(0.0)]
    segs = [_grasp_segment(objs[0], -0.06, 0.0), _grasp_segment(objs[1], -0.14, 0.0)]
    rec = SkillMixer(coefficients=_rule(2), symmetries=HAMMER_SYM).mix(_hammer_context(segs, objs)).record
    assert rec["weights"] == [0.5, 0.5]
    assert rec["symmetry"]["1"]["symmetry"] == "identity" and rec["symmetry"]["1"]["pos_cm"] == pytest.approx(8.0, abs=0.01)


def test_mimicgen_task_symmetries_name_reference_objects():
    declared = {}
    for path in sorted(Path("configs/experiment/generator/mimicgen/task").glob("*.yaml")):
        task = yaml.safe_load(path.read_text())
        for obj, transforms in (task.get("mixing_symmetries") or {}).items():
            assert obj in task["reference_objects"] and all(t["rot_axis"] in "xyz" for t in transforms)
            declared[(path.stem, obj)] = len(transforms)
    assert declared == {("hammer_cleanup", "hammer"): 1, ("square", "square_peg"): 3, ("stack_three", "cubeA"): 3,
                        ("stack_three", "cubeB"): 3, ("stack_three", "cubeC"): 3}
    method = yaml.safe_load(Path("configs/experiment/method/skillmix.yaml").read_text())
    assert method["mixing"]["symmetries"] == "${oc.select:task.mixing_symmetries,null}"


# per-arm random streams ---------------------------------------------------------------------------------------------
def test_single_arm_draw_streams_are_unchanged_by_the_arms():
    """MimicGen and CP-Gen (no named arm) and SoftMimicGen (one arm, robot0) keep SeedSequence([chunk seed, word])."""
    from tmg.generation.method.skillmix import SOURCE_SEED

    ref = np.random.default_rng(np.random.SeedSequence([11, RANDMIX_SEED]))
    c = float(np.exp(ref.uniform(np.log(0.05), np.log(20.0))))
    gamma = ref.gamma(c, size=4)
    for arms in (Arms(), Arms(("robot0",))):
        got = RandmixDraws(11, arms=arms).draw(arms.subtask(0, arms.names[0] if arms.names else None), 4)
        assert got[0] == c and np.array_equal(got[1], gamma)
    u = np.random.default_rng(np.random.SeedSequence([11, SOURCE_SEED])).random()
    assert SourceDraws(11).draw(S(0)) == u == SourceDraws(11, arms=["robot0"]).draw(S(0))


def test_two_arm_draws_come_from_per_arm_streams_independent_of_the_call_order():
    import zlib

    first = RandmixDraws(5, arms=["left", "right"])
    a_right, a_left = first.draw(S(0, "right"), 3)[0], first.draw(S(0, "left"), 3)[0]
    second = RandmixDraws(5, arms=["left", "right"])
    b_left, b_right = second.draw(S(0, "left"), 3)[0], second.draw(S(0, "right"), 3)[0]
    assert (a_right, a_left) == (b_right, b_left) and a_right != a_left
    ref = np.random.default_rng(np.random.SeedSequence([5, RANDMIX_SEED, zlib.crc32(b"right")]))
    assert a_right == float(np.exp(ref.uniform(np.log(0.05), np.log(20.0))))
    with pytest.raises(KeyError, match="stream"):
        first.draw(S(1), 3)                         # a two-arm generator must name the arm
    src = SourceDraws(5, arms=["left", "right"])
    assert src.draw(S(0, "left")) != src.draw(S(0, "right"))


def test_the_mixer_is_built_with_the_generators_arms():
    """The arms come from the generator's layout when the method is built, not from a later call on the mixer."""
    method = build_method({"name": "skillmix_randmix", "mixing": {"enabled": True, "type": "skillmix", "rule": "randmix"}},
                          chunk_seed=2, arms=["left", "right"])
    mixer = method.mixer
    assert mixer.arms == Arms(("left", "right")) and mixer.draws._streams.arms.per_arm and not hasattr(mixer, "set_end_effectors")
    assert mixer.describe()["arms"] == ["left", "right"]
    rec = mixer.mix(_context([_segment(0.0)] * 2, [pose()] * 2, 0, pose(), eef="left")).record
    assert rec["eef"] == "left"
    with pytest.raises(KeyError, match="arms"):
        mixer.mix(_context([_segment(0.0)] * 2, [pose()] * 2, 0, pose()))        # a call that names no arm
    one = build_method({"name": "skillmix_randmix", "mixing": {"enabled": True, "type": "skillmix", "rule": "randmix"}}, chunk_seed=2)
    assert one.mixer.arms == Arms() and not one.mixer.draws._streams.arms.per_arm


def test_a_two_armed_generator_refuses_a_rule_file_without_per_arm_blocks():
    mixer = SkillMixer(coefficients=_rule(2), arms=Arms(("left", "right")))
    with pytest.raises(KeyError, match="more than one end effector"):
        mixer.mix(_context([_segment(0.0)] * 2, [pose()] * 2, 0, pose(), eef="right"))
    with pytest.raises(ValueError, match="mixes per-arm keys"):
        SkillMixRule.from_mapping({"rule": "softmax", "num_sources": 1, "subtasks": {
            0: {"a": [0.0], "g": dict.fromkeys(("dx", "dy", "rotation", "inhand_position", "inhand_rotation"), 0.0), "d": 1, "c0": 0},
            "right/1": {"a": [0.0], "g": dict.fromkeys(("dx", "dy", "rotation", "inhand_position", "inhand_rotation"), 0.0), "d": 1, "c0": 0}}})


def test_the_mixer_checks_the_deformable_frame_against_the_rule_file(tmp_path):
    from tmg.generation.fit_coefficients import write_rule

    g = dict.fromkeys(("dx", "dy", "rotation", "inhand_position", "inhand_rotation"), 0.0)
    frame = {"version": "nodal_yaw_v1", "reference_sha256": "abc"}
    path = write_rule(tmp_path / "rope.yaml", generator="softmimicgen", task="rope", basis="t", num_sources=2,
                      blocks={S(0): {"a": [0.0, 0.0], "g": g, "d": 1.0, "c0": 0.0}}, object_frame=frame)
    rule = load_rule(str(path))
    assert rule.object_frame == frame
    segs, world = [_segment(0.0)] * 2, [np.eye(4)] * 2

    def mix(rule, frame):
        mixer = SkillMixer(coefficients=rule, arms=Arms(("robot0",)))
        return mixer.mix(_context(segs, world, 0, pose(), features=[pose(), pose(0.1)], eef="robot0", object_frame=frame))

    assert "weights" in mix(rule, frame).record
    with pytest.raises(ValueError, match="object frame"):
        mix(rule, {"version": "nodal_yaw_v1", "reference_sha256": "other"})
    with pytest.raises(ValueError, match="object frame"):
        mix(_rule(2), frame)                                          # a rule fitted without frames
    assert "weights" in mix(_rule(2), None).record                    # a rigid reference object: nothing to check
