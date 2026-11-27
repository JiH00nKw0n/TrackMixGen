"""SkillMix: the features, the rows of the fit, the softmax, sparsemax and randmix weights, the rule files, the mixer and the
fit."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from tmg.generation.method.interfaces import MixContext, build_method
from tmg.generation.method.skillmix import (
    MIN_WEIGHT,
    RANDMIX_SEED,
    DistanceCap,
    RandmixDraws,
    SkillMixer,
    SkillMixRule,
    SourceDraws,
    SubtaskRule,
    cap_weights,
    effective_floor,
    effective_sources,
    entropy,
    fitted_weights,
    gini,
    load_rule,
    randmix_weights,
    single_source,
)
from tmg.generation.method.success_model import (
    design_rows,
    in_hand,
    inhand_features,
    scene_features,
    subtask_outcomes,
)


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    p = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    p[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    p[:3, 3] = [x, y, z]
    return p


def flat(p):
    return [*p[:3, 3], *p[:3, :3].reshape(-1)]


def _rule(n, a=None, g=None, d=1.0, subtasks=(0,), rule="softmax"):
    a = np.zeros(n) if a is None else np.asarray(a, dtype=float)
    g = np.zeros(5) if g is None else np.asarray(g, dtype=float)
    return SkillMixRule(name="g/t", rule=rule, num_sources=n,
                        subtasks={k: SubtaskRule(a=a, g=g, d=float(d), c0=0.0) for k in subtasks})


def softmax_weights(scores, d, compatible):
    return fitted_weights("softmax", scores, d, compatible)


# features and rows ----------------------------------------------------------------------------------------------------
def test_scene_and_inhand_features():
    f = scene_features(pose(0.1, -0.2, yaw=0.3), pose(0.0, 0.1, yaw=0.1))
    assert np.allclose(f, [0.1, 0.3, 0.2])
    h1 = in_hand(pose(1, 0, 0), pose(1, 0, -0.1))
    h2 = in_hand(pose(0, 0, 0, yaw=0.5), pose(0.02, 0, -0.1, yaw=0.5))
    d = inhand_features(h1, h2)
    assert d[0] == pytest.approx(0.02) and d[1] == pytest.approx(0.0, abs=1e-9)


def test_subtask_outcomes_from_term_signals():
    steps = {"seg_sub": np.array([0, 0, 1, 1]), "seg_of_step": np.array([0, 0, 1, 2, 2, 3, 3]),
             "term__grasp": np.array([0, 0, 1, 1, 0, 0, 0])}
    assert subtask_outcomes({"success": True}, steps, ["grasp", None]) == [True, True]
    late = dict(steps, term__grasp=np.array([0, 0, 0, 0, 0, 1, 1]))      # fires only during the second subtask
    assert subtask_outcomes({"success": False}, late, ["grasp", None]) == [False, False]


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
    X, y, kept = design_rows([a, a], [[True, True], [False, False]], 1, 2, infos)
    assert kept == [0] and X.shape == (1, 2 + 3 + 2) and y.tolist() == [1.0]
    assert np.allclose(X[0, :2], [0, 1]) and X[0, 2] == pytest.approx(0.1) and X[0, 5] == pytest.approx(0.02)


def test_design_rows_label_is_task_success_not_the_subtask_success():
    objects = {"a": pose(z=-0.12), "b": pose(0.5)}
    ok = _attempt(0, 1, pose(0.1), pose(0.5), pose(), pose(0.4), pose(), objects)
    failed = dict(ok, success=False)                       # the grasp (subtask 0) succeeded, the task did not
    X, y, kept = design_rows([ok, failed], [[True, True], [True, False]], 0, 2, None)
    assert X.shape == (2, 2 + 3 + 2) and y.tolist() == [1.0, 0.0] and kept == [0, 1]
    assert np.all(X[:, 5:] == 0)                           # no in-hand features in the first subtask
    assert design_rows([ok], [[False, False]], 1, 2, None)[0].shape == (0, 2 + 3 + 2)      # nobody reached subtask 1


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


def test_sparsemax_weights_are_the_projection_of_the_scores_onto_the_simplex():
    s = np.array([1.0, 0.8, -1.0, 3.0])
    w = fitted_weights("sparsemax", s, 0.5, [0, 1, 2])     # z = s / (2 d) = s: 1.0, 0.8, -1.0 over the compatible sources
    assert np.allclose(w, [0.6, 0.4, 0.0, 0.0])            # threshold 0.4: the weakest source gets exactly 0
    far = fitted_weights("sparsemax", np.array([2.0, 0.0, 0.0]), 0.25, [0, 1, 2])
    assert far.tolist() == [1.0, 0.0, 0.0]                 # a clear best source takes all the weight, no small weights
    assert np.allclose(fitted_weights("sparsemax", s, 1e6, [0, 1, 2, 3]), 0.25)          # large d: uniform
    assert fitted_weights("sparsemax", s, -0.1, [0, 1, 2]).tolist() == [1.0, 0.0, 0.0, 0.0]   # d <= 0: the best alone
    small = fitted_weights("sparsemax", np.array([1.0, 0.0002, 0.0]), 0.5, [0, 1, 2])
    assert 0 < small[1] < 1e-3 and small[2] == 0.0         # no minimum weight: only the projection sets a weight to 0
    # the weights maximize sum w s + d (1 - |w|^2) over the simplex: no other point of the simplex does better
    rng = np.random.default_rng(0)
    best = w[:3] @ s[:3] + 0.5 * gini(w[:3])
    assert all(v @ s[:3] + 0.5 * gini(v) <= best + 1e-12 for v in rng.dirichlet(np.ones(3), size=2000))


def test_spreads_are_zero_for_a_single_source_and_largest_when_uniform():
    assert entropy([0.0, 1.0, 0.0]) == 0.0 and gini([0.0, 1.0, 0.0]) == 0.0
    assert entropy([0.5, 0.5]) == pytest.approx(np.log(2)) and gini([0.5, 0.5]) == pytest.approx(0.5)


def test_single_source_is_the_best_or_drawn_from_the_softmax_of_the_scores():
    s = np.array([0.3, 0.0, -0.2, 5.0])
    assert single_source(s, [0, 1, 2]) == 0                          # best: largest score among the compatible ones
    rng = np.random.default_rng(0)
    draws = np.array([single_source(s, [0, 1, 2], u) for u in rng.random(20000)])
    p = np.exp(s[:3]) / np.exp(s[:3]).sum()
    assert np.allclose([np.mean(draws == j) for j in range(3)], p, atol=0.015)     # sample: softmax(s), T = 1
    assert not np.any(draws == 3)                                    # an incompatible source is never drawn
    w = fitted_weights("softmax", s, -0.5, [0, 1, 2], u=0.999)
    assert w.tolist() == [0.0, 0.0, 1.0, 0.0]                         # the top of the cumulative distribution
    assert fitted_weights("sparsemax", s, -0.5, [0, 1, 2], u=0.0).tolist() == [1.0, 0.0, 0.0, 0.0]


def test_source_draws_are_reproducible_and_kept_within_an_attempt():
    a, b = SourceDraws(30001), SourceDraws(30001)
    first = a.draw(0)
    assert a.draw(0) == first and b.draw(0) == first                  # one draw per subtask; same chunk seed, same draws
    a.begin_attempt()
    assert a.draw(0) != first


def test_skillmix_sample_draws_one_source_only_where_the_fit_gives_no_spread():
    segs = [_segment(0.0), _segment(0.0), _segment(0.0)]
    objs = [pose(0.0), pose(0.0), pose(0.0)]
    rule = SkillMixRule(name="g/t", rule="softmax", num_sources=3, subtasks={
        0: SubtaskRule(a=np.array([0.2, 0.1, 0.0]), g=np.zeros(5), d=-0.3, c0=0.0),
        1: SubtaskRule(a=np.array([0.2, 0.1, 0.0]), g=np.zeros(5), d=0.5, c0=0.0)})
    picked = set()
    for seed in range(40):
        m = SkillMixer(coefficients=rule, chunk_seed=seed, single_source="sample")
        m.begin_attempt()
        w, rec = m.weights(0, np.zeros((3, 3)), None, [0, 1, 2])
        assert sorted(w.tolist()) == [0.0, 0.0, 1.0] and "source_draw" in rec
        picked.add(int(np.argmax(w)))
        w1, rec1 = m.weights(1, np.zeros((3, 3)), None, [0, 1, 2])
        assert (w1 > 0).sum() == 3 and "source_draw" not in rec1      # d > 0: blended as before
    assert picked == {0, 1, 2}
    best = SkillMixer(coefficients=rule, chunk_seed=0)
    assert best.weights(0, np.zeros((3, 3)), None, [0, 1, 2])[0].tolist() == [1.0, 0.0, 0.0]
    with pytest.raises(ValueError, match="single_source"):
        SkillMixer(coefficients=rule, single_source="random")


def test_rule_scores_add_the_feature_slopes_to_the_source_effects():
    block = SubtaskRule(a=np.array([0.5, -0.5]), g=np.array([-1.0, -2.0, -0.5, -10.0, -1.0]), d=0.3, c0=-1.0)
    scene = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.0]])
    assert np.allclose(block.scores(scene), [0.5 - 0.1 - 0.4 - 0.15, -0.5])          # no in-hand features: zeros
    assert np.allclose(block.scores(scene, np.array([[0.01, 0.1], [0.0, 0.0]])), [0.5 - 0.65 - 0.1 - 0.1, -0.5])


def test_randmix_draws_are_dirichlet_weights_from_their_own_stream():
    draws = RandmixDraws(chunk_seed=30001)
    first = draws.draw(0, 4)
    assert draws.draw(0, 4) is first                       # one draw per subtask and attempt (CP-Gen may re-solve)
    rng = np.random.default_rng(np.random.SeedSequence([30001, RANDMIX_SEED]))
    c = float(np.exp(rng.uniform(np.log(0.05), np.log(20.0))))
    gamma = rng.gamma(c, size=4)
    fallback = int(rng.integers(4))
    assert first[0] == c and np.array_equal(first[1], gamma) and first[2] == fallback
    w = randmix_weights(first, [0, 1, 3])
    expected = np.r_[gamma[[0, 1]], 0.0, gamma[3]] / gamma[[0, 1, 3]].sum()
    expected[expected < MIN_WEIGHT] = 0.0
    assert np.allclose(w, expected / expected.sum()) and w[2] == 0.0
    second = draws.draw(1, 4)
    draws.begin_attempt()
    assert draws.draw(0, 4)[0] != first[0] and second[0] != first[0]     # the stream moves on; a new attempt draws anew
    assert randmix_weights((0.1, np.zeros(4), 5), [1, 2]).tolist() == [0.0, 0.0, 1.0, 0.0]   # all gammas 0: fallback


# rule files -----------------------------------------------------------------------------------------------------------
COMMITTED = {"mimicgen": {"coffee": 2, "square": 2, "threading": 2, "stack_three": 4, "hammer_cleanup": 3, "mug_cleanup": 3,
                          "three_piece_assembly": 4},
             "cpgen": {"coffee": 3, "threading": 2, "hammer_cleanup": 4, "mug_cleanup": 4, "three_piece_assembly": 4}}


COMMITTED_RULES = {"mimicgen": ("softmax", "sparsemax"), "cpgen": ("softmax",)}


def test_committed_rule_files_load_and_give_weights():
    for generator, tasks in COMMITTED.items():
        for name in COMMITTED_RULES[generator]:
            for task, n in tasks.items():
                rule = load_rule(f"{generator}/{task}", name)
                assert rule.rule == name and rule.num_sources == 10 and sorted(rule.subtasks) == list(range(n))
                for k in range(n):
                    block = rule.for_subtask(k)
                    assert block.a.shape == (10,) and block.g.shape == (5,) and np.isfinite(block.d)
                    if k == 0:
                        assert np.allclose(block.g[3:], 0.0)          # no in-hand terms in the first subtask
                    scores = block.scores(np.full((10, 3), 0.05), np.full((10, 2), 0.01))
                    w = fitted_weights(name, scores, block.d, list(range(10)))
                    assert w.sum() == pytest.approx(1.0) and (w >= 0).all()
    # the MimicGen fits favour spreading the weight (d > 0) in every subtask, under both rules
    assert all(b.d > 0 for name in COMMITTED_RULES["mimicgen"] for t in COMMITTED["mimicgen"]
               for b in load_rule(f"mimicgen/{t}", name).subtasks.values())


def test_a_coefficient_file_of_the_removed_above_mean_rule_is_refused(tmp_path):
    old = tmp_path / "old.yaml"
    old.write_text(yaml.safe_dump({"generator": "g", "task": "t", "num_sources": 2, "subtasks": {
        0: {"alpha": {0: 0.1, 1: -0.2}, "slopes": {"dx": -1.0, "dy": -1.0, "rotation": -0.5}}}}))
    with pytest.raises(ValueError, match="not a SkillMix rule"):
        load_rule(str(old), "softmax")
    with pytest.raises(FileNotFoundError, match="make coefficients"):
        load_rule(str(tmp_path / "missing.yaml"), "softmax")


def test_build_method_reads_the_rule_file_and_keeps_randmix_out_of_the_skillmix_folders(tmp_path):
    from tmg.generation.fit_coefficients import write_rule

    block = {"a": [0.1, -0.2], "g": dict.fromkeys(("dx", "dy", "rotation", "inhand_position", "inhand_rotation"), 0.0),
             "d": 0.4, "c0": -1.0}
    path = write_rule(tmp_path / "rule.yaml", rule="softmax", generator="g", task="t", basis="test", num_sources=2,
                      blocks={0: block})
    sparse = write_rule(tmp_path / "sparse.yaml", rule="sparsemax", generator="g", task="t", basis="test", num_sources=2,
                        blocks={0: block})
    m = build_method({"name": "skillmix", "mixing": {"enabled": True, "type": "skillmix", "rule": "softmax",
                                                     "coefficients": str(path)}}, chunk_seed=0)
    assert isinstance(m.mixer, SkillMixer) and m.mixer.rule == "softmax" and m.mixer.coefficients.for_subtask(0).d == 0.4
    with pytest.raises(ValueError, match="skillmix_randmix"):
        build_method({"name": "skillmix", "mixing": {"enabled": True, "type": "skillmix", "rule": "randmix"}}, chunk_seed=0)
    with pytest.raises(ValueError, match="skillmix_randmix"):          # and no softmax pool in the randmix folder
        build_method({"name": "skillmix_randmix", "mixing": {"enabled": True, "type": "skillmix", "rule": "softmax",
                                                                 "coefficients": str(path)}}, chunk_seed=0)
    m = build_method({"name": "skillmix_randmix", "mixing": {"enabled": True, "type": "skillmix", "rule": "randmix",
                                                                 "coefficients": str(tmp_path / "missing.yaml")}}, chunk_seed=0)
    assert m.mixer.rule == "randmix" and m.mixer.coefficients is None          # randmix needs no rule file
    m = build_method({"name": "skillmix_sparsemax", "mixing": {"enabled": True, "type": "skillmix", "rule": "sparsemax",
                                                               "coefficients": str(sparse)}}, chunk_seed=0)
    assert m.mixer.rule == "sparsemax" and m.mixer.coefficients.rule == "sparsemax"
    with pytest.raises(ValueError, match="skillmix_sparsemax"):        # every rule has its own folder
        build_method({"name": "skillmix", "mixing": {"enabled": True, "type": "skillmix", "rule": "sparsemax",
                                                     "coefficients": str(sparse)}}, chunk_seed=0)
    m = build_method({"name": "skillmix_refit", "mixing": {"enabled": True, "type": "skillmix", "rule": "softmax",
                                                           "coefficients": str(path)}}, chunk_seed=0)
    assert m.mixer.rule == "softmax"                                   # a variant of a fitted rule under its own name
    with pytest.raises(ValueError, match="skillmix_randmix"):          # but random weights only in the randmix folder
        build_method({"name": "skillmix_refit", "mixing": {"enabled": True, "type": "skillmix", "rule": "randmix"}},
                     chunk_seed=0)
    with pytest.raises(ValueError, match="holds the softmax rule"):    # the file must hold the configured rule
        build_method({"name": "skillmix_sparsemax", "mixing": {"enabled": True, "type": "skillmix", "rule": "sparsemax",
                                                               "coefficients": str(path)}}, chunk_seed=0)
    with pytest.raises(ValueError, match="mixing.rule"):
        SkillMixer(rule="above_mean")
    with pytest.raises(ValueError, match="rule file"):
        SkillMixer(rule="softmax")


# the mixer ------------------------------------------------------------------------------------------------------------
def _segment(x, T=6):
    targets = np.stack([pose(x + 0.01 * t) for t in range(T)])
    return (pose(x), targets, np.full((T, 1), -1.0))


def _context(segs, objs, primary, current, subtask=0, extra=None):
    return MixContext(subtask_ind=subtask, primary=primary, primary_eef_start=segs[primary][0], primary_targets=segs[primary][1],
                      primary_gripper=segs[primary][2], src_object_poses=objs, src_segments=segs, num_sources=len(segs),
                      has_reference_object=True, extra={"current_object_pose": current, **(extra or {})})


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
                     has_reference_object=False, extra={})
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


# the fit --------------------------------------------------------------------------------------------------------------
def test_fit_recovers_the_sign_of_the_spread_effect():
    from tmg.generation.fit_coefficients import fit_subtask

    rng = np.random.default_rng(0)
    J, n = 3, 6000
    a = np.array([0.5, 0.0, -0.5])
    for rule, spread in (("softmax", entropy), ("sparsemax", gini)):
        for d_true in (1.5, -1.5):
            W = rng.dirichlet(np.full(J, 0.7), size=n)
            S = np.array([spread(w) for w in W])
            y = (rng.random(n) < 1 / (1 + np.exp(-(-0.5 + W @ a + d_true * S)))).astype(float)
            rows = [np.concatenate([w, np.zeros(5)]) for w in W]       # the fit adds the rule's spread column
            groups = list(np.arange(n) // 50)
            block = fit_subtask((rows, list(y), groups), ([], [], []), J, rule, with_cv=False)
            assert np.sign(block["d"]) == np.sign(d_true) and abs(block["d"] - d_true) < 0.6, (rule, d_true, block["d"])
            assert block["a"][0] > block["a"][2]


def _write_pool(folder, chunks, attempts):
    """A finished pool with ``attempts(chunk)`` as the attempt records of every chunk."""
    import json

    for c in range(chunks):
        records = folder / "chunks" / f"seed{c:05d}" / "records"
        records.mkdir(parents=True)
        (records.parent / "CHUNK_DONE.json").write_text("{}")
        (records / "attempts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in attempts(c)))


def test_fit_reads_the_pools_of_a_generator_that_stops_at_the_first_failed_subtask(tmp_path):
    """CP-Gen records: a unmix attempt reached subtask k when it has k + 1 subtask records; a randmix attempt when it
    has a mix record of k, also one whose motion plan to k failed (``unplanned_mixes``)."""
    from tmg.generation.fit_coefficients import fit_blocks, read_pools

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

    _write_pool(tmp_path / "randmix", 4, randmix)
    _write_pool(tmp_path / "unmix", 3, unmix)
    fitted = fit_blocks(read_pools([tmp_path / "randmix"]), read_pools([tmp_path / "unmix"]), num_sources=2,
                        signals=None, source=None, with_cv=False)
    assert sorted(fitted) == ["softmax", "sparsemax"]
    assert fitted["softmax"][0]["rows"] == fitted["sparsemax"][0]["rows"]      # the same rows, two spreads
    assert fitted["softmax"][0]["d"] != fitted["sparsemax"][0]["d"]
    only = fit_blocks(read_pools([tmp_path / "randmix"]), read_pools([tmp_path / "unmix"]), num_sources=2,
                      signals=None, source=None, rules=["sparsemax"], with_cv=False)
    assert only == {"sparsemax": fitted["sparsemax"]}                          # one rule alone: the same fit
    blocks = fitted["softmax"]
    assert sorted(blocks) == [0, 1] and blocks[0]["rows"] == {"randmix": 84, "unmix": 60}
    assert blocks[1]["rows"] == {"randmix": 84, "unmix": 48}       # the failed plans count, the 12 that stopped do not
    assert len(blocks[1]["a"]) == 2 and sorted(blocks[1]["g"]) == sorted(["dx", "dy", "rotation", "inhand_position", "inhand_rotation"])


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
    return MixContext(subtask_ind=0, primary=primary, primary_eef_start=segs[primary][0], primary_targets=segs[primary][1],
                      primary_gripper=segs[primary][2], src_object_poses=objs, src_segments=segs, num_sources=len(segs),
                      has_reference_object=True, extra={"current_object_pose": objs[primary], "reference_object": "hammer"})


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


# options of the factor experiment: effective-source floor, distance cap, derived rule files ---------------------------
def test_effective_floor_raises_the_temperature_only_where_the_weights_are_too_concentrated():
    scores, everyone = np.array([2.0, 1.0, 0.0, -1.0, -3.0]), [0, 1, 2, 3, 4]
    assert effective_sources([1.0, 0, 0]) == 1.0 and effective_sources([0.25] * 4) == pytest.approx(4.0)
    # the fitted weights already have 3 effective sources: the floor does not bind and they stand
    assert effective_sources(fitted_weights("softmax", scores, 5.0, everyone)) > 3
    assert effective_floor("softmax", scores, 5.0, everyone, 3) is None
    # too concentrated: the smallest temperature above the fitted one that reaches the floor
    assert effective_sources(fitted_weights("softmax", scores, 0.3, everyone)) < 1.2
    w, temperature = effective_floor("softmax", scores, 0.3, everyone, 3)
    assert temperature > 0.3 and effective_sources(w) == pytest.approx(3.0, abs=1e-3)
    assert np.allclose(w, fitted_weights("softmax", scores, temperature, everyone)) and list(np.argsort(-w)) == everyone
    # d <= 0 (one source by the fit) is blended too, over the compatible sources only
    w, temperature = effective_floor("softmax", scores, -0.5, [0, 2, 3, 4], 2)
    assert temperature > 0 and w[1] == 0 and effective_sources(w) == pytest.approx(2.0, abs=1e-3)
    # fewer compatible sources than the floor: they share the weight equally; one source needs no blend
    w, temperature = effective_floor("softmax", scores, 0.3, [0, 3], 3)
    assert temperature is None and w.tolist() == [0.5, 0, 0, 0.5, 0]
    assert effective_floor("softmax", scores, -0.5, [2], 3) is None and effective_floor("softmax", scores, 0.3, everyone, 1) is None
    w, temperature = effective_floor("sparsemax", scores, 0.3, everyone, 3)          # the sparsemax rule in the same way
    assert temperature > 0.3 and effective_sources(w) == pytest.approx(3.0, abs=1e-3)


def test_min_effective_sources_changes_the_mixer_weights_and_is_off_by_default():
    rule = _rule(4, a=[2.0, 1.0, 0.0, -1.0], d=0.2, subtasks=(0, 1))
    rule.subtasks[1] = SubtaskRule(a=rule.subtasks[0].a, g=np.zeros(5), d=-1.0, c0=0.0)
    scene, series = np.zeros((4, 3)), [np.full(4, -1.0)] * 4
    plain, rec = SkillMixer(coefficients=rule).decide(0, 0, scene, None, series)
    assert effective_sources(plain) < 1.1 and "temperature" not in rec and "min_effective_sources" not in rec
    mixer = SkillMixer(coefficients=rule, min_effective_sources=3)
    for k in (0, 1):                                      # the blended subtask and the one the fit gives one source
        w, rec = mixer.decide(k, 0, scene, None, series)
        assert effective_sources(w) == pytest.approx(3.0, abs=1e-3) and rec["min_effective_sources"] == 3
        assert rec["temperature"] > 0.2 and rec["d"] == (0.2, -1.0)[k]
    w, rec = SkillMixer(coefficients=_rule(4, d=1.0), min_effective_sources=3).decide(0, 0, scene, None, series)
    assert w.tolist() == [0.25] * 4 and "temperature" not in rec          # equal scores: 4 effective sources already
    assert mixer.describe()["min_effective_sources"] == 3 and SkillMixer(coefficients=rule).describe()["min_effective_sources"] is None
    with pytest.raises(ValueError, match="min_effective_sources"):
        SkillMixer(rule="randmix", min_effective_sources=3)


def test_cap_weights_drops_the_sources_beyond_the_cap_and_renormalizes():
    distances = {"1": {"pos_cm": 1.0, "rot_deg": 10.0}, "2": {"pos_cm": 1.0, "rot_deg": 31.0}, "3": {"pos_cm": 3.5, "rot_deg": 2.0}}
    cap = DistanceCap.from_config({"rot_deg": 30, "pos_cm": 3})
    w, dropped = cap_weights([0.4, 0.2, 0.2, 0.2], distances, cap)
    assert dropped == [2, 3] and w.tolist() == pytest.approx([2 / 3, 1 / 3, 0, 0])      # the selected source (0) has no entry
    w, dropped = cap_weights([0.4, 0.2, 0.2, 0.2], distances, DistanceCap.from_config({"rot_deg": 30, "pos_cm": None}))
    assert dropped == [2] and w.tolist() == pytest.approx([0.5, 0.25, 0, 0.25])          # no limit on the position
    w, dropped = cap_weights([0.4, 0.2, 0.2, 0.2], distances, DistanceCap.from_config({"rot_deg": 31, "pos_cm": 3.5}))
    assert dropped == [] and w.tolist() == pytest.approx([0.4, 0.2, 0.2, 0.2])           # at the limit is within it
    # the selected source has no weight and every retained source is beyond the cap: the largest weight stays alone
    w, dropped = cap_weights([0.0, 0.0, 0.3, 0.7], distances, cap)
    assert dropped == [2] and w.tolist() == [0, 0, 0, 1.0]
    assert DistanceCap.from_config(None) is None and DistanceCap.from_config({"rot_deg": None, "pos_cm": None}) is None
    with pytest.raises(ValueError, match="unknown keys"):
        DistanceCap.from_config({"rot": 30})


def _cap_case():
    """Four sources that grasp the hammer 0, 2, 8 and 2 cm (the last turned 40 degrees) from the selected source's grasp."""
    objs = [pose(0.0)] * 4
    segs = [_grasp_segment(objs[0], -0.06, 0.0), _grasp_segment(objs[1], -0.08, 0.0), _grasp_segment(objs[2], -0.14, 0.0),
            _grasp_segment(objs[3], -0.08, np.radians(40.0))]
    return segs, objs


def test_skillmix_mixer_leaves_out_the_sources_beyond_the_distance_cap():
    segs, objs = _cap_case()
    rule = _rule(4)                                       # equal scores: equal weights
    plain = SkillMixer(coefficients=rule).mix(_hammer_context(segs, objs))
    assert plain.record["weights"] == [0.25] * 4 and "distance_cap" not in plain.record
    capped = SkillMixer(coefficients=rule, max_source_distance={"rot_deg": 30, "pos_cm": 3}).mix(_hammer_context(segs, objs))
    rec = capped.record
    assert rec["symmetry"]["2"]["pos_cm"] == pytest.approx(8.0, abs=0.01) and rec["symmetry"]["3"]["rot_deg"] == pytest.approx(40.0, abs=0.1)
    assert rec["weights"] == [0.5, 0.5, 0.0, 0.0] and rec["mixed"]
    assert rec["distance_cap"] == {"rot_deg": 30.0, "pos_cm": 3.0, "dropped": [2, 3], "n_dropped": 2, "weights_before": [0.25] * 4}
    # the blend is the blend of the two sources within the cap: the same poses as a mixer given only those two
    two = SkillMixer(coefficients=_rule(2)).mix(_hammer_context(segs[:2], objs[:2]))
    assert np.allclose(capped.targets, two.targets) and np.allclose(capped.gripper, two.gripper)
    assert not np.allclose(capped.targets, plain.targets)
    # a cap that drops nothing leaves the blend exactly as it is without a cap
    loose = SkillMixer(coefficients=rule, max_source_distance={"rot_deg": 90, "pos_cm": 50}).mix(_hammer_context(segs, objs))
    assert loose.record["distance_cap"]["n_dropped"] == 0 and loose.record["weights"] == plain.record["weights"]
    assert np.array_equal(loose.targets, plain.targets) and np.array_equal(loose.gripper, plain.gripper)
    # every other source beyond the cap: the selected source alone, which is no mixing
    alone = SkillMixer(coefficients=rule, max_source_distance={"pos_cm": 1}).mix(_hammer_context(segs, objs))
    assert alone.record["weights"] == [1.0, 0.0, 0.0, 0.0] and not alone.record["mixed"]


def test_distance_cap_measures_sources_without_any_symmetry_candidate_too():
    # many-joint hand commands have no gripper flip, so without a cap no alignment record is written; the cap needs the
    # distances and gets them from the identity candidate
    def hand(n, close_at, closed):
        g = np.zeros((n, 3))
        g[close_at:] = closed
        return g

    segs = [(pose(0.02 + dx), np.stack([pose(dx + 0.01 * t) for t in range(6)]), hand(6, 3, [0.5, 0.6, 0.4])) for dx in (0.0, 0.08)]
    ctx = _context(segs, [pose(0.0), pose(0.0)], 0, pose(0.0))
    events_rule = {"type": "closure", "columns": [0, 1, 2]}
    assert "symmetry" not in SkillMixer(coefficients=_rule(2), gripper_events=events_rule).mix(ctx).record
    rec = SkillMixer(coefficients=_rule(2), gripper_events=events_rule, max_source_distance={"pos_cm": 3}).mix(ctx).record
    assert rec["symmetry"]["1"] == {"symmetry": "identity", "pos_cm": 8.0, "rot_deg": 0.0, "identity_pos_cm": 8.0, "identity_rot_deg": 0.0}
    assert rec["distance_cap"]["dropped"] == [1] and rec["weights"] == [1.0, 0.0] and not rec["mixed"]


def test_build_method_reads_a_derived_rule_file_and_the_options():
    base = {"enabled": True, "type": "skillmix", "rule": "softmax", "coefficients": "mimicgen/threading"}
    fitted = build_method({"name": "skillmix", "mixing": base}, chunk_seed=0).mixer
    assert fitted.max_distance is None and fitted.min_effective is None                    # both off unless set
    assert all(fitted.coefficients.for_subtask(k).d > 0 for k in (0, 1))
    derived = build_method({"name": "skillmix_select", "mixing": {**base, "derived": "select_only"}}, chunk_seed=0).mixer
    assert derived.coefficients.path.endswith("coefficients/derived/select_only/threading.yaml")
    assert [derived.coefficients.for_subtask(k).d for k in (0, 1)] == [-1.0, -1.0]
    assert np.array_equal(derived.coefficients.for_subtask(0).a, fitted.coefficients.for_subtask(0).a)
    with pytest.raises(FileNotFoundError, match="derive_coefficients.py"):
        build_method({"name": "skillmix_x", "mixing": {**base, "derived": "no_such_condition"}}, chunk_seed=0)
    options = build_method({"name": "skillmix_y", "mixing": {**base, "max_source_distance": {"rot_deg": 30, "pos_cm": 3},
                                                             "min_effective_sources": 3}}, chunk_seed=0).mixer
    assert options.max_distance == DistanceCap(rot_deg=30.0, pos_cm=3.0) and options.min_effective == 3.0
