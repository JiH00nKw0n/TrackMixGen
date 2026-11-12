"""Logistic source mixing: the success model and its coefficient file, the above-mean weights, the mixer, and
rank-based sampling."""
import numpy as np
import pytest
import yaml

from tmg.generation.method.interfaces import MixContext, build_method
from tmg.generation.method.logistic_mix import LogisticMixer, above_mean_weights
from tmg.generation.method.rank_sampling import RankBasedSampling, rank_based_sampling
from tmg.generation.method.ranking import RankedSources, SourceRanking
from tmg.generation.method.success_model import (
    SubtaskModel, SuccessModel, design_rows, fit_logistic, fit_subtask, in_hand, inhand_features, scene_features, sigmoid,
    subtask_outcomes)


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    p = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    p[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    p[:3, 3] = [x, y, z]
    return p


def flat(p):
    return [*p[:3, 3], *p[:3, :3].reshape(-1)]


def test_scene_and_inhand_features():
    f = scene_features(pose(0.1, -0.2, yaw=0.3), pose(0.0, 0.1, yaw=0.1))
    assert np.allclose(f, [0.1, 0.3, 0.2])
    h1 = in_hand(pose(1, 0, 0), pose(1, 0, -0.1))
    h2 = in_hand(pose(0, 0, 0, yaw=0.5), pose(0.02, 0, -0.1, yaw=0.5))
    d = inhand_features(h1, h2)
    assert d[0] == pytest.approx(0.02) and d[1] == pytest.approx(0.0, abs=1e-9)


def test_fit_logistic_recovers_coefficients():
    rng = np.random.default_rng(0)
    n, J = 20000, 3
    src = rng.integers(J, size=n)
    feats = rng.uniform(0, 0.3, size=(n, 3))
    alpha = np.array([1.0, 0.0, -1.0])
    slopes = np.array([-4.0, -2.0, -1.0])
    y = (rng.random(n) < sigmoid(alpha[src] + feats @ slopes)).astype(float)
    X = np.column_stack([np.eye(J)[src], feats])
    beta = fit_logistic(X, y)
    assert np.allclose(beta[:J], alpha, atol=0.15)
    assert np.allclose(beta[J:], slopes, atol=0.6)


def test_fit_logistic_stays_finite_when_a_source_never_succeeds():
    X = np.column_stack([np.eye(2)[[0, 0, 1, 1]], [0.1, 0.2, 0.1, 0.2]])
    beta = fit_logistic(X, np.array([1.0, 0.0, 0.0, 0.0]))
    assert np.all(np.isfinite(beta)) and beta[1] < -3


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
    X, y, carried = design_rows([a, a], [[True, True], [False, False]], 1, 2, infos)
    assert carried == "a" and X.shape == (1, 2 + 3 + 2) and y.tolist() == [1.0]
    assert np.allclose(X[0, :2], [0, 1]) and X[0, 2] == pytest.approx(0.1) and X[0, 5] == pytest.approx(0.02)
    block = fit_subtask(np.vstack([X, X]), np.array([1.0, 0.0]), 2, 1)
    assert set(block["slopes"]) == {"dx", "dy", "rotation", "inhand_position", "inhand_rotation"}


def test_above_mean_weights():
    w = above_mean_weights(np.array([0.9, 0.1, 0.5, 0.3]), [0, 1, 2, 3])
    assert np.allclose(w, [0.9 / 1.4, 0, 0.5 / 1.4, 0])
    w = above_mean_weights(np.array([0.9, 0.1, 0.5, 0.3]), [0, 1, 3])       # 2 incompatible
    assert np.allclose(w, [1, 0, 0, 0])


def _model(n=3, inhand=False):
    slopes = {"dx": -10.0, "dy": -10.0, "rotation": 0.0}
    if inhand:
        slopes.update(inhand_position=-5.0, inhand_rotation=0.0)
    return SuccessModel(name="g/t", num_sources=n, subtasks={0: SubtaskModel(np.zeros(n), slopes)}, carried={})


def _segment(x, T=6):
    targets = np.stack([pose(x + 0.01 * t) for t in range(T)])
    return (pose(x), targets, np.full((T, 1), -1.0))


def test_logistic_mixer_blends_sources_at_or_above_the_mean():
    n = 3
    segs = [_segment(0.0), _segment(0.0), _segment(0.0)]
    ctx = MixContext(subtask_ind=0, primary=2, primary_eef_start=segs[2][0], primary_targets=segs[2][1],
                     primary_gripper=segs[2][2], src_object_poses=[pose(0.0), pose(0.05), pose(0.3)], src_segments=segs,
                     num_sources=n, has_reference_object=True, extra={"current_object_pose": pose(0.0)})
    out = LogisticMixer(model=_model(n)).mix(ctx)
    rec = out.record
    assert rec["weights"][2] == 0 and rec["weights"][0] > rec["weights"][1] > 0
    assert sum(rec["weights"]) == pytest.approx(1.0)
    assert out.targets.shape[1:] == (4, 4)


def test_logistic_mixer_identity_without_reference_object():
    segs = [_segment(0.0)] * 2
    ctx = MixContext(subtask_ind=0, primary=0, primary_eef_start=segs[0][0], primary_targets=segs[0][1],
                     primary_gripper=segs[0][2], src_object_poses=[None, None], src_segments=segs, num_sources=2,
                     has_reference_object=False, extra={})
    rec = LogisticMixer(model=_model(2)).mix(ctx).record
    assert rec["mixed"] is False and "identity_reason" in rec


def _ranking():
    return SourceRanking(name="g/t", generator="g", task="t", basis="", default=None,
                         subtasks={0: RankedSources(order=(4, 2, 0, 1, 3)), 1: RankedSources(order=(1, 0, 2, 3, 4))}, path="")


def test_rank_based_sampling_draws_from_the_top_k():
    s = RankBasedSampling(ranking=_ranking(), top_k=3, chunk_seed=7)
    picks0 = {s.pick(0) for _ in range(200)}
    picks1 = {s.pick(1) for _ in range(200)}
    assert picks0 == {4, 2, 0} and picks1 == {1, 0, 2}
    assert RankBasedSampling(ranking=_ranking(), top_k=1, chunk_seed=0).pick(0) == 4
    a = [RankBasedSampling(ranking=_ranking(), top_k=5, chunk_seed=3).pick(0) for _ in range(1)]
    b = [RankBasedSampling(ranking=_ranking(), top_k=5, chunk_seed=3).pick(0) for _ in range(1)]
    assert a == b


def test_build_method_reads_coefficient_and_ranking_files(tmp_path, monkeypatch):
    coef = tmp_path / "coef.yaml"
    coef.write_text(yaml.safe_dump({"generator": "g", "task": "t", "num_sources": 2, "subtasks": {
        0: {"alpha": {0: 0.1, 1: -0.2}, "slopes": {"dx": -1.0, "dy": -1.0, "rotation": -0.5}},
        1: {"alpha": {0: 0.0, 1: 0.0}, "slopes": {"dx": 0.0, "dy": 0.0, "rotation": 0.0, "inhand_position": -3.0,
                                               "inhand_rotation": -1.0}, "carried_object": "a"}}}))
    m = build_method({"name": "x", "mixing": {"enabled": True, "type": "logistic", "coefficients": str(coef)}}, chunk_seed=0)
    assert isinstance(m.mixer, LogisticMixer) and m.mixer.model.carried == {1: "a"}
    with pytest.raises(FileNotFoundError, match="make coefficients"):
        build_method({"mixing": {"enabled": True, "type": "logistic", "coefficients": str(tmp_path / "missing.yaml")}}, chunk_seed=0)
    import tmg.generation.method.rank_sampling as rs
    monkeypatch.setattr(rs, "load_ranking", lambda name: _ranking())
    m = build_method({"name": "rank_top3", "sources": {"type": "rank_based", "top_k": 3, "ranking": "g/t"}}, chunk_seed=0)
    assert isinstance(m.sources, RankBasedSampling) and not m.is_identity
    assert isinstance(rank_based_sampling({"ranking": "g/t", "top_k": 1}, chunk_seed=0), RankBasedSampling)
