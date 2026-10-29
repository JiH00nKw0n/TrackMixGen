"""Mixup exploration variants: the geometric mixers (axes 2 and 3), top-K source selection, and their wiring into
the MimicGen and CP-Gen runtimes. Expected values are computed by hand in the comments."""
from types import SimpleNamespace

import numpy as np
import pytest

from tmg.generation.generators.cpgen import runtime as cpgen_rt
from tmg.generation.generators.mimicgen import runtime as mimicgen_rt
from tmg.generation.method.event_mix import blend_poses
from tmg.generation.method.geometric_mix import (
    GeometricMixConfig,
    JunctionMixer,
    TransformMixer,
    pose_distance,
)
from tmg.generation.method.interfaces import Method, MixContext, _mixer_types, build_method
from tmg.generation.method.ranking import RankedSources, SourceRanking, load_ranking
from tmg.generation.method.source_select import TopKSources


def ranking(order=(1, 2, 0)):
    return SourceRanking(name="t/t", generator="t", task="t", basis="", default=None,
                         subtasks={0: RankedSources(order=tuple(order))}, path="")


def pose(x=0.0, y=0.0, z=0.0, yaw=0.0):
    t = np.eye(4)
    c, s = np.cos(yaw), np.sin(yaw)
    t[:2, :2] = [[c, -s], [s, c]]
    t[:3, 3] = (x, y, z)
    return t


def segment(y: float, n: int = 10, frame=None):
    """n target poses moving along x at height y in ``frame`` (world = frame @ object-frame pose), one close event at
    step 5 (commands -1 then +1), start pose at the first target."""
    frame = np.eye(4) if frame is None else frame
    targets = np.stack([frame @ pose(0.01 * t, y) for t in range(n)])
    grip = np.r_[-np.ones(5), np.ones(n - 5)][:, None]
    return targets[0].copy(), targets, grip


def context(primary, object_poses, ys, extra, eef_starts=None):
    segs = [segment(y, frame=o) for y, o in zip(ys, object_poses)]
    if eef_starts is not None:
        segs = [(e, t, g) for e, (_, t, g) in zip(eef_starts, segs)]
    eef, targets, grip = segs[primary]
    return MixContext(subtask_ind=0, primary=primary, primary_eef_start=eef, primary_targets=targets, primary_gripper=grip,
                      src_object_poses=list(object_poses), src_segments=segs, num_sources=len(segs),
                      has_reference_object=True, extra=extra)


def make(cls, **values):
    return cls(ranking=ranking(), chunk_seed=3, config=GeometricMixConfig(max_anchor_rotation_deg=90.0, **values))


# --- axis 2 -----------------------------------------------------------------------------------------------------------
def test_transform_mixer_takes_the_smallest_transform_donor_and_weights_by_distance():
    # object poses at x = 0.30, 0.05, 0.20; current object at the origin -> delta = 0.30, 0.05, 0.20, median 0.20
    objs = [pose(0.30), pose(0.05), pose(0.20)]
    ctx = context(0, objs, ys=[0.0, 0.1, 0.05], extra={"current_object_pose": np.eye(4)})
    out = make(TransformMixer).mix(ctx)
    rec = out.record
    assert rec["mixed"] is True and rec["source_b"] == 1                      # argmin delta among {1, 2}
    assert rec["eligible_donors"] == [1, 2]                                   # every other source, never the primary
    assert rec["transform"]["distances"] == [0.3, 0.05, 0.2] and rec["transform"]["threshold"] == 0.2
    assert rec["transform"]["weight_a"] == 0.2                                 # 0.05 / 0.35 = 0.143, clipped to 0.2
    # object frame: y = 0.2 * 0.0 + 0.8 * 0.1 = 0.08; world = primary object frame (x + 0.30)
    np.testing.assert_allclose(out.targets[0][:3, 3], [0.30, 0.08, 0.0], atol=1e-12)
    np.testing.assert_allclose(out.targets[-1][:3, 3], [0.39, 0.08, 0.0], atol=1e-12)
    np.testing.assert_allclose(out.eef_start[:3, 3], [0.30, 0.08, 0.0], atol=1e-12)


def test_transform_mixer_rotation_counts_with_the_rotation_weight():
    a, b = pose(0.1), pose(0.1, yaw=0.5)
    assert pose_distance(a, b, 0.1) == pytest.approx(0.05)                     # 0 m + 0.1 m/rad * 0.5 rad
    assert pose_distance(pose(0.0), pose(0.3, 0.4), 0.1) == pytest.approx(0.5)  # 3-4-5 triangle


@pytest.mark.parametrize("primary, extra, reason", [
    (1, {"current_object_pose": np.eye(4)}, "primary transform distance not above the trigger quantile"),
    (2, {"current_object_pose": np.eye(4)}, "primary transform distance not above the trigger quantile"),   # = median
    (0, {}, "no current_object_pose in the mix context"),
])
def test_transform_mixer_identity_reasons(primary, extra, reason):
    objs = [pose(0.30), pose(0.05), pose(0.20)]
    ctx = context(primary, objs, ys=[0.0, 0.1, 0.05], extra=extra)
    out = make(TransformMixer).mix(ctx)
    assert out.record["mixed"] is False and out.record["identity_reason"] == reason
    np.testing.assert_array_equal(out.targets, ctx.primary_targets)


def test_transform_mixer_good_donors_only_skips_a_closer_bad_source():
    # ranking order (2, 0, 1) -> good = {2, 0}, so the closest source 1 is bad and donors: good leaves only source 2
    objs = [pose(0.30), pose(0.05), pose(0.20)]
    ctx = context(0, objs, ys=[0.0, 0.1, 0.05], extra={"current_object_pose": np.eye(4)})
    mixer = TransformMixer(ranking=ranking((2, 0, 1)), chunk_seed=3,
                           config=GeometricMixConfig(max_anchor_rotation_deg=90.0, donors="good"))
    rec = mixer.mix(ctx).record
    assert rec["source_b"] == 2 and rec["transform"]["weight_a"] == 0.4        # 0.20 / 0.50


# --- axis 3 -----------------------------------------------------------------------------------------------------------
def test_junction_mixer_blends_only_the_head_towards_the_nearest_start():
    # all object frames at the origin; first eef poses at x = 0.50, 0.02, 0.30; junction at the origin
    objs = [np.eye(4)] * 3
    starts = [pose(0.50), pose(0.02), pose(0.30)]
    ctx = context(0, objs, ys=[0.0, 0.1, 0.05], extra={"current_object_pose": np.eye(4), "junction_pose": np.eye(4)},
                  eef_starts=starts)
    out = make(JunctionMixer, head_steps=4).mix(ctx)
    rec = out.record
    assert rec["source_b"] == 1 and rec["junction"]["distances"] == [0.5, 0.02, 0.3]
    assert rec["junction"]["weight_a"] == 0.2                                  # 0.02 / 0.52 = 0.038, clipped
    assert rec["alignment"]["weights"] == [0.2, 0.4, 0.6, 0.8] + [1.0] * 6     # 1 - 0.8 * max(0, 1 - t/4)
    # y = (1 - w) * 0.1 on the head, the primary's y = 0 from step 4 on
    np.testing.assert_allclose(out.targets[:5, 1, 3], [0.08, 0.06, 0.04, 0.02, 0.0], atol=1e-12)
    np.testing.assert_array_equal(out.targets[4:], ctx.primary_targets[4:])
    np.testing.assert_allclose(out.gripper, ctx.primary_gripper)              # identical commands stay identical
    np.testing.assert_allclose(out.eef_start, blend_poses(starts[0][None], starts[1][None], 0.2)[0], atol=1e-12)


def test_junction_gap_is_measured_after_the_generator_transform():
    # object poses x = 0.1, 0.0, 0.4, current object at the origin: the start E_i moves by -x_i
    # E = 0.60, 0.02, 0.40 -> after the transform 0.50, 0.02, 0.00, so source 2 (not 1) is the nearest start
    objs = [pose(0.1), pose(0.0), pose(0.4)]
    starts = [pose(0.60), pose(0.02), pose(0.40)]
    ctx = context(0, objs, ys=[0.0, 0.1, 0.05], extra={"current_object_pose": np.eye(4), "junction_pose": np.eye(4)},
                  eef_starts=starts)
    rec = make(JunctionMixer, head_steps=4).mix(ctx).record
    assert rec["junction"]["distances"] == [0.5, 0.02, 0.0] and rec["source_b"] == 2


def test_junction_mixer_needs_the_junction_pose():
    objs = [np.eye(4)] * 3
    ctx = context(0, objs, ys=[0.0, 0.1, 0.05], extra={"current_object_pose": np.eye(4)})
    assert make(JunctionMixer).mix(ctx).record["identity_reason"] == "no junction_pose in the mix context"


def test_geometric_config_rejects_bad_values():
    with pytest.raises(ValueError, match="donors"):
        GeometricMixConfig.from_mapping({"donors": "best"})
    with pytest.raises(ValueError, match="weight_min"):
        GeometricMixConfig.from_mapping({"weight_min": 0.9, "weight_max": 0.5})


# --- top-K ------------------------------------------------------------------------------------------------------------
def test_top_k_draws_only_among_the_best_sources_from_its_private_stream():
    rule = TopKSources(ranking=ranking((4, 1, 3, 0, 2)), top_k=2, chunk_seed=7)
    assert rule.allowed(0) == [4, 1]
    picks = [rule.pick(0) for _ in range(20)]
    expected = [[4, 1][int(np.random.default_rng(np.random.SeedSequence([7, k, 424242])).integers(2))] for k in range(20)]
    assert picks == expected and set(picks) == {4, 1}
    with pytest.raises(ValueError, match="top_k"):
        TopKSources(ranking=ranking(), top_k=0, chunk_seed=7)


def test_build_method_registers_the_variants_and_the_source_rule():
    assert set(_mixer_types()) == {"guided", "transform", "junction"}
    m = build_method({"name": "mx_top3", "sources": {"top_k": 3, "ranking": "mimicgen/coffee"}}, chunk_seed=1)
    first_three = list(load_ranking("mimicgen/coffee").for_subtask(0).order[:3])       # the committed ranking file's order
    assert isinstance(m.sources, TopKSources) and m.sources.allowed(0) == first_three and not m.is_identity
    assert build_method({"name": "none"}, chunk_seed=1).sources is None


# --- runtimes ---------------------------------------------------------------------------------------------------------
def test_mimicgen_source_rule_replaces_the_official_pick_only_when_configured():
    rt = mimicgen_rt.MimicGenRuntime(Method(name="none", mixer=None, follower_factory=None), chunk_seed=1, records_dir=None)
    assert rt.select_source(0, 6) == 6
    rule = SimpleNamespace(pick=lambda subtask: 9 if subtask == 1 else 4)
    rt = mimicgen_rt.MimicGenRuntime(Method(name="mx_top1", mixer=None, follower_factory=None, sources=rule),
                                     chunk_seed=1, records_dir=None)
    assert rt.select_source(1, 6) == 9 and rt.select_source(0, 6) == 4


def test_mimicgen_scene_poses_follow_the_official_interpolation_start():
    info = SimpleNamespace(eef_pose=pose(0.1))
    prev = SimpleNamespace(last_waypoint=SimpleNamespace(pose=pose(0.2)))
    base = dict(cur_datagen_info=info, prev_executed_traj=prev, cur_object_pose=pose(0.3))
    later = mimicgen_rt.MimicGenRuntime.scene_poses({**base, "interpolate_from_last_target_pose": True, "is_first_subtask": False})
    first = mimicgen_rt.MimicGenRuntime.scene_poses({**base, "interpolate_from_last_target_pose": True, "is_first_subtask": True})
    no_interp = mimicgen_rt.MimicGenRuntime.scene_poses({**base, "interpolate_from_last_target_pose": False, "is_first_subtask": False})
    assert later["junction_pose"][0, 3] == 0.2 and first["junction_pose"][0, 3] == 0.1 and no_interp["junction_pose"][0, 3] == 0.1
    assert later["current_object_pose"][0, 3] == 0.3
    assert mimicgen_rt.MimicGenRuntime.scene_poses({**base, "cur_object_pose": None})["current_object_pose"] is None


def test_cpgen_source_rule_overrides_the_attempt_pick_and_keeps_the_stream_aligned():
    policy = SimpleNamespace(constraint_sequences=[0] * 10, current_constraint_sequence_idx=0, current_constraint_idx=3)
    plain = cpgen_rt.SourcePicker(True, 11)
    ruled = cpgen_rt.SourcePicker(True, 11, rule=SimpleNamespace(pick=lambda subtask: 7))
    ruled.pick(policy)
    assert policy.current_constraint_sequence_idx == 7 and policy.current_constraint_idx == 0
    plain.pick(SimpleNamespace(constraint_sequences=[0] * 10))
    assert plain._rng.integers(1000) == ruled._rng.integers(1000)             # the chunk's own draw was still taken
    with pytest.raises(ValueError, match="selection=ours"):
        cpgen_rt.SourcePicker(False, 11, rule=SimpleNamespace(pick=lambda s: 0))


def test_decide_matches_mix_without_blending():
    # same inputs, fresh mixers: decide picks the donor and weight mix uses, with the same draws and record keys
    ctx = context(0, [pose(x=0.3), pose(x=0.1), pose(x=0.2)], [0.0, 0.0, 0.0], {"current_object_pose": pose()})
    mixed = make(TransformMixer).mix(ctx)
    record, donor, weight = make(TransformMixer).decide(ctx)
    assert donor == mixed.record["source_b"] == 1 and weight == pytest.approx(mixed.record["transform"]["weight_a"], abs=1e-4)
    assert set(mixed.record) - set(record) == {"alignment", "mixed"} or set(mixed.record) - set(record) == {"alignment"}
    assert record["mixed"] is False                     # the caller sets it once its blend is done
    kept = make(TransformMixer).decide(context(1, [pose(x=0.3), pose(x=0.1), pose(x=0.2)], [0.0] * 3, {"current_object_pose": pose()}))
    assert kept[1] is None and kept[2] is None and "identity_reason" in kept[0]
