"""Adaptive Command follower: on a simulated hand that obeys the one-step response model it keeps one waypoint per step
where the path is reachable, tracks closer than Adaptive Pacing, and keeps the runtime's bookkeeping."""
import numpy as np
import pytest

from tmg.generation.method.adaptive_command import AdaptiveCommandFollower, rot_exp, rot_log
from tmg.generation.method.interfaces import build_method
from tmg.generation.method.tracking import EndEffectorPath, ErrorAdaptiveFollower

A_P, B_P, A_R, B_R, DP, DR = 0.629, 0.0936, 0.712, 0.0663, 0.05, 0.5


def path(n=60, spacing=0.007, close_at=None):
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * spacing
    poses[:, :3, :3] = np.stack([rot_exp([0, 0, 0.002 * i]) for i in range(n)])
    g = -np.ones((n, 1))
    if close_at is not None:
        g[close_at:] = 1.0
    return EndEffectorPath("robot0", poses, g)


def run(follower, eef, start_offset=0.0, max_steps=400):
    """Hand obeying the response model, driven by the controller rule action = clip((target - current) / D)."""
    follower.start_path([eef], [0.05] * len(eef.poses), [1] * len(eef.poses))
    x = eef.poses[0, :3, 3] + np.array([start_offset, 0, 0])
    R = eef.poses[0, :3, :3].copy()
    dx, w = np.zeros(3), np.zeros(3)
    errors, steps, slots = [], 0, []
    while not follower.finished() and steps < max_steps:
        meas = np.eye(4)
        meas[:3, 3], meas[:3, :3] = x, R
        targets, grippers, noise, rec = follower.next_target({"robot0": meas})
        t = targets["robot0"]
        up = np.clip((t[:3, 3] - x) / DP, -1, 1)
        ur = np.clip(rot_log(t[:3, :3] @ R.T) / DR, -1, 1)
        dx = A_P * dx + B_P * DP * up
        w = A_R * w + B_R * DR * ur
        x, R = x + dx, rot_exp(w) @ R
        ref = rec.get("ref", targets)["robot0"]
        errors.append(np.linalg.norm(x - ref[:3, 3]))
        slots += rec["new_slots"]
        steps += 1
    return steps, float(np.mean(errors[5:])), slots


def test_adaptive_command_keeps_pace_and_tracks_closer_than_pacing():
    eef = path()
    pacing = ErrorAdaptiveFollower(ds_min=0.25, regions={"default": {"alpha_p": 12.0, "alpha_r": 1.2}},
                                   approach_window=10, extra_steps_cap="segment")
    command = AdaptiveCommandFollower()
    s_p, e_p, _ = run(pacing, eef)
    s_c, e_c, slots = run(command, eef)
    assert s_c <= 1.15 * len(eef.poses) and s_c < s_p
    assert e_c < e_p and e_c < 0.004
    assert slots == list(range(len(eef.poses)))          # every original waypoint consumed exactly once, in order


def test_adaptive_command_slows_only_when_the_path_is_out_of_reach():
    far = path(n=30, spacing=0.03)                       # 30 mm per step, beyond the ~12.6 mm steady reach
    steps, _, _ = run(AdaptiveCommandFollower(), far)
    assert steps > 1.5 * 30
    near = path(n=30, spacing=0.004)
    steps, _, _ = run(AdaptiveCommandFollower(), near)
    assert steps <= 32


def test_adaptive_command_waits_for_arrival_before_a_gripper_event():
    eef = path(n=40, close_at=20)
    f = AdaptiveCommandFollower()
    f.start_path([eef], [0.05] * 40, [1] * 40)
    meas = np.eye(4)
    meas[:3, 3] = [0.5, 0, 0]                           # hand far from the path: arrival never holds
    f.s = 19.9                                          # the smallest advance (0.25) reaches the event at 20
    f.next_slot = 20
    targets, grippers, _, rec = f.next_target({"robot0": meas})
    assert rec["waiting"] and rec["k"] == 19 and grippers["robot0"][0] == -1.0


def test_build_method_picks_both_trackers():
    m = build_method({"name": "adaptive_command", "tracking": {"enabled": True, "type": "adaptive_command", "kappa": 0.3}}, chunk_seed=0)
    assert isinstance(m.follower_factory(), AdaptiveCommandFollower) and m.follower_factory().kappa == 0.3
    m = build_method({"name": "adaptive_pacing", "tracking": {"enabled": True, "type": "adaptive_pacing",
                                                             "regions": {"default": {"alpha_p": 1, "alpha_r": 1}}}}, chunk_seed=0)
    assert isinstance(m.follower_factory(), ErrorAdaptiveFollower)
    with pytest.raises(KeyError):
        AdaptiveCommandFollower(kapa=0.5)
