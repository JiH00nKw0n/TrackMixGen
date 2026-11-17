import numpy as np
import pytest

from tmg.generation.method.tracking import EndEffectorPath
from tmg.generation.method.transit_pacing import TransitPacingFollower

CFG = {"k_pos": 0.05, "k_rot": 0.1, "eps_pos": 0.1, "eps_rot": 0.1, "eps_settle_pos": 0.01, "eps_settle_rot": 0.1,
       "extra_steps_cap": "segment"}


def _follower(n_bridge=5, n_skill=5, step=0.1):
    n = n_bridge + n_skill
    poses = np.tile(np.eye(4), (n, 1, 1))
    poses[:, 0, 3] = np.arange(n) * step
    grips = -np.ones((n, 1))
    f = TransitPacingFollower.factory(CFG)()
    f.start_path([EndEffectorPath("arm", poses, grips)], [0.0] * n, [0] * n_bridge + [1] * n_skill)
    return f, poses


def _at(x):
    m = np.eye(4)
    m[0, 3] = x
    return {"arm": m}


def test_bridge_advance_keeps_predicted_error_within_bound():
    f, _ = _follower()
    f.next_target(_at(0.0))                                   # start: s = 0
    t, _, _, rec = f.next_target(_at(0.0))
    x = t["arm"][0, 3]
    assert rec["region"] == "bridge"
    assert (1 - 0.05) * x <= 0.1 + 1e-9                       # predicted next error within eps_pos
    assert x == pytest.approx(0.1 / 0.95, abs=0.1 / 40 * 0.1 + 1e-6) or x < 0.1 / 0.95


def test_waits_when_the_hand_lags_beyond_the_bound():
    f, _ = _follower()
    f.next_target(_at(0.0))
    f.next_target(_at(0.0))
    s = f.s
    _, _, _, rec = f.next_target(_at(-0.5))                    # hand far behind: no advance
    assert rec["ds_actual"] == 0.0 and f.s == s


def test_settles_at_the_bridge_end_then_runs_the_skill_at_full_speed():
    f, poses = _follower()
    f.next_target(_at(0.0))
    for _ in range(200):                                       # hand follows the target exactly
        if f.s >= 4 - 1e-9:
            break
        f.next_target(_at(f.targets(f.s)["arm"][0, 3]))
    assert f.s == pytest.approx(4.0)                           # stopped at the bridge end (waypoint 4)
    _, _, _, rec = f.next_target(_at(0.35))                    # 5 cm off: wait
    assert rec["region"] == "settle" and rec["ds_actual"] == 0.0
    _, _, _, rec = f.next_target(_at(0.4))                     # within 1 cm: the skill starts
    assert rec["ds_actual"] == pytest.approx(1.0)
    _, _, _, rec = f.next_target(_at(5.0))                     # skill: one waypoint per step whatever the error
    assert rec["region"] == "skill" and rec["ds_actual"] == pytest.approx(1.0)


def test_cap_ends_a_stuck_path():
    f, _ = _follower()
    f.next_target(_at(0.0))
    for _ in range(100):
        if f.finished():
            break
        f.next_target(_at(-5.0))
    assert f.finished()
