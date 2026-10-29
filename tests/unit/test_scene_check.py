"""Scene check bookkeeping (``tmg.generation.scene_check``) without a simulator: the tally, the report and the bounds
comparison are driven with arrays and a fake environment. Expected numbers are worked out by hand from the inputs
(table 0.8 x 0.8 m centred at the origin, top at 0.8 m, robot base at x = -0.56)."""
import types

import numpy as np

from tmg.generation.scene_check import (
    MAX_CONTACT_DETAILS,
    MAX_FAILURES,
    _build_report,
    _resets_beyond,
    _table_facts,
    _Tally,
    compare_bounds,
)


def _table():
    return {"centre_xy": [0.0, 0.0], "top_z": 0.8, "half_x": 0.4, "half_y": 0.4}


BASE_XY = (-0.56, 0.0)


def test_tally_classifies_placements_contacts_and_settling():
    tally = _Tally()
    table = _table()
    before = {"nut": np.array([0.1, 0.0, 0.83]), "peg": np.array([0.45, 0.0, 0.83])}      # the peg's centre is off the table
    tally.add_contacts(0, [("nut", "peg", ""), ("nut", "robot", "gripper0_finger")], before)
    assert dict(tally.contact_pairs) == {"nut & peg": 1, "nut & robot": 1}
    assert tally.contact_details == [
        {"reset": 0, "object": "nut", "other": "peg", "robot_geom": "", "object_xy": [0.1, 0.0], "other_xy": [0.45, 0.0]},
        {"reset": 0, "object": "nut", "other": "robot", "robot_geom": "gripper0_finger", "object_xy": [0.1, 0.0], "other_xy": None}]
    for name, p in before.items():
        tally.add_placement(name, p, radius=0.05, table=table, base_xy=BASE_XY, margin=0.02, reach_m=0.85)
    # a second reset puts the nut at (0.2, 0.1): still inside, 0.767 m from the base
    tally.add_placement("nut", np.array([0.2, 0.1, 0.83]), radius=0.05, table=table, base_xy=BASE_XY, margin=0.02, reach_m=0.85)
    nut, peg = tally.counts["nut"], tally.counts["peg"]
    assert nut["n"] == 2 and nut["centre_outside"] == 0 and nut["footprint_over_edge"] == 0 and nut["beyond_reach"] == 0
    assert peg["n"] == 1 and peg["centre_outside"] == 1 and peg["footprint_over_edge"] == 1       # 0.45 > 0.4; 0.5 > 0.38
    assert peg["beyond_reach"] == 1                                                            # 1.01 m > 0.85 m
    assert np.allclose(tally.dists["nut"], [0.66, 0.76655], atol=1e-5) and np.allclose(tally.dists["peg"], [1.01])
    assert tally.xy_min["nut"].tolist() == [0.1, 0.0] and tally.xy_max["nut"].tolist() == [0.2, 0.1]
    tally.add_settling("nut", before["nut"], np.array([0.1, 0.0, 0.81]), table, fall_thr=0.01, drift_thr=0.005)   # settled 2 cm
    tally.add_settling("peg", before["peg"], np.array([0.5, 0.0, 0.5]), table, fall_thr=0.01, drift_thr=0.005)    # fell off
    tally.add_settling("nut", before["nut"], np.array([0.1, 0.0, 0.825]), table, fall_thr=0.01, drift_thr=0.005)  # 5 mm: no drop
    assert nut["dropped"] == 1 and nut["off_table"] == 0 and nut["drifted"] == 0
    assert peg["dropped"] == 1 and peg["off_table"] == 1 and peg["drifted"] == 1
    objects = tally.objects({"nut": {"radius": 0.05}, "peg": {"radius": 0.0512345}})
    assert objects["nut"]["base_dist_m"] == {"min": 0.66, "mean": 0.713, "max": 0.767}
    assert objects["nut"]["xy_min"] == [0.1, 0.0] and objects["nut"]["xy_max"] == [0.2, 0.1]
    assert objects["nut"]["dropped"] == {"count": 1, "fraction": 0.5} and objects["nut"]["beyond_reach"] == {"count": 0, "fraction": 0.0}
    assert objects["peg"]["radius_m"] == 0.0512 and objects["peg"]["base_dist_m"] == {"min": 1.01, "mean": 1.01, "max": 1.01}
    assert objects["peg"]["off_table"] == {"count": 1, "fraction": 1.0} and objects["peg"]["drifted"] == {"count": 1, "fraction": 1.0}


def test_tally_keeps_every_contact_count_but_only_the_first_details():
    tally = _Tally()
    before = {"nut": np.array([0.1, 0.0, 0.83]), "peg": np.array([0.2, 0.0, 0.83])}
    for reset in range(MAX_CONTACT_DETAILS + 10):
        tally.add_contacts(reset, [("nut", "peg", "")], before)
    assert tally.contact_pairs["nut & peg"] == MAX_CONTACT_DETAILS + 10 == 60
    assert len(tally.contact_details) == MAX_CONTACT_DETAILS == 50
    assert [d["reset"] for d in tally.contact_details] == list(range(50))


def test_resets_beyond_counts_resets_with_any_far_object():
    # reset 1: a is far; reset 2: b is far; reset 3 (only b has a value) is not counted
    assert _resets_beyond({"a": [0.5, 0.9, 0.5], "b": [0.5, 0.5, 0.9, 0.2]}, 0.85) == 2
    assert _resets_beyond({"a": [0.85]}, 0.85) == 0                     # exactly the radius is within reach
    assert _resets_beyond({}, 0.85) == 0


def _report_inputs(**overrides):
    args = dict(env_name="Square_OURS", task={"name": "square"}, scene={"name": "ours"}, interface="MG_Square", resets=2,
                table=_table(), base_xy=BASE_XY, bounds_match={"match": True, "differences": []},
                constraint={"radius_m": 0.85}, reach_m=0.85, geometry={"nut": {"radius": 0.05}, "peg": {"radius": 0.05}})
    args.update(overrides)
    return args


def _two_reset_tally(peg_xy=(0.2, 0.0)):
    tally = _Tally()
    tally.done, tally.times, tally.rejects = 2, [0.5, 0.7], [0, 3]
    for _ in range(2):
        tally.add_placement("nut", np.array([0.1, 0.0, 0.83]), 0.05, _table(), BASE_XY, 0.02, 0.85)
        tally.add_placement("peg", np.array([*peg_xy, 0.83]), 0.05, _table(), BASE_XY, 0.02, 0.85)
    return tally


def test_build_report_fields_and_verdict():
    tally = _two_reset_tally()
    report = _build_report(tally, **_report_inputs())
    assert report["ok"] is True
    assert (report["env"], report["task"], report["scene"], report["interface"]) == ("Square_OURS", "square", "ours", "MG_Square")
    assert report["resets_requested"] == 2 and report["resets_done"] == 2 and report["placement_failures"] == 0
    assert report["table"] == _table() and report["robot_base_xy"] == [-0.56, 0.0]
    assert report["reach"] == {"constraint": {"radius_m": 0.85}, "radius_used_m": 0.85, "rejects": {"mean": 1.5, "max": 3},
                               "resets_with_any_object_beyond": 0}
    assert report["reset_time_s"] == {"mean": 0.6} and report["failures"] == []
    assert report["contacts_at_reset"] == {"resets_with_penetration": 0, "pairs": {}, "details": []}
    assert set(report["objects"]) == {"nut", "peg"} and report["objects"]["peg"]["base_dist_m"] == {"min": 0.76, "mean": 0.76, "max": 0.76}
    assert report["objects"]["peg"]["beyond_reach"] == {"count": 0, "fraction": 0.0}
    # a peg at x = 0.3 is 0.86 m from the base: beyond reach in both resets, information only (ok stays True)
    far = _build_report(_two_reset_tally(peg_xy=(0.3, 0.0)), **_report_inputs(constraint=None))
    assert far["reach"]["resets_with_any_object_beyond"] == 2 and far["reach"]["constraint"] is None and far["ok"] is True
    assert far["objects"]["peg"]["beyond_reach"] == {"count": 2, "fraction": 1.0}


def test_build_report_verdict_fails_on_each_problem_alone():
    base = _report_inputs()
    assert _build_report(_two_reset_tally(), **base)["ok"] is True
    failed = _two_reset_tally()
    failed.failures.append({"reset": 2, "error": "RandomizationError"})
    assert _build_report(failed, **base)["ok"] is False
    touching = _two_reset_tally()
    touching.add_contacts(0, [("nut", "peg", "")], {"nut": np.zeros(3), "peg": np.zeros(3)})
    report = _build_report(touching, **base)
    assert report["ok"] is False and report["contacts_at_reset"]["resets_with_penetration"] == 1
    mismatch = _build_report(_two_reset_tally(), **_report_inputs(bounds_match={"match": False, "differences": ["nut.x: ..."]}))
    assert mismatch["ok"] is False
    fell = _two_reset_tally()
    fell.add_settling("peg", np.array([0.3, 0.0, 0.83]), np.array([0.3, 0.0, 0.5]), _table(), 0.01, 0.005)   # off the table
    assert _build_report(fell, **base)["ok"] is False
    drifted = _two_reset_tally()
    drifted.add_settling("peg", np.array([0.3, 0.0, 0.83]), np.array([0.31, 0.0, 0.83]), _table(), 0.01, 0.005)
    drift_report = _build_report(drifted, **base)
    assert drift_report["ok"] is False and drift_report["objects"]["peg"]["drifted"] == {"count": 1, "fraction": 0.5}
    dropped = _two_reset_tally()
    dropped.add_settling("peg", np.array([0.3, 0.0, 0.83]), np.array([0.3, 0.0, 0.81]), _table(), 0.01, 0.005)  # settled, still on
    assert _build_report(dropped, **base)["ok"] is True                                                 # drops are information


def test_build_report_without_a_successful_reset_and_with_many_failures():
    tally = _Tally()
    tally.failures = [{"reset": i, "error": "RandomizationError"} for i in range(MAX_FAILURES + 5)]
    empty = _build_report(tally, **_report_inputs(table=None, base_xy=None, bounds_match=None, geometry=None))
    assert empty["objects"] == {} and empty["table"] is None and empty["robot_base_xy"] is None
    assert empty["placement_failures"] == 25 and len(empty["failures"]) == MAX_FAILURES == 20
    assert empty["reach"]["rejects"] is None and empty["reach"]["resets_with_any_object_beyond"] == 0
    assert empty["reset_time_s"] == {"mean": None} and empty["resets_done"] == 0
    assert empty["ok"] is False
    assert _build_report(_Tally(), **_report_inputs(table=None, base_xy=None, bounds_match=None, geometry=None))["ok"] is True


def test_table_facts_and_compare_bounds_with_a_fake_env():
    listed = {"nut": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z_rot": (0.0, 6.28), "reference": "table_offset", "rotation_axis": "z"}}
    actual = {"nut": {"x": (-0.1, 0.1), "y": (-0.2, 0.2), "z_rot": (0.0, 6.28), "reference": np.array([0.0, 0.0, 0.8]),
                      "rotation_axis": "z"}}
    env = types.SimpleNamespace(table_full_size=[0.8, 0.8, 0.05], table_offset=np.array([0.0, 0.0, 0.8]),
                                _get_initial_placement_bounds=lambda: actual)
    assert _table_facts(env) == _table()
    assert compare_bounds(env, listed) == {"match": True, "differences": []}
    assert compare_bounds(env, None) is None
    actual["nut"]["x"] = (-0.2, 0.2)
    actual["peg"] = dict(actual["nut"])
    result = compare_bounds(env, listed)
    assert not result["match"] and result["differences"] == ["nut.x: class (-0.2, 0.2) vs listed (-0.1, 0.1)",
                                                             "peg: sampled by the class but not listed"]
    actual["nut"]["x"] = (-0.1, 0.1)
    del actual["peg"]
    actual["nut"]["reference"] = np.array([0.0, 0.0, 0.82])
    actual["nut"]["rotation_axis"] = "y"
    listed["bolt"] = dict(listed["nut"])
    assert compare_bounds(env, listed)["differences"] == ["nut.reference: class [0.0, 0.0, 0.82] vs listed [0.0, 0.0, 0.8]",
                                                          "nut.rotation_axis: class y vs listed z",
                                                          "bolt: listed but not sampled by the class"]
