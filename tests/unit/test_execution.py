import numpy as np

from tmg.core.records import AttemptRecorder
from tmg.generation.execution import TrajectoryExecution, WaypointView
from tmg.generation.method.interfaces import Method
from tmg.generation.method.tracking import ErrorAdaptiveFollower

CFG = {"ds_min": 0.25, "regions": {"default": {"alpha_p": 12.0, "alpha_r": 1.2}, "approach": {"alpha_p": 50.0, "alpha_r": 5.0}},
       "approach_window": 10, "extra_steps_cap": "segment"}
EEF = "robot0"


def _sequences():
    def seq(n, x0):
        out = []
        for i in range(n):
            pose = np.eye(4)
            pose[0, 3] = x0 + 0.01 * i
            out.append(WaypointView(poses={EEF: pose}, grippers={EEF: np.array([-1.0])}, noise=0.05))
        return out
    return [seq(5, 0.0), seq(8, 0.05)]


class Draws:
    def __init__(self):
        self.global_calls = 0
        self.private_calls = 0

    def global_draw(self, n):
        self.global_calls += 1
        return np.zeros(n)

    def private_draw(self, n):
        self.private_calls += 1
        return np.zeros(n)


def _run(method, measured):
    draws = Draws()
    rec = AttemptRecorder()
    rec.begin_attempt()
    ex = TrajectoryExecution(method, rec, global_draw=draws.global_draw, private_draw=draws.private_draw)
    seqs = _sequences()
    ex.begin_trajectory(seqs)
    steps = 0
    for i, seq in enumerate(seqs):
        ex.enter_sequence(i)
        for j in ex.steps_of_sequence(i, len(seq)):
            target = ex.target(seq[j], None) if ex.follower is None else ex.target(None, measured)
            action = np.zeros(6)
            ex.apply_noise(action)
            ex.before_step(target_poses=target.poses, measured_poses={EEF: np.eye(4)},
                           executed_action=np.r_[action, target.grippers[EEF]], grippers=target.grippers,
                           noise_of_waypoint=0.05)
            ex.after_step(task_success=False, term_signals={"grasp": False})
            steps += 1
    return draws, rec, steps


def test_identity_draws_nothing_and_walks_every_waypoint():
    draws, rec, steps = _run(Method(name="none", mixer=None, follower_factory=None), None)
    assert steps == 13 and draws.global_calls == 0 and draws.private_calls == 0
    assert rec.seg_len == [5, 8] and rec.track["k"] == [-1] * 13
    assert rec.eef_names == [EEF]


def test_follower_consumes_one_global_draw_per_original_waypoint():
    method = Method(name="tracking_only", mixer=None, follower_factory=ErrorAdaptiveFollower.factory(CFG), labels="pre_noise")
    far = np.eye(4)
    far[0, 3] = -1.0
    draws, rec, steps = _run(method, {EEF: far})
    assert draws.global_calls == 13                 # exactly the official count, whatever the step count
    assert steps > 13 and draws.private_calls == steps - 13
    assert sum(rec.track["slot"]) == 13
    assert sorted(set(rec.seg_of_step)) == [0, 1] and rec.seg_len[0] + rec.seg_len[1] == steps


def test_segment_table_maps_waypoints_to_record_segments():
    from tmg.core.records import SEG_BRIDGE, SEG_SKILL
    from tmg.generation.execution import SegmentTable, slot_sources

    table = SegmentTable()
    assert len(table) == 0 and table.total == 0 and table.seg_id_of(0) == -1 and table.kinds_per_waypoint() == []
    table.add(3, seg_id=7, kind=SEG_BRIDGE)
    table.add(2, seg_id=9, kind=SEG_SKILL)
    assert table.total == 5 and table[1].seg_id == 9 and (table[1].start, table[1].end) == (3, 5)
    assert [table.seg_id_of(k) for k in range(6)] == [7, 7, 7, 9, 9, 9]        # out of range: the last segment
    assert table.kinds_per_waypoint() == [SEG_BRIDGE] * 3 + [SEG_SKILL] * 2
    # a step that newly covers its own waypoint takes the official draw; an extra step draws privately
    assert slot_sources({"k": 4, "new_slots": [3, 4]}) == (True, "global")
    assert slot_sources({"k": 4, "new_slots": []}) == (False, "private")
    assert slot_sources({"k": 4, "new_slots": [5]}) == (False, "private")
