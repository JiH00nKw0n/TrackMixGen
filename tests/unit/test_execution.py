import numpy as np

from tmg.core.records import AttemptRecorder
from tmg.generation.execution import TrajectoryExecution, WaypointView
from tmg.generation.method.interfaces import Method

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


def _run(method):
    rec = AttemptRecorder()
    rec.begin_attempt()
    ex = TrajectoryExecution(method, rec)
    seqs = _sequences()
    ex.begin_trajectory(seqs)
    steps = 0
    for i, seq in enumerate(seqs):
        ex.enter_sequence(i)
        for w in seq:
            action = np.full(6, 2.0)                       # clipped to 1 in the pre-noise label
            pre = ex.keep_pre_noise(action)
            ex.before_step(target_poses=w.poses, measured_poses={EEF: np.eye(4)},
                           executed_action=np.r_[action, w.grippers[EEF]], grippers=w.grippers, noise_of_waypoint=0.05)
            ex.after_step(task_success=False, term_signals={"grasp": False})
            assert np.array_equal(pre, np.full(6, 2.0))
            steps += 1
    return ex, rec, steps


def test_every_waypoint_is_recorded_once_in_its_segment():
    ex, rec, steps = _run(Method(name="none", mixer=None))
    assert steps == 13 and rec.seg_len == [5, 8] and rec.track["k"] == [-1] * 13
    assert rec.eef_names == [EEF]
    executed = np.zeros((13, 7))
    assert ex.label_actions(executed) is executed                      # executed labels: the actions as executed


def test_pre_noise_labels_are_the_clipped_arm_action_and_the_gripper_command():
    ex, _rec, _steps = _run(Method(name="m", mixer=None, labels="pre_noise"))
    labels = ex.label_actions(np.zeros((13, 7)))
    assert labels.shape == (13, 7) and np.allclose(labels[:, :6], 1.0) and np.allclose(labels[:, 6], -1.0)


def test_segment_table_maps_waypoints_to_record_segments():
    from tmg.core.records import SEG_BRIDGE, SEG_SKILL
    from tmg.generation.execution import SegmentTable

    table = SegmentTable()
    assert len(table) == 0 and table.total == 0 and table.seg_id_of(0) == -1
    table.add(3, seg_id=7, kind=SEG_BRIDGE)
    table.add(2, seg_id=9, kind=SEG_SKILL)
    assert table.total == 5 and table[1].seg_id == 9 and (table[1].start, table[1].end) == (3, 5)
    assert [table.seg_id_of(k) for k in range(6)] == [7, 7, 7, 9, 9, 9]        # out of range: the last segment
    assert [row.kind for row in table.rows] == [SEG_BRIDGE, SEG_SKILL]
