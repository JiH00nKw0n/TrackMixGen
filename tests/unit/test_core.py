import json

import numpy as np

from tmg.core.chunks import plan_chunks
from tmg.core.poses import invert, list_to_pose, pose_gap, pose_to_list
from tmg.core.records import AttemptRecorder, read_attempts


def test_plan_chunks():
    chunks = plan_chunks(600, 250, 30000)
    assert [(c.seed, c.attempts) for c in chunks] == [(30001, 250), (30002, 250), (30003, 100)]


def test_pose_roundtrip():
    pose = np.eye(4)
    pose[:3, 3] = [0.1, 0.2, 0.3]
    assert np.allclose(list_to_pose(pose_to_list(pose)), pose)
    assert np.allclose(invert(pose) @ pose, np.eye(4))
    assert pose_gap(pose, np.eye(4))[0] == np.linalg.norm([0.1, 0.2, 0.3])


def test_recorder_writes_npz_and_jsonl(tmp_path):
    rec = AttemptRecorder()
    rec.configure(tmp_path)
    rec.begin_attempt()
    rec.set_objects_init({"nut": np.eye(4)})
    rec.add_subtask(subtask_ind=0, object_name="nut", src_demo_ind=3, cur_obj_pose=np.eye(4))
    seg = rec.begin_segment(1)
    for t in range(3):
        rec.add_step(w_poses={"robot0": np.eye(4)}, e_poses={"robot0": np.eye(4)}, a_pre=np.zeros(6), a_label=np.zeros(7),
                     a_exec=np.zeros(7), noise=0.05, seg_id=seg, term_signals={"grasp": t == 2}, track=None)
        rec.add_success(t == 2)
    line = rec.flush(success=True, src_demo_inds=[3])
    assert line["n_steps"] == 3 and line["succ_first"] == 2 and line["term_first"]["grasp"] == 2
    lines = read_attempts(tmp_path)
    assert len(lines) == 1 and lines[0]["subtasks"][0]["src_demo_ind"] == 3
    arrays = np.load(tmp_path / lines[0]["npz"])
    assert arrays["seg_len"].tolist() == [3] and arrays["track_region"].tolist() == [-1, -1, -1]
    assert arrays["w_pos"].shape == (3, 1, 3) and arrays["eef_names"].tolist() == ["robot0"]
    assert json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[0])["success"] is True
