"""AutoData adapter units that run without Isaac Sim: chunk spec and descriptor, graft anchors on AutoData-shaped code,
image resize, camera lookup across observation groups."""
import ast
import json

import numpy as np
import pytest
import yaml

from tmg.config import compose, resolved
from tmg.core.chunks import plan_chunks
from tmg.data.images import resize_frames
from tmg.generation.generators import make_generator
from tmg.generation.generators.autodata.adapter import DESCRIPTOR, RUNTIME_MODULE, chunk_descriptor
from tmg.generation.graft import Graft

OFFICIAL = {"name": "gr1_open_microwave", "algo": "dexmimicgen",
            "generation_policy": {"name": "gr1_open_microwave", "num_trials": 50, "guarantee_success": False, "select_src_per_subtask": True},
            "subtasks": {"right": [{"object_ref": "microwave", "subtask_term_signal": "move_to_door", "action_noise": 0.005}],
                         "left": [{"object_ref": "microwave", "subtask_term_signal": None}]}}


def test_chunk_descriptor_changes_only_seed_trials_and_guarantee():
    out = chunk_descriptor(OFFICIAL, seed=30001, num_trials=250)
    assert out["generation_policy"] == {"name": "gr1_open_microwave", "num_trials": 250, "guarantee_success": True,
                                        "select_src_per_subtask": True, "seed": 30001}
    assert out["subtasks"] == OFFICIAL["subtasks"] and OFFICIAL["generation_policy"]["num_trials"] == 50


def test_autodata_chunk_spec(tmp_path, monkeypatch):
    src = tmp_path / "AutoData"
    (src / "autodata_examples" / "tasks").mkdir(parents=True)
    (src / "autodata_examples" / "tasks" / "gr1_open_microwave.yaml").write_text(yaml.safe_dump(OFFICIAL))
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    monkeypatch.setenv("TMG_AUTODATA_SRC", str(src))
    cfg = resolved(compose("generate", ["generator=autodata", "task=microwave", "scene=ours", "method=none", "generation=pool_isaac", "cluster=ada"]))
    assert cfg["task"]["horizon"] == 250 and cfg["task"]["source_revision"] == "arena_v0.2_lab_v3.0"
    gen = make_generator(cfg)
    chunk = plan_chunks(1000, 250, 30000)[1]
    spec = gen.prepare_chunk(chunk, tmp_path / "c")
    args = spec.extra["script_args"]
    assert args[args.index("--env_name") + 1] == "gr1_open_microwave" and args[args.index("--alg") + 1] == "dexmimicgen"
    assert args[args.index("--generation_num_trials") + 1] == "250" and "--enable_cameras" in args and "--seed" not in args[:args.index("--")]
    assert args[args.index("--") + 1:] == ["--seed", "30002"]
    assert spec.extra["runtime_module"] == RUNTIME_MODULE and spec.extra["package"] == "autodata_core"
    written = yaml.safe_load((tmp_path / "c" / DESCRIPTOR).read_text())
    assert written["generation_policy"]["seed"] == 30002 and written["generation_policy"]["guarantee_success"] is True
    assert spec.extra["layout"]["arm_action_dims"] == {"left": [0, 7], "right": [7, 14]} and spec.extra["layout"]["hand_prefixes"] is None
    assert gen.demo_layout()["image_key"] == "{cam}"


def test_graft_anchors_on_autodata_shaped_functions(tmp_path, monkeypatch):
    """The AutoData anchors (tuple assignment, two returns, a valued final return) on functions shaped like AutoData's."""
    import importlib
    import sys

    (tmp_path / "ad_shaped.py").write_text('''
async def generate(self, env_id):
    env_id_tensor, initial_state = await self.reset(env_id)
    buffers = Buf()
    while True:
        result = plan()
        if result is None:
            return Result(initial_state=initial_state, success=False)
        if done():
            break
    return Result(initial_state=initial_state, success=buffers.success)


async def execute(self, datastream, success_term, env_id=0):
    env = datastream.get_env()
    play_action = datastream.target_eef_pose_to_action(a=1)
    env.step(play_action)
    return bool(success_term.func(env, **success_term.params)[env_id])
''')
    monkeypatch.syspath_prepend(str(tmp_path))
    mod = importlib.import_module("ad_shaped")
    sys.modules.pop("ad_shaped", None)
    g = Graft(mod.generate, expected_hash=None, label="generate")
    g.insert_after_assign("initial_state", "_tmg.begin(self, env_id, initial_state)")
    g.insert_before_return("_tmg.end(False)", occurrence=0, expected_returns=2)
    g.insert_before_return("_tmg.end(buffers.success)", occurrence=1, expected_returns=2)
    code = ast.unparse(g.node)
    assert code.index("_tmg.begin(") < code.index("_tmg.end(False)") < code.index("_tmg.end(buffers.success)")
    assert code.count("return Result") == 2
    e = Graft(mod.execute, expected_hash=None, label="execute")
    e.insert_after_assign("play_action", "_tmg.cmd(self, datastream, env_id, play_action)", occurrence=0)
    e.name_final_return("success")
    e.insert_before_final_return("_tmg.step(self, env, env_id, None, success)")
    code = ast.unparse(e.node)
    assert "success = bool(success_term.func" in code and code.rstrip().endswith("return success")
    assert code.index("_tmg.cmd(") < code.index("success = bool") < code.index("_tmg.step(")
    compiled = e.compile({"_tmg": None})
    assert compiled.__name__ == "execute"


def test_resize_frames_matches_dataset_and_evaluator():
    frames = (np.random.default_rng(0).random((3, 512, 512, 3)) * 255).astype(np.uint8)
    small = resize_frames(frames, (128, 128))
    assert small.shape == (3, 128, 128, 3) and small.dtype == np.uint8
    assert np.array_equal(resize_frames(frames[0], (128, 128)), small[0])     # one frame resizes exactly like the batch
    assert resize_frames(small, (128, 128)) is small                            # already the right size: untouched


def test_observation_input_finds_cameras_in_other_groups_and_resizes():
    from tmg.evaluation.isaaclab_rollout import observation_input

    class T:
        def __init__(self, a):
            self.a = np.asarray(a)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.a

    big = (np.random.default_rng(2).random((1, 512, 512, 3)) * 255).astype(np.uint8)
    obs = {"policy": {"left_eef_pos": T([[0.0, 1.0, 2.0]]), "hand_joint_state": T(np.zeros((1, 22)))},
           "camera_obs": {"robot_pov_cam_rgb": T(big)}}
    observation = {"cameras": ["robot_pov_cam_rgb"], "image_size": [128, 128], "robot_state": ["left_eef_pos", "hand_joint_state"], "object_state": []}
    out = observation_input(obs, observation)
    assert out["observation.images.robot_pov_cam_rgb"].shape == (3, 128, 128) and out["observation.state"].shape == (25,)
    with pytest.raises(KeyError):
        observation_input({"policy": {"left_eef_pos": T([[0.0]])}}, {"cameras": ["missing"], "robot_state": [], "object_state": []})


def test_autodata_task_configs_and_driver_jobs():
    for task in ("microwave", "bottle_door"):
        cfg = resolved(compose("resets", ["generator=autodata", f"task={task}", "scene=ours"]))
        assert cfg["task"]["cameras"] == ["robot_pov_cam_rgb"] and cfg["task"]["image_size"] == [128, 128]
        assert cfg["task"]["policy_overrides"]["diffusion_policy"] == {"crop_shape": None}
    assert json.dumps(OFFICIAL)   # sanity: fixture serializable
