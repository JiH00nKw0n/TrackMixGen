"""The shared evaluation loop (``tmg.evaluation.session``) with a fake policy client: server checks, seeds, row layout."""
import numpy as np
import pytest

from tmg.evaluation import session
from tmg.evaluation.envs import ROBOSUITE_EVAL_KWARGS, is_isaaclab, isaaclab_task_id
from tmg.evaluation.resets import ENV_KWARGS


def test_check_policy_matches_cameras_and_object_state():
    obs = {"name": "image", "cameras": ["a", "b"], "object_state": []}
    session.check_policy_matches({"cameras": ["a", "b"], "env_state_dim": 0}, obs)
    with pytest.raises(RuntimeError, match="cameras"):
        session.check_policy_matches({"cameras": ["a"], "env_state_dim": 0}, obs)
    with pytest.raises(RuntimeError, match="env_state_dim"):
        session.check_policy_matches({"cameras": ["a", "b"], "env_state_dim": 14}, obs)
    low_dim = {"name": "low_dim", "cameras": [], "object_state": ["object"]}
    session.check_policy_matches({"cameras": [], "env_state_dim": 14}, low_dim)
    with pytest.raises(RuntimeError, match="env_state_dim"):          # an image checkpoint offered a low-dim observation
        session.check_policy_matches({"cameras": [], "env_state_dim": 0}, low_dim)
    with pytest.raises(RuntimeError, match="cameras"):                # camera order matters (the batch keys follow it)
        session.check_policy_matches({"cameras": ["b", "a"], "env_state_dim": 0}, obs)


def test_play_scenes_checks_the_server_then_records_rows_in_order():
    calls = []

    class Client:
        def __init__(self, url):
            calls.append(("connect", url))

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            calls.append(("close",))

        def info(self):
            return {"checkpoint": "/c/", "cameras": [], "env_state_dim": 3}

    rows = []
    scenes = [{"index": 4, "split": "test", "seed": 11}, {"index": 1, "split": "val", "seed": 12}]

    def run_one(policy, scene, seed):
        calls.append(("play", scene["index"], seed))
        return {"scene_seed": scene["seed"], "success": scene["index"] == 4, "steps": 3, "error": None}

    obs = {"name": "low_dim", "cameras": [], "object_state": ["object"]}
    session.play_scenes(scenes, server={"host": "h", "port": 1}, checkpoint="/c", obs_cfg=obs, eval_seed=5, training_seed=2,
                        run_one=run_one, record=rows.append, client=Client)
    # the committed seed rule, spelt out: SeedSequence([evaluation.seed, training_seed, reset_index])
    seeds = [int(np.random.SeedSequence([5, 2, i]).generate_state(1)[0]) for i in (4, 1)]
    assert calls == [("connect", "ws://h:1"), ("play", 4, seeds[0]), ("play", 1, seeds[1]), ("close",)]
    columns = ["reset_index", "split", "scene_seed", "success", "steps", "error", "policy_seed", "checkpoint", "seconds"]
    assert [list(r) for r in rows] == [columns] * 2
    assert rows[0]["reset_index"] == 4 and rows[0]["policy_seed"] == seeds[0] and rows[0]["checkpoint"] == "/c"
    assert rows[1]["success"] is False
    with pytest.raises(RuntimeError, match="another server"):
        session.play_scenes(scenes, server={"host": "h", "port": 1}, checkpoint="/other", obs_cfg=obs, eval_seed=5,
                            training_seed=2, run_one=run_one, record=rows.append, client=Client)


def test_env_flags_and_task_id_helpers():
    assert ENV_KWARGS == {"has_renderer": False, "has_offscreen_renderer": False, "use_camera_obs": False, "use_object_obs": True,
                          "ignore_done": True, "control_freq": 20, "reward_shaping": False}
    assert list(ENV_KWARGS) == ["has_renderer", "has_offscreen_renderer", "use_camera_obs", "use_object_obs", "ignore_done",
                                "control_freq", "reward_shaping"]           # json order of the stored env_kwargs attribute
    assert ROBOSUITE_EVAL_KWARGS == {"use_object_obs": True, "ignore_done": True, "control_freq": 20, "reward_shaping": False}
    assert is_isaaclab({"generator": {"simulator": "isaaclab"}}) and not is_isaaclab({"generator": {"name": "mimicgen"}})
    assert isaaclab_task_id({"env_name": "A", "camera_env": "B"}) == "B"
    assert isaaclab_task_id({"env_name": "A", "camera_env": None}) == "A"


def test_features_helpers():
    from tmg.core.features import camera_of, image_feature, is_image_key

    assert image_feature("agentview") == "observation.images.agentview"
    assert camera_of("observation.images.agentview") == "agentview"
    assert is_image_key("observation.images.x") and not is_image_key("observation.state")


def test_wilson_episode_seed_and_summary_keep_the_old_values():
    # Wilson 95% interval, z = 1.96 (hand-computed: p = 0.5, n = 10 -> centre 0.5, half width 0.26342)
    assert session.wilson(5, 10) == pytest.approx((0.23658, 0.76342), abs=1e-4)
    assert session.wilson(10, 10) == pytest.approx((0.72247, 1.0), abs=1e-4)
    assert session.wilson(0, 10) == pytest.approx((0.0, 0.27753), abs=1e-4)
    assert session.wilson(0, 0) == (0.0, 0.0)
    # policy seeds the committed evaluator produced (evaluation.seed 202600, training seed 5, reset indices 0..2)
    assert [session.episode_seed(202600, 5, i) for i in range(3)] == [3655276531, 1001985337, 2855745476]
    rows = [{"split": "val", "success": True, "steps": 3, "error": None}, {"split": "val", "success": False, "steps": 600, "error": None},
            {"split": "test", "success": False, "steps": 0, "error": "boom"}]
    s = session.summarize(rows)
    assert set(s) == {"all", "by_split"} and list(s["by_split"]) == ["test", "val"]
    assert s["all"] == {"n": 3, "n_success": 1, "success_rate": 1 / 3, "ci_low": pytest.approx(0.06149, abs=1e-4),
                        "ci_high": pytest.approx(0.79232, abs=1e-4), "mean_steps": 201.0, "errors": 1}
    assert s["by_split"]["test"]["success_rate"] == 0.0 and s["by_split"]["val"]["n_success"] == 1
    assert session.summarize([])["all"] == {"n": 0, "n_success": 0, "success_rate": None, "ci_low": 0.0, "ci_high": 0.0,
                                            "mean_steps": None, "errors": 0}
    assert session.server_url({"host": "127.0.0.1", "port": 8765}) == "ws://127.0.0.1:8765"
