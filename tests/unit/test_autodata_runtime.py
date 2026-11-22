"""Characterization of the AutoData runtime callbacks, without torch or Isaac Sim.

Uses the fake torch and tensor stand-ins of ``test_isaaclab_runtime``; the fake ``autodata_core`` modules below carry the
names the grafts read (``GenerationResult``, ``_optional_lock``, ``Waypoint``, ``MultiWaypoint``).
"""
import json
import sys
import types

import numpy as np
import pytest

from tests.unit.test_isaaclab_runtime import Method0, RecordingMixer, T, fake_torch, pose
from tmg.generation.generators.isaaclab.runtime import IsaacLabLayout


@pytest.fixture
def torch(monkeypatch):
    m = fake_torch()
    monkeypatch.setitem(sys.modules, "torch", m)
    return m


def dex_layout():
    return IsaacLabLayout(eefs=["left", "right"], arm_action_dims={"left": (0, 7), "right": (7, 14)}, hand_prefixes=None,
                          hand_term=None, measure_links=None, noise_draw_shapes=(3, 4, 3, 4))


def make_runtime(method, records_dir=None, cameras=()):
    from tmg.generation.generators.autodata.runtime import AutoDataRuntime

    return AutoDataRuntime(method, dex_layout(), records_dir=records_dir, cameras=cameras, image_size=None)


def datagen_info(num_steps, offset, eefs=("left", "right"), obj="object"):
    """One AutoData source demonstration: hand commands are the eef's own passthrough channel, plus a ``base`` channel."""
    poses = {e: T(np.stack([pose(offset + i + 50 * k) for i in range(num_steps)])) for k, e in enumerate(eefs)}
    return types.SimpleNamespace(eef_pose=poses, target_eef_pose={e: T(p.a + 0.5) for e, p in poses.items()},
                                 object_poses={obj: T(np.stack([pose(100 + offset + i) for i in range(num_steps)]))},
                                 passthrough_action={**{e: T(np.full((num_steps, 2), offset)) for e in eefs}, "base": T(np.zeros((num_steps, 3)))})


def dex_locals(infos, primary=0, with_base=True):
    bounds = T(np.array([[[0, 4], [4, 6]]] * len(infos)))
    info, start, end = infos[primary], 0, 4
    passthrough = {"right": info.passthrough_action["right"][start:end]}
    if with_base:
        passthrough["base"] = info.passthrough_action["base"][start:end]
    return {"src_subtask_eef_poses": info.eef_pose["right"][start:end], "src_subtask_target_poses": info.target_eef_pose["right"][start:end],
            "src_subtask_passthrough_actions": passthrough, "eef_name": "right", "subtask_ind": 0, "selected_src_demo_ind": primary,
            "all_randomized_subtask_boundaries": {"right": bounds}, "subtask_object_name": "object"}


class FakeDatastream:
    """``target_eef_pose_to_action`` (passthrough signature), the environment and the observed eef poses."""

    def __init__(self, torch, eefs=("left", "right"), width=14):
        self.torch, self.eefs, self.width, self.calls = torch, list(eefs), width, []
        self.last_target = {}
        self.env = types.SimpleNamespace(obs_buf={"policy": {}, "camera_obs": {"cam": T(np.full((1, 4, 4, 3), 0.5))}}, device="cpu")
        self.device = "cpu"

    def get_env(self):
        return self.env

    def target_eef_pose_to_action(self, target_eef_pose_dict, passthrough_action_dict, action_noise_dict=None, env_id=0):
        self.calls.append((dict(target_eef_pose_dict), action_noise_dict, env_id,
                           {e: {k: v.a.tolist() for k, v in channels.items()} for e, channels in passthrough_action_dict.items()}))
        action = np.zeros(self.width)
        for k, e in enumerate(self.eefs):
            if e in target_eef_pose_dict:
                action[7 * k] = float(target_eef_pose_dict[e].a[0, 3])
        if action_noise_dict is not None:          # as AutoData: pos then quat draw per hand, left then right
            for k, _ in enumerate(self.eefs):
                draw = np.concatenate([self.torch.randn_like(T(np.zeros(3))).a, self.torch.randn_like(T(np.zeros(4))).a])
                action[7 * k:7 * k + 7] += draw * 0.001
        return T(action)

    def get_robot_eef_pose(self, env_ids, eef_name):
        return T(self.last_target.get(eef_name, np.eye(4)))[None]


# source mixing ---------------------------------------------------------------------------------------------
def test_mix_segment_keeps_other_passthrough_channels_next_to_the_mixed_hand(torch):
    infos = [datagen_info(6, 0), datagen_info(6, 10)]
    local_vars = dex_locals(infos, primary=1)
    mixer = RecordingMixer()
    rt = make_runtime(Method0(mixer))
    gen = types.SimpleNamespace(src_demo_datagen_info_pool=types.SimpleNamespace(datagen_infos=infos))
    eef, targets, passthrough = rt.mix_segment(gen, local_vars)
    ctx = mixer.contexts[0]
    assert ctx.primary == 1 and ctx.extra == {"eef": "right", "soft": False, "reference_object": "object"} and ctx.rng.random() == np.random.default_rng(0).random()
    np.testing.assert_array_equal(ctx.primary_gripper, np.full((4, 2), 10))
    np.testing.assert_array_equal(ctx.primary_eef_start, pose(60))
    np.testing.assert_array_equal(ctx.primary_targets, infos[1].target_eef_pose["right"].a[0:4])
    assert ctx.src_segments[0][2].shape == (4, 2) and ctx.src_segments[0][2][0, 0] == 0.0 and ctx.src_object_poses[1][0, 3] == 110.0
    assert sorted(passthrough) == ["base", "right"] and passthrough["base"] is local_vars["src_subtask_passthrough_actions"]["base"]
    assert passthrough["right"].a.tolist() == [[0.5]] * 4 and targets.a[:, 0, 3].tolist() == [7.0] * 4
    np.testing.assert_array_equal(eef.a[0], np.eye(4) * 2)
    np.testing.assert_array_equal(eef.a[1:], infos[1].eef_pose["right"].a[1:4])
    assert eef.dtype == targets.dtype == passthrough["right"].dtype == "float32"
    assert rt._last_mix == {"mixed": True, "donor": 1}


def test_mix_segment_without_a_mixer_or_without_a_mix_returns_the_passthrough_dict(torch):
    infos = [datagen_info(6, 0), datagen_info(6, 10)]
    gen = types.SimpleNamespace(src_demo_datagen_info_pool=types.SimpleNamespace(datagen_infos=infos))
    for method in (Method0(None), Method0(RecordingMixer(mixed=False))):
        local_vars = dex_locals(infos)
        eef, targets, passthrough = make_runtime(method).mix_segment(gen, local_vars)
        assert eef is local_vars["src_subtask_eef_poses"] and targets is local_vars["src_subtask_target_poses"]
        assert passthrough == local_vars["src_subtask_passthrough_actions"] and passthrough is not local_vars["src_subtask_passthrough_actions"]


def test_mix_segment_refuses_other_channels_of_another_length(torch):
    infos = [datagen_info(6, 0), datagen_info(6, 10)]
    local_vars = dex_locals(infos)
    local_vars["src_subtask_passthrough_actions"]["base"] = T(np.zeros((3, 3)))
    gen = types.SimpleNamespace(src_demo_datagen_info_pool=types.SimpleNamespace(datagen_infos=infos))
    with pytest.raises(NotImplementedError):
        make_runtime(Method0(RecordingMixer())).mix_segment(gen, local_vars)


# attempt records --------------------------------------------------------------------------------------------
def test_attempt_records_labels_frames_subtasks_and_resets_the_success_term(torch, tmp_path):
    rt = make_runtime(Method0(), records_dir=tmp_path, cameras=["cam"])
    ds = FakeDatastream(torch)
    gen = types.SimpleNamespace(datastream=ds)
    resets = []
    term = types.SimpleNamespace(func=types.SimpleNamespace(reset=lambda env_ids: resets.append(env_ids)))
    rt.begin_attempt(gen, 0, term)
    assert resets == [[0]]
    spec = types.SimpleNamespace(num_interpolation_steps=3, action_noise=0.02)      # no num_fixed_steps: defaults to 0
    rt.record_subtask({"self": gen, "eef_name": "right", "subtask_ind": 0, "selected_src_demo_ind": 1, "selected_boundary": T([2, 8]),
                       "subtasks": [spec], "subtask_object_pose": T(pose(1.0)), "subtask_object_name": "object"})
    W = types.SimpleNamespace
    mw = types.SimpleNamespace(waypoints={"left": W(pose=T(pose(1.0)), passthrough_action={"left": T([1.0, 0.0]), "base": T([0.0])}, noise=0.01),
                                          "right": W(pose=T(pose(2.0)), passthrough_action={"right": T([0.0, 1.0])}, noise=0.02)})
    action = ds.target_eef_pose_to_action({e: w.pose for e, w in mw.waypoints.items()}, {}, {"left": 0.01}, 0)
    rt.record_command(mw, ds, 0, action)
    assert len(rt._frames["cam"]) == 1 and rt._frames["cam"][0].dtype == np.uint8 and rt._frames["cam"][0][0, 0, 0] == 127
    rt.record_step(mw, ds.get_env(), 0, None, True)
    rt.end_attempt(gen, 0, {"success": True})
    line = json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[0])
    assert line["src_demo_inds"] == [1] and line["n_steps"] == 1 and line["eef_names"] == ["left", "right"] and line["term_first"] == {}
    sub = line["subtasks"][0]
    assert (sub["src_subtask_start"], sub["src_subtask_end"], sub["num_interpolation_steps"], sub["num_fixed_steps"], sub["action_noise"]) == (2, 8, 3, 0, 0.02)
    rec = np.load(tmp_path / "attempt_000000.npz")
    assert rec["a_pre"].shape == (1, 14) and rec["a_pre"][0, 0] == 1.0 and rec["a_pre"][0, 7] == 2.0 and rec["noise"].tolist() == [np.float32(0.02)]
    assert rec["a_label"][0, 1] == 0.0 and rec["a_exec"][0, 1] != 0.0                     # the label is the noiseless action
    images = np.load(tmp_path / "images_000000.npz")
    assert images["cam"].shape == (1, 4, 4, 3) and rt._frames == {}
    assert ds.target_eef_pose_to_action.__func__ is FakeDatastream.target_eef_pose_to_action    # wrapper restored
    assert np.load(tmp_path / "labels_000000.npz")["actions_pre_noise"].shape == (1, 14)


def test_selected_sources_are_reset_at_every_attempt(torch, tmp_path):
    rt = make_runtime(Method0(), records_dir=tmp_path)
    gen = types.SimpleNamespace(datastream=FakeDatastream(torch))
    spec = types.SimpleNamespace(num_interpolation_steps=1, num_fixed_steps=0, action_noise=0.0)

    def subtask(eef, source):
        rt.record_subtask({"self": gen, "eef_name": eef, "subtask_ind": 0, "selected_src_demo_ind": source, "selected_boundary": T([0, 2]),
                           "subtasks": [spec], "subtask_object_name": None})

    rt.begin_attempt(gen, 0)
    subtask("left", 4)
    subtask("right", 5)
    rt.end_attempt(gen, 0, {"success": True})
    rt.begin_attempt(gen, 0)
    subtask("left", 6)
    rt.end_attempt(gen, 0, {"success": False})
    lines = [json.loads(s) for s in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [line["src_demo_inds"] for line in lines] == [[4, 5], [6]]


def test_end_attempt_writes_no_images_for_a_failed_attempt(torch, tmp_path):
    rt = make_runtime(Method0(), records_dir=tmp_path, cameras=["cam"])
    ds = FakeDatastream(torch)
    rt.begin_attempt(types.SimpleNamespace(datastream=ds), 0)
    rt.capture_frames(ds.get_env(), 0)
    rt.end_attempt(types.SimpleNamespace(datastream=ds), 0, {"success": False})
    assert not (tmp_path / "images_000000.npz").exists() and (tmp_path / "attempts.jsonl").exists()


def test_begin_attempt_discards_frames_of_an_aborted_attempt(torch):
    rt = make_runtime(Method0(), cameras=["cam"])
    ds = FakeDatastream(torch)
    rt.begin_attempt(types.SimpleNamespace(datastream=ds), 0)
    rt.capture_frames(ds.get_env(), 0)
    rt._restore_label_wrapper()                       # the attempt died before end_attempt (as the tracking loop's except does)
    rt.begin_attempt(types.SimpleNamespace(datastream=ds), 0)
    assert rt._frames == {"cam": []} and rt._labels == [] and rt._selected == {}


def test_measured_poses_prefer_the_configured_links(torch):
    rt = make_runtime(Method0())
    ds = FakeDatastream(torch)
    ds.last_target = {"left": pose(1.0), "right": pose(2.0)}
    out = rt.measured_poses(ds, 0, ["left", "right"])
    assert out["left"][0, 3] == 1.0 and out["right"][0, 3] == 2.0

    class Scene(dict):
        env_origins = T([[0, 0, 1]])

    robot = types.SimpleNamespace(data=types.SimpleNamespace(body_names=["l_link"], body_pos_w=T([[[1, 2, 3]]]), body_quat_w=T([[[1, 0, 0, 0]]])))
    ds.env = types.SimpleNamespace(scene=Scene(robot=robot))
    rt.layout.measure_links = {"left": "l_link"}
    assert rt.measured_poses(ds, 0, ["left"])["left"][:3, 3].tolist() == [1.0, 2.0, 2.0]       # link pose minus the environment origin


# grafts -----------------------------------------------------------------------------------------------------
AUTODATA_PACKAGE = '''
class GenerationResult:
    def __init__(self, initial_state, success):
        self.initial_state, self.success = initial_state, success


def _optional_lock(lock):
    return lock


class DataGenerator:
    async def generate(self, env_id, success_term, *, env_reset_queue, env_action_queue, pause_subtask=False, export_demo=True):
        env_id_tensor, initial_state = await self._reset_environment_for_generation(env_id=env_id, env_reset_queue=env_reset_queue)
        buffers = Buffers()
        while True:
            plan = self.plan()
            if plan is None:
                return GenerationResult(initial_state=initial_state, success=False)
            if buffers.done:
                break
        return GenerationResult(initial_state=initial_state, success=buffers.success)

    def generate_eef_subtask_trajectory(self, env_id, eef_name, subtask_ind, boundaries, pose_overrides, selected):
        src_subtask_eef_poses = 1
        src_subtask_target_poses = 2
        src_subtask_passthrough_actions = {}
        src_subtask_object_pose = None
        seq = WaypointSequence.from_poses(src_subtask_eef_poses, src_subtask_target_poses, src_subtask_passthrough_actions)
        return seq
'''
AUTODATA_WAYPOINT = '''
class Waypoint:
    pass


class MultiWaypoint:
    async def execute(self, *, datastream, success_term, env_id, env_action_queue):
        env = datastream.get_env()
        play_action = datastream.target_eef_pose_to_action(target_eef_pose_dict={}, passthrough_action_dict={}, action_noise_dict={}, env_id=env_id)
        env.step(play_action)
        return bool(success_term.func(env, **success_term.params)[env_id])
'''


def test_install_from_spec_grafts_autodata_and_wraps_the_success_term_setup(tmp_path, monkeypatch):
    from tests.unit.test_isaaclab_runtime import fake_package
    from tmg.generation.generators.autodata import runtime as rt

    dg, _wp = fake_package(tmp_path, monkeypatch, "autodata_core", AUTODATA_PACKAGE, AUTODATA_WAYPOINT)
    spec = {"chunk": {"seed": 31}, "records_dir": str(tmp_path / "records"), "task": {},
            "method": {"name": "none", "mixing": {"enabled": False}, "tracking": {"enabled": False}},
            "extra": {"soft": False, "package": "autodata_core", "cameras": ["cam"], "image_size": [8, 8], "graft_hashes": {},
                      "layout": {"eefs": ["left", "right"], "arm_action_dims": {"left": [0, 7], "right": [7, 14]}, "hand_prefixes": None,
                                 "hand_term": None, "measure_links": None, "noise_draw_shapes": [3, 4, 3, 4]}}}
    seen = []
    ns = {"setup_async_generation": lambda datastream, x, success_term: seen.append((datastream, success_term))}
    info = rt.install_from_spec(spec, ns)
    assert set(info) == {"generate", "generate_eef_subtask_trajectory", "execute", "hashes"}
    assert info["generate"] == ["after 'initial_state = ...': _tmg.begin_attempt(self, env_id, success_term)",
                                'before return #0: _tmg.end_attempt(self, env_id, {"success": False})',
                                'before return #1: _tmg.end_attempt(self, env_id, {"success": buffers.success})']
    assert info["generate_eef_subtask_trajectory"] == [
        "after 'src_subtask_object_pose = ...': src_subtask_eef_poses, src_subtask_target_poses, src_subtask_passthrough_actions = _tmg.mix_segment(self, locals())",
        "after 'seq = ...': _tmg.record_subtask(locals())"]
    assert info["execute"] == ["after 'play_action = ...': _tmg.record_command(self, datastream, env_id, play_action)",
                               "final return value bound to 'success'", "before final return: _tmg.record_step(self, env, env_id, None, success)"]
    assert dg.DataGenerator.generate.__name__ == "generate" and len(info["hashes"]) == 3
    runtime = dg.DataGenerator.generate_eef_subtask_trajectory.__globals__["_tmg"]
    assert type(runtime) is rt.AutoDataRuntime and runtime.cameras == ["cam"] and runtime.image_size == (8, 8)
    assert runtime.layout.arm_action_dims == {"left": (0, 7), "right": (7, 14)} and runtime.layout.noise_draw_shapes == (3, 4, 3, 4)
    assert dg.DataGenerator.generate.__globals__["_tmg"] is runtime          # the official generate, grafted

    class Cfg:
        func = type("Term", (), {"__init__": lambda self, cfg, env: setattr(self, "env", env)})
        params = {}  # noqa: RUF012  test stub

    cfg = Cfg()
    ds = types.SimpleNamespace(get_env=lambda: "E")
    ns["setup_async_generation"](ds, 0, cfg)
    assert seen == [(ds, cfg)] and cfg.func.env == "E"
