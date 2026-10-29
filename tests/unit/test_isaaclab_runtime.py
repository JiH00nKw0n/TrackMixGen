"""Characterization of the Isaac Lab runtime callbacks and the shared-path tracking loop, without torch or Isaac Sim.

A fake ``torch`` module (numpy underneath) stands in for the real one, so ``mix_segment``, the pre-noise label wrapper,
the noise ledger and the whole tracking loop run on plain numpy arrays. The fake generator/environment objects below
mirror the attributes the callbacks read from the official classes (``softmimicgen.datagen`` shape).
"""
import asyncio
import sys
import types

import numpy as np
import pytest

from tmg.generation.method.interfaces import Method, MixedSegment
from tmg.generation.method.tracking import ErrorAdaptiveFollower


# fake torch -----------------------------------------------------------------------------------------
class T:
    """A tensor stand-in over a numpy array with the few methods the runtime touches."""

    def __init__(self, a, dtype="float32", device="cpu"):
        self.a = np.array(a, dtype=float)
        self.dtype, self.device = dtype, device

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.a

    def clone(self):
        return T(self.a.copy(), self.dtype, self.device)

    def __getitem__(self, key):
        return T(self.a[key], self.dtype, self.device)

    def __len__(self):
        return len(self.a)

    def __bool__(self):
        return bool(self.a.reshape(-1)[0]) if self.a.size == 1 else bool(self.a.any())

    @property
    def shape(self):
        return self.a.shape

    @property
    def ndim(self):
        return self.a.ndim


def fake_torch(seed=0):
    """A module named ``torch`` whose random draws come from one numpy generator (the "global" stream)."""
    m = types.ModuleType("torch")
    m.global_rng = np.random.default_rng(seed)
    m.rng_state = [0]            # a counter standing in for the CPU RNG state: every draw advances it
    m.int64, m.bool = "int64", "bool"

    class Generator:
        def __init__(self, device=None):
            self.rng = None

        def manual_seed(self, seed):
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)

    def randn(k, device=None, dtype=None, generator=None):
        if generator is None:
            m.rng_state[0] += 1
            return T(m.global_rng.standard_normal(k), dtype or "float32", device or "cpu")
        return T(generator.rng.standard_normal(k), dtype or "float32", device or "cpu")

    def randn_like(x):
        m.rng_state[0] += 1
        return T(m.global_rng.standard_normal(x.shape), x.dtype, x.device)

    m.Generator, m.randn, m.randn_like = Generator, randn, randn_like
    m.as_tensor = lambda x, dtype=None, device=None: T(np.asarray(x), dtype or "float32", device or "cpu")
    m.tensor = lambda x, dtype=None, device=None: T(np.asarray(x), dtype or "float32", device or "cpu")
    m.cat = lambda ts, dim=0: T(np.concatenate([t.a for t in ts], axis=dim), ts[0].dtype, ts[0].device)
    def get_rng_state():
        return (m.rng_state[0], m.global_rng.bit_generator.state)

    def set_rng_state(state):
        m.rng_state[0] = state[0]
        m.global_rng.bit_generator.state = state[1]

    m.random = types.SimpleNamespace(get_rng_state=get_rng_state, set_rng_state=set_rng_state)
    m.cuda = types.SimpleNamespace(is_available=lambda: False, get_rng_state_all=lambda: None, set_rng_state_all=lambda s: None)
    return m


@pytest.fixture
def torch(monkeypatch):
    m = fake_torch()
    monkeypatch.setitem(sys.modules, "torch", m)
    return m


# fake generator pieces -------------------------------------------------------------------------------
class Method0:
    """A method with a recording mixer (or none) and executed labels."""

    labels = "executed"
    follower_factory = None

    def __init__(self, mixer=None):
        self.mixer = mixer


class RecordingMixer:
    """Returns a fixed mixed segment and keeps the context it was given."""

    def __init__(self, mixed=True):
        self.mixed, self.contexts = mixed, []

    def mix(self, context):
        self.contexts.append(context)
        n = len(context.primary_targets)
        targets = np.tile(np.eye(4), (n, 1, 1))
        targets[:, 0, 3] = 7.0
        return MixedSegment(eef_start=np.eye(4) * 2, targets=targets, gripper=np.full((n, 1), 0.5),
                            record={"mixed": self.mixed, "donor": 1})


def pose(x):
    p = np.eye(4)
    p[0, 3] = x
    return p


def datagen_info(num_steps, offset, eef="robot0", obj="rope", hand="gripper_action", hand_width=1, nodal=False):
    """One source demonstration as the datagen info pool holds it (tensor-like fields per end effector / object)."""
    poses = T(np.stack([pose(offset + i) for i in range(num_steps)]))
    info = types.SimpleNamespace(eef_pose={eef: poses}, target_eef_pose={eef: T(poses.a + 0.5)},
                                 object_poses={obj: T(np.stack([pose(100 + offset + i) for i in range(num_steps)]))})
    setattr(info, hand, {eef: T(np.full((num_steps, hand_width), offset))})
    if nodal:
        info.object_nodal_positions = {obj: T(np.full((num_steps, 3, 3), offset))}
    return info


def soft_layout():
    from tmg.generation.generators.isaaclab.runtime import IsaacLabLayout

    return IsaacLabLayout(eefs=["robot0"], arm_action_dims={"robot0": (0, 6)}, hand_prefixes=None, hand_term=None,
                          measure_links=None, noise_draw_shapes=(6,))


def soft_locals(infos, primary=0, subtask=0, soft=False):
    bounds = T(np.array([[[0, 4], [4, 6]]] * len(infos)))
    start, end = 0, 4
    info = infos[primary]
    local_vars = {"src_subtask_eef_poses": info.eef_pose["robot0"][start:end], "src_subtask_target_poses": info.target_eef_pose["robot0"][start:end],
                  "src_subtask_gripper_actions": info.gripper_action["robot0"][start:end], "eef_name": "robot0", "subtask_ind": subtask,
                  "selected_src_demo_ind": primary, "all_randomized_subtask_boundaries": {"robot0": bounds}, "subtask_object_name": "rope",
                  "subtask_object_soft": soft}
    if soft:
        local_vars["src_subtask_nodal_positions"] = info.object_nodal_positions["rope"][start]
    return local_vars


def make_runtime(method, records_dir=None):
    from tmg.generation.generators.isaaclab.runtime import SoftMimicGenRuntime

    return SoftMimicGenRuntime(method, soft_layout(), records_dir=records_dir)


# mix_segment ------------------------------------------------------------------------------------------
def test_mix_segment_without_a_mixer_returns_the_inputs_untouched(torch):
    infos = [datagen_info(6, 0), datagen_info(6, 10)]
    local_vars = soft_locals(infos)
    rt = make_runtime(Method0(mixer=None))
    out = rt.mix_segment(types.SimpleNamespace(src_demo_datagen_info_pool=types.SimpleNamespace(datagen_infos=infos)), local_vars)
    assert out[0] is local_vars["src_subtask_eef_poses"] and out[1] is local_vars["src_subtask_target_poses"]
    assert out[2] is local_vars["src_subtask_gripper_actions"] and rt._last_mix is None


def test_mix_segment_hands_the_mixer_every_source_and_repacks_the_mixed_segment(torch):
    infos = [datagen_info(6, 0), datagen_info(6, 10)]
    local_vars = soft_locals(infos, primary=1)
    mixer = RecordingMixer()
    rt = make_runtime(Method0(mixer))
    gen = types.SimpleNamespace(src_demo_datagen_info_pool=types.SimpleNamespace(datagen_infos=infos))
    eef, targets, grippers = rt.mix_segment(gen, local_vars)
    ctx = mixer.contexts[0]
    assert ctx.subtask_ind == 0 and ctx.primary == 1 and ctx.num_sources == 2 and ctx.has_reference_object is True
    assert ctx.extra == {"eef": "robot0", "soft": False}
    assert ctx.rng.random() == np.random.default_rng(0).random()                     # the mixer's stream is always default_rng(0)
    np.testing.assert_array_equal(ctx.primary_eef_start, pose(10))
    np.testing.assert_array_equal(ctx.primary_targets, infos[1].target_eef_pose["robot0"].a[0:4])
    np.testing.assert_array_equal(ctx.primary_gripper, np.full((4, 1), 10))
    assert len(ctx.src_segments) == 2 and ctx.src_segments[0][1].shape == (4, 4, 4) and ctx.src_segments[0][0][0, 3] == 0.0
    np.testing.assert_array_equal(ctx.src_object_poses[0], pose(100))
    np.testing.assert_array_equal(ctx.src_object_poses[1], pose(110))
    np.testing.assert_array_equal(ctx.src_segments[1][2], np.full((4, 1), 10))
    assert rt._last_mix == {"mixed": True, "donor": 1}
    # the mixed start replaces the first eef pose, the rest of the eef poses are kept; targets and grippers are the mixer's
    np.testing.assert_array_equal(eef.a[0], np.eye(4) * 2)
    np.testing.assert_array_equal(eef.a[1:], infos[1].eef_pose["robot0"].a[1:4])
    assert eef.dtype == targets.dtype == "float32" and targets.a[:, 0, 3].tolist() == [7.0] * 4 and grippers.a.tolist() == [[0.5]] * 4


def test_mix_segment_keeps_the_primary_when_the_mixer_declines(torch):
    infos = [datagen_info(6, 0), datagen_info(6, 10)]
    local_vars = soft_locals(infos)
    rt = make_runtime(Method0(RecordingMixer(mixed=False)))
    gen = types.SimpleNamespace(src_demo_datagen_info_pool=types.SimpleNamespace(datagen_infos=infos))
    out = rt.mix_segment(gen, local_vars)
    assert out[0] is local_vars["src_subtask_eef_poses"] and out[2] is local_vars["src_subtask_gripper_actions"]
    assert rt._last_mix == {"mixed": False, "donor": 1}


def test_mix_segment_registers_donor_segments_into_the_primary_rope(torch, monkeypatch):
    """Soft objects: donors go through the official nodal registration, the primary does not, object frames are identity."""
    calls = []

    def register(*, tgt_obj_nodal_pos, src_eef_poses, src_obj_nodal_pos, use_rotation_transform, bend_coef, rot_coef):
        calls.append((float(tgt_obj_nodal_pos.a[0, 0]), float(src_obj_nodal_pos.a[0, 0]), use_rotation_transform, bend_coef, rot_coef))
        return T(src_eef_poses.a + 1000)

    for name in ("softmimicgen", "softmimicgen.datagen"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    dg = types.ModuleType("softmimicgen.datagen.data_generator")
    dg.transform_source_data_segment_using_nodal_registration = register
    monkeypatch.setitem(sys.modules, "softmimicgen.datagen.data_generator", dg)
    infos = [datagen_info(6, 0, nodal=True), datagen_info(6, 10, nodal=True), datagen_info(6, 20, nodal=True)]
    mixer = RecordingMixer()
    rt = make_runtime(Method0(mixer))
    gen = types.SimpleNamespace(src_demo_datagen_info_pool=types.SimpleNamespace(datagen_infos=infos))
    rt.mix_segment(gen, soft_locals(infos, primary=1, soft=True))
    ctx = mixer.contexts[0]
    assert calls == [(10.0, 0.0, True, 0.1, 1e-3), (10.0, 20.0, True, 0.1, 1e-3)]
    assert ctx.extra == {"eef": "robot0", "soft": True}
    assert all(np.array_equal(p, np.eye(4)) for p in ctx.src_object_poses)
    assert ctx.src_segments[0][1][0, 0, 3] == 1000.5 and ctx.src_segments[1][1][0, 0, 3] == 10.5 and ctx.src_segments[2][1][0, 0, 3] == 1020.5


# pre-noise labels --------------------------------------------------------------------------------------
class FakeEnv:
    """Only ``target_eef_pose_to_action`` (draws noise when a noise dict is given) and the observed eef pose."""

    def __init__(self, torch, width=6):
        self.torch, self.width, self.calls = torch, width, []
        self.last_target = {}

    def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=0):
        self.calls.append((dict(target_eef_pose_dict), action_noise_dict, env_id))
        action = np.zeros(self.width)
        action[0] = sum(float(v.a[0, 3]) for v in target_eef_pose_dict.values())
        action[-1] = float(next(iter(gripper_action_dict.values())).a.reshape(-1)[0])
        if action_noise_dict is not None:
            action = action + self.torch.randn_like(T(np.zeros(self.width))).a * 0.001
        return T(action)

    def get_robot_eef_pose(self, eef_name, env_ids):
        return T(self.last_target.get(eef_name, np.eye(4)))[None]


def test_label_wrapper_records_the_noiseless_action_without_moving_the_official_stream(torch):
    rt = make_runtime(Method0())
    env = FakeEnv(torch)
    gen = types.SimpleNamespace(env=env)
    rt.begin_attempt(gen, 0)
    target = {"robot0": T(pose(3.0))}
    out = env.target_eef_pose_to_action(target_eef_pose_dict=target, gripper_action_dict={"robot0": T([1.0])},
                                        action_noise_dict={"robot0": 0.01}, env_id=0)
    assert len(env.calls) == 2 and env.calls[0][1] is None and env.calls[1][1] == {"robot0": 0.01}
    assert env.calls[0][0]["robot0"] is not target["robot0"] and env.calls[1][0]["robot0"] is target["robot0"]   # clean call sees a clone
    assert torch.rng_state[0] == 1                                     # exactly one official draw happened (the noisy call)
    assert rt._labels[0].tolist() == [3.0, 0, 0, 0, 0, 1.0] and rt._executed[0][0] != 3.0 and rt._executed[0] is not out
    np.testing.assert_array_equal(rt._executed[0], out.a)
    rt._restore_label_wrapper()
    assert env.target_eef_pose_to_action.__func__ is FakeEnv.target_eef_pose_to_action


class DrawingEnv(FakeEnv):
    """An action conversion that draws its noise even when no noise dict is given (and scales it by zero)."""

    def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=0):
        self.calls.append((dict(target_eef_pose_dict), action_noise_dict, env_id))
        action = np.zeros(self.width)
        action[0] = sum(float(v.a[0, 3]) for v in target_eef_pose_dict.values())
        action[-1] = float(next(iter(gripper_action_dict.values())).a.reshape(-1)[0])
        scale = 0.001 if action_noise_dict is not None else 0.0
        return T(action + self.torch.randn_like(T(np.zeros(self.width))).a * scale)


def test_label_wrapper_puts_the_rng_state_back_when_the_clean_call_draws(torch):
    rt = make_runtime(Method0())
    env = DrawingEnv(torch)
    rt.begin_attempt(types.SimpleNamespace(env=env), 0)
    first_draw = np.random.default_rng(0).standard_normal(6)      # what the official (unwrapped) call would have drawn
    out = env.target_eef_pose_to_action({"robot0": T(pose(3.0))}, {"robot0": T([1.0])}, {"robot0": 0.01}, 0)
    assert len(env.calls) == 2 and torch.rng_state[0] == 1          # the clean call's draw left no trace in the state
    assert rt._labels[0].tolist() == [3.0, 0, 0, 0, 0, 1.0]
    np.testing.assert_allclose((out.a - rt._labels[0]) / 0.001, first_draw, atol=1e-9)   # the noisy call drew the first value


def test_end_attempt_flushes_sources_subtasks_and_labels(torch, tmp_path):
    rt = make_runtime(Method0(), records_dir=tmp_path)
    env = FakeEnv(torch)
    gen = types.SimpleNamespace(env=env, env_cfg=types.SimpleNamespace(subtask_configs={"robot0": [
        types.SimpleNamespace(num_interpolation_steps=5, num_fixed_steps=2, action_noise=0.03)]}))
    rt.begin_attempt(gen, 0)
    rt.record_subtask({"self": gen, "eef_name": "robot0", "subtask_ind": 0, "selected_src_demo_ind": 2,
                       "selected_src_subtask_boundary": T([3, 9]), "subtask_object_pose": T(pose(1.0)), "subtask_object_name": "rope"})
    env.target_eef_pose_to_action({"robot0": T(pose(3.0))}, {"robot0": T([1.0])}, {"robot0": 0.01}, 0)
    W = types.SimpleNamespace
    mw = types.SimpleNamespace(waypoints={"robot0": W(pose=T(pose(3.0)), gripper_action=T([1.0]), noise=0.03)})
    rt.record_command(mw, env, 0, None, T(np.arange(6.0)))
    rt.record_step(mw, env, 0, {"subtask_terms": {"grasp": T([[1.0]])}}, False)
    rt.end_attempt(gen, 0, {"success": True})
    import json

    line = json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[0])
    assert line["success"] is True and line["src_demo_inds"] == [2] and line["n_steps"] == 1 and line["labels"] == "executed"
    assert line["num_label_steps"] == 1 and line["executed_steps"] == 1 and line["term_first"] == {"grasp": 0}
    sub = line["subtasks"][0]
    assert (sub["src_demo_ind"], sub["src_subtask_start"], sub["src_subtask_end"]) == (2, 3, 9)
    assert (sub["num_interpolation_steps"], sub["num_fixed_steps"], sub["action_noise"], sub["mix"]) == (5, 2, 0.03, None)
    labels = np.load(tmp_path / "labels_000000.npz")
    assert labels["actions_pre_noise"].shape == (1, 6) and labels["actions_pre_noise"][0, 0] == 3.0
    rec = np.load(tmp_path / "attempt_000000.npz")
    assert rec["a_pre"].tolist() == [[3.0, 0, 0, 0, 0, 1.0]] and rec["noise"].tolist() == [np.float32(0.03)] and rec["a_exec"][0, 1] == 1.0
    assert rec["w_pos"][0, 0].tolist() == [3.0, 0.0, 0.0] and rec["seg_kind"].tolist() == [1]


def test_selected_sources_are_reset_at_every_attempt(torch, tmp_path):
    """An attempt that records a subtask for one hand only must not inherit the other hand's choice from the attempt before."""
    import json

    rt = make_runtime(Method0(), records_dir=tmp_path)
    rt.layout.eefs = ["left", "right"]
    rt.layout.arm_action_dims = {"left": (0, 3), "right": (3, 6)}
    cfg = types.SimpleNamespace(num_interpolation_steps=1, num_fixed_steps=0, action_noise=0.0)
    gen = types.SimpleNamespace(env=FakeEnv(torch), env_cfg=types.SimpleNamespace(subtask_configs={"left": [cfg], "right": [cfg]}))

    def subtask(eef, source):
        rt.record_subtask({"self": gen, "eef_name": eef, "subtask_ind": 0, "selected_src_demo_ind": source,
                           "selected_src_subtask_boundary": T([0, 2]), "subtask_object_name": None})

    rt.begin_attempt(gen, 0)
    subtask("left", 4)
    subtask("right", 5)
    rt.end_attempt(gen, 0, {"success": True})
    rt.begin_attempt(gen, 0)
    subtask("right", 6)
    rt.end_attempt(gen, 0, {"success": False})
    lines = [json.loads(s) for s in (tmp_path / "attempts.jsonl").read_text().splitlines()]
    assert [line["src_demo_inds"] for line in lines] == [[4, 5], [6]]


# noise bookkeeping --------------------------------------------------------------------------------------
def test_noise_ledger_draws_every_bundle_up_front_and_private_draws_for_extra_steps(torch):
    from tmg.generation.generators.isaaclab.runtime import NoiseLedger, _wrap_randn_like

    expected = np.random.default_rng(0).standard_normal((3, 6))
    ledger = NoiseLedger(3, (6,), "cpu", "float32", private_seed=1000 + 7919)
    assert torch.rng_state[0] == 3 and np.array_equal(np.stack([b[0].a for b in ledger.original]), expected)
    assert ledger.bundle(1)[0].a.tolist() == expected[1].tolist()
    with pytest.raises(RuntimeError):
        ledger.bundle(1)
    with pytest.raises(RuntimeError):
        ledger.bundle(3)
    private = ledger.bundle(None)
    assert ledger.extra == 1 and np.array_equal(private[0].a, np.random.default_rng(8919).standard_normal(6)) and torch.rng_state[0] == 3
    before = torch.randn_like
    with _wrap_randn_like(torch, ledger.bundle(0)):
        v = torch.randn_like(T(np.zeros(6)))
        assert v.a.tolist() == expected[0].tolist()
        with pytest.raises(RuntimeError):
            torch.randn_like(T(np.zeros(6)))            # one draw per shape in the bundle
    assert torch.randn_like is before
    with pytest.raises(RuntimeError), _wrap_randn_like(torch, ledger.bundle(2)):
        torch.randn_like(T(np.zeros(5)))                # a different shape than the bundled draw
    assert torch.randn_like is before                   # restored although the block raised
    assert ledger.used == {0, 1, 2} and ledger.extra == 1
    from tmg.generation.generators.isaaclab.tracking_loop import replay_randn_like

    assert _wrap_randn_like is replay_randn_like


def test_cut_interval_builds_a_linear_bridge_for_the_hand_without_its_own(torch):
    from tmg.generation.generators.isaaclab.runtime import _cut_interval

    class W:
        def __init__(self, pose_, grip, noise):
            self.pose, self.gripper_action, self.noise = pose_, grip, noise

    def wps(n, start, noise=0.003):
        return [W(T(pose(start + i)), T([1.0]), noise) for i in range(n)]

    rot = np.eye(4)
    rot[:3, :3] = -np.eye(3)
    rot[0, 3] = 10.0
    # right: official bridge of 3 then source steps 6..10; left: no bridge of its own, previous waypoint at x=0, body starts at x=10
    active = {"right": {"raw": wps(7, 6), "begin": 6, "end": 10, "prefix": 3, "subtask": 1},
              "left": {"raw": [W(T(rot), T([1.0]), 0.0)] + wps(3, 11), "begin": 6, "end": 10, "prefix": 0, "subtask": 1}}
    interval = _cut_interval(active, ["left", "right"], 6, W, {"left": W(T(pose(0.0)), T([0.0]), 0.0)})
    assert interval.kinds == [0, 0, 0, 1, 1, 1, 1] and interval.source_indices == [6, 6, 6, 6, 7, 8, 9] and interval.subtasks == {"left": 1, "right": 1}
    bridge = interval.waypoints["left"][:3]
    assert [round(w.pose.a[0, 3], 3) for w in bridge] == [2.5, 5.0, 7.5]                 # positions interpolate linearly
    assert bridge[0].pose.a[0, 0] == 1.0 and bridge[1].pose.a[0, 0] == -1.0 and bridge[2].pose.a[0, 0] == -1.0   # rotation switches at t >= 0.5
    assert all(w.gripper_action.a.tolist() == [1.0] and w.noise == 0.0 for w in bridge) and bridge[0].gripper_action is not active["left"]["raw"][0].gripper_action
    assert interval.waypoints["right"][0] is active["right"]["raw"][0] and interval.waypoints["left"][3] is active["left"]["raw"][0]
    with pytest.raises(RuntimeError):
        _cut_interval(active, ["left", "right"], 6, W, {})       # nothing to bridge the left hand from
    from tmg.generation.generators.isaaclab.tracking_loop import cut_interval

    again = cut_interval(active, ["left", "right"], 6, W, {"left": W(T(pose(0.0)), T([0.0]), 0.0)}, lambda w: w.gripper_action.clone())
    assert (again.start, again.end, again.kinds, again.source_indices, again.subtasks) == (6, 10, interval.kinds, interval.source_indices, interval.subtasks)
    for e in ("left", "right"):
        assert [w.pose.a.tolist() for w in again.waypoints[e]] == [w.pose.a.tolist() for w in interval.waypoints[e]]
        assert [w.gripper_action.a.tolist() for w in again.waypoints[e]] == [w.gripper_action.a.tolist() for w in interval.waypoints[e]]


# the tracking loop on a fake Soft-shaped generator --------------------------------------------------------
TRACK_CFG = {"ds_min": 0.25, "regions": {"default": {"alpha_p": 12.0, "alpha_r": 1.2}, "approach": {"alpha_p": 50.0, "alpha_r": 5.0}},
             "approach_window": 2, "extra_steps_cap": "segment"}


class FakeQueue:
    async def put(self, item):
        pass

    async def join(self):
        pass


class SoftGenerator:
    """The attributes of SoftMimicGen's ``DataGenerator`` that the tracking loop reads; one source demonstration.

    ``subtasks[eef]`` lists ``(begin, end, interpolation, fixed)`` per subtask; the official builder would return the
    bridge (interpolation + fixed waypoints) followed by the transformed source steps ``begin..end``.
    """

    def __init__(self, torch, subtasks: dict, waypoint_cls, hooks: dict, measured_error=0.0):
        self.torch, self.W, self.err, self.hooks = torch, waypoint_cls, measured_error, hooks
        self.env = FakeEnv(torch)
        self.env.device = "cpu"
        self.env.recorder_manager = types.SimpleNamespace(calls=[])
        for name in ("reset", "set_success_to_episodes", "export_episodes"):
            setattr(self.env.recorder_manager, name, lambda *a, _n=name, **k: self.env.recorder_manager.calls.append((_n, a, k)))
        self.env.scene = types.SimpleNamespace(get_state=lambda is_relative: {"relative": is_relative})
        self.env_cfg = types.SimpleNamespace(subtask_configs={
            e: [types.SimpleNamespace(num_interpolation_steps=interp, num_fixed_steps=fixed) for _, _, interp, fixed in subs]
            for e, subs in subtasks.items()})
        self.bounds = {e: np.array([[[b, en] for b, en, _, _ in subs]]) for e, subs in subtasks.items()}
        self.src_demo_datagen_info_pool = types.SimpleNamespace(asyncio_lock=asyncio.Lock())
        self.built = []

    def randomize_subtask_boundaries(self):
        return self.bounds

    def generate_trajectory(self, env_id, eef_name, subtask_ind, boundaries, pose_overrides, selected, prev):
        selected[eef_name] = 0
        begin, end = boundaries[eef_name][0, subtask_ind]
        cfg = self.env_cfg.subtask_configs[eef_name][subtask_ind]
        self.built.append((eef_name, subtask_ind, prev is None))
        x0 = 1000 * (list(self.env_cfg.subtask_configs).index(eef_name) + 1)
        bridge = [self.W(T(pose(x0 + begin - 0.5)), T([1.0]), 0.0) for _ in range(cfg.num_interpolation_steps + cfg.num_fixed_steps)]
        cfg.action_noise = 0.003
        # the official builder carries the grafted ``_tmg.record_subtask(locals())`` after ``transformed_seq``
        self.hooks["runtime"].record_subtask({"self": self, "eef_name": eef_name, "subtask_ind": subtask_ind, "selected_src_demo_ind": 0,
                                              "selected_src_subtask_boundary": T([begin, end]), "subtask_object_name": None})
        return bridge + [self.W(T(pose(x0 + s)), T([1.0]), 0.003) for s in range(begin, end)]


def soft_waypoint_module(torch, hooks, log):
    """``Waypoint`` and ``MultiWaypoint`` as the official module has them; ``execute`` draws the noise like the official one and
    carries the two grafted ``_tmg`` calls (``hooks["runtime"]``)."""
    W = types.SimpleNamespace

    def Waypoint(pose_, gripper_action, noise):
        return W(pose=pose_, gripper_action=gripper_action, noise=noise)

    class MultiWaypoint:
        def __init__(self, waypoints):
            self.waypoints = waypoints

        async def execute(self, env, success_term, env_id, env_action_queue):
            if hooks.get("fail_at") == len(log) + 1:
                raise RuntimeError("the simulator died")
            env.last_target = {e: w.pose.a for e, w in self.waypoints.items()}
            noise = {e: w.noise for e, w in self.waypoints.items()}
            action = env.target_eef_pose_to_action(target_eef_pose_dict={e: w.pose for e, w in self.waypoints.items()},
                                                   gripper_action_dict={e: w.gripper_action for e, w in self.waypoints.items()},
                                                   action_noise_dict=noise, env_id=env_id)
            hooks["runtime"].record_command(self, env, env_id, None, action)
            log.append(action.a.copy())
            success = len(log) >= 6
            hooks["runtime"].record_step(self, env, env_id, {"subtask_terms": {"done": T([[float(success)]])}}, success)
            return {"states": [len(log)], "observations": [{}], "actions": [action[None]], "success": success}

    mod = types.ModuleType("tests.unit.waypoint")
    mod.Waypoint, mod.MultiWaypoint = Waypoint, MultiWaypoint
    return mod


def run_soft_tracking(torch, monkeypatch, tmp_path, subtasks, error=0.0, follower_factory=None, fail_at=None):
    from tmg.generation.generators.isaaclab import runtime as rt

    log, hooks = [], {"fail_at": fail_at}
    wp = soft_waypoint_module(torch, hooks, log)
    monkeypatch.setitem(sys.modules, "tests.unit.waypoint", wp)        # the loop imports ``<generator package>.waypoint``
    gen = SoftGenerator(torch, subtasks, wp.Waypoint, hooks, measured_error=error)
    if error:
        gen.env.get_robot_eef_pose = lambda eef_name, env_ids: T(gen.env.last_target.get(eef_name, np.eye(4)) + np.array([[0, 0, 0, error]] + [[0] * 4] * 3))[None]
    layout = soft_layout()
    layout.eefs = sorted(subtasks)
    layout.arm_action_dims = {e: (0, 6) for e in layout.eefs}
    runtime = rt.SoftMimicGenRuntime(Method(name="tracking", mixer=None, follower_factory=ErrorAdaptiveFollower.factory(TRACK_CFG)),
                                     layout, records_dir=tmp_path)
    hooks["runtime"] = runtime
    generate = rt.make_tracking_generate(runtime, follower_factory or ErrorAdaptiveFollower.factory(TRACK_CFG), private_seed=11 + 7919)
    if fail_at is not None:
        return gen, runtime, generate, log, None
    results = asyncio.run(generate(gen, 0, "success_term", env_reset_queue=FakeQueue(), env_action_queue=None))
    import json

    line = json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[0])
    return gen, runtime, results, log, line


def test_tracking_loop_replays_one_official_noise_bundle_per_base_waypoint(torch, monkeypatch, tmp_path):
    subtasks = {"robot0": [(0, 5, 1, 1), (5, 9, 2, 0)]}
    gen, _runtime, results, log, line = run_soft_tracking(torch, monkeypatch, tmp_path, subtasks)
    # 14 bundles are drawn up front: (5 - 0) + 1 + 2 for the first subtask, (9 - 5) + 0 + 2 for the second
    expected = np.random.default_rng(0).standard_normal((14, 6))
    # 7 + 6 steps at zero error, every step one base waypoint, consuming slots 0..12 in order
    assert len(log) == 13 and line["noise_slots_used"] == 13 and line["private_noise_steps"] == 0
    assert torch.rng_state[0] == 14                 # the ledger's up-front draws are the only global draws of the attempt
    used = np.stack([(a - np.array([x, 0, 0, 0, 0, 1.0])) / 0.001 for a, x in zip(log, [1000 - 0.5] * 2 + list(range(1000, 1005)) + [1004.5] * 2 + list(range(1005, 1009)))])
    np.testing.assert_allclose(used, expected[:13], atol=1e-6)
    assert gen.built == [("robot0", 0, True), ("robot0", 1, False)]
    assert results["success"] is True and results["initial_state"] == {"relative": True} and len(results["states"]) == 13
    assert results["actions"].a.shape == (13, 6)
    assert [c[0] for c in gen.env.recorder_manager.calls] == ["reset", "set_success_to_episodes", "export_episodes"]
    assert line["interval_start"] == 5 and line["interval_end"] == 9 and line["interval_subtasks"] == {"robot0": 1}
    assert line["num_label_steps"] == 13 and line["executed_steps"] == 13 and line["src_demo_inds"] == [0] and line["n_steps"] == 13
    assert [s["subtask_ind"] for s in line["subtasks"]] == [0, 1] and line["term_first"] == {"done": 5} and line["succ_first"] == 5
    rec = np.load(tmp_path / "attempt_000000.npz")
    assert rec["track_slot"].tolist() == [1] * 13 and rec["track_k"].tolist() == list(range(7)) + list(range(6))
    assert rec["seg_of_step"].tolist() == [0] * 7 + [1] * 6 and rec["seg_kind"].tolist() == [1, 1] and rec["seg_sub"].tolist() == [0, 1]
    assert rec["track_region"].tolist() == [2, 2, 2, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0]      # region of the waypoint before the step (bridge = 2)
    assert rec["e_pos"][:, 0, 0].tolist() == rec["w_pos"][:, 0, 0].tolist()              # the fake env sits on the commanded target
    assert gen.env.target_eef_pose_to_action.__func__ is FakeEnv.target_eef_pose_to_action    # wrapper restored


def test_tracking_loop_takes_private_draws_for_extra_steps_and_bridges_the_other_hand(torch, monkeypatch, tmp_path):
    subtasks = {"left": [(0, 4, 1, 1), (4, 8, 0, 0)], "right": [(0, 4, 1, 1), (4, 8, 2, 0)]}
    gen, runtime, _results, log, line = run_soft_tracking(torch, monkeypatch, tmp_path, subtasks, error=0.05)
    expected = np.random.default_rng(0).standard_normal((12, 6))
    # a 5 cm error slows the follower down: more steps than base waypoints; each base waypoint still consumes its official
    # bundle exactly once, the extra steps draw from the private stream seeded with chunk seed + 7919
    assert len(log) > 12 and line["noise_slots_used"] == 12 and line["private_noise_steps"] == len(log) - 12
    assert torch.rng_state[0] == 13                 # right hand: (4 + 1 + 1 + 1) + (4 + 2) base waypoints, the larger count
    assert gen.built == [("left", 0, True), ("right", 0, True), ("left", 1, False), ("right", 1, False)]
    assert line["interval_start"] == 4 and line["interval_subtasks"] == {"left": 1, "right": 1}
    rec = np.load(tmp_path / "attempt_000000.npz")
    assert rec["track_slot"].sum() == 12 and rec["track_k"].max() == 5          # second interval: bridge of 2 + 4 source steps
    assert len(rec["track_slot"]) == len(log) and line["src_demo_inds"] == [0, 0] and rec["eef_names"].tolist() == ["left", "right"]
    assert rec["w_pos"][-1].tolist() == [[1007.0, 0.0, 0.0], [2007.0, 0.0, 0.0]]       # both hands end on their last source step
    bridge_steps = (rec["seg_of_step"] == 1) & (rec["track_k"] < 2)                 # the left hand's linear bridge 1003 -> 1004
    assert bridge_steps.any() and np.all((rec["w_pos"][bridge_steps, 0, 0] > 1003.0) & (rec["w_pos"][bridge_steps, 0, 0] < 1004.0))
    # action[1:5] carries nothing but noise * 0.001: official steps replay the bundles in slot order (slots a step skips are
    # marked used without being drawn), private steps take the private stream of chunk seed + 7919 in order
    private = np.random.default_rng(11 + 7919)
    next_slot = 0
    for a, used_slot in zip(log, rec["track_slot"].tolist()):
        drawn = a[1:5] / 0.001
        if used_slot:
            while not np.allclose(drawn, expected[next_slot, 1:5], atol=1e-9):
                next_slot += 1
                assert next_slot < 12, "an official step used noise that is no bundle of the ledger"
            next_slot += 1
        else:
            np.testing.assert_allclose(drawn, private.standard_normal(6)[1:5], atol=1e-9)
    assert next_slot == 12 and (rec["track_slot"] == 0).sum() == line["private_noise_steps"] > 0
    assert runtime._original_action_fn is None


class TwoAtATime:
    """A follower that commands every second waypoint, so each step covers two base waypoints (two noise slots)."""

    started = []  # noqa: RUF012  (noises, kinds) of every path handed to a follower

    def start_path(self, paths, noises, kinds, previous_states):
        self.paths, self.n, self.k = {p.name: p for p in paths}, len(paths[0].poses), -1
        TwoAtATime.started.append((list(noises), list(kinds)))

    def finished(self):
        return self.k >= self.n - 1

    def next_target(self, measured):
        before, self.k = self.k, min(self.k + 2, self.n - 1)
        row = {"s_before": float(before), "s_after": float(self.k), "ds_proposed": 2.0, "ds_actual": float(self.k - before), "k": self.k,
               "region": "rest", "capped": False, "new_slots": list(range(before + 1, self.k + 1))}
        return ({e: p.poses[self.k] for e, p in self.paths.items()}, {e: p.grippers[self.k] for e, p in self.paths.items()}, 0.003, row)

    def final_gripper_states(self):
        return {}


def test_tracking_loop_marks_every_slot_a_step_covers_and_replays_the_first(torch, monkeypatch, tmp_path):
    from tmg.generation.method.tracking import EndEffectorPath  # noqa: F401  (the scripted follower takes the same paths)

    subtasks = {"robot0": [(0, 4, 0, 0)]}
    _gen, _runtime, results, log, line = run_soft_tracking(torch, monkeypatch, tmp_path, subtasks, follower_factory=TwoAtATime)
    # 5 bundles up front ((4 - 0) + 1 for the first subtask); two steps, each covering two base waypoints
    expected = np.random.default_rng(0).standard_normal((5, 6))
    assert torch.rng_state[0] == 5 and len(log) == 2 and results["success"] is False
    assert line["noise_slots_used"] == 4 and line["private_noise_steps"] == 0       # slots 1 and 3 are consumed although never drawn
    used = np.stack([(a - np.array([x, 0, 0, 0, 0, 1.0])) / 0.001 for a, x in zip(log, [1001.0, 1003.0])])
    np.testing.assert_allclose(used, expected[[0, 2]], atol=1e-6)                   # a step replays the bundle of its first new slot
    rec = np.load(tmp_path / "attempt_000000.npz")
    assert rec["track_k"].tolist() == [1, 3] and rec["track_slot"].tolist() == [1, 1] and rec["track_ds_actual"].tolist() == [2.0, 2.0]
    assert TwoAtATime.started[-1] == ([0.003] * 4, [1] * 4)           # per-waypoint noise levels and kinds reach the follower


def test_tracking_loop_restores_the_label_wrapper_when_a_step_raises(torch, monkeypatch, tmp_path):
    subtasks = {"robot0": [(0, 4, 0, 0)]}
    gen, runtime, generate, log, _line = run_soft_tracking(torch, monkeypatch, tmp_path, subtasks, fail_at=3)
    with pytest.raises(RuntimeError, match="simulator died"):
        asyncio.run(generate(gen, 0, "success_term", env_reset_queue=FakeQueue(), env_action_queue=None))
    assert len(log) == 2 and runtime._original_action_fn is None
    assert gen.env.target_eef_pose_to_action.__func__ is FakeEnv.target_eef_pose_to_action    # the official method is back
    assert not (tmp_path / "attempts.jsonl").exists()                                        # no record of the aborted attempt


# grafts and the entrypoint dispatch -------------------------------------------------------------------
SOFT_PACKAGE = '''
class DataGenerator:
    async def generate(self, env_id, success_term, env_reset_queue=None, env_action_queue=None):
        new_initial_state = self.env.scene.get_state(is_relative=True)
        results = {"success": False}
        return results

    def generate_trajectory(self, env_id, eef_name, subtask_ind, boundaries, pose_overrides, selected, prev):
        src_subtask_eef_poses = 1
        src_subtask_target_poses = 2
        src_subtask_gripper_actions = 3
        src_subtask_nodal_positions = 4
        transformed_seq = WaypointSequence.from_poses(src_subtask_eef_poses, src_subtask_target_poses, src_subtask_gripper_actions)
        return transformed_seq
'''
WAYPOINT_MODULE = '''
class MultiWaypoint:
    async def execute(self, env, success_term, env_id, env_action_queue):
        state = env.scene.get_state(is_relative=True)
        play_action = env.target_eef_pose_to_action(target_eef_pose_dict={}, gripper_action_dict={}, action_noise_dict={}, env_id=env_id)
        obs = env.step(play_action)
        success = bool(success_term.func(env, **success_term.params)[env_id])
        return {"success": success}
'''


def fake_package(tmp_path, monkeypatch, name, generator_source, waypoint_source):
    """Register ``<name>.data_generator`` and ``<name>.waypoint`` as importable modules written to ``tmp_path``."""
    import importlib

    pkg = tmp_path / name
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "data_generator.py").write_text(generator_source)
    (pkg / "waypoint.py").write_text(waypoint_source)
    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in (name, f"{name}.data_generator", f"{name}.waypoint"):
        sys.modules.pop(mod, None)
    return importlib.import_module(f"{name}.data_generator"), importlib.import_module(f"{name}.waypoint")


def test_install_grafts_the_soft_builder_at_its_class_level_anchors(tmp_path, monkeypatch):
    import inspect

    from tmg.generation.generators.isaaclab import runtime as rt

    dg, wp = fake_package(tmp_path, monkeypatch, "fakesoft", SOFT_PACKAGE, WAYPOINT_MODULE)
    runtime = rt.SoftMimicGenRuntime(Method0(), soft_layout(), records_dir=None)
    info = rt.install("fakesoft", runtime, {"generate": None, "trajectory": None, "execute": None})
    assert set(info) == {"generate", "generate_trajectory", "execute"}
    assert dg.DataGenerator.generate.__name__ == "generate" and dg.DataGenerator.generate_trajectory.__name__ == "generate_trajectory"
    assert info["generate"] == ["after 'new_initial_state = ...': _tmg.begin_attempt(self, env_id)",
                                "before final return: _tmg.end_attempt(self, env_id, results)"]
    assert info["generate_trajectory"] == [
        "after 'src_subtask_nodal_positions = ...': src_subtask_eef_poses, src_subtask_target_poses, src_subtask_gripper_actions = _tmg.mix_segment(self, locals())",
        "after 'transformed_seq = ...': _tmg.record_subtask(locals())"]
    assert info["execute"] == ["after 'play_action = ...': _tmg.record_command(self, env, env_id, state, play_action)",
                               "after 'success = ...': _tmg.record_step(self, env, env_id, obs, success)"]
    assert dg.DataGenerator.generate_trajectory.__globals__["_tmg"] is runtime and dg.DataGenerator.generate.__tmg_graft__["edits"] == info["generate"]
    assert inspect.iscoroutinefunction(wp.MultiWaypoint.execute)
    with pytest.raises(Exception):      # noqa: B017  Isaac Lab Mimic's builder name does not exist in the Soft-shaped package
        rt.install("fakesoft", rt.IsaacLabRuntime(Method0(), soft_layout(), records_dir=None), {})


MIMIC_PACKAGE = '''
class DataGenerator:
    async def generate(self, env_id, success_term, env_reset_queue=None, env_action_queue=None):
        new_initial_state = self.env.scene.get_state(is_relative=True)
        results = {"success": False}
        return results

    def generate_eef_subtask_trajectory(self, env_id, eef_name, subtask_ind, boundaries, pose_overrides, selected):
        src_subtask_eef_poses = 1
        src_subtask_target_poses = 2
        src_subtask_gripper_actions = 3
        src_subtask_object_pose = 4
        transformed_seq = WaypointSequence.from_poses(src_subtask_eef_poses, src_subtask_target_poses, src_subtask_gripper_actions)
        return transformed_seq
'''


def tracking_loop_of(generate):
    """The ``TrackingLoop`` a replacement ``generate`` closes over."""
    from tmg.generation.generators.isaaclab.tracking_loop import TrackingLoop

    return next(c.cell_contents for c in generate.__closure__ if isinstance(c.cell_contents, TrackingLoop))


@pytest.mark.parametrize("soft, package_source, builder", [(True, SOFT_PACKAGE, "generate_trajectory"),
                                                           (False, MIMIC_PACKAGE, "generate_eef_subtask_trajectory")])
def test_install_from_spec_picks_the_runtime_by_soft_and_seeds_the_private_stream(tmp_path, monkeypatch, soft, package_source, builder):
    from tmg.generation.generators.isaaclab import runtime as rt

    dg, _wp = fake_package(tmp_path, monkeypatch, "fakepkg", package_source, WAYPOINT_MODULE)
    spec = {"chunk": {"seed": 31}, "records_dir": str(tmp_path / "records"), "task": {},
            "method": {"name": "tracking_only", "tracking": {"enabled": True, "regions": TRACK_CFG["regions"]}},
            "extra": {"soft": soft, "package": "fakepkg", "graft_hashes": {},
                      "layout": {"eefs": ["robot0"], "arm_action_dims": {"robot0": [0, 6]}, "hand_prefixes": None, "hand_term": None,
                                 "measure_links": None, "noise_draw_shapes": [6]}}}
    info = rt.install_from_spec(spec, None)
    assert set(info) == {"generate", builder, "execute", "tracking"}
    runtime = getattr(dg.DataGenerator, builder).__globals__["_tmg"]
    assert type(runtime) is (rt.SoftMimicGenRuntime if soft else rt.IsaacLabRuntime)
    assert runtime.layout.arm_action_dims == {"robot0": (0, 6)} and runtime.layout.noise_draw_shapes == (6,) and runtime.records_dir == tmp_path / "records"
    assert info[builder][0].startswith("after 'src_subtask_nodal_positions = ...'" if soft else "after 'src_subtask_object_pose = ...'")
    loop = tracking_loop_of(dg.DataGenerator.generate)
    assert type(loop) is rt.IsaacLabTrackingLoop and loop.runtime is runtime and loop.private_seed == 31 + 7919
    assert loop.follower_factory is runtime.method.follower_factory and callable(loop.follower_factory)


def test_layout_from_spec_restores_the_tuples():
    from tmg.generation.generators.isaaclab.runtime import IsaacLabLayout

    layout = IsaacLabLayout.from_spec({"eefs": ["left", "right"], "arm_action_dims": {"left": [0, 7], "right": [7, 14]}, "hand_prefixes": None,
                                       "hand_term": None, "measure_links": {"left": "l"}, "noise_draw_shapes": [3, 4, 3, 4]})
    assert layout.arm_action_dims == {"left": (0, 7), "right": (7, 14)} and layout.noise_draw_shapes == (3, 4, 3, 4)
    assert layout.eefs == ["left", "right"] and layout.measure_links == {"left": "l"}


def test_entrypoint_dispatches_to_the_runtime_module_of_the_spec(monkeypatch, capsys):
    from tmg.generation.generators.isaaclab import entrypoint

    calls = []
    for name in ("fake_runtime_a", "fake_runtime_b"):
        mod = types.ModuleType(name)
        mod.install_from_spec = lambda spec, namespace, _n=name: calls.append((_n, spec["chunk"]["seed"], namespace)) or {"ok": _n}
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(entrypoint, "DEFAULT_RUNTIME", "fake_runtime_a")
    ns = {}
    entrypoint.install_runtime({"extra": {}, "chunk": {"seed": 5}}, ns)
    entrypoint.install_runtime({"extra": {"runtime_module": "fake_runtime_b"}, "chunk": {"seed": 6}}, ns)
    entrypoint.install_runtime({"extra": {"disable_graft": True}, "chunk": {"seed": 7}}, ns)
    assert calls == [("fake_runtime_a", 5, ns), ("fake_runtime_b", 6, ns)]
    out = capsys.readouterr().out
    assert "grafts installed (fake_runtime_a)" in out and "grafts installed (fake_runtime_b)" in out and "DISABLED" in out
