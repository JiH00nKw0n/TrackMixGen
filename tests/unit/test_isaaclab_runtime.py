"""Characterization of the Isaac Lab runtime callbacks, without torch or Isaac Sim.

A fake ``torch`` module (numpy underneath) stands in for the real one, so ``mix_segment`` and the pre-noise label wrapper
run on plain numpy arrays. The fake generator/environment objects below
mirror the attributes the callbacks read from the official classes (``softmimicgen.datagen`` shape).
"""
import sys
import types

import numpy as np
import pytest

from tmg.generation.method.interfaces import MixedSegment


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

    def __init__(self, mixer=None):
        self.mixer = mixer


class RecordingMixer:
    """Returns a fixed mixed segment and keeps the context it was given."""

    def __init__(self, mixed=True):
        self.mixed, self.contexts = mixed, []

    def begin_attempt(self):
        pass

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
                  "subtask_object_soft": soft, "env_id": 0, "subtask_object_pose": T(pose(105.0))}
    if soft:
        local_vars["src_subtask_nodal_positions"] = info.object_nodal_positions["rope"][start]
        local_vars["subtask_nodal_positions"] = info.object_nodal_positions["rope"][start]
    return local_vars


class SceneEnv:
    """The environment's pose getters the scene of a subtask reads (eef pose now, rigid object poses now)."""

    def __init__(self, eef=None, objects=None):
        self.eef = {} if eef is None else eef
        self.objects = {} if objects is None else objects

    def get_robot_eef_pose(self, eef_name, env_ids):
        return T(self.eef.get(eef_name, np.eye(4)))[None]

    def get_object_poses(self, env_ids):
        return {k: T(v)[None] for k, v in self.objects.items()}

    def target_eef_pose_to_action(self, target_eef_pose_dict, gripper_action_dict, action_noise_dict=None, env_id=0):
        return T(np.zeros(6))


def soft_generator(infos, env=None):
    return types.SimpleNamespace(src_demo_datagen_info_pool=types.SimpleNamespace(datagen_infos=infos),
                                 env=env or SceneEnv())


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
    gen = soft_generator(infos)
    eef, targets, grippers = rt.mix_segment(gen, local_vars)
    ctx = mixer.contexts[0]
    assert ctx.subtask_ind == 0 and ctx.primary == 1 and ctx.num_sources == 2 and ctx.has_reference_object is True
    assert _same_extra(ctx.extra, {"eef": "robot0", "soft": False, "reference_object": "rope", "current_object_pose": pose(105.0),
                                   "carried_object": None})         # subtask 0: no carried object, no in-hand poses
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
    out = rt.mix_segment(soft_generator(infos), local_vars)
    assert out[0] is local_vars["src_subtask_eef_poses"] and out[2] is local_vars["src_subtask_gripper_actions"]
    assert rt._last_mix == {"mixed": False, "donor": 1}


def test_mix_segment_registers_donor_segments_into_the_primary_rope(torch, monkeypatch):
    """Soft objects: donors go through the official nodal registration, the primary does not, object frames are identity."""
    calls = []

    def register(*, tgt_obj_nodal_pos, src_eef_poses, src_obj_nodal_pos, use_rotation_transform, bend_coef, rot_coef):
        calls.append((float(tgt_obj_nodal_pos.a[0, 0]), float(src_obj_nodal_pos.a[0, 0]), use_rotation_transform, bend_coef, rot_coef))
        return T(src_eef_poses.a + 1000)

    install_fake_registration(monkeypatch, register)
    infos = [rope_info(6, 0), rope_info(6, 10), rope_info(6, 20)]
    mixer = RecordingMixer()
    rt = make_runtime(Method0(mixer))
    rt.mix_segment(soft_generator(infos), soft_locals(infos, primary=1, soft=True))
    ctx = mixer.contexts[0]
    assert [c[2:] for c in calls] == [(True, 0.1, 1e-3)] * 2 and calls[0][0] == calls[1][0] == float(infos[1].object_nodal_positions["rope"].a[0, 0, 0])
    assert ctx.extra["eef"] == "robot0" and ctx.extra["soft"] is True and ctx.extra["reference_object"] == "rope"
    assert all(np.array_equal(p, np.eye(4)) for p in ctx.src_object_poses)          # blending frames: the world after registration
    assert ctx.src_segments[0][1][0, 0, 3] == 1000.5 and ctx.src_segments[1][1][0, 0, 3] == 10.5 and ctx.src_segments[2][1][0, 0, 3] == 1020.5
    # the donors' start poses are registered too (the primary's is its own)
    assert ctx.src_segments[0][0][0, 3] == 1000.0 and ctx.src_segments[1][0][0, 3] == 10.0 and ctx.src_segments[2][0][0, 3] == 1020.0
    # features: the nodal frames of soft_frame (source 0's first cloud is the reference, so source 0 has the identity rotation)
    from tmg.generation.method.soft_frame import nodal_frame
    ref = infos[0].object_nodal_positions["rope"].a[0]
    for j, info in enumerate(infos):
        np.testing.assert_allclose(ctx.extra["feature_object_poses"][j], nodal_frame(info.object_nodal_positions["rope"].a[0], ref))
    np.testing.assert_allclose(ctx.extra["current_object_pose"], nodal_frame(infos[1].object_nodal_positions["rope"].a[0], ref))


def _same_extra(a, b):
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) if isinstance(b[k], np.ndarray) else a[k] == b[k] for k in b)


def install_fake_registration(monkeypatch, register):
    for name in ("softmimicgen", "softmimicgen.datagen"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    dg = types.ModuleType("softmimicgen.datagen.data_generator")
    dg.transform_source_data_segment_using_nodal_registration = register
    monkeypatch.setitem(sys.modules, "softmimicgen.datagen.data_generator", dg)


def rope_cloud(offset, yaw=0.0, n=12):
    """A bent rope of n nodes, turned by ``yaw`` and shifted by ``offset`` along x."""
    t = np.linspace(-0.2, 0.2, n)
    c, s = np.cos(yaw), np.sin(yaw)
    pts = np.stack([t, 2.0 * t ** 2, np.full(n, 0.02)], axis=1) @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T
    return pts + [offset, 0.0, 0.0]


def rope_info(num_steps, offset, yaw_step=0.1):
    """A SoftMimicGen source whose rope turns by ``yaw_step`` per step (and per source offset)."""
    info = datagen_info(num_steps, offset)
    info.object_nodal_positions = {"rope": T(np.stack([rope_cloud(offset, yaw_step * (i + offset)) for i in range(num_steps)]))}
    return info


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


# scene of a subtask: mixer inputs and record fields ----------------------------------------------------
def rope_generator(infos, eef_pose=None):
    cfg = [types.SimpleNamespace(num_interpolation_steps=5, num_fixed_steps=0, action_noise=0.0, subtask_term_signal="grasp"),
           types.SimpleNamespace(num_interpolation_steps=5, num_fixed_steps=0, action_noise=0.0, subtask_term_signal=None)]
    gen = soft_generator(infos, SceneEnv(eef={"robot0": pose(0.3) if eef_pose is None else eef_pose}))
    gen.env_cfg = types.SimpleNamespace(subtask_configs={"robot0": cfg})
    return gen


def rope_subtask(rt, gen, infos, subtask, primary, current_nodes):
    local_vars = soft_locals(infos, primary=primary, subtask=subtask, soft=True)
    local_vars.update({"self": gen, "subtask_nodal_positions": T(current_nodes), "selected_src_subtask_boundary": T([0, 4] if subtask == 0 else [4, 6])})
    local_vars.pop("subtask_object_pose")                              # a deformable object has no rigid pose
    out = rt.mix_segment(gen, local_vars)
    rt.record_subtask(local_vars)
    return out


def test_rope_records_the_nodal_frames_and_in_hand_poses_the_fit_needs(torch, tmp_path, monkeypatch):
    """Method none (baseline): the records carry the scene of every subtask; nothing reaches the generation."""
    import json

    from tmg.generation.method.soft_frame import FRAME_VERSION, nodal_frame, reference_hash
    from tmg.generation.method.success_model import in_hand
    from tmg.generation.method.success_model import pose12 as P

    infos = [rope_info(6, 0), rope_info(6, 10), rope_info(6, 20)]
    gen = rope_generator(infos)
    rt = make_runtime(Method0(None), records_dir=tmp_path)
    rt.begin_attempt(gen, 0)
    now0, now1 = rope_cloud(0.5, 0.7), rope_cloud(0.6, 0.9)
    out = rope_subtask(rt, gen, infos, 0, 2, now0)
    assert out[0] is not None and rt._last_mix is None                 # no mixer: the builder's segment untouched
    rope_subtask(rt, gen, infos, 1, 2, now1)
    rt.end_attempt(gen, 0, {"success": True})
    line = json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[0])
    s0, s1 = line["subtasks"]
    ref = infos[0].object_nodal_positions["rope"].a[0]
    assert "scene_error" not in s0 and s0["term_signal"] == "grasp" and s1["term_signal"] is None
    assert s0["object_frame"] == {"version": FRAME_VERSION, "reference_sha256": reference_hash(ref)}
    np.testing.assert_allclose(P(s0["cur_obj_pose"]), nodal_frame(now0, ref), atol=1e-6)
    np.testing.assert_allclose(P(s0["src_obj_pose"]), nodal_frame(infos[2].object_nodal_positions["rope"].a[0], ref), atol=1e-6)
    np.testing.assert_allclose(P(s1["src_obj_pose"]), nodal_frame(infos[2].object_nodal_positions["rope"].a[4], ref), atol=1e-6)
    np.testing.assert_allclose(P(s0["eef_pose"]), pose(0.3))
    np.testing.assert_allclose(P(s0["src_eef_pose"]), pose(20.0))
    assert s0["carried_object"] is None and s1["carried_object"] == "rope"
    np.testing.assert_allclose(P(s1["objects"]["rope"]), nodal_frame(now1, ref), atol=1e-6)
    np.testing.assert_allclose(P(s1["src_objects"]["rope"]), P(s1["src_obj_pose"]))
    # the in-hand features the fit computes from these fields are non-zero for this scene
    assert np.linalg.norm(in_hand(P(s1["eef_pose"]), P(s1["objects"]["rope"]))[:3, 3] - in_hand(P(s1["src_eef_pose"]), P(s1["src_objects"]["rope"]))[:3, 3]) > 0


def test_rope_mixer_gets_nodal_scene_and_in_hand_inputs_from_the_second_subtask(torch, monkeypatch):
    install_fake_registration(monkeypatch, lambda **kw: kw["src_eef_poses"])
    infos = [rope_info(6, 0), rope_info(6, 10), rope_info(6, 20)]
    gen = rope_generator(infos)
    mixer = RecordingMixer(mixed=False)
    rt = make_runtime(Method0(mixer))
    rt.begin_attempt(gen, 0)
    rope_subtask(rt, gen, infos, 0, 1, rope_cloud(0.5, 0.7))
    rope_subtask(rt, gen, infos, 1, 1, rope_cloud(0.6, 0.9))
    first, second = (c.extra for c in mixer.contexts)
    assert "current_in_hand" not in first and first["current_object_pose"] is not None and len(first["feature_object_poses"]) == 3
    assert second["carried_object"] == "rope" and second["current_in_hand"].shape == (4, 4) and len(second["source_in_hand"]) == 3
    rots = [p[:2, :2] for p in second["feature_object_poses"]]
    assert not np.allclose(rots[0], rots[1])                           # the sources' frames differ: non-zero scene features


def test_a_scene_that_cannot_be_read_is_noted_without_a_mixer_and_raises_with_one(torch):
    infos = [rope_info(6, 0), rope_info(6, 10)]
    gen = rope_generator(infos)
    for method, raises in ((Method0(None), False), (Method0(RecordingMixer()), True)):
        rt = make_runtime(method)
        rt.begin_attempt(gen, 0)
        local_vars = soft_locals(infos, soft=True)
        local_vars.update({"self": gen, "selected_src_subtask_boundary": T([0, 4])})
        local_vars.pop("subtask_nodal_positions")
        if raises:
            with pytest.raises(KeyError, match="subtask_nodal_positions"):
                rt.mix_segment(gen, local_vars)
            continue
        rt.mix_segment(gen, local_vars)
        rt.record_subtask(local_vars)
        assert "subtask_nodal_positions" in rt.rec.subtasks[0]["scene_error"]


def test_rigid_objects_use_the_builder_pose_and_the_environment_for_another_carried_object(torch):
    infos = [datagen_info(6, 0, obj="bottle"), datagen_info(6, 10, obj="bottle")]
    for info, offset in zip(infos, (0, 10)):
        info.object_poses["fridge"] = T(np.stack([pose(200 + offset + i) for i in range(6)]))
    env = SceneEnv(eef={"robot0": pose(1.0)}, objects={"bottle": pose(0.9), "fridge": pose(300.0)})
    gen = soft_generator(infos, env)
    gen.env_cfg = types.SimpleNamespace(subtask_configs={"robot0": [types.SimpleNamespace(
        num_interpolation_steps=0, num_fixed_steps=0, action_noise=0.0, subtask_term_signal="grasp")] * 2})
    from tmg.generation.method.success_model import pose12 as P

    rt = make_runtime(Method0(None))
    rt.begin_attempt(gen, 0)
    for k, obj in enumerate(("bottle", "fridge")):
        local_vars = soft_locals(infos, primary=1, subtask=k)
        local_vars.update({"self": gen, "subtask_object_name": obj, "subtask_object_pose": T(pose(150.0 + k)),
                           "selected_src_subtask_boundary": T([0, 4] if k == 0 else [4, 6])})
        rt.record_subtask(local_vars)
    s0, s1 = rt.rec.subtasks
    np.testing.assert_allclose(P(s0["cur_obj_pose"]), pose(150.0))
    np.testing.assert_allclose(P(s0["src_obj_pose"]), pose(110.0))
    assert s1["carried_object"] == "bottle" and sorted(s1["objects"]) == ["bottle", "fridge"]
    np.testing.assert_allclose(P(s1["objects"]["bottle"]), pose(0.9))
    np.testing.assert_allclose(P(s1["src_objects"]["bottle"]), pose(114.0))
    np.testing.assert_allclose(P(s1["cur_obj_pose"]), pose(151.0))
    np.testing.assert_allclose(P(s1["src_obj_pose"]), pose(214.0))


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


@pytest.mark.parametrize("soft, package_source, builder", [(True, SOFT_PACKAGE, "generate_trajectory"),
                                                           (False, MIMIC_PACKAGE, "generate_eef_subtask_trajectory")])
def test_install_from_spec_picks_the_runtime_by_soft(tmp_path, monkeypatch, soft, package_source, builder):
    from tmg.generation.generators.isaaclab import runtime as rt

    dg, _wp = fake_package(tmp_path, monkeypatch, "fakepkg", package_source, WAYPOINT_MODULE)
    spec = {"chunk": {"seed": 31}, "records_dir": str(tmp_path / "records"), "task": {},
            "method": {"name": "none", "mixing": {"enabled": False}, "tracking": {"enabled": False}},
            "extra": {"soft": soft, "package": "fakepkg", "graft_hashes": {},
                      "layout": {"eefs": ["robot0"], "arm_action_dims": {"robot0": [0, 6]}, "hand_prefixes": None, "hand_term": None,
                                 "measure_links": None, "noise_draw_shapes": [6]}}}
    info = rt.install_from_spec(spec, None)
    assert set(info) == {"generate", builder, "execute"}
    runtime = getattr(dg.DataGenerator, builder).__globals__["_tmg"]
    assert type(runtime) is (rt.SoftMimicGenRuntime if soft else rt.IsaacLabRuntime)
    assert runtime.layout.arm_action_dims == {"robot0": (0, 6)} and runtime.layout.noise_draw_shapes == (6,) and runtime.records_dir == tmp_path / "records"
    assert info[builder][0].startswith("after 'src_subtask_nodal_positions = ...'" if soft else "after 'src_subtask_object_pose = ...'")
    assert dg.DataGenerator.generate.__globals__["_tmg"] is runtime          # the official generate, grafted


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
