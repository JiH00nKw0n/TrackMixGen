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


HELD_RULE = {"version": 1, "rigid_m": 0.20, "nodes_m": 0.10}       # scene_reading.held_rule of the generator configs
# the literal settings of the official builder's registration call (SoftMimicGen c9d146ba, read 2026-11-27), which are also
# the values the code of 2fcfc0d passed for the donors of the running Rope randmix pool
OFFICIAL_REGISTRATION = {"use_rotation_transform": True, "bend_coef": 0.1, "rot_coef": 1e-3}


def make_runtime(method, records_dir=None, held_rule=HELD_RULE):
    from tmg.generation.generators.isaaclab.runtime import SoftMimicGenRuntime
    from tmg.generation.method.interfaces import HeldRule

    runtime = SoftMimicGenRuntime(method, soft_layout(), records_dir=records_dir, held_rule=HeldRule.from_mapping(held_rule))
    runtime.registration_kwargs = dict(OFFICIAL_REGISTRATION)        # as bind_official reads them from the official builder
    return runtime


# mix_segment ------------------------------------------------------------------------------------------
def test_mix_segment_without_a_mixer_returns_the_inputs_untouched(torch):
    infos = [datagen_info(6, 0), datagen_info(6, 10)]
    local_vars = soft_locals(infos)
    rt = make_runtime(Method0(mixer=None))
    out = rt.mix_segment(types.SimpleNamespace(src_demo_datagen_info_pool=types.SimpleNamespace(datagen_infos=infos)), local_vars)
    assert out[0] is local_vars["src_subtask_eef_poses"] and out[1] is local_vars["src_subtask_target_poses"]
    assert out[2] is local_vars["src_subtask_gripper_actions"] and rt.last_mix("robot0") is None


def test_mix_segment_hands_the_mixer_every_source_and_repacks_the_mixed_segment(torch):
    infos = [datagen_info(6, 0), datagen_info(6, 10)]
    local_vars = soft_locals(infos, primary=1)
    mixer = RecordingMixer()
    rt = make_runtime(Method0(mixer))
    gen = soft_generator(infos)
    eef, targets, grippers = rt.mix_segment(gen, local_vars)
    ctx = mixer.contexts[0]
    assert ctx.subtask_ind == 0 and ctx.primary == 1 and ctx.num_sources == 2 and ctx.has_reference_object is True
    scene = ctx.scene                                               # subtask 0: no carried object, no in-hand poses
    assert (scene.eef, scene.subtask, scene.reference_object, scene.carried_object, scene.held) == ("robot0", 0, "rope", None, None)
    np.testing.assert_array_equal(scene.current_object_pose, pose(105.0))
    np.testing.assert_array_equal(scene.source_object_poses, ctx.src_object_poses)     # a rigid object: the blending frames
    assert scene.held_rule.as_dict() == HELD_RULE and scene.object_frame is None
    assert not hasattr(ctx, "rng")                                  # the mixers draw from their own seeded streams only
    np.testing.assert_array_equal(ctx.primary_eef_start, pose(10))
    np.testing.assert_array_equal(ctx.primary_targets, infos[1].target_eef_pose["robot0"].a[0:4])
    np.testing.assert_array_equal(ctx.primary_gripper, np.full((4, 1), 10))
    assert len(ctx.src_segments) == 2 and ctx.src_segments[0][1].shape == (4, 4, 4) and ctx.src_segments[0][0][0, 3] == 0.0
    np.testing.assert_array_equal(ctx.src_object_poses[0], pose(100))
    np.testing.assert_array_equal(ctx.src_object_poses[1], pose(110))
    np.testing.assert_array_equal(ctx.src_segments[1][2], np.full((4, 1), 10))
    assert rt.last_mix("robot0") == {"mixed": True, "donor": 1}
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
    assert rt.last_mix("robot0") == {"mixed": False, "donor": 1}


def test_mix_segment_registers_donor_segments_into_the_primary_rope(torch, monkeypatch):
    """Soft objects: donors go through the official nodal registration, the primary does not, object frames are identity."""
    calls = []

    def register(*, tgt_obj_nodal_pos, src_eef_poses, src_obj_nodal_pos, use_rotation_transform=None, bend_coef=None, rot_coef=None):
        calls.append((float(tgt_obj_nodal_pos.a[0, 0]), float(src_obj_nodal_pos.a[0, 0]), use_rotation_transform, bend_coef, rot_coef))
        return T(src_eef_poses.a + 1000)

    install_fake_registration(monkeypatch, register)
    infos = [rope_info(6, 0), rope_info(6, 10), rope_info(6, 20)]
    mixer = RecordingMixer()
    rt = make_runtime(Method0(mixer))
    rt.registration_kwargs = {"use_rotation_transform": True, "bend_coef": 0.2, "rot_coef": 5e-3}    # as read from the official call
    rt.mix_segment(soft_generator(infos), soft_locals(infos, primary=1, soft=True))
    ctx = mixer.contexts[0]
    assert ctx.scene.donor_registration == rt.registration_kwargs    # handed to the mixer, which checks and records it
    unbound = make_runtime(Method0(RecordingMixer()))
    unbound.registration_kwargs = None                               # the official call was never read: no donor is registered
    with pytest.raises(RuntimeError, match="were not read"):
        unbound.mix_segment(soft_generator(infos), soft_locals(infos, primary=1, soft=True))
    assert [c[2:] for c in calls] == [(True, 0.2, 5e-3)] * 2 and calls[0][0] == calls[1][0] == float(infos[1].object_nodal_positions["rope"].a[0, 0, 0])
    assert ctx.scene.eef == "robot0" and ctx.scene.reference_object == "rope" and ctx.scene.object_frame["version"] == "nodal_yaw_v1"
    assert all(np.array_equal(p, np.eye(4)) for p in ctx.src_object_poses)          # blending frames: the world after registration
    assert ctx.src_segments[0][1][0, 0, 3] == 1000.5 and ctx.src_segments[1][1][0, 0, 3] == 10.5 and ctx.src_segments[2][1][0, 0, 3] == 1020.5
    # the donors' start poses are registered too (the primary's is its own)
    assert ctx.src_segments[0][0][0, 3] == 1000.0 and ctx.src_segments[1][0][0, 3] == 10.0 and ctx.src_segments[2][0][0, 3] == 1020.0
    # features: the nodal frames of soft_frame (source 0's first cloud is the reference, so source 0 has the identity rotation)
    from tmg.generation.method.soft_frame import nodal_frame
    ref = infos[0].object_nodal_positions["rope"].a[0]
    for j, info in enumerate(infos):
        np.testing.assert_allclose(ctx.scene.source_object_poses[j], nodal_frame(info.object_nodal_positions["rope"].a[0], ref))
    np.testing.assert_allclose(ctx.scene.current_object_pose, nodal_frame(infos[1].object_nodal_positions["rope"].a[0], ref))


def official_builder(call: str):
    """A class whose ``generate_trajectory`` holds the registration call ``call`` (the official builder's shape)."""
    import linecache

    source = ("class DataGenerator:\n    def generate_trajectory(self, tgt, src, nodes, bend, kw):\n"
              f"        transformed = {call}\n        return transformed\n")
    name = f"<official {abs(hash(call))}>"
    linecache.cache[name] = (len(source), None, source.splitlines(keepends=True), name)      # inspect.getsource reads it
    namespace = {}
    exec(compile(source, name, "exec"), namespace)
    return namespace["DataGenerator"]


REGISTER = "transform_source_data_segment_using_nodal_registration"
DATA = "tgt_obj_nodal_pos=tgt, src_eef_poses=src, src_obj_nodal_pos=nodes"


def test_the_registration_settings_are_read_from_the_official_builders_own_call(torch):
    """The official builder registers the selected source with literal coefficients in its own call; the donors are
    registered with the same ones, read from that call, not with numbers of ours. The call as it is at SoftMimicGen
    c9d146ba gives exactly the values 2fcfc0d hard-coded for the donors (True, 0.1, 1e-3)."""
    from tmg.generation.generators.isaaclab import runtime as rt

    official = official_builder(f"{REGISTER}({DATA}, use_rotation_transform=True, bend_coef=0.1, rot_coef=1e-3)")
    passed = rt.SoftMimicGenRuntime.registration_data
    assert rt.literal_call_keywords(official.generate_trajectory, REGISTER, passed) == OFFICIAL_REGISTRATION
    runtime = rt.SoftMimicGenRuntime(Method0(None), soft_layout(), None)
    assert runtime.registration_kwargs is None and runtime.official_settings() == {}
    runtime.bind_official(official)
    assert runtime.registration_kwargs == OFFICIAL_REGISTRATION and runtime.official_settings() == {REGISTER: OFFICIAL_REGISTRATION}
    assert runtime.donor_registration(soft=True) == OFFICIAL_REGISTRATION and runtime.donor_registration(soft=False) is None
    assert rt.SoftMimicGenRuntime(Method0(None), soft_layout(), None).donor_registration(soft=True) is None    # not read: unmixed
    # literals in other spellings are read: a signed number, `not`, a tuple
    spelled = official_builder(f"{REGISTER}({DATA}, use_rotation_transform=not False, bend_coef=-0.5, rot_coef=+1e-3, shape=(1, 2))")
    assert rt.literal_call_keywords(spelled.generate_trajectory, REGISTER, passed) == {
        "use_rotation_transform": True, "bend_coef": -0.5, "rot_coef": 1e-3, "shape": (1, 2)}
    rt.IsaacLabRuntime(Method0(None), soft_layout(), None).bind_official(object)       # a rigid-object runtime reads nothing


@pytest.mark.parametrize("call, why", [
    (f"{REGISTER}({DATA}, bend_coef=self.bend, rot_coef=1e-3)", "bend_coef is 'self.bend', not a literal"),
    (f"{REGISTER}({DATA}, bend_coef=bend, rot_coef=1e-3)", "bend_coef is 'bend', not a literal"),
    (f"{REGISTER}({DATA}, bend_coef=2 * 0.05)", r"bend_coef is '2 \* 0.05', not a literal"),
    (f"{REGISTER}({DATA}, bend_coef=float('0.1'))", "not a literal"),
    (f"{REGISTER}(src, nodes, tgt, True, 0.1, 1e-3)", "6 positional arguments"),
    (f"{REGISTER}(src, {DATA.split(', ', 1)[1]}, bend_coef=0.1)", "1 positional arguments"),
    (f"{REGISTER}({DATA}, **kw)", r"passes \*\*kw"),
    (f"{REGISTER}(tgt_obj_nodal_pos=tgt, src_eef_poses=src, bend_coef=0.1)", r"does not pass \['src_obj_nodal_pos'\] by keyword"),
    ("other_registration(tgt, src, nodes)", "expected one call, found 0"),
    (f"{REGISTER}({DATA}) if bend else {REGISTER}({DATA}, bend_coef=0.2)", "expected one call, found 2"),
])
def test_an_official_call_whose_settings_cannot_be_read_is_refused_not_guessed(torch, call, why):
    """Strict: a setting that is not a literal, a positional argument, ``**kwargs``, a missing data argument, no call or
    two calls all raise. Dropping such an argument would register the donors with the function's defaults while the
    selected source uses the official values, with no trace."""
    from tmg.generation.generators.isaaclab import runtime as rt

    runtime = rt.SoftMimicGenRuntime(Method0(RecordingMixer()), soft_layout(), None)
    with pytest.raises(RuntimeError, match=why):
        runtime.bind_official(official_builder(call))
    assert runtime.registration_kwargs is None


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


def node_pose(cloud, shift=0.0):
    """A pose at the first node of a cloud (an end effector holding the rope there), moved by ``shift`` along x."""
    p = np.eye(4)
    p[:3, 3] = np.asarray(cloud)[0] + [shift, 0.0, 0.0]
    return p


def hand_on_rope(infos, shift=0.0):
    """Sources whose end effector is at the rope's first node at every step (they hold the rope)."""
    for info in infos:
        nodes = info.object_nodal_positions["rope"].a
        info.eef_pose = {"robot0": T(np.stack([node_pose(n, shift) for n in nodes]))}
        info.target_eef_pose = {"robot0": T(info.eef_pose["robot0"].a.copy())}
    return infos


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

    infos = hand_on_rope([rope_info(6, 0), rope_info(6, 10), rope_info(6, 20)])
    infos[1].eef_pose["robot0"].a[:, 0, 3] += 1.0                      # source 1 has let go of the rope
    now0, now1 = rope_cloud(0.5, 0.7), rope_cloud(0.6, 0.9)
    gen = rope_generator(infos, eef_pose=node_pose(now1, 0.03))        # the hand holds the rope 3 cm from a node
    rt = make_runtime(Method0(None), records_dir=tmp_path)
    rt.begin_attempt(gen, 0)
    out = rope_subtask(rt, gen, infos, 0, 2, now0)
    assert out[0] is not None and rt.last_mix("robot0") is None                 # no mixer: the builder's segment untouched
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
    np.testing.assert_allclose(P(s0["eef_pose"]), node_pose(now1, 0.03), atol=1e-6)
    np.testing.assert_allclose(P(s0["src_eef_pose"]), node_pose(infos[2].object_nodal_positions["rope"].a[0]), atol=1e-6)
    assert s0["carried_object"] is None and s1["carried_object"] == "rope" and s1["carried_candidate"] == "rope"
    assert s1["held_distance"]["now"] <= 0.03 and s1["held_distance"]["source"] < 1e-3
    assert s1["held_distance"]["deformable"] is True and "held_distance" not in s0 and "held" not in s1   # tested on the nearest node; no stored verdict
    assert s0["held_rule"] == s1["held_rule"] == HELD_RULE and line["record_version"] == 3 and "invalid" not in line
    np.testing.assert_allclose(P(s1["objects"]["rope"]), nodal_frame(now1, ref), atol=1e-6)
    np.testing.assert_allclose(P(s1["src_objects"]["rope"]), P(s1["src_obj_pose"]))
    # the in-hand features the fit computes from these fields are non-zero for this scene
    assert np.linalg.norm(in_hand(P(s1["eef_pose"]), P(s1["objects"]["rope"]))[:3, 3] - in_hand(P(s1["src_eef_pose"]), P(s1["src_objects"]["rope"]))[:3, 3]) > 0


def test_rope_mixer_gets_nodal_scene_and_in_hand_inputs_from_the_second_subtask(torch, monkeypatch):
    install_fake_registration(monkeypatch, lambda **kw: kw["src_eef_poses"])
    infos = hand_on_rope([rope_info(6, 0), rope_info(6, 10), rope_info(6, 20)])
    infos[1].eef_pose["robot0"].a[:, 0, 3] += 1.0                      # source 1 has let go of the rope
    now1 = rope_cloud(0.6, 0.9)
    gen = rope_generator(infos, eef_pose=node_pose(now1))
    mixer = RecordingMixer(mixed=False)
    rt = make_runtime(Method0(mixer))
    rt.begin_attempt(gen, 0)
    rope_subtask(rt, gen, infos, 0, 1, rope_cloud(0.5, 0.7))
    rope_subtask(rt, gen, infos, 1, 1, now1)
    from tmg.generation.method.success_model import current_in_hand, held_flags, source_in_hand

    first, second = (c.scene for c in mixer.contexts)
    assert first.carried_object is None and current_in_hand(first) is None and first.current_object_pose is not None
    assert len(first.source_object_poses) == 3
    assert second.carried_object == "rope" and current_in_hand(second).shape == (4, 4) and len(second.source_carried_poses) == 3
    assert source_in_hand(second, 1) is None and source_in_hand(second, 0) is not None      # only where the rope is held
    assert held_flags(second) == (True, [True, False, True]) and second.held.deformable and second.held.sources == [True, False, True]
    rots = [p[:2, :2] for p in second.source_object_poses]
    assert not np.allclose(rots[0], rots[1])                           # the sources' frames differ: non-zero scene features


def unreadable_subtask(rt, gen, infos, subtask=0):
    """One builder call whose scene cannot be read (the current node positions are missing)."""
    local_vars = soft_locals(infos, subtask=subtask, soft=True)
    local_vars.update({"self": gen, "selected_src_subtask_boundary": T([0, 4])})
    local_vars.pop("subtask_nodal_positions")
    out = rt.mix_segment(gen, local_vars)
    rt.record_subtask(local_vars)
    return local_vars, out


@pytest.mark.parametrize("mixing", [False, True])
def test_a_scene_that_cannot_be_read_fails_the_attempt_only_never_the_chunk(torch, tmp_path, capsys, mixing):
    """Without a mixer the record notes it and the generation goes on. With one the subtask is NOT mixed and says so (no
    silent identity): the mix record names the scene error and the attempt is marked invalid; the next attempt is mixed
    again."""
    import json

    install_fake_registration(monkeypatch=pytest.MonkeyPatch(), register=lambda **kw: kw["src_eef_poses"])
    infos = [rope_info(6, 0), rope_info(6, 10)]
    gen = rope_generator(infos)
    mixer = RecordingMixer() if mixing else None
    rt = make_runtime(Method0(mixer), records_dir=tmp_path)
    rt.begin_attempt(gen, 0)
    local_vars, out = unreadable_subtask(rt, gen, infos)
    assert out[0] is local_vars["src_subtask_eef_poses"] and out[1] is local_vars["src_subtask_target_poses"]   # the source's own segment
    assert "subtask_nodal_positions" in rt.rec.subtasks[0]["scene_error"] and "WARNING" in capsys.readouterr().out
    assert rt.rec.subtasks[0]["term_signal"] == "grasp"                 # what could be read is still recorded
    rt.end_attempt(gen, 0, {"success": True})
    rt.begin_attempt(gen, 0)                                            # the chunk goes on: a readable attempt
    rope_subtask(rt, gen, infos, 0, 1, rope_cloud(0.5, 0.7))
    rt.end_attempt(gen, 0, {"success": True})
    bad, good = (json.loads(text) for text in (tmp_path / "attempts.jsonl").read_text().splitlines())
    assert bad["scene_errors"] == 1 and good["scene_errors"] == 0 and "invalid" not in good and "scene_error" not in good["subtasks"][0]
    assert bad["success"] is True                                       # the official verdict is not rewritten
    if mixing:
        mix = bad["subtasks"][0]["mix"]
        assert bad["invalid"] == "scene_error" and mix["mixed"] is False and mix["identity_reason"].startswith("scene_error: scene: KeyError")
        assert len(mixer.contexts) == 1 and good["subtasks"][0]["mix"] == {"mixed": True, "donor": 1}     # only the good attempt was mixed
    else:
        assert "invalid" not in bad and bad["subtasks"][0]["mix"] is None


def test_a_term_signal_that_cannot_be_read_is_a_record_error_but_the_subtask_is_still_mixed(torch, tmp_path, monkeypatch):
    import json

    install_fake_registration(monkeypatch, lambda **kw: kw["src_eef_poses"])
    infos = [rope_info(6, 0), rope_info(6, 10)]
    gen = rope_generator(infos)
    del gen.env_cfg                                                     # the term signal lives there
    mixer = RecordingMixer()
    rt = make_runtime(Method0(mixer), records_dir=tmp_path)
    rt.begin_attempt(gen, 0)
    local_vars = soft_locals(infos, soft=True)
    local_vars.update({"self": gen, "selected_src_subtask_boundary": T([0, 4])})
    rt.mix_segment(gen, local_vars)
    monkeypatch.setattr(rt, "subtask_settings", lambda *a: (5, 0, 0.0))
    rt.record_subtask(local_vars)
    rt.end_attempt(gen, 0, {"success": False})
    line = json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[0])
    s = line["subtasks"][0]
    assert len(mixer.contexts) == 1 and s["mix"]["mixed"] is True and "invalid" not in line
    assert s["scene_error"].startswith("term_signal: AttributeError") and line["scene_errors"] == 1 and "src_obj_pose" in s


def test_rigid_objects_use_the_builder_pose_and_the_environment_for_another_carried_object(torch):
    infos = [datagen_info(6, 0, obj="bottle"), datagen_info(6, 10, obj="bottle")]
    for info, offset in zip(infos, (0, 10)):
        info.object_poses["fridge"] = T(np.stack([pose(200 + offset + i) for i in range(6)]))
        info.object_poses["bottle"] = T(np.stack([pose(offset + i + 0.05) for i in range(6)]))   # in the hand
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
    np.testing.assert_allclose(P(s0["src_obj_pose"]), pose(10.05))
    assert s1["carried_object"] == "bottle" and sorted(s1["objects"]) == ["bottle"]    # the carried object only
    np.testing.assert_allclose(P(s1["objects"]["bottle"]), pose(0.9))
    np.testing.assert_allclose(P(s1["src_objects"]["bottle"]), pose(14.05))
    np.testing.assert_allclose(P(s1["cur_obj_pose"]), pose(151.0))
    np.testing.assert_allclose(P(s1["src_obj_pose"]), pose(214.0))


def test_an_object_the_hand_does_not_hold_gives_no_in_hand_pose(torch):
    """GR1 Bottle Placement: the left hand's subtask 0 references the bottle (it supports the placement) and its subtask 1
    the fridge; the bottle is 0.6 m from the left hand, so subtask 1 has no carried object (pilot 2026-11-27)."""
    infos = [datagen_info(6, 0, obj="bottle"), datagen_info(6, 10, obj="bottle")]
    for info, offset in zip(infos, (0, 10)):
        info.object_poses["fridge"] = T(np.stack([pose(200 + offset + i) for i in range(6)]))
    env = SceneEnv(eef={"robot0": pose(1.0)}, objects={"bottle": pose(1.6), "fridge": pose(300.0)})
    gen = soft_generator(infos, env)
    gen.env_cfg = types.SimpleNamespace(subtask_configs={"robot0": [types.SimpleNamespace(
        num_interpolation_steps=0, num_fixed_steps=0, action_noise=0.0, subtask_term_signal="support")] * 2})
    rt = make_runtime(Method0(None))
    rt.begin_attempt(gen, 0)
    for k, obj in enumerate(("bottle", "fridge")):
        local_vars = soft_locals(infos, primary=1, subtask=k)
        local_vars.update({"self": gen, "subtask_object_name": obj, "subtask_object_pose": T(pose(150.0 + k)),
                           "selected_src_subtask_boundary": T([0, 4] if k == 0 else [4, 6])})
        rt.record_subtask(local_vars)
    s1 = rt.rec.subtasks[1]
    assert s1["carried_object"] is None and s1["carried_candidate"] == "bottle" and s1["objects"] == {} and s1["src_objects"] == {}
    from tmg.generation.method.success_model import held_test_of_record

    test = held_test_of_record(s1)
    assert s1["held_distance"]["now"] == pytest.approx(0.6) and (test.now, test.sources, test.deformable) == (False, [False], False)


def test_the_hold_limits_come_from_the_rule_not_from_the_code(torch):
    """The same scene (the bottle 0.6 m from the hand) under three rules: the configured 0.20 m (not held), a 1 m rule
    (held: the in-hand poses appear), and no rule at all (no test: the carried object's poses are kept)."""
    from tmg.generation.generators.isaaclab import runtime as module

    assert not hasattr(module, "HOLD_RIGID_M") and not hasattr(module, "HOLD_NODES_M")
    infos = [datagen_info(6, 0, obj="bottle"), datagen_info(6, 10, obj="bottle")]
    for info, offset in zip(infos, (0, 10)):
        info.object_poses["fridge"] = T(np.stack([pose(200 + offset + i) for i in range(6)]))
    env = SceneEnv(eef={"robot0": pose(1.0)}, objects={"bottle": pose(1.6), "fridge": pose(300.0)})
    gen = soft_generator(infos, env)
    gen.env_cfg = types.SimpleNamespace(subtask_configs={"robot0": [types.SimpleNamespace(
        num_interpolation_steps=0, num_fixed_steps=0, action_noise=0.0, subtask_term_signal="support")] * 2})
    seen = {}
    for name, rule in (("configured", HELD_RULE), ("wide", {"version": 7, "rigid_m": 1.0, "nodes_m": 0.1}), ("none", None)):
        rt = make_runtime(Method0(None), held_rule=rule)
        rt.begin_attempt(gen, 0)
        for k, obj in enumerate(("bottle", "fridge")):
            local_vars = soft_locals(infos, primary=1, subtask=k)
            local_vars.update({"self": gen, "subtask_object_name": obj, "subtask_object_pose": T(pose(150.0 + k)),
                               "selected_src_subtask_boundary": T([0, 4] if k == 0 else [4, 6])})
            rt.record_subtask(local_vars)
        seen[name] = rt.rec.subtasks[1]
    assert seen["configured"]["carried_object"] is None and seen["configured"]["held_rule"] == HELD_RULE
    assert seen["wide"]["carried_object"] == "bottle" and seen["wide"]["held_rule"]["version"] == 7
    from tmg.generation.method.success_model import held_test_of_record

    wide = held_test_of_record(seen["wide"])
    assert wide.now is True and wide.sources == [False]                 # the source's bottle is 100 m away
    assert sorted(seen["wide"]["objects"]) == ["bottle"] and seen["wide"]["src_objects"] == {}
    assert seen["none"]["carried_object"] == "bottle" and "held_distance" not in seen["none"] and "held_rule" not in seen["none"]
    assert sorted(seen["none"]["src_objects"]) == ["bottle"]


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
        transformed_eef_poses = transform_source_data_segment_using_nodal_registration(
            tgt_obj_nodal_pos=None, src_eef_poses=src_subtask_eef_poses, src_obj_nodal_pos=src_subtask_nodal_positions,
            use_rotation_transform=True, bend_coef=0.1, rot_coef=1e-3)
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
    assert set(info) == {"generate", "generate_trajectory", "execute", "hashes", "official_settings"}
    assert set(info["hashes"]) == {"generate", "trajectory", "execute"} and all(len(h) == 64 for h in info["hashes"].values())
    assert info["official_settings"] == {}                              # an unmixed run reads nothing from the official builder
    assert dg.DataGenerator.generate.__name__ == "generate" and dg.DataGenerator.generate_trajectory.__name__ == "generate_trajectory"
    assert info["generate"] == ["after 'new_initial_state = ...': _tmg.begin_attempt(self, env_id)",
                                "before final return: _tmg.end_attempt(self, env_id, results)"]
    assert info["generate_trajectory"] == [
        "after 'src_subtask_nodal_positions = ...': src_subtask_eef_poses, src_subtask_target_poses, src_subtask_gripper_actions = _tmg.mix_segment(self, locals())",
        "after 'transformed_seq = ...': _tmg.record_subtask(locals())"]
    assert info["execute"] == ["after 'play_action = ...': _tmg.record_command(self, env, env_id, state, play_action)",
                               "after 'success = ...': _tmg.record_step(self, env, env_id, obs, success)"]
    assert dg.DataGenerator.generate_trajectory.__globals__["_tmg"] is runtime and dg.DataGenerator.generate.__tmg_graft__["edits"] == info["generate"]
    assert runtime.registration_kwargs is None
    mixing = rt.SoftMimicGenRuntime(Method0(RecordingMixer()), soft_layout(), records_dir=None)
    fake_package(tmp_path, monkeypatch, "fakesoft2", SOFT_PACKAGE, WAYPOINT_MODULE)
    logged = rt.install("fakesoft2", mixing, {})
    assert mixing.registration_kwargs == OFFICIAL_REGISTRATION       # the official call's, and in the graft log
    assert logged["official_settings"] == {"transform_source_data_segment_using_nodal_registration": OFFICIAL_REGISTRATION}
    fake_package(tmp_path, monkeypatch, "fakemimic", MIMIC_PACKAGE.replace("generate_eef_subtask_trajectory", "generate_trajectory")
                 .replace("src_subtask_object_pose = 4", "src_subtask_nodal_positions = 4"), WAYPOINT_MODULE)
    rt.install("fakemimic", rt.SoftMimicGenRuntime(Method0(None), soft_layout(), records_dir=None), {})   # no such call: unmixed is unaffected
    with pytest.raises(RuntimeError, match="expected one call"):
        rt.install("fakemimic", rt.SoftMimicGenRuntime(Method0(RecordingMixer()), soft_layout(), records_dir=None), {})
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
    spec = {"chunk": {"seed": 31}, "records_dir": str(tmp_path / "records"), "task": {}, "scene_reading": {"held_rule": HELD_RULE},
            "method": {"name": "none", "mixing": {"enabled": False}, "tracking": {"enabled": False}},
            "extra": {"soft": soft, "package": "fakepkg", "graft_hashes": {},
                      "layout": {"eefs": ["robot0"], "arm_action_dims": {"robot0": [0, 6]}, "hand_prefixes": None, "hand_term": None,
                                 "measure_links": None, "noise_draw_shapes": [6]}}}
    info = rt.install_from_spec(spec, None)
    assert set(info) == {"generate", builder, "execute", "hashes", "official_settings"}
    runtime = getattr(dg.DataGenerator, builder).__globals__["_tmg"]
    assert type(runtime) is (rt.SoftMimicGenRuntime if soft else rt.IsaacLabRuntime)
    assert runtime.layout.arm_action_dims == {"robot0": (0, 6)} and runtime.layout.noise_draw_shapes == (6,) and runtime.records_dir == tmp_path / "records"
    assert runtime.held_rule.as_dict() == HELD_RULE
    assert rt.held_rule_of({"chunk": {}}) is None                       # a spec written before scene_reading existed: no hold test
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


# the features at generation equal the fit's ----------------------------------------------------------------------------
@pytest.mark.parametrize("soft", [False, True])
def test_mixer_features_at_generation_equal_the_fits_design_rows(torch, tmp_path, monkeypatch, soft):
    """One attempt of two subtasks with the real SkillMixer (randmix): the scene and in-hand features the mixer scored
    for the selected source equal the columns tmg fit_coefficients builds from the same attempt's record."""
    import json

    from tmg.generation.method.skillmix import SkillMixer
    from tmg.generation.method.interfaces import Arms
    from tmg.generation.method.success_model import HELD_FEATURES, design_rows, mix_record_features

    torch.Tensor = T                       # scipy's array-API check looks the fake module's Tensor up
    install_fake_registration(monkeypatch, lambda **kw: kw["src_eef_poses"])
    infos = hand_on_rope([rope_info(6, 0), rope_info(6, 10), rope_info(6, 20)])
    for info in infos:
        eef = info.eef_pose["robot0"].a
        info.object_poses["rope"] = T(eef @ pose(0.05))                    # the rigid stand-in is held 5 cm from the hand
        info.gripper_action = {"robot0": T(np.full((6, 1), -1.0))}       # open throughout: every source compatible
    hand = node_pose(rope_cloud(0.04, 0.6))
    env = SceneEnv(eef={"robot0": hand}, objects={})
    gen = rope_generator(infos, eef_pose=hand)
    gen.env = env
    mixer = SkillMixer(rule="randmix", chunk_seed=4, arms=Arms(("robot0",)))
    rt = make_runtime(Method0(mixer), records_dir=tmp_path)
    rt.begin_attempt(gen, 0)
    primary = 2
    for k, nodes in enumerate((rope_cloud(0.03, 0.4), rope_cloud(0.04, 0.6))):
        local_vars = soft_locals(infos, primary=primary, subtask=k, soft=soft)
        local_vars.update({"self": gen, "selected_src_subtask_boundary": T([0, 4] if k == 0 else [4, 6]),
                           "subtask_object_pose": T(hand @ pose(0.02 + 0.01 * k)), "subtask_nodal_positions": T(nodes)})
        if soft:
            local_vars.pop("subtask_object_pose")
        rt.mix_segment(gen, local_vars)
        rt.record_subtask(local_vars)
    rt.end_attempt(gen, 0, {"success": True})
    line = json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[0])
    for k in (0, 1):
        mix = line["subtasks"][k]["mix"]
        assert "weights" in mix and "identity_reason" not in mix
        X, _, kept = design_rows([line], [[True, True]], k, 3, eef="robot0", features=HELD_FEATURES)
        assert kept == [0]
        fit_features = X[0, 3:]
        mixer_features = mix_record_features(mix, HELD_FEATURES, 3)[primary]     # what the mixer scored for that source
        np.testing.assert_allclose(fit_features, mixer_features, atol=2e-4, err_msg=str(HELD_FEATURES))
        if k == 1:
            assert np.abs(mixer_features[3:5]).sum() > 0 and np.abs(mixer_features[:3]).sum() > 0   # not a trivial match
            assert mix.get("registration") == (OFFICIAL_REGISTRATION if soft else None)     # deformable: how the donors were registered
            assert mix["held"] == {"now": True, "sources": [True, True, True]} and mixer_features[5] == 0


# the unmixed baseline: nothing reaches the generation --------------------------------------------------------------
@pytest.mark.parametrize("held_rule", [HELD_RULE, {"version": 9, "rigid_m": 5.0, "nodes_m": 5.0}, None])
def test_unmixed_generation_returns_the_builders_own_objects_and_draws_nothing(torch, tmp_path, monkeypatch, held_rule):
    """Method none: ``mix_segment`` hands back the very objects the official builder sliced without reading the scene, and
    the scene reading for the record (whatever the hold rule, and when it fails) draws from no random stream, so the
    official code runs on exactly what it would without the grafts."""
    import random

    from tmg.generation.generators.isaaclab import runtime as module

    infos = hand_on_rope([rope_info(6, 0), rope_info(6, 10), rope_info(6, 20)])
    gen = rope_generator(infos, eef_pose=node_pose(rope_cloud(0.6, 0.9)))
    rt = make_runtime(Method0(None), records_dir=tmp_path, held_rule=held_rule)
    reads = []
    original = module.IsaacLabRuntime.subtask_scene
    monkeypatch.setattr(module.IsaacLabRuntime, "subtask_scene", lambda self, g, lv: reads.append(lv["subtask_ind"]) or original(self, g, lv))
    np.random.seed(5)
    random.seed(5)
    before = (np.random.get_state()[1].copy(), random.getstate(), torch.random.get_rng_state())
    rt.begin_attempt(gen, 0)
    for k, nodes in enumerate((rope_cloud(0.5, 0.7), rope_cloud(0.6, 0.9))):
        local_vars = soft_locals(infos, primary=2, subtask=k, soft=True)
        local_vars.update({"self": gen, "subtask_nodal_positions": T(nodes), "selected_src_subtask_boundary": T([0, 4] if k == 0 else [4, 6])})
        if k == 1:
            local_vars.pop("subtask_nodal_positions")              # the scene of the second subtask cannot be read
        out = rt.mix_segment(gen, local_vars)
        assert reads == list(range(k))                             # mix_segment itself read no scene
        assert out[0] is local_vars["src_subtask_eef_poses"] and out[1] is local_vars["src_subtask_target_poses"]
        assert out[2] is local_vars["src_subtask_gripper_actions"] and rt.last_mix("robot0") is None
        rt.record_subtask(local_vars)
    rt.end_attempt(gen, 0, {"success": True})
    after = (np.random.get_state()[1], random.getstate(), torch.random.get_rng_state())
    assert np.array_equal(before[0], after[0]) and before[1] == after[1] and before[2] == after[2]
    assert reads == [0, 1] and "scene_error" in rt.rec.subtasks[1] and "scene_error" not in rt.rec.subtasks[0]


def test_a_deformable_object_carried_to_another_reference_object_is_a_scene_error_not_a_silent_zero(torch, tmp_path, monkeypatch):
    """The in-hand pose of a deformable object against another reference object is not implemented. That is said: the
    subtask's scene is an error (NotImplementedError), so a mixing attempt is marked invalid, instead of the subtask
    being mixed with in-hand features of 0."""
    import json

    install_fake_registration(monkeypatch, lambda **kw: kw["src_eef_poses"])
    infos = hand_on_rope([rope_info(6, 0), rope_info(6, 10)])
    for info, offset in zip(infos, (0, 10)):
        info.object_poses["peg"] = T(np.stack([pose(200 + offset + i) for i in range(6)]))
    gen = rope_generator(infos, eef_pose=node_pose(rope_cloud(0.6, 0.9)))
    mixer = RecordingMixer()
    rt = make_runtime(Method0(mixer), records_dir=tmp_path)
    rt.begin_attempt(gen, 0)
    rope_subtask(rt, gen, infos, 0, 1, rope_cloud(0.5, 0.7))            # grasp the rope (deformable)
    local_vars = soft_locals(infos, primary=1, subtask=1, soft=False)   # then a subtask whose reference object is a rigid peg
    local_vars.update({"self": gen, "subtask_object_name": "peg", "subtask_object_pose": T(pose(300.0)),
                       "selected_src_subtask_boundary": T([4, 6])})
    out = rt.mix_segment(gen, local_vars)
    rt.record_subtask(local_vars)
    rt.end_attempt(gen, 0, {"success": True})
    line = json.loads((tmp_path / "attempts.jsonl").read_text().splitlines()[0])
    s1 = line["subtasks"][1]
    assert out[0] is local_vars["src_subtask_eef_poses"] and len(mixer.contexts) == 1          # only the first subtask was mixed
    assert s1["scene_error"].startswith("scene: NotImplementedError") and "deformable" in s1["scene_error"]
    assert line["invalid"] == "scene_error" and s1["mix"]["mixed"] is False and "inhand_features" not in s1["mix"]
