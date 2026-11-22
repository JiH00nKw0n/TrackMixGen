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
    assert ctx.extra == {"eef": "robot0", "soft": False, "reference_object": "rope"}
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
    assert ctx.extra == {"eef": "robot0", "soft": True, "reference_object": "rope"}
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
