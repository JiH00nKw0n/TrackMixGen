"""Isaac Lab adapter units that run without Isaac Sim: chunk spec, entrypoint grafting, interval cutting."""
import ast
import types

import numpy as np
import pytest

from tmg.config import compose, resolved
from tmg.core.chunks import plan_chunks
from tmg.generation.generators import make_generator
from tmg.generation.generators.isaaclab.runtime import IsaacLabLayout, _cut_interval


def test_dex_chunk_spec_carries_layout_script_and_overrides(tmp_path, monkeypatch):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    cfg = resolved(compose("generate", ["generator=dexmimicgen", "task=pick_place", "scene=ours", "method=trackmix", "generation=pilot20"]))
    gen = make_generator(cfg)
    chunk = plan_chunks(20, 20, 9000)[0]
    spec = gen.prepare_chunk(chunk, tmp_path / "chunk")
    assert spec.extra["package"] == "isaaclab_mimic.datagen" and spec.extra["soft"] is False
    assert spec.extra["layout"]["hand_prefixes"] == {"left": "L_", "right": "R_"} and spec.extra["layout"]["noise_draw_shapes"] == [3, 3, 4, 4]
    args = spec.extra["script_args"]
    assert "--enable_pinocchio" in args and args[args.index("--generation_num_trials") + 1] == "20"
    assert args[args.index("--task") + 1] == "TMG-PickPlace-GR1T2-Images-Mimic-v0"
    assert spec.method["mixing"]["max_anchor_rotation_deg"] == 40.0 and spec.scene["override_bounds"] is False
    assert gen.chunk_command(spec, tmp_path / "chunk" / "spec.json")[-2:] == ["--spec", str(tmp_path / "chunk" / "spec.json")]
    assert gen.demo_layout()["labels_from_records"] is True


def test_soft_generator_uses_the_soft_builder(tmp_path, monkeypatch):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    cfg = resolved(compose("generate", ["generator=softmimicgen", "task=rope", "scene=official", "method=none", "generation=pilot20"]))
    gen = make_generator(cfg)
    spec = gen.prepare_chunk(plan_chunks(5, 5, 1)[0], tmp_path / "c")
    assert spec.extra["soft"] is True and spec.extra["package"] == "softmimicgen.datagen" and "--task" not in spec.extra["script_args"]


def test_entrypoint_inserts_install_before_main_guard(tmp_path):
    script = tmp_path / "generate_dataset.py"
    script.write_text("import sys\nCALLS = []\ndef setup_env_config(a):\n    return a\ndef main():\n    CALLS.append(setup_env_config(1))\nif __name__ == '__main__':\n    main()\n")
    from tmg.generation.generators.isaaclab import entrypoint

    seen = {}

    def fake_install(namespace, spec):
        seen["spec"] = spec
        original = namespace["setup_env_config"]
        namespace["setup_env_config"] = lambda a: original(a) + 41

    tree = ast.parse(script.read_text())
    index = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.If))
    tree.body.insert(index, ast.parse("_tmg_install(globals(), _tmg_spec)").body[0])
    ast.fix_missing_locations(tree)
    g = {"__name__": "__main__", "__file__": str(script), "_tmg_install": fake_install, "_tmg_spec": {"x": 1}}
    exec(compile(tree, str(script), "exec"), g)
    assert g["CALLS"] == [42] and seen["spec"] == {"x": 1}
    assert entrypoint.install.__doc__


def test_cut_interval_aligns_two_hands_to_the_common_source_interval():
    W = types.SimpleNamespace

    def wps(n, start):
        return [W(pose=np.eye(4) * (start + i), gripper_action=np.zeros(11), noise=0.003) for i in range(n)]

    # right: bridge of 2 then source steps 0..9 (end 10); left: source steps 0..5 (end 6); both bridges start at 0
    active = {"right": {"raw": wps(12, 0), "begin": 0, "end": 10, "prefix": 2, "subtask": 0},
              "left": {"raw": wps(8, 100), "begin": 0, "end": 6, "prefix": 2, "subtask": 0}}
    interval = _cut_interval(active, ["left", "right"], 0, W, {})
    assert (interval.start, interval.end) == (0, 6) and interval.kinds == [0, 0, 1, 1, 1, 1, 1, 1]
    assert interval.source_indices == [0, 0, 0, 1, 2, 3, 4, 5] and len(interval.waypoints["right"]) == 8
    assert interval.waypoints["right"][2] is active["right"]["raw"][2] and interval.waypoints["left"][7] is active["left"]["raw"][7]
    short = {"left": active["left"], "right": {**active["right"], "raw": wps(3, 0)}}
    with pytest.raises(RuntimeError):
        _cut_interval(short, ["left", "right"], 0, W, {})
    unequal = {"left": {**active["left"], "prefix": 1, "raw": wps(7, 100)}, "right": active["right"]}
    with pytest.raises(NotImplementedError):
        _cut_interval(unequal, ["left", "right"], 0, W, {})


def test_layout_hand_layout_reads_the_action_term():
    layout = IsaacLabLayout(eefs=["left", "right"], arm_action_dims={"left": (0, 7), "right": (7, 14)},
                            hand_prefixes={"left": "L_", "right": "R_"}, hand_term="upper_body_ik", measure_links=None)
    names = [f"{h}_{j}" for j in ("a", "b") for h in ("L", "R")]
    term = types.SimpleNamespace(_hand_joint_names=names, cfg=types.SimpleNamespace(hand_joint_names=names))
    env = types.SimpleNamespace(action_manager=types.SimpleNamespace(get_term=lambda n: term))
    hl = layout.hand_layout(env)
    assert hl.indices == {"left": [0, 2], "right": [1, 3]} and hl.arm_dim == 14
    assert IsaacLabLayout(eefs=["robot0"], arm_action_dims={"robot0": (0, 6)}, hand_prefixes=None, hand_term=None, measure_links=None).hand_layout(env) is None


def test_chunk_seed_goes_into_datagen_config_not_the_command_line(tmp_path, monkeypatch):
    """The official generate_dataset.py has no --seed; the entrypoint writes the chunk seed into datagen_config.seed."""
    import json

    from tmg.generation.generators.isaaclab import entrypoint

    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    cfg = resolved(compose("generate", ["generator=softmimicgen", "task=rope", "scene=ours", "method=none", "generation=pilot20"]))
    gen = make_generator(cfg)
    spec = gen.prepare_chunk(plan_chunks(20, 20, 9000)[0], tmp_path / "c")
    assert "--seed" not in spec.extra["script_args"] and spec.chunk["seed"] == 9001
    assert spec.extra["script"].startswith("/opt/src/SoftMimicGen/") or "TMG_SOFTMIMICGEN_SRC" in str(spec.extra["script"])

    class Datagen:
        seed = 1

    import types

    class EnvCfg:
        datagen_config = Datagen()
        events = types.SimpleNamespace(reset_object_position=types.SimpleNamespace(
            params={"pose_range": {"x": (0.0, 0.1)}, "partial_pose_range": {"x": (0.0, 0.1), "y": (0.0, 0.1)}}))

    calls = []
    ns = {"setup_env_config": lambda **kw: (EnvCfg(), "success_term")}
    spec_dict = json.loads(json.dumps(spec.__dict__))
    spec_dict["extra"]["disable_graft"] = True
    entrypoint.install(ns, spec_dict)
    env_cfg, term = ns["setup_env_config"](env_name="x")
    assert env_cfg.events.reset_object_position.params["pose_range"]["x"] == (-0.05, 0.15)      # the ours scene is applied
    calls.append(env_cfg.datagen_config.seed)
    assert calls == [9001] and term == "success_term"
