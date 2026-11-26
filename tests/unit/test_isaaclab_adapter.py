"""Isaac Lab adapter units that run without Isaac Sim: chunk spec, entrypoint grafting, interval cutting."""
import ast
import json
import types
from pathlib import Path


from tmg.config import compose, resolved
from tmg.core.chunks import plan_chunks
from tmg.generation.generators import make_generator
from tmg.generation.generators.isaaclab.runtime import IsaacLabLayout


def test_soft_generator_uses_the_soft_builder(tmp_path, monkeypatch):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    cfg = resolved(compose("generate", ["generator=softmimicgen", "task=rope", "scene=official", "method=none", "generation=test/pilot20"]))
    gen = make_generator(cfg)
    spec = gen.prepare_chunk(plan_chunks(5, 5, 1)[0], tmp_path / "c")
    assert spec.extra["soft"] is True and spec.extra["package"] == "softmimicgen.datagen" and "--task" not in spec.extra["script_args"]
    # the whole chunk spec of a SoftMimicGen chunk, as the committed adapter wrote it (the entrypoint reads these keys)
    assert spec.extra == {
        "package": "softmimicgen.datagen", "soft": True,
        "script": str(Path(gen.generator_cfg["source_root"]) / "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py"),
        "script_args": ["--headless", "--enable_cameras", "--device", "cuda:0", "--num_envs", "1", "--generation_num_trials", "5",
                        "--input_file", str(tmp_path / "sources/softmimicgen/rope/source.hdf5"), "--output_file", str(tmp_path / "c/generated.hdf5"),
                        "--kit_args", "--/renderer/multiGpu/enabled=false --/renderer/activeGpu=0"],
        "layout": {"eefs": ["robot0"], "arm_action_dims": {"robot0": [0, 6]}, "hand_prefixes": None, "hand_term": None, "measure_links": None,
                   "noise_draw_shapes": [6]},
        "graft_hashes": {"generate": None, "trajectory": None, "execute": None},
        "camera_module": None, "env_name": "Isaac-Rope-Franka-IK-Rel-Mimic-v0"}
    assert list(spec.extra) == ["package", "soft", "script", "script_args", "layout", "graft_hashes", "camera_module", "env_name"]
    assert spec.generator_config == "" and spec.generator == "softmimicgen" and spec.chunk == {"index": 0, "seed": 2, "attempts": 5}
    assert spec.records_dir == str(tmp_path / "c/records") and spec.expected_outputs == [str(tmp_path / "c/records/attempts.jsonl")]
    assert json.loads((tmp_path / "c/chunk_args.json").read_text()) == spec.extra["script_args"]


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
    cfg = resolved(compose("generate", ["generator=softmimicgen", "task=rope", "scene=ours", "method=none", "generation=test/pilot20"]))
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
        observations = types.SimpleNamespace(policy=types.SimpleNamespace())

    import sys

    managers = types.ModuleType("isaaclab.managers")           # the low-dim term is written with Isaac Lab's term config class
    managers.ObservationTermCfg = lambda func, params: types.SimpleNamespace(func=func, params=params)
    monkeypatch.setitem(sys.modules, "isaaclab", types.ModuleType("isaaclab"))
    monkeypatch.setitem(sys.modules, "isaaclab.managers", managers)

    calls = []
    ns = {"setup_env_config": lambda **kw: (EnvCfg(), "success_term")}
    spec_dict = json.loads(json.dumps(spec.__dict__))
    spec_dict["extra"]["disable_graft"] = True
    entrypoint.install(ns, spec_dict)
    env_cfg, term = ns["setup_env_config"](env_name="x")
    assert env_cfg.events.reset_object_position.params["pose_range"]["x"] == (-0.05, 0.15)      # the ours scene is applied
    assert env_cfg.observations.policy.rope_state.params["counts"] == [16]                       # the task's low-dim term is added
    calls.append(env_cfg.datagen_config.seed)
    assert calls == [9001] and term == "success_term"


def _chunk_with_records(folder: Path, subtasks):
    from tmg.generation.base import ChunkSpec

    records = folder / "records"
    records.mkdir(parents=True)
    lines = [{"attempt": i, "success": False, "subtasks": subtasks, "npz": f"attempt_{i:06d}.npz"} for i in range(2)]
    (records / "attempts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in lines))
    for i in range(2):
        (records / f"attempt_{i:06d}.npz").write_bytes(b"")
    return ChunkSpec(chunk={"index": 0, "seed": 1, "attempts": 2}, generator="softmimicgen", task={}, scene={}, method={},
                     protocol={}, generator_config="", folder=str(folder), records_dir=str(records), expected_outputs=[], extra={})


def test_scene_errors_fail_a_calibration_chunk_and_stop_its_pool_but_only_warn_elsewhere(tmp_path, monkeypatch, capsys):
    from tmg.generation.generators.isaaclab import adapter

    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(adapter, "validate_records", lambda folder, attempts: None)
    broken = [{"eef": "robot0", "subtask_ind": 0, "scene_error": "KeyError: 'subtask_nodal_positions'"}]
    for generation, fails in (("test/pilot20", False), ("pool_unmix", True)):
        cfg = resolved(compose("generate", ["generator=softmimicgen", "task=rope", "scene=official", "method=none",
                                            f"generation={generation}"]))
        gen = make_generator(cfg)
        result = gen.chunk_result(_chunk_with_records(tmp_path / generation.replace("/", "_"), broken))
        assert result["scene_errors"] == 2 and ("error" in result) == fails and bool(result.get("stop_pool")) == fails
        assert "WARNING" in capsys.readouterr().out
    clean = gen.chunk_result(_chunk_with_records(tmp_path / "clean", [{"eef": "robot0", "subtask_ind": 0}]))
    assert "scene_errors" not in clean and "error" not in clean
