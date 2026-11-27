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


def _chunk_with_records(folder: Path, lines):
    from tmg.generation.base import ChunkSpec

    records = folder / "records"
    records.mkdir(parents=True)
    lines = [{"attempt": i, "npz": f"attempt_{i:06d}.npz", **line} for i, line in enumerate(lines)]
    (records / "attempts.jsonl").write_text("".join(json.dumps(a) + "\n" for a in lines))
    for i in range(len(lines)):
        (records / f"attempt_{i:06d}.npz").write_bytes(b"")
    return ChunkSpec(chunk={"index": 0, "seed": 1, "attempts": len(lines)}, generator="softmimicgen", task={}, scene={}, method={},
                     protocol={}, generator_config="", folder=str(folder), records_dir=str(records), expected_outputs=[], extra={})


RULE = {"version": 1, "rigid_m": 0.2, "nodes_m": 0.1}


def rope_line(success=True, grasp_step=5, held=True, scene_error=None, invalid=None, mixed=None):
    """One Rope attempt record of record version 2: grasp (signal grasp, fired at ``grasp_step``), then shaping from step 10."""
    def sub(k):
        s = {"eef": "robot0", "subtask_ind": k, "object_name": "object", "src_demo_ind": 0, "start_step": 10 * k,
             "term_signal": "grasp" if k == 0 else None, "held_rule": RULE, "mix": None}
        if k == 1:
            s["held"] = {"now": held, "source": True, "deformable": True}
        if mixed is not None:
            s["mix"] = {"subtask": k, "mixed": mixed, "weights": [1.0]} if mixed else {
                "subtask": k, "mixed": False, "identity_reason": "scene_error: scene: ValueError: no yaw"}
        return s

    subs = [sub(0), sub(1)]
    if scene_error:
        subs[1] = {"eef": "robot0", "subtask_ind": 1, "object_name": "object", "start_step": 10, "scene_error": scene_error,
                   "mix": subs[1]["mix"]}
    line = {"success": success, "subtasks": subs, "n_steps": 30, "term_first": {"grasp": grasp_step}, "record_version": 2,
            "scene_errors": int(bool(scene_error))}
    return dict(line, invalid=invalid) if invalid else line


def soft_generator(tmp_path, monkeypatch, generation, method="none", task="rope", overrides=()):
    from tmg.generation.generators.isaaclab import adapter

    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    monkeypatch.setattr(adapter, "validate_records", lambda folder, attempts: None)
    return make_generator(resolved(compose("generate", ["generator=softmimicgen", f"task={task}", "scene=official",
                                                        f"method={method}", f"generation={generation}", *overrides])))


RANDMIX = ("generation.name=pool_randmix", "method.mixing.rule=randmix", "method.name=skillmix_randmix")


def test_which_pools_need_their_subtask_scenes_is_a_generation_flag_and_the_method_not_a_pool_name(tmp_path, monkeypatch):
    """``generation.purpose: calibration`` (pool_unmix.yaml, kept by the randmix pool that overrides only its name) or a
    mixing method; a pool that merely has the name of a calibration pool does not count."""
    from tmg.generation.generators.isaaclab import adapter

    assert not hasattr(adapter, "CALIBRATION_POOLS")
    assert soft_generator(tmp_path, monkeypatch, "pool_unmix").needs_subtask_scenes()
    assert soft_generator(tmp_path, monkeypatch, "pool_unmix", "skillmix", overrides=RANDMIX).needs_subtask_scenes()
    assert not soft_generator(tmp_path, monkeypatch, "test/pilot20").needs_subtask_scenes()
    assert not soft_generator(tmp_path, monkeypatch, "test/pilot20", overrides=("generation.name=pool_unmix",)).needs_subtask_scenes()
    assert soft_generator(tmp_path, monkeypatch, "test/pilot20", overrides=("generation.purpose=calibration",)).needs_subtask_scenes()
    assert soft_generator(tmp_path, monkeypatch, "test/pilot20", "skillmix", overrides=RANDMIX).needs_subtask_scenes()   # a mixing pool


def test_scene_errors_stop_a_pool_only_above_the_configured_share_of_a_chunk(tmp_path, monkeypatch, capsys):
    """One unreadable scene in a chunk (a folded towel) costs that attempt; the pool stops when more than
    ``scene_reading.max_error_rate`` (0.1) of a chunk's attempts are affected, and only a pool that needs the scenes."""
    error = "scene: ValueError: the planar rotation of this node cloud is undefined"
    few = [rope_line() for _ in range(19)] + [rope_line(scene_error=error)]                # 1 of 20: 5 %
    many = [rope_line() for _ in range(17)] + [rope_line(scene_error=error)] * 3           # 3 of 20: 15 %
    calibration = soft_generator(tmp_path, monkeypatch, "pool_unmix")
    assert calibration.scene_reading()["max_error_rate"] == 0.1
    result = calibration.chunk_result(_chunk_with_records(tmp_path / "few", few))
    assert result["attempts_without_scene"] == 1 and result["scene_errors"] == 1 and "error" not in result and "stop_pool" not in result
    assert "WARNING" in capsys.readouterr().out
    result = calibration.chunk_result(_chunk_with_records(tmp_path / "many", many))
    assert result["stop_pool"] is True and "3 of 20 attempts" in result["error"] and "max_error_rate" in result["error"]
    baseline = soft_generator(tmp_path, monkeypatch, "test/pilot20")                         # an unmixed baseline pool: warned only
    result = baseline.chunk_result(_chunk_with_records(tmp_path / "baseline", many))
    assert result["attempts_without_scene"] == 3 and "error" not in result and "WARNING" in capsys.readouterr().out
    clean = calibration.chunk_result(_chunk_with_records(tmp_path / "clean", [rope_line() for _ in range(4)]))
    assert "scene_errors" not in clean and "error" not in clean and clean["sanity"]["errors"] == []


def test_a_mixing_pool_counts_its_invalid_attempts_and_stops_above_the_share(tmp_path, monkeypatch, capsys):
    error = "scene: ValueError: no yaw"
    bad = rope_line(scene_error=error, invalid="scene_error", mixed=False)
    mixing = soft_generator(tmp_path, monkeypatch, "pool_unmix", "skillmix", overrides=RANDMIX)
    result = mixing.chunk_result(_chunk_with_records(tmp_path / "one", [rope_line(mixed=True) for _ in range(19)] + [bad]))
    assert result["invalid_attempts"] == 1 and "error" not in result and result["successes"] == 20     # the official count stays
    assert result["sanity"]["blocks"]["0"]["mix"] == {"records": 19, "identity": 0, "identity_rate": 0.0, "identity_reasons": {}}
    result = mixing.chunk_result(_chunk_with_records(tmp_path / "many", [rope_line(mixed=True) for _ in range(5)] + [bad] * 5))
    assert result["stop_pool"] is True and "5 of 10 attempts" in result["error"]


def test_the_chunk_result_carries_the_sanity_report_and_a_calibration_pool_stops_on_its_errors(tmp_path, monkeypatch, capsys):
    """The first chunk already shows a reached rule that misreads the records (every success "grasped" after shaping began),
    a carried object that is never held, and the declared fallback of a task without signals."""
    late = [rope_line(grasp_step=25) for _ in range(12)]
    calibration = soft_generator(tmp_path, monkeypatch, "pool_unmix")
    result = calibration.chunk_result(_chunk_with_records(tmp_path / "late", late))
    assert result["stop_pool"] is True and "12 of 12 successful attempts count as not having reached it" in result["error"]
    assert result["sanity"]["blocks"]["1"]["successes_unreached_rate"] == 1.0 and "ERROR sanity" in capsys.readouterr().out
    baseline = soft_generator(tmp_path, monkeypatch, "test/pilot20")
    result = baseline.chunk_result(_chunk_with_records(tmp_path / "late_baseline", late))
    assert "error" not in result and result["sanity"]["errors"]                             # reported, the baseline pool goes on
    loose = [rope_line(held=False) for _ in range(12)]
    result = calibration.chunk_result(_chunk_with_records(tmp_path / "loose", loose))
    assert "error" not in result and result["sanity"]["blocks"]["1"]["held"]["now"] == 0.0
    assert any("holds the carried object in none of 12" in w for w in result["sanity"]["warnings"])
    assert "WARNING sanity" in capsys.readouterr().out
    few = calibration.chunk_result(_chunk_with_records(tmp_path / "few", late[:4]))          # under min_successes: not judged
    assert "error" not in few
    # Towel records carry no signal at all: its task declares task_success, so its chunks pass; Rope's would not
    silent = [dict(rope_line(), term_first={}) for _ in range(12)]
    towel = soft_generator(tmp_path, monkeypatch, "pool_unmix", task="towel")
    assert towel.fit_layout().reached == "task_success"
    assert "error" not in towel.chunk_result(_chunk_with_records(tmp_path / "towel", silent))
    result = calibration.chunk_result(_chunk_with_records(tmp_path / "silent_rope", silent))
    assert result["stop_pool"] is True and "cannot read the records" in result["error"]


def test_the_chunk_spec_carries_the_hold_rule_of_the_generator_config_outside_its_identity(tmp_path, monkeypatch):
    gen = soft_generator(tmp_path, monkeypatch, "test/pilot20")
    spec = gen.prepare_chunk(plan_chunks(5, 5, 1)[0], tmp_path / "c")
    assert spec.scene_reading == {"held_rule": RULE} and "scene_reading" not in spec.identity()
    assert "held_rule" not in json.dumps(spec.extra) and "fit" not in spec.identity()["task"]
    from tmg.generation.generators.isaaclab.runtime import held_rule_of

    assert held_rule_of(json.loads(json.dumps(spec.__dict__))).as_dict() == RULE          # what the chunk process reads
