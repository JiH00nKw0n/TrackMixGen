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
REGISTRATION = {"use_rotation_transform": True, "bend_coef": 0.1, "rot_coef": 0.001}


def finish(spec, result):
    """Write the chunk's CHUNK_DONE.json, as the runner does for a chunk without an error."""
    from tmg.core.chunks import CHUNK_DONE, write_json

    write_json(Path(spec.folder) / CHUNK_DONE, result)
    return result


def chunk(generator, pool, name, lines):
    """One more chunk of ``pool``: its result, kept as finished when it has no error."""
    result = generator.chunk_result(_chunk_with_records(pool / name, lines))
    return result if "error" in result else finish(_spec_of(pool / name), result)


def _spec_of(folder):
    from types import SimpleNamespace

    return SimpleNamespace(folder=str(folder))


def rope_line(success=True, grasp_step=5, held=True, scene_error=None, invalid=None, mixed=None):
    """One Rope attempt record of record version 2: grasp (signal grasp, fired at ``grasp_step``), then shaping from step 10."""
    def sub(k):
        s = {"eef": "robot0", "subtask_ind": k, "object_name": "object", "src_demo_ind": 0, "start_step": 10 * k,
             "term_signal": "grasp" if k == 0 else None, "held_rule": RULE, "mix": None}
        if k == 1:
            s["held_distance"] = {"now": 0.005 if held else 0.35, "source": 0.002, "deformable": True}
        if mixed is not None:
            s["mix"] = {"subtask": k, "mixed": mixed, "weights": [1.0], "registration": REGISTRATION} if mixed else {
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


def test_scene_errors_are_judged_on_the_pools_finished_chunks_not_on_one_chunks_share(tmp_path, monkeypatch, capsys):
    """A chunk with 3 unreadable scenes of 20 (15 %, three times the 5 % limit) does not stop a pool: a 5 % rate does that
    to one chunk in thirteen. The same share chunk after chunk does, once the pool's counts show it; a chunk in which
    every attempt fails stops the pool at once; and only a pool that needs the scenes is stopped."""
    error = "scene: ValueError: the planar rotation of this node cloud is undefined"
    unlucky = [rope_line() for _ in range(17)] + [rope_line(scene_error=error)] * 3          # 3 of 20
    clean = [rope_line() for _ in range(20)]
    calibration = soft_generator(tmp_path, monkeypatch, "pool_unmix")
    assert calibration.fit_layout().sanity.max_scene_error_rate == 0.05
    pool = tmp_path / "healthy" / "chunks"
    first = chunk(calibration, pool, "seed1", unlucky)
    assert first["attempts_without_scene"] == 3 and "error" not in first and first["sanity"]["without_scene"] == 3
    assert first["sanity_verdict"] == {"chunks": 1, "errors": [], "warnings": []} and "tripod" not in str(first["scene_error_examples"])
    for index in range(2, 8):                                        # the rest of the pool is clean: 3 of 140 in the end
        result = chunk(calibration, pool, f"seed{index}", clean)
        assert "error" not in result and result["sanity_verdict"]["chunks"] == index
    assert "3 of 140 (2.1 %) attempts without the scene" in capsys.readouterr().out
    pool = tmp_path / "sick" / "chunks"                              # 15 % in every chunk: shown by the fourth chunk
    results = [chunk(calibration, pool, f"seed{index}", unlucky) for index in range(1, 5)]
    assert ["error" in r for r in results] == [False, False, False, True] and results[3]["stop_pool"] is True
    assert "pool up to chunk seed4 (4 chunks)" in results[3]["error"] and "12 of 80 (15.0 %) attempts without the scene" in results[3]["error"]
    assert "record_checks.max_scene_error_rate = 0.05" in results[3]["error"]
    broken = calibration.chunk_result(_chunk_with_records(tmp_path / "broken" / "chunks" / "seed1", [rope_line(scene_error=error)] * 20))
    assert broken["stop_pool"] is True and "20 of 20 (100.0 %)" in broken["error"]            # systematic: at the first chunk
    baseline = soft_generator(tmp_path, monkeypatch, "test/pilot20")                         # an unmixed baseline pool: said, not stopped
    result = baseline.chunk_result(_chunk_with_records(tmp_path / "baseline" / "chunks" / "seed1", [rope_line(scene_error=error)] * 20))
    assert "error" not in result and result["sanity_verdict"]["errors"] and "ERROR sanity" in capsys.readouterr().out


def test_a_mixing_pool_counts_its_invalid_attempts_and_records_the_donor_registration(tmp_path, monkeypatch, capsys):
    bad = rope_line(scene_error="scene: ValueError: no yaw", invalid="scene_error", mixed=False)
    mixing = soft_generator(tmp_path, monkeypatch, "pool_unmix", "skillmix", overrides=RANDMIX)
    pool = tmp_path / "mix" / "chunks"
    result = chunk(mixing, pool, "seed1", [rope_line(mixed=True) for _ in range(19)] + [bad])
    assert result["invalid_attempts"] == 1 and "error" not in result and result["successes"] == 20     # the official count stays
    assert result["sanity"]["blocks"]["0"]["mix"] == {"records": 19, "identity": 0, "identity_reasons": {}}
    assert result["donor_registration"] == [REGISTRATION]              # the settings the donors were registered with
    result = mixing.chunk_result(_chunk_with_records(pool / "seed2", [rope_line(mixed=True) for _ in range(5)] + [bad] * 10))
    assert result["stop_pool"] is True and "11 of 35 (31.4 %) attempts without the scene" in result["error"]
    unmixed = soft_generator(tmp_path, monkeypatch, "pool_unmix").chunk_result(
        _chunk_with_records(tmp_path / "plain" / "chunks" / "seed1", [rope_line() for _ in range(3)]))
    assert "donor_registration" not in unmixed


def test_unreached_successes_stop_a_calibration_pool_and_only_warn_in_a_data_pool(tmp_path, monkeypatch, capsys):
    """The share of successes that count as not reached is judged on the pool's finished chunks. A reached rule that
    misreads every record stops a calibration pool at its first chunk; a mixing data pool (softmax, not read by the fit)
    gets a warning; a task whose environment publishes no signals passes through its declared rule, and the same records
    under a rule that cannot read them stop at once."""
    late = [rope_line(grasp_step=25) for _ in range(12)]              # every success "grasped" after shaping began
    few = [rope_line() for _ in range(18)] + [rope_line(grasp_step=25)] * 2            # 2 of 20: nothing shown
    calibration = soft_generator(tmp_path, monkeypatch, "pool_unmix")
    result = calibration.chunk_result(_chunk_with_records(tmp_path / "late" / "chunks" / "seed1", late))
    assert result["stop_pool"] is True and "block 1: 12 of 12 (100.0 %) successful attempts count as not having reached it" in result["error"]
    assert result["sanity"]["blocks"]["1"]["successes_unreached"] == 12 and "ERROR sanity" in capsys.readouterr().out
    pool = tmp_path / "few" / "chunks"
    assert all("error" not in chunk(calibration, pool, f"seed{index}", few) for index in range(1, 4))      # 6 of 60
    (tmp_path / "rule.yaml").write_text("rule: softmax\nnum_sources: 1\nsubtasks: {0: {a: [0.0], g: {dx: 0, dy: 0, rotation: 0, "
                                        "inhand_position: 0, inhand_rotation: 0}, d: 1, c0: 0}}\n")
    data = soft_generator(tmp_path, monkeypatch, "test/pilot20", "skillmix", overrides=(f"method.mixing.coefficients={tmp_path / 'rule.yaml'}",))
    assert data.needs_subtask_scenes() and data.generation.get("purpose") != "calibration"
    result = data.chunk_result(_chunk_with_records(tmp_path / "data" / "chunks" / "seed1", late))
    assert "error" not in result and result["sanity_verdict"]["errors"] == []
    assert "12 of 12" in result["sanity_verdict"]["warnings"][0] and "WARNING sanity" in capsys.readouterr().out
    loose = [rope_line(held=False) for _ in range(12)]
    result = calibration.chunk_result(_chunk_with_records(tmp_path / "loose" / "chunks" / "seed1", loose))
    assert "error" not in result and result["sanity"]["blocks"]["1"]["held"] == {"tested": 12, "now": 0, "sources": {"0": [12, 12]}}
    assert any("holds the carried object in none of 12" in w for w in result["sanity_verdict"]["warnings"])
    # Towel records carry no signal at all: its task declares task_success, so its chunks pass; Rope's would not
    silent = [dict(rope_line(), term_first={}) for _ in range(12)]
    towel = soft_generator(tmp_path, monkeypatch, "pool_unmix", task="towel")
    assert towel.fit_layout().reached == "task_success"
    assert "error" not in towel.chunk_result(_chunk_with_records(tmp_path / "towel" / "chunks" / "seed1", silent))
    result = calibration.chunk_result(_chunk_with_records(tmp_path / "silent_rope" / "chunks" / "seed1", silent))
    assert result["stop_pool"] is True and "cannot read the records" in result["error"]


def test_the_chunk_spec_carries_the_hold_rule_of_the_generator_config_outside_its_identity(tmp_path, monkeypatch):
    gen = soft_generator(tmp_path, monkeypatch, "test/pilot20")
    spec = gen.prepare_chunk(plan_chunks(5, 5, 1)[0], tmp_path / "c")
    assert spec.scene_reading == {"held_rule": RULE} and "scene_reading" not in spec.identity()
    assert "held_rule" not in json.dumps(spec.extra) and "fit" not in spec.identity()["task"]
    from tmg.generation.generators.isaaclab.runtime import held_rule_of

    assert held_rule_of(json.loads(json.dumps(spec.__dict__))).as_dict() == RULE          # what the chunk process reads
