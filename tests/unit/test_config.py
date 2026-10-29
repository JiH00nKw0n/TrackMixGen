
import pytest

from tmg.config import compose, resolved


def test_generate_composes_and_resolves(monkeypatch, tmp_path):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    cfg = compose("generate", ["task=square"])
    plain = resolved(cfg)
    assert plain["generator"]["name"] == "mimicgen"
    assert plain["task"]["name"] == "square"
    assert plain["method"]["name"] == "none"
    assert plain["output"] == f"{tmp_path}/pools/mimicgen/square/official_d1/none/pilot20"
    assert plain["task"]["source"].startswith(str(tmp_path))


def test_unknown_key_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    with pytest.raises(Exception):
        compose("generate", ["task=square", "generation.attemptz=3"])


def test_plus_override_is_refused():
    with pytest.raises(ValueError):
        compose("generate", ["task=square", "+extra=1"])


def test_method_variants_compose(monkeypatch, tmp_path):
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    for name, mixing, tracking in [("trackmix", True, True), ("mixup_only", True, False), ("tracking_only", False, True)]:
        plain = resolved(compose("generate", ["task=square", f"method={name}"]))
        assert plain["method"]["name"] == name
        assert plain["method"]["mixing"]["enabled"] is mixing
        assert plain["method"]["tracking"]["enabled"] is tracking
        assert plain["method"]["mixing"]["ranking"] == "mimicgen/square" and plain["method"]["mixing"]["good_fraction"] == 0.5


def test_check_scene_composes_for_every_mimicgen_task(tmp_path):
    for task in ("square", "coffee", "threading", "stack_three", "hammer_cleanup", "mug_cleanup", "three_piece_assembly"):
        cfg = resolved(compose("check_scene", [f"task={task}", "scene=ours", f"paths.root={tmp_path}"]))
        assert cfg["output"].endswith(f"scene_check/mimicgen/{task}/ours") and cfg["resets"] == 200


def test_generator_method_overrides_merge():
    from tmg.generation.base import apply_method_overrides

    method = {"name": "trackmix", "mixing": {"weight_a": 0.8, "max_anchor_rotation_deg": 20.0}, "tracking": {"ds_min": 0.25}}
    gen = {"name": "dexmimicgen", "method_overrides": {"trackmix": {"mixing": {"max_anchor_rotation_deg": 40.0}}}}
    merged = apply_method_overrides(method, gen)
    assert merged["mixing"] == {"weight_a": 0.8, "max_anchor_rotation_deg": 40.0} and merged["tracking"] == {"ds_min": 0.25}
    assert merged["overrides_from_generator"] == "dexmimicgen" and apply_method_overrides(method, {"name": "mimicgen"}) == method


def test_serve_port_override():
    from tmg.config import compose

    cfg = compose("serve", ["checkpoint=/x", "server.port=8772"])
    assert cfg.server.port == 8772 and cfg.server.host == cfg.evaluation.server.host


def test_select_checkpoint_uses_the_run_path_of_train():
    from tmg.config import compose

    ov = ["task=square", "observation=low_dim", "policy=diffusion_policy", "seed=3"]
    assert compose("select_checkpoint", ov).output == compose("train", ov).output


def test_every_generation_config_composes_and_is_named_after_its_file(tmp_path, monkeypatch):
    """Generation configs live flat (full pools) or under test/ (small runs); each composes alone and its `name`, which
    names the pool folder, equals the file name (the test/ folder is not part of the path)."""
    import pathlib
    monkeypatch.setenv("TMG_DATA_ROOT", str(tmp_path))
    root = pathlib.Path(__file__).resolve().parents[2] / "configs/experiment/generation"
    files = sorted(root.glob("*.yaml")) + sorted((root / "test").glob("*.yaml"))
    assert len(files) >= 9
    for f in files:
        option = f.stem if f.parent == root else f"test/{f.stem}"
        plain = resolved(compose("generate", ["generator=mimicgen", "task=square", "scene=ours", "method=none", f"generation={option}"]))
        g = plain["generation"]
        assert g["name"] == f.stem and {"attempts", "chunk", "seed_offset", "protocol"} <= set(g), option
        assert plain["output"].endswith(f"/none/{f.stem}"), option
