"""The Makefile refuses the simulator targets of a generator that takes one GPU (generator config ``gpu_variable``) on a
machine with shared GPUs (cluster config ``shared_gpus``) without that variable, and verify-baseline for a generator that
does not declare ``byte_verify`` (dry runs, nothing executes)."""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(shutil.which("make") is None, reason="make not installed")


def dry_run(*args, isaac_gpu=None):
    env = {k: v for k, v in os.environ.items() if k not in ("ISAAC_GPU", "CLUSTER", "MAKEFLAGS")}
    if isaac_gpu is not None:
        env["ISAAC_GPU"] = isaac_gpu
    return subprocess.run(["make", "-n", "-C", str(REPO), *args], env=env, capture_output=True, text=True, check=True).stdout


@pytest.mark.parametrize("target", ["generate", "resets", "eval", "check-scene", "validate", "evaluate"])
@pytest.mark.parametrize("generator", ["softmimicgen", "dexmimicgen"])
def test_isaac_sim_targets_on_the_isaac_server_need_isaac_gpu(target, generator):
    refused = dry_run(target, "CLUSTER=isaac", f"GENERATOR={generator}")
    assert "set ISAAC_GPU" in refused and "exit 1" in refused.splitlines()[0]     # before anything else, the server too
    allowed = dry_run(target, "CLUSTER=isaac", f"GENERATOR={generator}", isaac_gpu="6")
    assert "set ISAAC_GPU" not in allowed
    if target not in ("validate", "evaluate"):                     # those two run a script that calls make eval
        assert "device=6" in allowed


def test_other_clusters_and_images_are_not_affected():
    assert "set ISAAC_GPU" not in dry_run("generate", "CLUSTER=isaac", "GENERATOR=mimicgen")
    assert "set ISAAC_GPU" not in dry_run("generate", "CLUSTER=local", "GENERATOR=softmimicgen")


def test_the_isaac_host_is_recognised_by_its_data_root_whatever_the_cluster(tmp_path):
    """CLUSTER left at its default (local) on the Isaac server: the isaac cluster's data_root exists there."""
    import yaml

    root = yaml.safe_load((REPO / "configs/machine/cluster/isaac.yaml").read_text())["data_root"]
    out = dry_run("generate", "GENERATOR=softmimicgen", f"SHARED_GPU_HOST_ROOTS={tmp_path}")       # stands in for that root
    assert "set ISAAC_GPU" in out and "CLUSTER=local" in out
    assert "set ISAAC_GPU" not in dry_run("generate", "GENERATOR=softmimicgen", f"SHARED_GPU_HOST_ROOTS={tmp_path / 'absent'}")
    if not Path(root).exists():                                     # not on the Isaac server: the default run is allowed
        assert "set ISAAC_GPU" not in dry_run("generate", "GENERATOR=softmimicgen")


def test_verify_baseline_is_refused_for_generators_that_do_not_declare_byte_verify():
    for generator in ("softmimicgen", "dexmimicgen", "cpgen"):
        out = dry_run("verify-baseline", f"GENERATOR={generator}", "CLUSTER=isaac", isaac_gpu="6")
        assert "does not declare byte_verify" in out.splitlines()[0] and "exit 1" in out.splitlines()[0]
    assert "does not declare byte_verify" not in dry_run("verify-baseline", "GENERATOR=mimicgen")


def test_the_guard_reads_the_configs_not_generator_cluster_or_image_names():
    """Which generators need a GPU choice and which machines share their GPUs are config keys: the Makefile's guard names
    no simulator, image or cluster."""
    import yaml

    text = (REPO / "Makefile").read_text()
    guard = text[text.index("GPU_CHOICE_CHECK = @true"):text.index("ifneq (,$(filter isaaclab-%,$(IMAGE)))")]
    assert "isaac" not in guard.lower().replace("isaac_gpu", "") and "gpu_variable" in guard and "shared_gpus" in guard
    assert "ISAAC_GPU_CHECK" not in text and "ON_ISAAC_HOST" not in text
    generators = {p.stem: yaml.safe_load(p.read_text()).get("gpu_variable") for p in (REPO / "configs/experiment/generator").glob("*.yaml")}
    assert generators == {"softmimicgen": "ISAAC_GPU", "dexmimicgen": "ISAAC_GPU", "mimicgen": None, "cpgen": None}
    clusters = {p.stem: yaml.safe_load(p.read_text()).get("shared_gpus", False) for p in (REPO / "configs/machine/cluster").glob("*.yaml")}
    assert [name for name, shared in clusters.items() if shared] == ["isaac"]
