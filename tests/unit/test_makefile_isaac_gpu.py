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


def test_the_shared_gpu_machine_is_recognised_by_its_declared_hostname_whatever_the_cluster():
    """CLUSTER left at its default (local) on the Isaac server: its cluster config declares the machine's hostname, an
    explicit setting, instead of the Makefile guessing from a folder that happens to exist."""
    import yaml

    isaac = yaml.safe_load((REPO / "configs/machine/cluster/isaac.yaml").read_text())
    assert isaac["shared_gpus"] is True and isaac["hostname"]
    out = dry_run("generate", "GENERATOR=softmimicgen", f"HOSTNAME={isaac['hostname']}")
    assert "set ISAAC_GPU" in out and "CLUSTER=local" in out
    assert "set ISAAC_GPU" not in dry_run("generate", "GENERATOR=softmimicgen", "HOSTNAME=some-other-machine")
    assert "set ISAAC_GPU" not in dry_run("generate", "GENERATOR=mimicgen", f"HOSTNAME={isaac['hostname']}")
    assert "set ISAAC_GPU" not in dry_run("generate", "GENERATOR=softmimicgen", f"HOSTNAME={isaac['hostname']}", isaac_gpu="6")
    text = (REPO / "Makefile").read_text()
    assert "data_root" not in text[text.index("GPU_CHOICE_CHECK = @true"):text.index("ifneq ($(GPU_VARIABLE),)")]


def test_the_host_name_is_asked_of_the_machine_and_cut_at_the_first_dot():
    """A HOSTNAME in the environment (often fully qualified, or simply wrong) is not used: the Makefile asks the machine
    (`hostname -s`, assigned with `:=`). A name given on the command line is cut at its first dot, so a fully qualified
    name still matches the short name the cluster config declares and cannot switch the guard off."""
    import socket
    import yaml

    name = yaml.safe_load((REPO / "configs/machine/cluster/isaac.yaml").read_text())["hostname"]
    assert "set ISAAC_GPU" in dry_run("generate", "GENERATOR=softmimicgen", f"HOSTNAME={name}.example.org")
    assert "set ISAAC_GPU" in dry_run("generate", "GENERATOR=softmimicgen", f"HOSTNAME={name}")
    text = (REPO / "Makefile").read_text()
    assert "HOSTNAME             := $(shell hostname -s" in text and "HOSTNAME ?=" not in text and "HOSTNAME             ?=" not in text

    def with_environment(value):
        env = {k: v for k, v in os.environ.items() if k not in ("ISAAC_GPU", "CLUSTER", "MAKEFLAGS")}
        return subprocess.run(["make", "-n", "-C", str(REPO), "generate", "GENERATOR=softmimicgen"], env={**env, "HOSTNAME": value},
                              capture_output=True, text=True, check=True).stdout

    if socket.gethostname().split(".")[0] != name:                  # not on that machine: the environment cannot claim to be it
        assert "set ISAAC_GPU" not in with_environment(name) and "set ISAAC_GPU" not in with_environment(f"{name}.example.org")


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
    generators = {p.stem: yaml.safe_load(p.read_text()).get("gpu_variable") for p in (REPO / "configs/experiment/generator").glob("*.yaml")
                  if "name" in yaml.safe_load(p.read_text())}
    assert generators == {"softmimicgen": "ISAAC_GPU", "dexmimicgen": "ISAAC_GPU", "mimicgen": None, "cpgen": None}
    clusters = {p.stem: yaml.safe_load(p.read_text()).get("shared_gpus", False) for p in (REPO / "configs/machine/cluster").glob("*.yaml")}
    assert [name for name, shared in clusters.items() if shared] == ["isaac"]
