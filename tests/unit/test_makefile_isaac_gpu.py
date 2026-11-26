"""The Makefile refuses the Isaac Sim targets on the shared Isaac server without ISAAC_GPU (dry runs, nothing executes)."""
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
    return subprocess.run(["make", "-n", "-C", str(REPO), *args], env=env, capture_output=True, text=True).stdout


@pytest.mark.parametrize("target", ["generate", "verify-baseline", "resets", "eval", "check-scene"])
@pytest.mark.parametrize("generator", ["softmimicgen", "dexmimicgen"])
def test_isaac_sim_targets_on_the_isaac_server_need_isaac_gpu(target, generator):
    refused = dry_run(target, "CLUSTER=isaac", f"GENERATOR={generator}")
    assert "set ISAAC_GPU" in refused and "exit 1" in refused.splitlines()[0]
    allowed = dry_run(target, "CLUSTER=isaac", f"GENERATOR={generator}", isaac_gpu="6")
    assert "set ISAAC_GPU" not in allowed and "device=6" in allowed


def test_other_clusters_and_images_are_not_affected():
    assert "set ISAAC_GPU" not in dry_run("generate", "CLUSTER=isaac", "GENERATOR=mimicgen")
    assert "set ISAAC_GPU" not in dry_run("generate", "CLUSTER=local", "GENERATOR=softmimicgen")
