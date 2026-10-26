"""What produced a result folder: code revision, installed packages, machine, image, time."""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from importlib import metadata
from pathlib import Path

from tmg import REPO_ROOT, __version__

TRACKED_PACKAGES = ("tmg", "mimicgen", "robomimic", "robosuite", "mujoco", "numpy", "h5py", "hydra-core",
                    "omegaconf", "lerobot", "torch", "isaaclab", "isaaclab_mimic", "softmimicgen", "cpgen")
REVISION_ENV = "TMG_REVISION"
IMAGE_ENV = "TMG_IMAGE"


def code_revision() -> dict:
    """Git revision of the repository, or the revision baked into the image (``TMG_REVISION``)."""
    baked = os.environ.get(REVISION_ENV)
    try:
        head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(REPO_ROOT), "status", "--porcelain"], capture_output=True,
                               text=True, check=True, timeout=10).stdout.strip() != ""
        return {"git": head, "dirty": dirty, "baked": baked}
    except Exception:  # noqa: BLE001 - no git in the image is the normal case
        return {"git": None, "dirty": None, "baked": baked}


def package_versions() -> dict:
    """Installed versions of the tracked packages (``TRACKED_PACKAGES``); absent packages are skipped."""
    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def provenance(extra: dict | None = None) -> dict:
    """Everything that identifies how a result was produced.

    Args:
        extra: additional keys to record (for example the stage name and output folder).

    Returns:
        A JSON-able dict with the tmg version, code revision, package versions, image name, host, platform,
        Python version and timestamp.
    """
    data = {
        "tmg_version": __version__,
        "revision": code_revision(),
        "packages": package_versions(),
        "image": os.environ.get(IMAGE_ENV),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if extra:
        data.update(extra)
    return data


def write_provenance(folder: Path, extra: dict | None = None) -> Path:
    """Write :func:`provenance` as ``<folder>/provenance.json`` and return the path."""
    path = Path(folder) / "provenance.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(provenance(extra), indent=2, sort_keys=True) + "\n")
    return path
