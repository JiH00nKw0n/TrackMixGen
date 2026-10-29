#!/usr/bin/env python3
"""Install a native environment under a user-owned directory. Never install a kernel driver or accept an EULA.

One stage per image (``train``, ``isaaclab-soft``, ``isaaclab-dex``; ``IMAGES`` lists what each one needs), plus
``tools`` (uv and make), ``graphics`` (the host driver's user-space libraries) and ``verify`` (the installed pins).
The exact command sequence of every image is frozen in ``tests/unit/test_native_bootstrap.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SOFT_COMMIT = "c9d146ba57358a544167de8ebe946caaac8f6220"
SOFT_LAB_COMMIT = "25c69e088fa8122ad189b3a566d38ef24fa74d6f"
DEX_LAB_COMMIT = "3c6e67bb5c7ada942a6d1884ab69338f57596f77"
SOFT_PILLOW_FILE = "source/isaaclab/setup.py"

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
ISAAC_TORCH, ISAAC_TORCHVISION = "2.7.0", "0.22.0"       # the Isaac Sim 5.1 images; the train image pins torch in requirements/
ISAAC_PINS = {"isaacsim": "5.1.0", "torch": f"{ISAAC_TORCH}+cu128", "torchvision": f"{ISAAC_TORCHVISION}+cu128",
              "torchaudio": f"{ISAAC_TORCH}+cu128"}
# Unpinned helpers are resolved together with the exact Isaac Sim requirements: installing the pins alone leaves recent
# pytest/IPython/wheel/Hugging Face releases with incompatible requirements. These are the versions that resolution gives.
ISAAC_HELPER_PINS = {"packaging": "23.0", "click": "8.1.7", "idna": "3.10", "psutil": "5.9.8"}


@dataclass(frozen=True)
class ImageSpec:
    """What ``install`` and ``verify_installation`` need to know about one image."""

    name: str
    python: str                                   # interpreter version of the venv
    expected: dict[str, str]                      # pins verify_installation checks, in the order they are reported
    lab_repo: str | None = None                   # Isaac Lab repository to check out (Isaac Lab images only)
    lab_commit: str | None = None
    source_packages: tuple[str, ...] = field(default=("isaaclab", "isaaclab_assets", "isaaclab_tasks", "isaaclab_mimic"))

    @property
    def isaaclab(self) -> bool:
        return self.lab_repo is not None

    @property
    def python_tuple(self) -> list[int]:
        """``[3, 11]`` for ``"3.11"`` (what ``sys.version_info[:2]`` must equal)."""
        return [int(v) for v in self.python.split(".")]


IMAGES = {
    "train": ImageSpec("train", "3.12", {"torch": "2.8.0+cu128", "torchvision": "0.23.0+cu128", "lerobot": "0.6.1"}),
    "isaaclab-soft": ImageSpec("isaaclab-soft", "3.11",
                               {**ISAAC_PINS, "qpsolvers": "4.8.1", **ISAAC_HELPER_PINS, "pillow": "11.3.0"},
                               lab_repo="https://github.com/masoudmoghani/IsaacLab", lab_commit=SOFT_LAB_COMMIT),
    "isaaclab-dex": ImageSpec("isaaclab-dex", "3.11",
                              {**ISAAC_PINS, "qpsolvers": "4.3.1", **ISAAC_HELPER_PINS, "pillow": "11.3.0"},
                              lab_repo="https://github.com/isaac-sim/IsaacLab", lab_commit=DEX_LAB_COMMIT),
}


def uv_path(root: Path) -> Path:
    return root / "bin/uv"


def venv_python(root: Path, image: str) -> Path:
    return root / "envs" / image / "bin/python"


def apt_cmd(root: Path) -> list[str]:
    """``apt-get`` with its lists and cache under the native root, so no system directory is touched."""
    return ["apt-get", "-o", f"Dir::State::lists={root / 'tools/apt/lists'}", "-o", f"Dir::Cache={root / 'tools/apt/cache'}"]


def soft_pillow_contents(path):
    """Describe the one authorized compatibility change against the pinned upstream source."""
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    if head != SOFT_LAB_COMMIT:
        raise RuntimeError(f"unexpected Soft Isaac Lab revision: {head}")
    original = subprocess.check_output(
        ["git", "-C", str(path), "show", f"HEAD:{SOFT_PILLOW_FILE}"], text=True)
    if original.count("pillow==11.2.1") != 1:
        raise RuntimeError("expected exactly one upstream Pillow 11.2.1 requirement")
    return original, original.replace("pillow==11.2.1", "pillow==11.3.0")


def align_soft_pillow(path):
    original, aligned = soft_pillow_contents(path)
    target = path / SOFT_PILLOW_FILE
    if target.read_text() not in (original, aligned):
        raise RuntimeError(f"refusing to replace other edits in {target}")
    target.write_text(aligned)


def run(args, **kwargs):
    print("RUN", " ".join(map(str, args)), flush=True)
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def download(url, dest):
    dest = Path(dest)
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    print("DOWNLOAD", url, flush=True)
    with urllib.request.urlopen(url, timeout=90) as response, partial.open("wb") as output:
        shutil.copyfileobj(response, output)
    partial.replace(dest)


def checkout(url, path, revision, *, soft_pillow_patch=False):
    if not path.exists():
        run(["git", "clone", "--filter=blob:none", url, path])
    dirty = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True)
    if dirty:
        allowed = False
        if soft_pillow_patch and revision == SOFT_LAB_COMMIT and dirty == f" M {SOFT_PILLOW_FILE}\n":
            _, aligned = soft_pillow_contents(path)
            allowed = (path / SOFT_PILLOW_FILE).read_text() == aligned
        if not allowed:
            raise RuntimeError(f"refusing to replace modified checkout {path}")
    run(["git", "-C", path, "checkout", "--detach", revision])


def tools(root):
    uv = uv_path(root)
    if not uv.exists():
        bootstrap = root / "tools/bootstrap"
        if not (bootstrap / "bin/python").exists():
            run([sys.executable, "-m", "venv", bootstrap])
        run([bootstrap / "bin/python", "-m", "pip", "--isolated", "install", "--no-user", "uv"])
        uv.parent.mkdir(parents=True, exist_ok=True)
        uv.symlink_to(bootstrap / "bin/uv")
    if not shutil.which("make") and not (root / "tools/usr/bin/make").exists():
        folder = root / "downloads/deb"
        folder.mkdir(parents=True, exist_ok=True)
        lists = root / "tools/apt/lists"
        (lists / "partial").mkdir(parents=True, exist_ok=True)
        (root / "tools/apt/cache" / "archives/partial").mkdir(parents=True, exist_ok=True)
        apt = apt_cmd(root)
        if not any(lists.glob("*Packages*")):
            run([*apt, "update"])
        run([*apt, "download", "make"], cwd=folder)
        for deb in folder.glob("make_*.deb"):
            run(["dpkg-deb", "-x", deb, root / "tools"])
    return uv


def graphics(root):
    """Extract the installed host driver's user-space libraries. Do not run nvidia-installer."""
    # Kit's renderer loads GLU even in headless mode. Extract Ubuntu libraries into the user directory.
    packages = root / "downloads/graphics-deb"
    packages.mkdir(parents=True, exist_ok=True)
    for package in ("libglu1-mesa", "libopengl0"):
        archives = list(packages.glob(f"{package}_*.deb"))
        if not archives:
            run([*apt_cmd(root), "download", package], cwd=packages)
            archives = list(packages.glob(f"{package}_*.deb"))
        for archive in archives:
            run(["dpkg-deb", "-x", archive, root / "tools"])
    versions = set(subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).split())
    if len(versions) != 1:
        raise RuntimeError(f"expected one host driver version, got {versions}")
    version = versions.pop()
    if not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", version):
        raise ValueError(version)
    package = root / "downloads" / f"NVIDIA-Linux-x86_64-{version}.run"
    url = f"https://download.nvidia.com/XFree86/Linux-x86_64/{version}/{package.name}"
    download(url, package)
    extracted = root / "graphics" / version
    extracted.parent.mkdir(parents=True, exist_ok=True)
    if not (extracted / f"libGLX_nvidia.so.{version}").exists():
        run(["sh", package, "--extract-only", "--target", extracted])
    libs = root / "graphics/lib"
    libs.mkdir(parents=True, exist_ok=True)
    allowed = ("libGLX_nvidia", "libEGL_nvidia", "libGLES", "libnvoptix", "libnvidia-egl",
               "libnvidia-gl", "libnvidia-gpucomp", "libnvidia-rtcore", "libnvidia-allocator", "libnvidia-tls",
               "libnvidia-ngx")
    for source in extracted.glob("lib*.so*"):
        if source.name.startswith(allowed) and source.is_file():
            target = libs / source.name
            if not target.exists():
                target.symlink_to(source)
    for name, soname in {"libGLX_nvidia": "0", "libEGL_nvidia": "0", "libGLESv1_CM_nvidia": "1",
                         "libGLESv2_nvidia": "2", "libnvoptix": "1", "libnvidia-allocator": "1",
                         "libnvidia-ngx": "1"}.items():
        source = libs / f"{name}.so.{version}"
        target = libs / f"{name}.so.{soname}"
        if source.exists() and not target.exists():
            target.symlink_to(source.name)
    icd = root / "graphics/nvidia_icd.json"
    icd.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {
        "library_path": str(libs / "libEGL_nvidia.so.0"), "api_version": "1.3.0"}}, indent=2))
    egl = root / "graphics/nvidia_egl.json"
    egl.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {
        "library_path": str(libs / "libEGL_nvidia.so.0")}}, indent=2))
    env = {"LD_LIBRARY_PATH": f"{libs}:{root / 'tools/usr/lib/x86_64-linux-gnu'}",
           "VK_DRIVER_FILES": str(icd), "VK_ICD_FILENAMES": str(icd),
           "__EGL_VENDOR_LIBRARY_FILENAMES": str(egl)}
    (root / "graphics/environment.json").write_text(json.dumps(env, indent=2))
    with package.open("rb") as downloaded:
        digest = hashlib.file_digest(downloaded, "sha256").hexdigest()
    (root / "graphics/provenance.json").write_text(json.dumps(
        {"driver_version": version, "download": url, "sha256": digest, "kernel_driver_changed": False}, indent=2))
    run(["python3", REPO / "scripts/probe_native_graphics.py"], env=dict(os.environ, **env))


def install_isaaclab_sources(root, spec: ImageSpec, python, pip):
    """Isaac Sim, the pinned Isaac Lab checkout and the generator's own packages of an Isaac Lab image."""
    pip("--index-strategy", "unsafe-best-match", "--extra-index-url", "https://pypi.nvidia.com",
        f"isaacsim[all,extscache]=={ISAAC_PINS['isaacsim']}")
    pip("--index-url", TORCH_INDEX, f"torch=={ISAAC_TORCH}", f"torchvision=={ISAAC_TORCHVISION}")
    src = root / "src" / spec.name
    src.mkdir(parents=True, exist_ok=True)
    lab = src / "IsaacLab"
    soft = spec.name == "isaaclab-soft"
    checkout(spec.lab_repo, lab, spec.lab_commit, soft_pillow_patch=soft)
    if soft:
        align_soft_pillow(lab)
        checkout("https://github.com/NVlabs/SoftMimicGen", src / "SoftMimicGen", SOFT_COMMIT)
    for name in spec.source_packages:
        pip("--no-build-isolation", "-e", lab / "source" / name)
    if soft:
        # The original Dockerfile follows rapprentice HEAD. Record its actual commit below.
        if not (src / "rapprentice").exists():
            run(["git", "clone", "--depth", "1", "https://github.com/masoudmoghani/rapprentice", src / "rapprentice"])
        pip("-e", src / "rapprentice", f"qpsolvers=={spec.expected['qpsolvers']}", "h5py", "huggingface_hub")
        for name in ["softmimicgen", "softmimicgen_assets", "softmimicgen_tasks"]:
            pip("--no-build-isolation", "-e", src / "SoftMimicGen/source" / name)
        run([python, "-c", ("from huggingface_hub import snapshot_download; import sys; "
             "snapshot_download('masoudmoghani/SoftMimicGen', repo_type='dataset', "
             "allow_patterns='data/*', local_dir=sys.argv[1])"),
             src / "SoftMimicGen/source/softmimicgen_assets"], env=pip.env)
    else:
        pip("h5py", "robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0",
            "osqp==0.6.7.post3", f"qpsolvers=={spec.expected['qpsolvers']}")
    # torch a second time, now as the explicit +cu128 builds (kept from the original sequence: the source installs
    # above may have replaced the wheels; test_native_bootstrap freezes this command list)
    pip("--index-url", TORCH_INDEX, f"torch=={ISAAC_PINS['torch']}", f"torchvision=={ISAAC_PINS['torchvision']}",
        f"torchaudio=={ISAAC_PINS['torchaudio']}")


def install(root, image):
    spec = IMAGES[image]
    uv = tools(root)
    venv = root / "envs" / image
    python = venv_python(root, image)
    env = dict(os.environ, UV_CACHE_DIR=str(root / "cache/uv"), UV_PYTHON_INSTALL_DIR=str(root / "python"),
               UV_LINK_MODE="hardlink",
               PATH=f"{venv / 'bin'}:{root / 'tools/usr/bin'}:{root / 'bin'}:{os.environ.get('PATH', '')}")
    env.setdefault("UV_HTTP_TIMEOUT", "180")
    env.setdefault("UV_CONCURRENT_DOWNLOADS", "4")
    if not python.exists():
        run([uv, "venv", "--python", spec.python, "--seed", venv], env=env)

    def pip(*args):
        run([uv, "pip", "install", "--python", python, *args], env=env)
    pip.env = env                 # the one non-pip command of the Isaac Lab images runs with the same environment
    pip("pip<26", "setuptools<77", "wheel", "toml", "ninja", "cmake<4")
    if spec.isaaclab:
        install_isaaclab_sources(root, spec, python, pip)
    else:
        pip("-r", REPO / "requirements/train-torch.txt")
        pip("-r", REPO / "requirements/train.txt")
    pip("--no-deps", "-e", REPO)
    pip("hydra-core>=1.3,<1.4", "omegaconf>=2.3,<2.4", "pyyaml", "scipy", "msgpack", "websockets")
    if spec.isaaclab:
        pip(*(f"{name}=={version}" for name, version in ISAAC_HELPER_PINS.items()),
            "pytest", "ipython", "wheel", "transformers", "huggingface_hub")
        prune_unused_httpx2(root, python)
    verify_installation(root, image)
    print("INSTALLED", image, flush=True)


def prune_unused_httpx2(root, python):
    """A newer Hugging Face install can leave this incompatible dependency after the resolver downgrades it."""
    code = """
import importlib.metadata as m
import json
from packaging.requirements import Requirement
users = []
for dist in m.distributions():
    for text in dist.requires or []:
        req = Requirement(text)
        if req.name == 'httpx2' and (req.marker is None or req.marker.evaluate({'extra': ''})):
            users.append(dist.metadata['Name'])
print(json.dumps(users))
"""
    users = json.loads(subprocess.check_output([str(python), "-c", code], text=True))
    if not users:
        run([uv_path(root), "pip", "uninstall", "--python", python, "httpx2"])


def verify_installation(root, image):
    """Check the installed pins and source revisions without reinstalling or importing Isaac Sim."""
    spec = IMAGES[image]
    uv = uv_path(root)
    python = venv_python(root, image)
    evidence = root / "install-records"
    evidence.mkdir(exist_ok=True)
    with (evidence / f"{image}.freeze.txt").open("w") as output:
        run([uv, "pip", "freeze", "--python", python], stdout=output)
    check = subprocess.run([str(uv), "pip", "check", "--python", str(python)],
                           check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (evidence / f"{image}.check.txt").write_text(check.stdout)
    expected = dict(spec.expected)
    actual = json.loads(subprocess.check_output([
        str(python), "-c", ("import sys,json,importlib.metadata as m; from packaging.version import Version; "
        "expected=json.loads(sys.argv[1]); actual={n:m.version(n) for n in expected}; "
        "print(json.dumps({'python':list(sys.version_info[:3]),'packages':actual,"
        "'passed':all(Version(actual[n])==Version(v) for n,v in expected.items())}))"),
        json.dumps(expected)], text=True))
    actual["expected_packages"] = expected
    actual["passed"] = actual["passed"] and actual["python"][:2] == spec.python_tuple
    actual["pinned_versions_match"] = actual["passed"]
    actual["dependency_check_passed"] = check.returncode == 0
    actual["passed"] = actual["pinned_versions_match"] and actual["dependency_check_passed"]
    actual["source_commits"] = {}
    actual["source_changes"] = {}
    for source in (root / "src" / image).glob("*/.git"):
        actual["source_commits"][source.parent.name] = subprocess.check_output(
            ["git", "-C", str(source.parent), "rev-parse", "HEAD"], text=True).strip()
        actual["source_changes"][source.parent.name] = subprocess.check_output(
            ["git", "-C", str(source.parent), "diff", "HEAD"], text=True)
    (evidence / f"{image}.versions.json").write_text(json.dumps(actual, indent=2))
    if not actual["pinned_versions_match"]:
        raise RuntimeError(f"installed versions differ from the original pins: {actual}")
    if check.returncode != 0:
        raise RuntimeError(f"unexpected dependency conflicts: {check.stdout}")
    print("VERIFIED PINS", image, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["tools", "graphics", *IMAGES, "verify"])
    parser.add_argument("--image", choices=list(IMAGES))
    args = parser.parse_args()
    root = Path(os.environ.get("TMG_NATIVE_ROOT") or "~/.local/share/tmg-native").expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "tools":
        tools(root)
    elif args.stage == "graphics":
        graphics(root)
    elif args.stage == "verify":
        if not args.image:
            parser.error("verify requires --image")
        verify_installation(root, args.image)
    else:
        install(root, args.stage)


if __name__ == "__main__":
    main()
