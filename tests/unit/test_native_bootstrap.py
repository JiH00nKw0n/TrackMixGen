"""The Pillow compatibility patch must preserve unrelated edits on later installs."""
import subprocess

import pytest

from scripts import bootstrap_native as bootstrap


@pytest.fixture
def soft_checkout(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / bootstrap.SOFT_PILLOW_FILE
    source.parent.mkdir(parents=True)
    source.write_text('INSTALL_REQUIRES = ["pillow==11.2.1"]\n')
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-qm", "Original requirement"], check=True)
    revision = subprocess.check_output(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True).strip()
    monkeypatch.setattr(bootstrap, "SOFT_LAB_COMMIT", revision)
    return tmp_path, source, revision


def test_repeated_checkout_keeps_only_authorized_pillow_patch(soft_checkout):
    path, source, revision = soft_checkout
    bootstrap.align_soft_pillow(path)
    bootstrap.checkout("unused", path, revision, soft_pillow_patch=True)
    bootstrap.align_soft_pillow(path)
    assert source.read_text() == 'INSTALL_REQUIRES = ["pillow==11.3.0"]\n'


def test_other_source_edits_are_preserved_and_rejected(soft_checkout):
    path, source, revision = soft_checkout
    bootstrap.align_soft_pillow(path)
    changed = source.read_text() + "OTHER_USER_SETTING = True\n"
    source.write_text(changed)
    with pytest.raises(RuntimeError, match="modified checkout"):
        bootstrap.checkout("unused", path, revision, soft_pillow_patch=True)
    with pytest.raises(RuntimeError, match="other edits"):
        bootstrap.align_soft_pillow(path)
    assert source.read_text() == changed


def test_untracked_user_file_is_not_accepted_as_a_clean_checkout(soft_checkout):
    path, _, revision = soft_checkout
    bootstrap.align_soft_pillow(path)
    extra = path / "user-notes.txt"
    extra.write_text("keep this")
    with pytest.raises(RuntimeError, match="modified checkout"):
        bootstrap.checkout("unused", path, revision, soft_pillow_patch=True)
    assert extra.read_text() == "keep this"


# Every command `install` issues for each image, with the native root as /ROOT, the repository as /REPO and the
# bootstrapping interpreter as {PY}. Frozen before the install steps were reorganised: a change here changes what
# gets installed on the cluster, so it must be deliberate.
INSTALL_COMMANDS = {
    "train": [
        "{PY} -m venv /ROOT/tools/bootstrap",
        "/ROOT/tools/bootstrap/bin/python -m pip --isolated install --no-user uv",
        "/ROOT/bin/uv venv --python 3.12 --seed /ROOT/envs/train",
        "/ROOT/bin/uv pip install --python /ROOT/envs/train/bin/python pip<26 setuptools<77 wheel toml ninja cmake<4",
        "/ROOT/bin/uv pip install --python /ROOT/envs/train/bin/python -r /REPO/requirements/train-torch.txt",
        "/ROOT/bin/uv pip install --python /ROOT/envs/train/bin/python -r /REPO/requirements/train.txt",
        "/ROOT/bin/uv pip install --python /ROOT/envs/train/bin/python --no-deps -e /REPO",
        ("/ROOT/bin/uv pip install --python /ROOT/envs/train/bin/python hydra-core>=1.3,<1.4 omegaconf>=2.3,<2.4 pyyaml scipy "
        "msgpack websockets"),
        "VERIFY /ROOT train",
    ],
    "isaaclab-soft": [
        "{PY} -m venv /ROOT/tools/bootstrap",
        "/ROOT/tools/bootstrap/bin/python -m pip --isolated install --no-user uv",
        "/ROOT/bin/uv venv --python 3.11 --seed /ROOT/envs/isaaclab-soft",
        "/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python pip<26 setuptools<77 wheel toml ninja cmake<4",
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python --index-strategy unsafe-best-match "
        "--extra-index-url https://pypi.nvidia.com isaacsim[all,extscache]==5.1.0"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python "
        "--index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0"),
        ("CHECKOUT https://github.com/masoudmoghani/IsaacLab /ROOT/src/isaaclab-soft/IsaacLab "
        "25c69e088fa8122ad189b3a566d38ef24fa74d6f soft_pillow_patch"),
        "ALIGN /ROOT/src/isaaclab-soft/IsaacLab",
        ("CHECKOUT https://github.com/NVlabs/SoftMimicGen /ROOT/src/isaaclab-soft/SoftMimicGen "
        "c9d146ba57358a544167de8ebe946caaac8f6220"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python --no-build-isolation -e "
        "/ROOT/src/isaaclab-soft/IsaacLab/source/isaaclab"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python --no-build-isolation -e "
        "/ROOT/src/isaaclab-soft/IsaacLab/source/isaaclab_assets"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python --no-build-isolation -e "
        "/ROOT/src/isaaclab-soft/IsaacLab/source/isaaclab_tasks"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python --no-build-isolation -e "
        "/ROOT/src/isaaclab-soft/IsaacLab/source/isaaclab_mimic"),
        "git clone --depth 1 https://github.com/masoudmoghani/rapprentice /ROOT/src/isaaclab-soft/rapprentice",
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python -e /ROOT/src/isaaclab-soft/rapprentice "
        "qpsolvers==4.8.1 h5py huggingface_hub"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python --no-build-isolation -e "
        "/ROOT/src/isaaclab-soft/SoftMimicGen/source/softmimicgen"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python --no-build-isolation -e "
        "/ROOT/src/isaaclab-soft/SoftMimicGen/source/softmimicgen_assets"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python --no-build-isolation -e "
        "/ROOT/src/isaaclab-soft/SoftMimicGen/source/softmimicgen_tasks"),
        ("/ROOT/envs/isaaclab-soft/bin/python -c from huggingface_hub import snapshot_download; import sys; "
         "snapshot_download('masoudmoghani/SoftMimicGen', repo_type='dataset', allow_patterns='data/*', local_dir=sys.argv[1]) "
         "/ROOT/src/isaaclab-soft/SoftMimicGen/source/softmimicgen_assets"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python "
        "--index-url https://download.pytorch.org/whl/cu128 torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128"),
        "/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python --no-deps -e /REPO",
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python hydra-core>=1.3,<1.4 omegaconf>=2.3,<2.4 pyyaml "
        "scipy msgpack websockets"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-soft/bin/python packaging==23.0 click==8.1.7 idna==3.10 "
        "psutil==5.9.8 pytest ipython wheel transformers huggingface_hub"),
        "PRUNE /ROOT /ROOT/envs/isaaclab-soft/bin/python",
        "VERIFY /ROOT isaaclab-soft",
    ],
    "isaaclab-dex": [
        "{PY} -m venv /ROOT/tools/bootstrap",
        "/ROOT/tools/bootstrap/bin/python -m pip --isolated install --no-user uv",
        "/ROOT/bin/uv venv --python 3.11 --seed /ROOT/envs/isaaclab-dex",
        "/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-dex/bin/python pip<26 setuptools<77 wheel toml ninja cmake<4",
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-dex/bin/python --index-strategy unsafe-best-match "
        "--extra-index-url https://pypi.nvidia.com isaacsim[all,extscache]==5.1.0"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-dex/bin/python --index-url https://download.pytorch.org/whl/cu128 "
        "torch==2.7.0 torchvision==0.22.0"),
        "CHECKOUT https://github.com/isaac-sim/IsaacLab /ROOT/src/isaaclab-dex/IsaacLab 3c6e67bb5c7ada942a6d1884ab69338f57596f77",
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-dex/bin/python --no-build-isolation -e "
        "/ROOT/src/isaaclab-dex/IsaacLab/source/isaaclab"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-dex/bin/python --no-build-isolation -e "
        "/ROOT/src/isaaclab-dex/IsaacLab/source/isaaclab_assets"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-dex/bin/python --no-build-isolation -e "
        "/ROOT/src/isaaclab-dex/IsaacLab/source/isaaclab_tasks"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-dex/bin/python --no-build-isolation -e "
        "/ROOT/src/isaaclab-dex/IsaacLab/source/isaaclab_mimic"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-dex/bin/python h5py "
        "robomimic @ git+https://github.com/ARISE-Initiative/robomimic.git@v0.4.0 osqp==0.6.7.post3 qpsolvers==4.3.1"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-dex/bin/python --index-url https://download.pytorch.org/whl/cu128 "
        "torch==2.7.0+cu128 torchvision==0.22.0+cu128 torchaudio==2.7.0+cu128"),
        "/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-dex/bin/python --no-deps -e /REPO",
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-dex/bin/python hydra-core>=1.3,<1.4 omegaconf>=2.3,<2.4 pyyaml "
        "scipy msgpack websockets"),
        ("/ROOT/bin/uv pip install --python /ROOT/envs/isaaclab-dex/bin/python packaging==23.0 click==8.1.7 idna==3.10 "
        "psutil==5.9.8 pytest ipython wheel transformers huggingface_hub"),
        "PRUNE /ROOT /ROOT/envs/isaaclab-dex/bin/python",
        "VERIFY /ROOT isaaclab-dex",
    ],
}


# The environment the committed ``install`` gave its commands: the variables it set on top of ``os.environ``, the same
# for every command that gets an environment, and the positions (in the lists above) of the commands run without one.
UV_ENV = {"UV_CACHE_DIR": "/ROOT/cache/uv", "UV_PYTHON_INSTALL_DIR": "/ROOT/python", "UV_LINK_MODE": "hardlink",
          "UV_HTTP_TIMEOUT": "180", "UV_CONCURRENT_DOWNLOADS": "4",
          "PATH": "/ROOT/envs/{image}/bin:/ROOT/tools/usr/bin:/ROOT/bin:$PATH"}
WITHOUT_ENV = {"train": [0, 1], "isaaclab-soft": [0, 1, 10], "isaaclab-dex": [0, 1]}


def _record_install(monkeypatch, image, envs=None):
    """Run ``install`` on an empty native root with every side effect replaced by a recorder; returns the command list
    (and appends to ``envs``, per command, the variables its environment changes, or None when it gets none)."""
    import os
    import shutil
    import sys
    from pathlib import Path

    calls = []
    for name in ("UV_HTTP_TIMEOUT", "UV_CONCURRENT_DOWNLOADS"):       # install only sets these when they are unset
        monkeypatch.delenv(name, raising=False)

    def run(args, **kwargs):
        assert set(kwargs) <= {"env"}, kwargs      # install's commands run from the current directory
        calls.append(" ".join(str(a) for a in args))
        if envs is not None:
            env = kwargs.get("env")
            envs.append(None if env is None else {k: v.replace(os.environ["PATH"], "$PATH") for k, v in env.items()
                                                  if os.environ.get(k) != v})

    monkeypatch.setattr(bootstrap, "run", run)
    monkeypatch.setattr(bootstrap, "checkout", lambda url, path, rev, **kw: calls.append(
        f"CHECKOUT {url} {path} {rev}" + (" soft_pillow_patch" if kw.get("soft_pillow_patch") else "")))
    monkeypatch.setattr(bootstrap, "align_soft_pillow", lambda path: calls.append(f"ALIGN {path}"))
    monkeypatch.setattr(bootstrap, "prune_unused_httpx2", lambda root, python: calls.append(f"PRUNE {root} {python}"))
    monkeypatch.setattr(bootstrap, "verify_installation", lambda root, image: calls.append(f"VERIFY {root} {image}"))
    monkeypatch.setattr(Path, "exists", lambda self: False)        # nothing installed yet: every step runs
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: None)
    monkeypatch.setattr(Path, "symlink_to", lambda self, *a, **k: None)
    monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/make")
    bootstrap.install(Path("/ROOT"), image)
    return [c.replace(str(bootstrap.REPO), "/REPO").replace(sys.executable, "{PY}") for c in calls]


@pytest.mark.parametrize("image", sorted(INSTALL_COMMANDS))
def test_install_issues_the_frozen_command_list(monkeypatch, image):
    assert _record_install(monkeypatch, image) == INSTALL_COMMANDS[image]


@pytest.mark.parametrize("image", sorted(INSTALL_COMMANDS))
def test_install_gives_its_commands_the_frozen_environment(monkeypatch, image):
    envs = []
    _record_install(monkeypatch, image, envs)
    assert [i for i, env in enumerate(envs) if env is None] == WITHOUT_ENV[image]
    expected = {k: v.format(image=image) for k, v in UV_ENV.items()}
    assert all(env == expected for env in envs if env is not None)
