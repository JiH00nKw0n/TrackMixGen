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
