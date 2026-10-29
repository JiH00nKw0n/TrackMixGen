"""Native dispatch must preserve host paths and never stop a server it did not start."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def test_native_make_dispatches_without_docker(tmp_path):
    root = tmp_path / "native"
    executable = root / "envs/isaaclab-soft/bin/python"
    executable.parent.mkdir(parents=True)
    output = tmp_path / "call.json"
    executable.write_text(
        f"#!{sys.executable}\nimport json,os,sys\n"
        f"open({str(output)!r},'w').write(json.dumps(dict(args=sys.argv[1:],data=os.environ['TMG_DATA_ROOT'],"
        "source=os.environ['TMG_ISAACLAB_SRC'],soft_source=os.environ['TMG_SOFTMIMICGEN_SRC'])))\n"
    )
    executable.chmod(0o755)
    data = tmp_path / "data"
    env = dict(os.environ, TMG_NATIVE_ROOT=str(root), TMG_DATA_ROOT=str(data))
    subprocess.run(["make", "-s", "sources", "CLUSTER=pod", "IMAGE=isaaclab-soft",
                    "GENERATOR=softmimicgen", "TASK=rope"], cwd=REPO, env=env, check=True)
    call = json.loads(output.read_text())
    assert call["args"] == ["-m", "tmg", "sources", "generator=softmimicgen", "task=rope"]
    assert call["data"] == str(data)
    assert call["source"] == str(root / "src/isaaclab-soft/IsaacLab")
    assert call["soft_source"] == str(root / "src/isaaclab-soft/SoftMimicGen")


def test_runtime_paths_keep_native_root_but_translate_containers(tmp_path):
    env = dict(os.environ, TMG_DATA_ROOT=str(tmp_path))
    for cluster, expected in [("pod", str(tmp_path)), ("ada", "/data")]:
        result = subprocess.check_output(["make", "-s", "print-runtime-data-root", f"CLUSTER={cluster}"],
                                         cwd=REPO, env=env, text=True)
        assert result.strip() == expected


def test_failed_server_start_does_not_stop_existing_server(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    calls = tmp_path / "calls"
    fake_make = binary / "make"
    fake_make.write_text(f'''#!/bin/sh
echo "$*" >> '{calls}'
case "$*" in
  *print-data-root*|*print-runtime-data-root*) echo '{tmp_path}/data' ;;
  *serve-detached*) exit 1 ;;
esac
''')
    fake_make.chmod(0o755)
    env = dict(os.environ, PATH=f"{binary}:{os.environ['PATH']}", RUN=f"{tmp_path}/data/run/seed202601",
               CKPT=f"{tmp_path}/data/run/seed202601/checkpoints/0010000", TASK="rope", CLUSTER="pod")
    result = subprocess.run(["bash", "scripts/evaluate_checkpoint.sh"], cwd=REPO, env=env,
                            capture_output=True, check=False)
    assert result.returncode != 0
    assert "serve-detached" in calls.read_text()
    assert "serve-stop" not in calls.read_text()


def test_autodata_uses_the_main_pinned_runtime_paths(tmp_path, monkeypatch):
    import importlib.util
    spec=importlib.util.spec_from_file_location('native_autodata',REPO/'scripts/native_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    root=tmp_path/'native';py=root/'envs/isaaclab-autodata/bin/python'
    py.parent.mkdir(parents=True);py.touch()
    monkeypatch.setenv('TMG_NATIVE_ROOT',str(root))
    actual,env=module.runtime_environment('isaaclab-autodata',str(tmp_path/'data'))
    assert actual==py
    assert env['TMG_ISAACLAB_SRC']==str(root/'src/isaaclab-autodata/IsaacLab-Arena/submodules/IsaacLab')
    assert env['TMG_ARENA_SRC']==str(root/'src/isaaclab-autodata/IsaacLab-Arena')
    assert env['TMG_AUTODATA_SRC']==str(root/'src/isaaclab-autodata/AutoData')
