#!/usr/bin/env python3
"""Run an installed tmg environment without a container; manage only owned policy servers."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

REPO = Path(__file__).resolve().parents[1]


def runtime_environment(image: str, data_root: str) -> tuple[Path, dict[str, str]]:
    """Return an environment's interpreter and process-local paths, without changing the host."""
    if image not in {"isaaclab-soft", "isaaclab-dex", "train"}:
        raise ValueError(f"native runtime does not support {image!r}")
    root = Path(os.environ.get("TMG_NATIVE_ROOT") or "~/.local/share/tmg-native").expanduser().resolve()
    venv = root / "envs" / image
    python = venv / "bin/python"
    if not python.is_file():
        raise FileNotFoundError(f"Run scripts/bootstrap_native.py {image} first: {python}")
    env = dict(os.environ)
    env.update(TMG_DATA_ROOT=str(Path(data_root).resolve()), TMG_IMAGE=f"native/{image}",
               VIRTUAL_ENV=str(venv), PYTHONUNBUFFERED="1", HF_HUB_DISABLE_TELEMETRY="1")
    env["PATH"] = f"{venv / 'bin'}:{root / 'tools/usr/bin'}:{env.get('PATH', '')}"
    env["PYTHONPATH"] = str(REPO) + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("HF_HOME", str(Path(data_root) / ".cache/huggingface"))
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    graphics = root / "graphics/environment.json"
    if graphics.is_file():
        additions = json.loads(graphics.read_text())
        for key, value in additions.items():
            if key == "LD_LIBRARY_PATH" and env.get(key):
                value += ":" + env[key]
            env[key] = value
    if image != "train":
        env["TMG_ISAACLAB_SRC"] = str(root / "src" / image / "IsaacLab")
    return python, env


def process_identity(pid: int):
    """Use Linux start ticks to reject stale or recycled PIDs."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text.rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else int(fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def server(args):
    """Start or stop the exact process recorded for this server name, under a file lock."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", args.name):
        raise ValueError("unsafe server name")
    folder = Path(args.data_root).resolve() / "native_servers"
    folder.mkdir(parents=True, exist_ok=True)
    record = folder / f"{args.name}.json"
    with (folder / f"{args.name}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        previous = json.loads(record.read_text()) if record.exists() else None
        alive = previous and process_identity(previous["pid"]) == previous["start_ticks"]
        if args.action == "serve-stop":
            if alive:
                os.killpg(previous["pid"], signal.SIGTERM)
                for _ in range(100):
                    if process_identity(previous["pid"]) is None:
                        break
                    time.sleep(0.1)
                else:
                    raise RuntimeError("owned policy server did not exit; record preserved")
            if previous:
                previous["stopped_at"] = time.time()
                record.write_text(json.dumps(previous, indent=2))
            return
        if alive:
            raise RuntimeError(f"server already running: {record}")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))  # refuse a port owned by another process
        python, env = runtime_environment("train", args.data_root)
        command = [str(python), "-m", "tmg", "serve", f"checkpoint={args.checkpoint}",
                   "server.host=127.0.0.1", f"server.port={args.port}"]
        with (folder / f"{args.name}.log").open("ab") as log:
            proc = subprocess.Popen(command, env=env, cwd=REPO, stdin=subprocess.DEVNULL,
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ticks = process_identity(proc.pid)
        record.write_text(json.dumps(dict(pid=proc.pid, start_ticks=ticks, command=command,
                                         started_at=time.time()), indent=2))
        if ticks is None:
            raise RuntimeError(f"policy server exited at launch; see {folder / (args.name + '.log')}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["run", "python", "serve-start", "serve-stop"])
    parser.add_argument("--image", default="isaaclab-soft")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--name")
    parser.add_argument("--port", type=int)
    parser.add_argument("--checkpoint")
    args, rest = parser.parse_known_args()
    if args.action.startswith("serve-"):
        server(args)
    else:
        python, env = runtime_environment(args.image, args.data_root)
        rest = rest[1:] if rest[:1] == ["--"] else rest
        argv = [str(python)] + (["-m", "tmg"] if args.action == "run" else []) + rest
        os.execve(python, argv, env)


if __name__ == "__main__":
    main()
