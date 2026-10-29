#!/usr/bin/env python3
"""Continue a campaign once an existing, unchanged pool process finishes successfully.

The old generation checkout must remain untouched. Stop only its campaign dispatcher
before using this continuation. This helper never signals or retries a physical job.
"""
import argparse
import datetime
import fcntl
import json
from pathlib import Path
import subprocess
import time

from native_runtime import process_identity


def completed_pool(path, expected_successes):
    result = json.loads((path / "DONE.json").read_text())
    chunks = result.get("chunks", [])
    if not chunks or result.get("successes") != expected_successes:
        raise ValueError("Predecessor pool did not produce the requested successes")
    if sum(c.get("successes", 0) for c in chunks) != expected_successes:
        raise ValueError("Pool and chunk success counts disagree")
    if any(not c.get("ok") or c.get("exit_code") != 0 or
           c.get("exported_episodes") != c.get("successes") for c in chunks):
        raise ValueError("A predecessor chunk failed or has incomplete exports")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid', required=True, type=int)
    p.add_argument('--start-ticks', required=True, type=int)
    p.add_argument('--pool', required=True, type=Path)
    p.add_argument('--successes', default=1000, type=int)
    p.add_argument('--status', required=True, type=Path)
    p.add_argument('--log', required=True, type=Path)
    p.add_argument('command', nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        p.error('A continuation command is required')
    args.status.parent.mkdir(parents=True, exist_ok=True)
    with args.status.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # A prior attempt is evidence and must never be silently overwritten/retried.
        with args.status.open('x') as f:
            json.dump({}, f)
        state = dict(predecessor_pid=args.pid, predecessor_start_ticks=args.start_ticks,
                     pool=str(args.pool), command=command, status='waiting_for_pool')
        def save():
            state['observed_at'] = datetime.datetime.now().astimezone().isoformat()
            temporary = args.status.with_suffix('.pending')
            temporary.write_text(json.dumps(state, indent=2))
            temporary.replace(args.status)
        save()
        try:
            while process_identity(args.pid) == args.start_ticks:
                time.sleep(30)
            state['completed_pool'] = completed_pool(args.pool, args.successes)
            args.log.parent.mkdir(parents=True, exist_ok=True)
            with args.log.open('xb') as log:
                child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', child_pid=child.pid,
                         child_start_ticks=process_identity(child.pid),
                         child_started_at=datetime.datetime.now().astimezone().isoformat())
            save()
            code = child.wait()
            state.update(status='completed' if code == 0 else 'failed', exit_code=code)
            save()
            return code
        except Exception as exc:
            state.update(status='blocked', error=repr(exc))
            save()
            raise


if __name__ == '__main__':
    raise SystemExit(main())
