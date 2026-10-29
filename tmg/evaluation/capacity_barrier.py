"""Optional startup rendezvous for isolated capacity measurements only."""
import json
import os
import time
from pathlib import Path


def wait_for_capacity_workers():
    directory = os.environ.get('TMG_CAPACITY_BARRIER')
    if not directory:
        return
    root = Path(directory)
    parties = int(os.environ['TMG_CAPACITY_PARTIES'])
    root.mkdir(parents=True, exist_ok=True)
    record = root / f'{os.getpid()}.json'
    record.write_text(json.dumps(dict(pid=os.getpid(), ready_at=time.time())))
    deadline = time.monotonic() + 1800
    while len(list(root.glob('*.json'))) < parties:
        if time.monotonic() > deadline:
            raise TimeoutError('Capacity workers did not all finish environment preparation')
        time.sleep(.2)
    print(f'[capacity] all {parties} environments ready; starting rollouts', flush=True)
