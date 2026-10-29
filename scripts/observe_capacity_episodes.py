#!/usr/bin/env python3
"""Timestamp newly completed probe episodes without changing the evaluation."""
import json
import os
import time
from pathlib import Path

root=Path(os.environ['TMG_RESULTS_ROOT'])/'reports/evaluation_capacity'
output=root/'episode_completion_observations.jsonl'
assert not output.exists(), 'observer already ran; preserve its evidence'
seen=set()
with output.open('x') as f:
    while True:
        for p in root.glob('concurrent_*/*/episodes.jsonl'):
            for text in p.read_text().splitlines():
                try: row=json.loads(text)
                except ValueError: continue
                key=(str(p),row['reset_index'])
                if key in seen: continue
                seen.add(key)
                f.write(json.dumps(dict(stage=p.parent.parent.name,policy=p.parent.name,
                     reset_index=row['reset_index'],seconds=row['seconds'],steps=row['steps'],
                     error=row.get('error'),observed_completed_at=time.time(),
                     observation_interval_seconds=1))+ '\n');f.flush()
        if (root/'FINISHED.json').exists() or (root/'ERROR.json').exists(): break
        time.sleep(1)
