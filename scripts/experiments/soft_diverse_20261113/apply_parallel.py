from pathlib import Path
import json, subprocess, time
from throughput import records
while True:
    p=Path('tuning/windows.json')
    if p.exists() and 'four' in json.loads(p.read_text()): break
    time.sleep(15)
w=json.loads(p.read_text()); obs=records(); cutoff=float(Path('tuning/base_end').read_text())
rates={}; limits={}
for task,first in [('rope',0),('towel',2)]:
    base_rates=[]
    for g in (first,first+1):
        events=obs[g][30001+g%2]
        duration=cutoff-events[0][0]
        base_rates.append(sum(s for t,s in events if events[0][0]<t<=cutoff)*3600/duration)
    per={1:sum(base_rates)/2}
    for g in (first,first+1):
        r=w['two_three']['gpu'][str(g)]; per[r['workers']]=r['per_hour']
    r=w['four']['gpu'][str(first)]; per[4]=r['per_hour']
    # If the observed yields tie, keep the lower concurrency.
    limits[task]=max(per,key=per.get); rates[task]=per
Path('tuning/rates.json').write_text(json.dumps(rates,indent=2))
Path('tuning/targets.json').write_text(json.dumps(limits,indent=2))
subprocess.run(['python3','-u','tuning/finish_parallel.py'],check=True)
