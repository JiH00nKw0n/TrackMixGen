"""Compare equal-duration windows after every contributing process has generated an attempt."""
import json, os, subprocess, time
from pathlib import Path
from throughput import records
root=Path.cwd()
window=720
phasepath=Path('tuning/windows.json')
phases=json.loads(phasepath.read_text()) if phasepath.exists() else {}
def save():
    tmp=phasepath.with_suffix('.tmp'); tmp.write_text(json.dumps(phases,indent=2)); tmp.replace(phasepath)
def measure(label,groups):
    while True:
        data=records(); ready=True
        for gpu,seeds in groups.items():
            seq=data[gpu]
            if not all(seed in seq for seed in seeds): ready=False; break
        if ready: break
        time.sleep(15)
    start=max(data[g][s][0][0] for g,seeds in groups.items() for s in seeds)
    end=start+window
    while time.time()<end: time.sleep(min(30,end-time.time()))
    data=records()
    res={}
    for gpu,seeds in groups.items():
        attempts=[(t,s) for seed in seeds for t,s in data[gpu][seed] if start < t <= end]
        res[gpu]={'workers':len(seeds),'seeds':seeds,'n':len(attempts),'s':sum(s for _,s in attempts),'per_hour':sum(s for _,s in attempts)*3600/window}
    return {'start':start,'end':end,'gpu':res}
def seeds(g,n):return [30001+g%2]+[41000+100*g+i for i in range(1,n)]
if 'two_three' not in phases:
    phases['two_three']=measure('two_three',{g:seeds(g,2+g%2) for g in range(4)})
    save()
if 'four' not in phases:
    for seq in [2,3]:
        for g in [0,2]:
            folder=Path(f'data/tuning/{"rope" if g==0 else "towel"}/gpu{g}/seed{41000+g*100+seq}')
            if folder.exists(): continue
            while int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))<40*1024**2: time.sleep(15)
            out=open(f'tuning/{g}_{seq}.log','a')
            subprocess.Popen(['bash','tuning/start',str(g),str(seq),'25'],stdout=out,stderr=subprocess.STDOUT)
            time.sleep(10)
    phases['four']=measure('four',{g:seeds(g,4) for g in (0,2)})
    save()
