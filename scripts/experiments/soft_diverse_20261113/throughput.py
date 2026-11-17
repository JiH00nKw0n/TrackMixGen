"""Count timestamped completed attempts across all workers sharing each GPU."""
import json, time
from collections import defaultdict
from pathlib import Path

root=Path.cwd()
def records():
    out=defaultdict(dict)
    for p in [*Path('data/pools').rglob('attempts.jsonl'), *Path('data/tuning').rglob('attempts.jsonl')]:
        seed=int(p.parents[1].name[4:])
        if seed>=41000:
            gpu=(seed-41000)//100
        else:
            gpu=(0 if 'rope' in p.parts else 2)+(seed-30001)%2
        events=[]
        for line in p.read_text().splitlines():
            try: r=json.loads(line)
            except json.JSONDecodeError: continue
            events.append(((p.parent/r['npz']).stat().st_mtime,int(r['success'])))
        if events: out[gpu][seed]=events
    return out

if __name__=='__main__':
    import sys
    rows=records()
    now=time.time()
    for gpu,byseed in sorted(rows.items()):
        samples=sorted(byseed.items())
        # All times are server filesystem wall-clock seconds. Start after every worker
        # produced one record and stop before any newly added worker entered.
        base_seed=samples[0][0]
        base=byseed[base_seed]
        cutoff=Path('tuning/0_1.log').stat().st_ctime
        # File ctime changes on output; use container start timestamp from logs instead.
        cutoff=float(Path('tuning/base_end').read_text())
        def rate(start,end,events):
            values=[s for seq in events for t,s in seq if start < t <= end]
            return {'minutes':round((end-start)/60,1),'success':sum(values),'attempts':len(values),'success_per_h':round(sum(values)*3600/(end-start),1)}
        bounds=[(1,base[0][0],cutoff,[base])]
        if len(samples)>1:
            st=max(vals[0][0] for seed,vals in samples)
            if now>st: bounds.append((len(samples),st,now,[v for _,v in samples]))
        print(gpu,[(n,rate(a,b,events)) for n,a,b,events in bounds],flush=True)
