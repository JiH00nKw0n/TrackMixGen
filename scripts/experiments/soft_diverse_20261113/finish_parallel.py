"""Finish the first two baseline chunks and exactly 500 additional successes per task."""
from pathlib import Path
import json, os, shutil, subprocess, time, fcntl
root=Path.cwd()
lock=open('tuning/finish.lock','w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
targets=json.loads(Path('tuning/targets.json').read_text())
primaries=json.loads(Path('tuning/primaries.json').read_text())
def read(p): return json.loads(p.read_text())
def checkchunk(p, expected):
    result=read(p/'CHUNK_DONE.json')
    assert result['ok'] and result['successes']==result['exported_episodes']==expected and result['exit_code']==0,p
    return result
def containers():
    ids=subprocess.check_output(['docker','ps','-q','--no-trunc'],text=True).split()
    return json.loads(subprocess.check_output(['docker','inspect',*ids])) if ids else []
def finish(task):
    pool=root/'data/pools/softmimicgen'/task/'diverse20/official/none/pool_isaac'
    dst=root/'data/tuning/combined'/task
    if (dst/'DATASET_DONE').exists(): return True
    live=containers()
    liveids={c['Id'] for c in live}
    prim=[]
    gpus=range(2) if task=='rope' else range(2,4)
    active={}
    for gpu in gpus:
        first=pool/'chunks'/f'seed{30001+gpu%2}'
        prim.append(first)
        cid=primaries[str(gpu)]
        if (first/'CHUNK_DONE.json').exists():
            checkchunk(first,250)
            if cid in liveids:
                # The original supervisor also scheduled a second 250-success chunk.
                # Its first chunk is closed and verified; the unique smaller seeds replace the second.
                subprocess.run(['docker','stop','-t','10',cid],check=True)
        active[gpu]=int(not (first/'CHUNK_DONE.json').exists())
    path=root/'data/tuning'/task
    parts=list(path.glob('gpu*/seed*'))
    planned=sum(read(p/'plan.json')['chunks'][0]['attempts'] for p in parts)
    assert planned<=500
    for gpu in gpus:
        jobs=[p for p in parts if p.parent.name==f'gpu{gpu}']
        for p in jobs:
            if (p/'DONE.json').exists(): continue
            arg='output=/'+str(p.relative_to(root))
            if not any(arg in (c['Config']['Cmd'] or []) for c in live):
                if time.time()-p.stat().st_mtime<120: continue
                raise RuntimeError(f'Unfinished worker is absent: {p}')
            active[gpu]+=1
        if active[gpu]>=targets[task] or planned>=500: continue
        ram=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))/1024**2
        recent=Path('tuning/last_launch')
        if recent.exists() and time.time()-recent.stat().st_mtime<180: continue
        if ram<(40 if task=='towel' else 32) or shutil.disk_usage(path).free<50*1024**3: continue
        recent.touch()
        seq=max([int(p.name[4:])%100 for p in jobs]+[0])+1
        quota=min(50,500-planned)
        log=open(f'tuning/{gpu}_{seq}.log','a')
        proc=subprocess.Popen(['bash','tuning/start',str(gpu),str(seq),str(quota)],stdout=log,stderr=subprocess.STDOUT)
        seed=41000+gpu*100+seq
        for _ in range(120):
            if (path/f'gpu{gpu}/seed{seed}/plan.json').exists(): break
            if proc.poll() is not None: raise RuntimeError(f'worker failed to start, {log.name}')
            time.sleep(1)
        else: raise RuntimeError('worker did not publish a plan')
        return False
    if planned!=500 or not all((p/'DONE.json').exists() for p in parts) or not all((p/'CHUNK_DONE.json').exists() for p in prim): return False
    chunks=prim+[next((p/'chunks').glob('seed*')) for p in sorted(parts)]
    (dst/'chunks').mkdir(parents=True,exist_ok=True)
    results=[]; plan=[]
    for idx,p in enumerate(chunks):
        spec=read(p/'chunk_spec.json'); quota=spec['chunk']['attempts']
        result=checkchunk(p,quota)
        link=dst/'chunks'/p.name
        if not link.exists(): link.symlink_to(os.path.relpath(p,link.parent))
        results.append(result); plan.append({'index':idx,'seed':int(p.name[4:]),'attempts':quota})
    assert sum(r['successes'] for r in results)==1000 and len({c['seed'] for c in plan})==len(plan)
    summary=dict(task=task,generator='softmimicgen',scene='diverse20',method='none',generation='pool_isaac',chunks=results,plan=plan,successes=1000,attempts=sum(r['attempts'] for r in results))
    summary['success_rate']=1000/summary['attempts']
    (dst/'DONE.json').write_text(json.dumps(summary,indent=2))
    subprocess.run(['make','-s','-C','branch','dataset','DATA_ROOT='+str(root/'data'),'IMAGE=isaaclab-soft','GENERATOR=softmimicgen','TASK='+task,'CLUSTER=isaac','SCENE=diverse20','SELECTION=official','METHOD=none','GENERATION=pool_isaac','OBSERVATION=image',f'OVERRIDES=task.num_sources=10 size=1000 pool=/data/tuning/combined/{task}'],check=True)
    ds=root/'data/datasets/softmimicgen'/task/'diverse20/official/none/pool_isaac/image/n1000_seed0.DATASET.json'
    assert read(ds)['demos']==1000
    (dst/'DATASET_DONE').touch()
    return True
while True:
    done=[finish(t) for t in ('rope','towel')]
    if all(done): break
    time.sleep(20)
