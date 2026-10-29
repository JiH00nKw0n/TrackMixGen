#!/usr/bin/env python3
"""Reserve memory before new Pod jobs; never signal an existing job to make room."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from native_runtime import process_identity
from profile_pod_evaluation import GIB, resources, write


def permits(sample, leases, request, max_evaluations, max_generations=3):
    if sum(x['evaluations'] for x in leases)+request['evaluations'] > max_evaluations:
        return False
    if (sum(x.get('kind') == 'generate' for x in leases)
            + (request.get('kind') == 'generate') > max_generations):
        return False
    # Reserve the part of each admitted job's peak that it has not allocated yet.
    pending_ram=sum(max(0,x['ram_bytes']-x['observed_ram_bytes']) for x in leases)
    pending_gpu=sum(max(0,x['gpu_mib']-x['observed_gpu_mib']) for x in leases)
    return (sample['memory_working']+pending_ram+request['ram_bytes']+4*GIB <= sample['memory_limit']
            and sample['gpu_used_mib']+pending_gpu+request['gpu_mib']+4096 <= sample['gpu_total_mib'])


def usage(records, data):
    parents={}
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit(): continue
        try:
            fields=(entry/'stat').read_text().rsplit(')',1)[1].split()
            parents[int(entry.name)]=int(fields[1])
        except (OSError,ValueError): pass
    raw=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory',
                                 '--format=csv,noheader,nounits'],text=True)
    gpu={}
    for row in raw.splitlines():
        try:
            pid,mem=row.split(',');gpu[int(pid)]=float(mem)
        except ValueError: pass
    for r in records:
        pids={r['pid']}
        if r.get('child_pid') and process_identity(r['child_pid'])==r.get('child_start_ticks'):
            pids.add(r['child_pid'])
        for f in (data/'native_servers').glob('*.json'):
            try:
                s=json.loads(f.read_text())
                if s.get('resource_owner')==r['pid'] and process_identity(s['pid'])==s['start_ticks']:
                    pids.add(s['pid'])
            except (OSError,ValueError,KeyError): continue
        while True:
            extra={pid for pid,parent in parents.items() if parent in pids}
            if extra <= pids: break
            pids |= extra
        ram=0
        for pid in pids:
            try:
                for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
                    if line.startswith('Pss:'): ram+=int(line.split()[1])*1024
            except OSError: pass
        r.update(observed_ram_bytes=ram,observed_gpu_mib=sum(gpu.get(p,0) for p in pids))
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=['train','eval','generate'],required=True)
    parser.add_argument('--evaluations',type=int,default=0)
    parser.add_argument('--task',required=True)
    args,command=parser.parse_known_args()
    command=command[1:] if command[:1]==['--'] else command
    assert command
    data=Path(os.environ['TMG_DATA_ROOT'])
    root=Path(os.environ['TMG_RESULTS_ROOT'])/'reports/evaluation_capacity'
    config=json.loads((root/'selected.json').read_text())
    assert config['verified']
    selected=config['six'] if config['max_evaluations']==6 else config['four']
    n=selected['concurrent'];base=selected['baseline']
    if args.kind=='generate':
        # Separate processes, each with one unchanged simulator environment.
        # Admit gradually from actual usage; three is a cap, not a guarantee.
        ram=(12 if args.task in ('rope','towel') else 16)*GIB
        gpu=10240 if args.task in ('rope','towel') else 14336
        count=0
    elif args.kind=='train':
        ram=12*GIB;gpu=12288;count=0
    else:
        count=args.evaluations;assert 1 <= count <= config['max_evaluations']
        ram=max(6*GIB,1.15*(selected['peak_memory_working']-base['memory_working'])/n)*count
        gpu=max(6144,1.15*(selected['peak_gpu_mib']-base['gpu_used_mib'])/n)*count
        # Unmeasured tasks may have larger scenes; reserve at least 8 GiB for each.
        budget=config.get('minimum_eval_gib',{}).get(args.task,8 if args.task!='rope' else 0)
        ram=max(ram,budget*GIB*count);gpu=max(gpu,budget*1024*count)
    leases=root/'leases';leases.mkdir(exist_ok=True)
    pid=os.getpid();record=leases/f'{pid}.json'
    request=dict(pid=pid,start_ticks=process_identity(pid),kind=args.kind,task=args.task,
                 evaluations=count,ram_bytes=ram,gpu_mib=gpu,command=command)
    last_report=0
    while True:
        with (leases/'admission.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            active=[]
            for p in leases.glob('*.json'):
                r=json.loads(p.read_text())
                alive=process_identity(r['pid'])==r['start_ticks']
                if r.get('child_pid'):
                    alive=alive or process_identity(r['child_pid'])==r.get('child_start_ticks')
                if not r.get('released_at') and alive:
                    active.append(r)
            sample=resources();active=usage(active,data)
            if permits(sample,active,request,config['max_evaluations'],config.get('max_generations',3)):
                request['admitted_at']=time.time();request['admission_sample']=sample
                write(record,request)
                break
        if time.time()-last_report>60:
            print(f'[capacity] waiting for memory or evaluation slots: {args.kind} {args.task}',flush=True)
            last_report=time.time()
        time.sleep(5)
    threads='2' if args.kind=='generate' else '1'
    env=dict(os.environ,TMG_RESOURCE_OWNER=str(pid),OMP_NUM_THREADS=threads,MKL_NUM_THREADS=threads,OPENBLAS_NUM_THREADS=threads)
    child=subprocess.Popen(command,env=env)
    request.update(child_pid=child.pid,child_start_ticks=process_identity(child.pid))
    write(record,request)
    try:
        code=child.wait()
        request.update(exit_code=code)
    finally:
        if child.poll() is not None: request['released_at']=time.time()
        write(record,request)
    raise SystemExit(code)


if __name__=='__main__': main()
