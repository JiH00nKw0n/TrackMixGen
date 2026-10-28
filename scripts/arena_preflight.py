#!/usr/bin/env python3
"""Run each newly connected task once before formal scheduling; preserve failures without retrying."""
from __future__ import annotations

import concurrent.futures
import json
import os
import subprocess
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
TASKS = ('bottle_placement_door_closing', 'microwave_opening')


def check_dataset(path, output):
    with h5py.File(path, 'r') as f:
        assert len(f['data']) == 1
        demo = f['data'][next(iter(f['data']))]
        n = len(demo['actions'])
        assert n > 0
        for key in ('actions', 'actions_pre_noise', 'actions_executed'):
            x = demo[key][:]
            assert x.shape == (n,36) and np.isfinite(x).all(), (key,x.shape)
        images = demo['obs/robot_pov_cam_rgb_image']
        assert images.shape == (n,128,128,3) and images.dtype == np.uint8, images.shape
        samples = images[np.unique(np.linspace(0,n-1,min(3,n)).astype(int))]
        assert all(float(frame.std()) > 1 for frame in samples), 'blank camera images'
        Image.fromarray(np.concatenate(list(samples),axis=1)).save(output/'camera_samples.png')
        assert np.isfinite(demo['obs/robot_state'][:]).all()
        return {'steps': n, 'actions': 36, 'camera_shape': list(images.shape),
                'finite_labels': True, 'nonblank_camera': True}


def task_preflight(task, root, reports):
    output = reports/task
    output.mkdir(parents=True,exist_ok=True)
    record = output/'result.json'
    if record.exists():
        prior = json.loads(record.read_text())
        if prior.get('passed'):
            return prior
        raise RuntimeError(f'{task}: previous failure preserved; review required before another attempt')
    result = {'task':task,'started_at':time.time(),'passed':False,'formal_pool':False}
    record.write_text(json.dumps(result,indent=2))
    common = ['GENERATOR=dexmimicgen','IMAGE=isaaclab-arena',f'TASK={task}',
              'CLUSTER=pod','SCENE=ours','METHOD=none','GENERATION=arena_probe','OBSERVATION=image']
    try:
        for stage in ('sources','generate','dataset'):
            cmd = ['make','-s','-C',str(REPO),stage,*common]
            if stage == 'dataset': cmd += ['OVERRIDES=size=1']
            with (output/f'{stage}.log').open('ab') as log:
                subprocess.run(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        dataset = root/f'datasets/dexmimicgen/{task}/ours/none/arena_probe/image/n1_seed0.hdf5'
        result.update(check_dataset(dataset,output))
        result.update(passed=True,dataset=str(dataset),finished_at=time.time())
    except Exception as exc:
        result.update(error=repr(exc),finished_at=time.time())
        raise
    finally:
        record.write_text(json.dumps(result,indent=2))
    return result


def main():
    root=Path(os.environ['TMG_DATA_ROOT'])
    reports=Path(os.environ.get('TMG_RESULTS_ROOT',root))/'reports/arena_preflight'
    reports.mkdir(parents=True,exist_ok=True)
    failures=[]; results=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures={executor.submit(task_preflight,t,root,reports):t for t in TASKS}
        for future in concurrent.futures.as_completed(futures):
            try: results.append(future.result())
            except Exception as exc: failures.append({'task':futures[future],'error':repr(exc)})
    summary={'at':time.time(),'passed':not failures,'tasks':results,'failures':failures,
             'max_simultaneous_tasks':2,'formal_training_runs':0}
    (reports/'SUMMARY.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    if failures: raise SystemExit(1)


if __name__=='__main__': main()
