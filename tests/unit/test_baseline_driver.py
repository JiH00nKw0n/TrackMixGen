"""The driver shares one dataset and caps combined policy/seed concurrency."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_two_policies_share_dataset_and_use_at_most_two_slots(tmp_path):
    repo = tmp_path/'repo'
    scripts = repo/'scripts'
    scripts.mkdir(parents=True)
    source = Path(__file__).resolve().parents[2]/'scripts/run_isaaclab_baseline.sh'
    shutil.copyfile(source, scripts/source.name)
    task_config = repo/'configs/experiment/generator/softmimicgen/task/rope.yaml'
    task_config.parent.mkdir(parents=True)
    task_config.write_text('name: rope\n')
    data = tmp_path/'data'
    ds = data/'datasets/softmimicgen/rope/ours/none/pool_isaac/image/n1000_seed0.hdf5'
    ds.parent.mkdir(parents=True)
    ds.touch()
    resets = data/'resets/softmimicgen/rope/ours/val25_test300_seed202600.json'
    resets.parent.mkdir(parents=True)
    resets.touch()
    binary = tmp_path/'bin'
    binary.mkdir()
    make = binary/'make'
    make.write_text(f'#!/bin/sh\necho "{data}"\n')
    make.chmod(0o755)
    sleep = binary/'sleep'
    sleep.write_text('#!/bin/sh\nexit 0\n')
    sleep.chmod(0o755)
    state = tmp_path/'state.json'
    state.write_text(json.dumps({'active':0,'peak':0,'calls':[]}))
    pipeline = scripts/'task_policy_pipeline.sh'
    pipeline.write_text(f'''#!{sys.executable}
import fcntl,json,os,sys,time
from pathlib import Path
p=Path({str(state)!r})
with open(str(p)+'.lock','w') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 s=json.loads(p.read_text()); s['active']+=1; s['peak']=max(s['active'],s['peak'])
 s['calls'].append([os.environ['POLICY'],os.environ['SEED'],sys.argv[2]])
 p.write_text(json.dumps(s))
time.sleep(.15)
with open(str(p)+'.lock','w') as f:
 fcntl.flock(f,fcntl.LOCK_EX)
 s=json.loads(p.read_text()); s['active']-=1; p.write_text(json.dumps(s))
''')
    pipeline.chmod(0o755)
    (scripts/'collect_results.py').write_text('pass\n')
    env=dict(os.environ,PATH=f'{binary}:'+os.environ['PATH'],SOFT_TASKS='rope',DEX_TASKS='',
             POLICIES='bc_rnn diffusion_policy',SEEDS='202601 202602 202603',SEED_PARALLEL='2',
             SEQUENTIAL_TASKS='1',TMG_RETIRE_COMPLETED='0',TMG_RESULTS_ROOT=str(tmp_path/'results'))
    result=subprocess.run(['bash',str(scripts/source.name)],env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode == 0, result.stdout+result.stderr
    s=json.loads(state.read_text())
    assert s['peak']==2 and s['active']==0
    assert {(p,k) for p,k,_ in s['calls']} == {(p,k) for p in ['bc_rnn','diffusion_policy'] for k in ['202601','202602','202603']}
    assert len({port for _,_,port in s['calls']})==6
    assert 'generation skipped' in result.stdout

    # A requested but unimplemented task must remain visible and prevent a false
    # campaign-complete marker, without running a physical job for that task.
    env['DEX_TASKS'] = 'bottle_placement_door_closing microwave_opening'
    result=subprocess.run(['bash',str(scripts/source.name)],env=env,capture_output=True,text=True,timeout=20)
    assert result.returncode == 2, result.stdout+result.stderr
    log=Path(env['TMG_RESULTS_ROOT'])/'logs/isaaclab_baseline'
    assert not (log/'DONE').exists()
    assert set((log/'BLOCKED_TASKS').read_text().splitlines()) == {
        'dexmimicgen/bottle_placement_door_closing', 'dexmimicgen/microwave_opening'}
