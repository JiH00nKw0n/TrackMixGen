"""Capacity decisions must respect both RAM and GPU limits, not utilization alone."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from profile_pod_evaluation import GIB, six_fits, unsafe
from pod_resource_guard import permits
from review_pod_capacity import review


def baseline():
    return dict(memory_working=3*GIB, memory_limit=46*GIB, gpu_used_mib=500,
                gpu_total_mib=49140, oom=0, oom_kill=0)


def test_six_requires_ram_and_gpu_reserve():
    four=dict(baseline=baseline(),peak_memory_working=28*GIB,peak_gpu_mib=25000)
    assert six_fits(four)[0]
    four['peak_memory_working']=31*GIB
    assert not six_fits(four)[0]
    four['peak_memory_working']=28*GIB
    four['peak_gpu_mib']=32000
    assert not six_fits(four)[0]


def test_oom_and_low_headroom_abort_benchmark():
    old=baseline()
    assert unsafe(old,old) is None
    assert unsafe(dict(old,oom=1),old)
    assert unsafe(dict(old,memory_working=45*GIB),old)
    assert unsafe(dict(old,gpu_used_mib=48000),old)


def test_admission_reserves_starting_workers_without_double_counting():
    job=dict(evaluations=2,ram_bytes=12*GIB,gpu_mib=12000)
    sample=baseline()
    active=dict(job,observed_ram_bytes=0,observed_gpu_mib=0)
    assert permits(sample,[active],job,4)
    assert not permits(sample,[active],job,2)
    active.update(ram_bytes=36*GIB)
    assert not permits(sample,[active],job,4)
    active.update(ram_bytes=12*GIB,observed_ram_bytes=12*GIB,observed_gpu_mib=12000)
    sample.update(memory_working=15*GIB,gpu_used_mib=12500)
    assert permits(sample,[active],job,4)


def test_review_requires_real_overlap_speed_and_reserve(tmp_path):
    observations=[]
    for count,seconds in [(4,10),(6,12)]:
        folder=tmp_path/f'concurrent_{count}';folder.mkdir()
        result=dict(passed=True,concurrent=count,baseline=baseline(),peak_memory_working=25*GIB,
                    peak_gpu_mib=24000,episodes_per_hour=100)
        (folder/'result.json').write_text(json.dumps(result))
        for policy in ['bc_rnn','diffusion_policy']:
            p=folder/policy;p.mkdir()
            rows=[dict(reset_index=i,seconds=seconds) for i in range(count//2)]
            (p/'episodes.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            observations += [dict(stage=folder.name,policy=policy,reset_index=i,seconds=seconds,
                observed_completed_at=100,observation_interval_seconds=1) for i in range(count//2)]
    (tmp_path/'episode_completion_observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in observations))
    review(tmp_path)
    assert json.loads((tmp_path/'selected.json').read_text())['max_evaluations']==6
    p=tmp_path/'concurrent_6/result.json';r=json.loads(p.read_text());r['peak_memory_working']=44*GIB;p.write_text(json.dumps(r))
    review(tmp_path)
    assert json.loads((tmp_path/'selected.json').read_text())['max_evaluations']==4


def test_generation_slots_and_startup_memory_are_both_reserved():
    job=dict(kind='generate',evaluations=0,ram_bytes=12*GIB,gpu_mib=10240)
    starting=dict(job,observed_ram_bytes=0,observed_gpu_mib=0)
    assert permits(baseline(),[starting,starting],job,4,3)
    assert not permits(baseline(),[starting,starting],job,4,2)
    assert not permits(baseline(),[starting,starting,starting],job,4,3)
    busy=dict(baseline(),memory_working=20*GIB)
    assert not permits(busy,[starting],job,4,3)
