"""Capacity decisions must respect both RAM and GPU limits, not utilization alone."""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from profile_pod_evaluation import GIB, six_fits, unsafe
from pod_resource_guard import permits


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
