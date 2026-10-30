"""Admission budgets must not mistake reclaimable disk cache for job memory."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from pod_resource_metrics import GIB, admission_memory
from pod_resource_guard import permits


def stats(**changes):
    values = dict(file=38, shmem=0, active_file=29, inactive_file=9,
                  file_dirty=0, file_writeback=0, file_mapped=0)
    values.update(changes)
    return {key: value * GIB for key, value in values.items()}


def test_active_clean_cache_does_not_block_a_twenty_gib_test():
    admission, cache = admission_memory(40 * GIB, stats())
    assert (admission, cache) == (2 * GIB, 38 * GIB)
    sample = dict(memory_working=31 * GIB, memory_admission=admission,
                  memory_limit=46 * GIB, gpu_used_mib=0, gpu_total_mib=49140)
    request = dict(evaluations=2, ram_bytes=20 * GIB, gpu_mib=20480)
    assert permits(sample, [], request, 4)
    del sample["memory_admission"]
    assert not permits(sample, [], request, 4)


def test_dirty_shared_and_mapped_memory_still_counts():
    admission, cache = admission_memory(40 * GIB, stats(
        shmem=6, file_dirty=2, file_writeback=1, file_mapped=3))
    assert cache == 26 * GIB
    assert admission == 14 * GIB


def test_file_lru_bounds_discount_and_preserves_unlisted_memory():
    assert admission_memory(40 * GIB, stats(active_file=10)) == (21 * GIB, 19 * GIB)
    assert admission_memory(3 * GIB, stats()) == (0, 3 * GIB)


@pytest.mark.parametrize("key", list(stats()))
def test_missing_counter_fails_closed(key):
    sample = stats()
    del sample[key]
    assert admission_memory(40 * GIB, sample) == (40 * GIB, 0)


def test_pending_reservations_headroom_and_gpu_limits_are_preserved():
    sample = dict(memory_working=31 * GIB, memory_admission=3 * GIB,
                  memory_limit=46 * GIB, gpu_used_mib=0, gpu_total_mib=49140)
    request = dict(evaluations=2, ram_bytes=20 * GIB, gpu_mib=20480)
    lease = dict(evaluations=2, ram_bytes=20 * GIB, observed_ram_bytes=0,
                 gpu_mib=20480, observed_gpu_mib=0)
    assert not permits(sample, [lease], request, 4)  # 3 + 20 + 20 + 4 > 46
    sample["memory_admission"] = 2 * GIB
    assert permits(sample, [lease], request, 4)
    sample["gpu_used_mib"] = 4090
    assert not permits(sample, [lease], request, 4)
    assert not permits(sample, [lease], request, 2)
