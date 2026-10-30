"""Read-only KHDP admission metrics; no probe or simulator launch."""
import json
import subprocess
import time
from pathlib import Path
GIB=1024**3

def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def admission_memory(current, stat):
    """Discount clean disk cache, retaining shared, mapped and dirty memory.

    Active file pages are reclaimable too. Bound the discount by both the
    file-backed LRU and file-minus-shmem counters, which need not be equal.
    Missing counters fail closed instead of treating unknown memory as free.
    """
    keys = ("file", "shmem", "active_file", "inactive_file", "file_dirty",
            "file_writeback", "file_mapped")
    if any(key not in stat for key in keys):
        return current, 0
    values = {key: max(0, int(stat[key])) for key in keys}
    reclaimable = max(0, min(values["file"] - values["shmem"],
                             values["active_file"] + values["inactive_file"])
                      - values["file_dirty"] - values["file_writeback"]
                      - values["file_mapped"])
    reclaimable = min(current, reclaimable)
    return current - reclaimable, reclaimable


def resources():
    cg = Path("/sys/fs/cgroup")
    stat = dict(line.split() for line in (cg / "memory.stat").read_text().splitlines())
    events = dict(line.split() for line in (cg / "memory.events").read_text().splitlines())
    current = int((cg / "memory.current").read_text())
    admission, reclaimable = admission_memory(current, stat)
    row = subprocess.check_output([
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
    util, used, total, power = map(float, row.split(","))
    return dict(at=time.time(), gpu_util=util, gpu_used_mib=used, gpu_total_mib=total,
                gpu_power_w=power, memory_current=current, memory_inactive_file=int(stat["inactive_file"]),
                memory_working=current-int(stat["inactive_file"]), memory_anon=int(stat["anon"]),
                memory_admission=admission, memory_reclaimable_file=reclaimable,
                memory_stat={key: int(value) for key, value in stat.items()},
                memory_limit=int((cg / "memory.max").read_text()),
                oom=int(events["oom"]), oom_kill=int(events["oom_kill"]))


