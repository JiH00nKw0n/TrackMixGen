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


def resources():
    cg = Path("/sys/fs/cgroup")
    stat = dict(line.split() for line in (cg / "memory.stat").read_text().splitlines())
    events = dict(line.split() for line in (cg / "memory.events").read_text().splitlines())
    current = int((cg / "memory.current").read_text())
    row = subprocess.check_output([
        "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,power.draw",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
    util, used, total, power = map(float, row.split(","))
    return dict(at=time.time(), gpu_util=util, gpu_used_mib=used, gpu_total_mib=total,
                gpu_power_w=power, memory_current=current, memory_inactive_file=int(stat["inactive_file"]),
                memory_working=current-int(stat["inactive_file"]), memory_anon=int(stat["anon"]),
                memory_limit=int((cg / "memory.max").read_text()),
                oom=int(events["oom"]), oom_kill=int(events["oom_kill"]))


