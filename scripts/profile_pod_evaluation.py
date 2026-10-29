#!/usr/bin/env python3
"""Measure isolated Rope evaluation capacity; never write formal training/evaluation outputs."""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from native_runtime import REPO, process_identity, runtime_environment

GIB = 1024**3
POLICIES = ("bc_rnn", "diffusion_policy")


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


def unsafe(sample, baseline):
    if sample["oom"] > baseline["oom"] or sample["oom_kill"] > baseline["oom_kill"]:
        return "cgroup recorded an OOM event"
    if sample["memory_limit"] - sample["memory_working"] < 2 * GIB:
        return "less than 2 GiB of RAM headroom after reclaimable inactive cache"
    if sample["gpu_total_mib"] - sample["gpu_used_mib"] < 2048:
        return "less than 2 GiB of GPU headroom"
    return None


def six_fits(four):
    """Conservative projection from simultaneous 4-worker peak, with 4 GiB reserve."""
    baseline = four["baseline"]
    projected_ram = baseline["memory_working"] + 1.5 * max(0, four["peak_memory_working"]-baseline["memory_working"])
    projected_gpu = baseline["gpu_used_mib"] + 1.5 * max(0, four["peak_gpu_mib"]-baseline["gpu_used_mib"])
    return (projected_ram <= baseline["memory_limit"]-4*GIB
            and projected_gpu <= baseline["gpu_total_mib"]-4096), dict(
                projected_working_ram_bytes=projected_ram, projected_gpu_mib=projected_gpu,
                reserve_ram_bytes=4*GIB, reserve_gpu_mib=4096)


class Profile:
    def __init__(self, root, data):
        self.root, self.data = root, data
        self.owned = []
        self.envs = {}
        for image in ("train", "isaaclab-soft"):
            py, env = runtime_environment(image, str(data))
            env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            self.envs[image] = py, env

    def launch(self, image, args, log):
        py, env = self.envs[image]
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("xb") as f:
            p = subprocess.Popen([str(py), "-m", "tmg", *args], cwd=REPO, env=env,
                                 stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        ticks = process_identity(p.pid)
        self.owned.append((p, ticks))
        write(log.with_suffix(".process.json"), dict(pid=p.pid, start_ticks=ticks, args=args, at=time.time()))
        return p

    def stop_owned(self):
        # Only this invocation's process groups, verified against start ticks.
        for p, ticks in self.owned:
            if ticks is not None and process_identity(p.pid) == ticks:
                os.killpg(p.pid, signal.SIGTERM)
        deadline = time.monotonic()+20
        while any(p.poll() is None for p, _ in self.owned) and time.monotonic() < deadline:
            time.sleep(.5)
        for p, ticks in self.owned:
            if ticks is not None and process_identity(p.pid) == ticks:
                os.killpg(p.pid, signal.SIGKILL)
            p.wait()
        self.owned.clear()

    def train(self):
        checkpoints = {}
        # A small read-only view is sufficient to prepare inference weights; formal data is untouched.
        source=self.data/'datasets/softmimicgen/rope/ours/none/pool_isaac/image/n1000_seed0.hdf5'
        subset=self.root/'probe_20_demos.hdf5'
        py,env=self.envs['train']
        subprocess.run([str(py),'-c',
            'import h5py,sys; src,out=sys.argv[1:]; '
            'f=h5py.File(src,"r"); g=h5py.File(out,"x"); d=g.create_group("data"); '
            'd.attrs.update(dict(f["data"].attrs)); '
            'keys=sorted(f["data"],key=lambda k:int(k.split("_")[1]))[:20]; '
            '[(d.__setitem__(k,h5py.ExternalLink(src,"/data/"+k))) for k in keys]; '
            'g.close(); f.close()',str(source),str(subset)],env=env,check=True)
        for policy in POLICIES:
            run = self.root / "probe_training" / policy / "seed202699"
            p = self.launch("train", ["train", "generator=softmimicgen", "task=rope", "scene=ours",
                "method=none", "generation=pool_isaac", "observation=image", f"policy={policy}",
                "cluster=pod", "seed=202699", "updates=200", "training.updates_per_epoch=100",
                "training.checkpoint_every_epochs=2", "training.cache=hdf5", "training.num_threads=2",
                "training.keep_training_states=last", f"dataset.path={subset}", f"output={run}"], self.root / f"train_{policy}.log")
            baseline=resources()
            while p.poll() is None:
                sample=resources()
                write(self.root / "LIVE.json", dict(phase="probe_training", policy=policy, sample=sample))
                reason=unsafe(sample, baseline)
                if reason: raise RuntimeError(reason)
                time.sleep(3)
            if p.returncode: raise RuntimeError(f"probe training failed: {policy}, exit {p.returncode}")
            ckpts=list((run / "checkpoints").glob("*/checkpoint.json"))
            if len(ckpts) != 1: raise RuntimeError(f"expected one isolated checkpoint: {ckpts}")
            checkpoints[policy]=str(ckpts[0].parent)
        return checkpoints

    def stage(self, count, checkpoints, scenes):
        folder=self.root / f"concurrent_{count}"
        folder.mkdir()
        baseline=resources(); samples=[]; workers=[]; servers=[]
        started=time.time(); next_launch=0; index=0; all_alive_at=None
        try:
            while True:
                now=time.time()
                if index<count and now>=next_launch:
                    policy=POLICIES[index%2]; rank=index//2; shards=count//2; port=29400+count*10+index
                    with socket.socket() as sock: sock.bind(("127.0.0.1",port))
                    server=self.launch("train", ["serve", f"checkpoint={checkpoints[policy]}",
                        "server.host=127.0.0.1", f"server.port={port}"], folder/f"server_{index}.log")
                    servers.append((server,port,index,policy,rank,shards))
                    index+=1; next_launch=now+15
                for server,port,i,policy,rank,shards in list(servers):
                    if any(w[0]==i for w in workers): continue
                    if server.poll() is not None: raise RuntimeError(f"policy server {i} exited {server.returncode}")
                    with socket.socket() as sock:
                        sock.settimeout(.1); ready=sock.connect_ex(("127.0.0.1",port))==0
                    if ready:
                        args=["eval","generator=softmimicgen","task=rope","scene=ours","observation=image",
                              "cluster=pod",f"checkpoint={checkpoints[policy]}","split=val","training_seed=202699",
                              f"evaluation.server.port={port}",f"max_scenes={scenes}",f"shard=[{rank},{shards}]",
                              f"output={folder/policy}"]
                        p=self.launch("isaaclab-soft",args,folder/f"eval_{i}.log")
                        workers.append((i,p))
                sample=resources(); sample["live_eval_processes"]=sum(p.poll() is None for _,p in workers)
                sample["launched_eval_processes"]=len(workers); samples.append(sample)
                with (folder/"resources.jsonl").open("a") as f: f.write(json.dumps(sample)+"\n")
                write(self.root/"LIVE.json",dict(phase="evaluation",concurrent=count,elapsed_s=now-started,sample=sample))
                if sample["live_eval_processes"]==count and all_alive_at is None: all_alive_at=now
                reason=unsafe(sample,baseline)
                if reason: raise RuntimeError(reason)
                if any(p.poll() not in (None,0) for _,p in workers): raise RuntimeError("an evaluation process failed")
                if len(workers)==count and all(p.poll() is not None for _,p in workers): break
                if now-started>3600: raise RuntimeError("capacity stage exceeded one hour")
                time.sleep(3)
            elapsed=time.time()-started
            rows=[]
            for policy in POLICIES:
                rs=[json.loads(x) for x in (folder/policy/"episodes.jsonl").read_text().splitlines()]
                indices=[r["reset_index"] for r in rs]
                assert len(rs)==scenes and len(set(indices))==scenes
                assert all(not r.get("error") and r["steps"]>0 and r["checkpoint"]==checkpoints[policy] for r in rs)
                rows+=rs
            assert all_alive_at is not None, "requested concurrency never overlapped"
            result=dict(passed=True,concurrent=count,episodes=len(rows),steps=sum(r["steps"] for r in rows),
                        wall_seconds=elapsed,episodes_per_hour=len(rows)*3600/elapsed,
                        step_seconds=sum(r["seconds"] for r in rows),all_alive_at=all_alive_at,
                        peak_gpu_mib=max(s["gpu_used_mib"] for s in samples),
                        peak_memory_working=max(s["memory_working"] for s in samples),
                        peak_memory_current=max(s["memory_current"] for s in samples),
                        mean_gpu_util=sum(s["gpu_util"] for s in samples)/len(samples),
                        peak_gpu_util=max(s["gpu_util"] for s in samples),baseline=baseline,
                        finished_at=time.time(),formal_results=False)
            write(folder/"result.json",result)
            return result
        except BaseException as e:
            write(folder/"result.json",dict(passed=False,error=repr(e),concurrent=count,finished_at=time.time()))
            raise
        finally: self.stop_owned()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenes-per-policy",type=int,default=6)
    args=parser.parse_args()
    data=Path(os.environ["TMG_DATA_ROOT"]).resolve()
    root=Path(os.environ["TMG_RESULTS_ROOT"])/"reports/evaluation_capacity"
    assert (root/"ACTIVE.json").exists(), "formal pipeline hold must be installed first"
    assert not (root/"RUN.json").exists(), "do not automatically repeat a capacity attempt"
    write(root/"RUN.json",dict(pid=os.getpid(),start_ticks=process_identity(os.getpid()),started_at=time.time(),scenes_per_policy=args.scenes_per_policy))
    profile=Profile(root,data)
    try:
        # Arena preflight is allowed to finish normally; formal training is held by ACTIVE.json.
        preflight=root.parent/"arena_preflight/SUMMARY.json"
        while not preflight.exists():
            write(root/"LIVE.json",dict(phase="waiting_for_arena_preflight",at=time.time()))
            time.sleep(10)
        summary=json.loads(preflight.read_text())
        write(root/"arena_preflight_state.json",summary)
        # Rope measurements remain useful when an unrelated Arena preflight failed.
        checkpoints=profile.train()
        write(root/"checkpoints.json",checkpoints)
        four=profile.stage(4,checkpoints,args.scenes_per_policy)
        fits,projection=six_fits(four)
        write(root/"six_admission.json",dict(admitted=fits,**projection))
        selected=4; six=None
        if fits:
            six=profile.stage(6,checkpoints,args.scenes_per_policy)
            if six["episodes_per_hour"]>=four["episodes_per_hour"]*1.1: selected=6
        write(root/"selected.json",dict(verified=True,max_evaluations=selected,
              per_policy_pipeline=selected//2,training_pipelines=2,task_measured="softmimicgen/rope",
              models_measured=list(POLICIES),comparison="same scenes and checkpoints, startup included",
              six_attempted=six is not None,four=four,six=six,at=time.time()))
        write(root/"FINISHED.json",dict(passed=True,selected=selected,at=time.time()))
        # Keep formal runs held until a human-readable review and resource admission are installed.
    except BaseException as e:
        write(root/"ERROR.json",dict(error=repr(e),at=time.time()))
        raise
    finally: profile.stop_owned()


if __name__=="__main__": main()
