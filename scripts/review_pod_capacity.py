#!/usr/bin/env python3
"""Review measured rollout overlap and throughput before releasing formal work."""
import json
import os
from pathlib import Path
from profile_pod_evaluation import GIB, POLICIES, write


def review(root):
    observations=[json.loads(x) for x in (root/'episode_completion_observations.jsonl').read_text().splitlines()]
    metrics={}
    for count in (4,6):
        folder=root/f'concurrent_{count}'
        if not (folder/'result.json').exists(): continue
        result=json.loads((folder/'result.json').read_text())
        if not result.get('passed'):
            metrics[str(count)]=result;continue
        rows={p:[json.loads(x) for x in (folder/p/'episodes.jsonl').read_text().splitlines()] for p in POLICIES}
        means={p:sum(r['seconds'] for r in rs)/len(rs) for p,rs in rows.items()}
        # Conservative active intervals, allowing one second of observation delay.
        events=[]
        for r in observations:
            if r['stage']!=folder.name: continue
            worker=(r['policy'],r['reset_index']%(count//2))
            end=r['observed_completed_at']-r['observation_interval_seconds']
            start=r['observed_completed_at']-r['seconds']
            if end>start:events.extend([(start,1,worker),(end,-1,worker)])
        active=set();peak=0
        for _,change,worker in sorted(events):
            if change>0:active.add(worker)
            else:active.discard(worker)
            peak=max(peak,len(active))
        result.update(mean_episode_seconds=means,verified_simultaneous_rollouts=peak,
                      balanced_steady_episodes_per_hour=count*3600/max(means.values()))
        metrics[str(count)]=result
    four=metrics['4'];assert four['passed'] and four['verified_simultaneous_rollouts']==4
    selected=4;reason='Four evaluations passed; six was not both faster and within the production reserve.'
    six=metrics.get('6')
    if six and six.get('passed') and six['verified_simultaneous_rollouts']==6:
        baseline=six['baseline']
        ram_budget=max(6*GIB,1.15*(six['peak_memory_working']-baseline['memory_working'])/6)*6
        gpu_budget=max(6144,1.15*(six['peak_gpu_mib']-baseline['gpu_used_mib'])/6)*6
        reserve_ok=(ram_budget+baseline['memory_working']+4*GIB<=baseline['memory_limit']
                    and gpu_budget+baseline['gpu_used_mib']+4096<=baseline['gpu_total_mib'])
        if reserve_ok and six['balanced_steady_episodes_per_hour']>=four['balanced_steady_episodes_per_hour']*1.1:
            selected=6;reason='Six concurrent rollouts were verified, improved steady throughput by at least 10%, and fit the memory reserve.'
    config=dict(verified=True,max_evaluations=selected,per_policy_pipeline=selected//2,training_pipelines=2,
                task_measured='softmimicgen/rope',models_measured=list(POLICIES),four=four,six=six,
                six_attempted=six is not None,reason=reason,other_task_cap=4,other_tasks_require_memory_admission=True)
    old=root/'selected.json'
    if old.exists() and not (root/'initial_selection.json').exists():old.rename(root/'initial_selection.json')
    write(old,config);write(root/'REVIEWED.json',config)
    print(json.dumps(dict(selected=selected,reason=reason,metrics=metrics),indent=2))


if __name__=='__main__':
    review(Path(os.environ['TMG_RESULTS_ROOT'])/'reports/evaluation_capacity')
