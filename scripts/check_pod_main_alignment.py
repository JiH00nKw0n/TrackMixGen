#!/usr/bin/env python3
"""Record scientific source/config parity and the explicit Pod execution exceptions."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
EXCEPTIONS = {
    'configs/experiment/table23_scope.json': 'Adds the five-task, two-policy, three-seed campaign inventory.',
    'configs/experiment/training/default.yaml': 'Adds optimizer-state retention and bounded HDF5 reading options; learning parameters unchanged.',
    'tmg/data/loader.py': 'Reads identical selected frames on demand; main float32 normalization reductions retained and tested for equality.',
    'tmg/policy/trainer.py': 'Selects the HDF5 reading option and removes only obsolete optimizer/RNG states after checkpoint writes.',
    'tmg/policy/storage.py': 'Preserves all inference weights and the newest optimizer/RNG state.',
    'tmg/evaluation/capacity_barrier.py': 'Unused by formal evaluation; retained for historical capacity evidence.',
}

def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], stderr=subprocess.DEVNULL)


def check(ref):
    reference = git('rev-parse', ref).decode().strip()
    paths = set(git('ls-tree', '-r', '--name-only', reference, 'tmg', 'configs/experiment', 'requirements', 'docker').decode().splitlines())
    paths |= set(git('ls-files', 'tmg', 'configs/experiment', 'requirements', 'docker').decode().splitlines())
    files = []
    for name in sorted(paths):
        try: original = git('show', f'{reference}:{name}')
        except subprocess.CalledProcessError: original = None
        current = (ROOT / name).read_bytes() if (ROOT / name).is_file() else None
        files.append({'path': name, 'main_sha256': hashlib.sha256(original).hexdigest() if original is not None else None,
                      'pod_sha256': hashlib.sha256(current).hexdigest() if current is not None else None,
                      'identical': original == current, 'execution_exception': EXCEPTIONS.get(name)})
    unexpected = [x['path'] for x in files if not x['identical'] and not x['execution_exception']]
    from tmg.config import compose, resolved
    configurations = []
    for gen, task, horizon in [('softmimicgen','rope',400),('softmimicgen','towel',400),('dexmimicgen','pick_place',600),('autodata','microwave',600),('autodata','bottle_door',600)]:
        for policy in ['bc_rnn','diffusion_policy']:
            c = resolved(compose('train', [f'generator={gen}', f'task={task}', 'scene=ours', 'method=none', 'generation=pool_isaac',
                                            'observation=image', f'policy={policy}', 'seed=202601', 'cluster=pod', 'dataset.size=1000']))
            assert c['task']['horizon'] == horizon
            assert c['task']['policy_overrides'][policy]['crop_shape'] is None
            configurations.append(c)
    return {'measured_at': datetime.datetime.now().astimezone().isoformat(), 'main_reference': reference,
            'branch_head': git('rev-parse','HEAD').decode().strip(), 'files': files, 'unexpected_differences': unexpected,
            'resolved_configurations': configurations, 'passed': not unexpected,
            'formal_scope': {'tasks':5,'policies':['bc_rnn','diffusion_policy'],'training_seeds':[202601,202602,202603],
                             'trainings':30,'updates_per_training':100000,'validation_per_checkpoint':25,'checkpoints':10,'tests_per_training':300}}

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main-ref',default='1ef296bdfa2f141a4fd93f049e63dd2ad2509542')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=check(args.main_ref)
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2))
    print(json.dumps({'passed':result['passed'],'main_reference':result['main_reference'],
                      'files_checked':len(result['files']),'unexpected_differences':result['unexpected_differences'],'out':str(args.out)}))
    raise SystemExit(0 if result['passed'] else 1)
