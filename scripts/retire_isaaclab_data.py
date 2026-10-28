#!/usr/bin/env python3
"""Opt-in storage retirement after complete, error-free Isaac Lab evaluations.

Keep all JSON/YAML/log evidence and selected inference weights. Never remove data
for an incomplete policy/seed. Record file hashes before unlinking large files.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def check_eval(folder, scenes, split, checkpoint):
    summary = read(folder / 'eval_summary.json')
    episodes = rows(folder / 'episodes.jsonl')
    expected = {(s['index'], s['seed']) for s in scenes if s['split'] == split}
    actual = [(e['reset_index'], e['scene_seed']) for e in episodes]
    expected_checkpoint = (folder.parents[2] / 'checkpoints' / checkpoint).resolve()
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError(f'incomplete or duplicate {split} scenes: {folder}')
    if any(e.get('error') is not None or e['steps'] <= 0 or e['split'] != split
           or Path(e['checkpoint']).resolve() != expected_checkpoint for e in episodes):
        raise ValueError(f'error or mismatched checkpoint in {folder}')
    if summary['split'] != split or Path(summary['checkpoint']).resolve() != expected_checkpoint:
        raise ValueError(f'wrong summary: {folder}')
    n_success = sum(bool(e['success']) for e in episodes)
    if summary['all']['n'] != len(expected) or summary['all']['n_success'] != n_success or summary['all']['errors'] != 0:
        raise ValueError(f'summary disagrees with episodes: {folder}')
    return n_success


def check_run(run, resets):
    result = read(run / 'train_result.json')
    if result['updates'] != 100000 or result['demos'] != 1000:
        raise ValueError(f'full training has not finished: {run}')
    scenes = read(resets)['scenes']
    if sum(s['split'] == 'val' for s in scenes) != 25 or sum(s['split'] == 'test' for s in scenes) != 300:
        raise ValueError('expected 25 validation and 300 test scenes')
    expected = {f'{i:07d}' for i in range(10000, 100001, 10000)}
    validation = rows(run / 'validation.jsonl')
    by_checkpoint = {Path(r['checkpoint']).name: r for r in validation}
    if len(validation) != 10 or set(by_checkpoint) != expected:
        raise ValueError(f'incomplete validation: {run}')
    scores = {}
    for name, row in by_checkpoint.items():
        folders = list((run / 'eval').glob(f'val_h*/{name}'))
        if len(folders) != 1:
            raise ValueError(f'ambiguous or missing validation: {run}/{name}')
        scores[name] = check_eval(folders[0], scenes, 'val', name)
        if row['n'] != 25 or row['n_success'] != scores[name] or abs(row['success_rate'] - scores[name] / 25) > 1e-9:
            raise ValueError('validation row disagrees with episodes')
    selected = read(run / 'SELECTED.json')
    best = max(scores, key=lambda name: (scores[name], -int(name)))
    if selected['rule'] != 'best_validation' or Path(selected['checkpoint']).resolve() != (run / 'checkpoints' / best).resolve():
        raise ValueError('selection differs from best validation / earliest tie')
    if not (run / 'checkpoints' / best / 'pretrained_model' / 'model.safetensors').is_file():
        raise ValueError('selected inference weights missing')
    tests = list((run / 'eval').glob(f'test_h*/{best}'))
    if len(tests) != 1:
        raise ValueError('ambiguous or missing test')
    check_eval(tests[0], scenes, 'test', best)
    return best


def delete_files(files, scope, receipt):
    """Reject symlinks, record bytes/SHA256, then remove only enumerated files."""
    scope = scope.resolve()
    entries = []
    for path in sorted(set(files)):
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(scope):
            raise ValueError(f'unsafe removal path: {path}')
        # Reject a symlink in any directory below the allowed scope as well.
        parent = path.parent
        while parent != scope:
            if parent.is_symlink():
                raise ValueError(f'symlink parent: {parent}')
            parent = parent.parent
        digest = hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda: f.read(8 * 1024**2), b''):
                digest.update(block)
        entries.append({'path': str(path), 'bytes': path.stat().st_size, 'sha256': digest.hexdigest()})
    receipt.parent.mkdir(parents=True, exist_ok=True)
    record = {'status': 'verified_before_removal', 'files': entries}
    receipt.write_text(json.dumps(record, indent=2))
    for item in entries:
        Path(item['path']).unlink()
    record['status'] = 'removed'
    receipt.write_text(json.dumps(record, indent=2))
    print(f'retired {len(entries)} files, {sum(e["bytes"] for e in entries) / 1024**3:.2f} GiB; {receipt}')


def retire_run(run, resets):
    run = run.resolve()
    best = check_run(run, resets)
    marker = run / 'STORAGE_RETIRED.json'
    if marker.exists() and read(marker)['status'] == 'removed':
        return
    files = []
    for folder in (run / 'checkpoints').glob('[0-9]*'):
        files.extend(folder.glob('training_state.pt'))
        if folder.name != best:
            files.extend(folder.rglob('*.safetensors'))
            files.extend(folder.rglob('*.bin'))
    delete_files(files, run.resolve(), marker)


def retire_task(root, gen, task, policies, seeds):
    suffix = Path(gen) / task / 'ours/none/pool_isaac/image'
    resets = root / 'resets' / gen / task / 'ours/val25_test300_seed202600.json'
    for policy in policies:
        for seed in seeds:
            retire_run(root / 'runs' / suffix / policy / f'seed{seed}', resets)
    pool = root / 'pools' / gen / task / 'ours/none/pool_isaac'
    if read(pool / 'DONE.json')['successes'] < 1000:
        raise ValueError('generation incomplete')
    marker = root / 'retired' / gen / f'{task}.json'
    if marker.exists() and read(marker)['status'] == 'removed':
        return
    files = list(pool.rglob('*.hdf5')) + list(pool.rglob('*.npz'))
    files += list((root / 'datasets' / suffix).glob('*.hdf5'))
    delete_files(files, root, marker)
    record = read(marker)
    record.update(policies=policies, seeds=seeds)
    marker.write_text(json.dumps(record, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    r = sub.add_parser('run')
    r.add_argument('--run', type=Path, required=True)
    r.add_argument('--resets', type=Path, required=True)
    t = sub.add_parser('task')
    t.add_argument('--root', type=Path, required=True)
    t.add_argument('--generator', choices=['softmimicgen', 'dexmimicgen'], required=True)
    t.add_argument('--task', choices=['rope', 'towel', 'jenga', 'pick_place', 'bottle_placement_door_closing', 'microwave_opening'], required=True)
    t.add_argument('--policies', nargs='+', choices=['bc_rnn', 'diffusion_policy'], required=True)
    t.add_argument('--seeds', nargs='+', type=int, required=True)
    a = p.parse_args()
    if a.mode == 'run':
        retire_run(a.run.resolve(), a.resets.resolve())
    else:
        retire_task(a.root.resolve(), a.generator, a.task, a.policies, a.seeds)


if __name__ == '__main__':
    main()
