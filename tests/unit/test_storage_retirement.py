import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('retirement', Path(__file__).resolve().parents[2] / 'scripts/retire_isaaclab_data.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture_run(root):
    run = root / 'run'
    resets = root / 'resets.json'
    scenes = [{'index': i, 'seed': 1000+i, 'split': 'val' if i < 25 else 'test'} for i in range(325)]
    dump(resets, {'scenes': scenes})
    dump(run / 'train_result.json', {'updates': 100000, 'demos': 1000})
    validation = []
    for i in range(1, 11):
        name = f'{i*10000:07d}'
        ckpt = run / 'checkpoints' / name
        dump(ckpt / 'checkpoint.json', {'update': i*10000})
        dump(ckpt / 'stats.json', {'kept': True})
        dump(ckpt / 'pretrained_model/config.json', {'kept': True})
        (ckpt / 'pretrained_model/model.safetensors').write_bytes(b'model')
        (ckpt / 'training_state.pt').write_bytes(b'resume')
        for split in (['val', 'test'] if i == 1 else ['val']):
            folder = run / 'eval' / f'{split}_h400' / name
            episodes = [{'reset_index': s['index'], 'scene_seed': s['seed'], 'split': split, 'checkpoint': str(ckpt),
                         'success': False, 'steps': 400, 'error': None} for s in scenes if s['split'] == split]
            dump(folder / 'eval_summary.json', {'split': split, 'checkpoint': str(ckpt), 'all': {'n': len(episodes), 'n_success': 0, 'errors': 0}})
            (folder / 'episodes.jsonl').write_text('\n'.join(map(json.dumps, episodes)))
        validation.append({'checkpoint': str(ckpt), 'n': 25, 'n_success': 0, 'success_rate': 0})
    (run / 'validation.jsonl').write_text('\n'.join(map(json.dumps, validation)))
    dump(run / 'SELECTED.json', {'checkpoint': str(run/'checkpoints/0010000'), 'rule': 'best_validation'})
    return run, resets


def test_retirement_preserves_selected_weights_and_all_evidence(tmp_path):
    run, resets = fixture_run(tmp_path)
    mod.retire_run(run, resets)
    assert (run/'checkpoints/0010000/pretrained_model/model.safetensors').exists()
    assert not (run/'checkpoints/0020000/pretrained_model/model.safetensors').exists()
    assert not list(run.glob('checkpoints/*/training_state.pt'))
    assert len(list(run.glob('checkpoints/*/checkpoint.json'))) == 10
    assert len(list(run.glob('eval/*/*/episodes.jsonl'))) == 11
    receipt = mod.read(run/'STORAGE_RETIRED.json')
    assert receipt['status'] == 'removed' and len(receipt['files']) == 19
    mod.retire_run(run, resets)  # completed work stays usable on restart


@pytest.mark.parametrize('corruption', ['episode_error', 'duplicate_scene', 'wrong_selection', 'short_training'])
def test_invalid_result_never_removes_weights(tmp_path, corruption):
    run, resets = fixture_run(tmp_path)
    episode_file = run/'eval/test_h400/0010000/episodes.jsonl'
    episodes = mod.rows(episode_file)
    if corruption == 'episode_error':
        episodes[0]['error'] = 'renderer failed'
    elif corruption == 'duplicate_scene':
        episodes[1] = episodes[0]
    elif corruption == 'wrong_selection':
        dump(run/'SELECTED.json', {'checkpoint': str(run/'checkpoints/0020000'), 'rule': 'best_validation'})
    else:
        dump(run/'train_result.json', {'updates': 500, 'demos': 1000})
    episode_file.write_text('\n'.join(map(json.dumps, episodes)))
    with pytest.raises(ValueError):
        mod.retire_run(run, resets)
    assert len(list(run.glob('checkpoints/*/pretrained_model/model.safetensors'))) == 10
    assert len(list(run.glob('checkpoints/*/training_state.pt'))) == 10


def test_outside_and_symlink_paths_are_not_deleted(tmp_path):
    scope = tmp_path/'allowed'
    scope.mkdir()
    outside = tmp_path/'outside.bin'
    outside.write_bytes(b'keep')
    link = scope/'linked'
    link.symlink_to(outside)
    for target in (outside, link):
        with pytest.raises(ValueError):
            mod.delete_files([target], scope, scope/'receipt.json')
    assert outside.read_bytes() == b'keep'


def test_only_old_resume_states_are_pruned(tmp_path):
    from tmg.policy.storage import prune_training_states
    run, _ = fixture_run(tmp_path)
    prune_training_states(run, run/'checkpoints/0100000')
    assert len(list(run.glob('checkpoints/*/training_state.pt'))) == 1
    assert len(list(run.glob('checkpoints/*/pretrained_model/model.safetensors'))) == 10
    assert (run/'checkpoints/0100000/training_state.pt').exists()


def test_task_data_waits_for_every_requested_policy(tmp_path):
    root = tmp_path/'data'
    root.mkdir()
    suffix = Path('softmimicgen/rope/ours/none/pool_isaac/image')
    source_run, source_resets = fixture_run(tmp_path/'fixture')
    import shutil
    run = root/'runs'/suffix/'diffusion_policy/seed202601'
    shutil.copytree(source_run, run)
    # Replace fixture paths with the actual run path, including every episode.
    for path in run.rglob('*'):
        if path.suffix in {'.json', '.jsonl'}:
            path.write_text(path.read_text().replace(str(source_run), str(run)))
    resets = root/'resets/softmimicgen/rope/ours/val25_test300_seed202600.json'
    resets.parent.mkdir(parents=True)
    shutil.copyfile(source_resets, resets)
    pool = root/'pools/softmimicgen/rope/ours/none/pool_isaac'
    dump(pool/'DONE.json', {'successes': 1000})
    bulk = pool/'generated.hdf5'
    bulk.write_bytes(b'keep until all models finish')
    with pytest.raises(FileNotFoundError):
        mod.retire_task(root, 'softmimicgen', 'rope', ['diffusion_policy', 'bc_rnn'], [202601])
    assert bulk.exists()
    mod.retire_task(root, 'softmimicgen', 'rope', ['diffusion_policy'], [202601])
    assert not bulk.exists()
    assert (run/'checkpoints/0010000/pretrained_model/model.safetensors').exists()
    assert mod.read(root/'retired/softmimicgen/rope.json')['policies'] == ['diffusion_policy']
