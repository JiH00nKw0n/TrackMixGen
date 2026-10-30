from pathlib import Path
import pytest


def test_legacy_paths_are_explicit_overrides(tmp_path, monkeypatch):
    from tmg.config import compose, resolved
    monkeypatch.setenv('TMG_DATA_ROOT', str(tmp_path))
    for task in ('rope', 'towel'):
        base = f'{tmp_path}/softmimicgen/{task}/ours/none/pool_isaac'
        common = ['generator=softmimicgen', f'task={task}', 'scene=ours', 'method=none', 'generation=pool_isaac']
        cfg = resolved(compose('train', common+['observation=image', 'policy=bc_rnn',
            f'dataset.path={base}/image/n1000_seed0.hdf5', f'output={base}/run',
            'training.cache=hdf5', 'training.keep_training_states=last']))
        assert cfg['dataset']['path'] == f'{base}/image/n1000_seed0.hdf5'
        assert cfg['output'] == f'{base}/run'
        assert cfg['selection']['name'] == 'official'
        assert cfg['task']['policy_overrides']['bc_rnn']['crop_shape'] is None
        gen = resolved(compose('generate',common+[f'output={base}', 'task.object_state=[]',
            'task.lowdim_terms=null', 'task.num_sources=null', 'generation.protocol.keep_failed=true', 'generation.protocol.select_src_per_subtask=true', 'generation.protocol.selection_strategy=random', 'cluster.max_workers=3', 'cluster.workers=3']))
        assert gen['output'] == base
        assert gen['task']['object_state'] == []
        assert gen['task']['lowdim_terms'] is None
        assert gen['generation']['attempts'] == 1000
        assert gen['generation']['chunk'] == 250
        assert gen['generation']['protocol']['guarantee']


def test_completed_spec_relocates_nested_launch_arguments(tmp_path):
    from tmg.generation.runner import _relocate_spec
    from tmg.generation.base import ChunkSpec
    scratch=tmp_path/'scratch'; final=tmp_path/'existing'
    spec=ChunkSpec(generator='softmimicgen', chunk={}, folder=str(scratch), records_dir=str(scratch/'records'),
       task={}, scene={}, method={}, protocol={}, generator_config=str(scratch/'args.json'),
       expected_outputs=[str(scratch/'records/a')], extra={'script_args':['--output',str(scratch/'generated.hdf5')]})
    _relocate_spec(spec,scratch,final)
    assert spec.extra['script_args'][-1] == str(final/'generated.hdf5')


def test_disabled_optional_terms_match_but_enabled_terms_do_not(tmp_path):
    from dataclasses import asdict
    from tmg.core.chunks import write_json, CHUNK_SPEC, CHUNK_DONE
    from tmg.generation.base import ChunkSpec
    from tmg.generation.runner import _chunk_state
    spec=ChunkSpec(generator='softmimicgen', chunk={}, folder=str(tmp_path), records_dir=str(tmp_path/'records'),
       task={'name':'towel'}, scene={}, method={}, protocol={}, generator_config='')
    write_json(tmp_path/CHUNK_SPEC, asdict(spec)); write_json(tmp_path/CHUNK_DONE,{})
    spec.task['lowdim_terms']=None
    assert _chunk_state(tmp_path,spec)[0] == 'done'
    spec.task['lowdim_terms']={'towel_state':{'func':'new observation'}}
    assert _chunk_state(tmp_path,spec)[0] == 'changed'
