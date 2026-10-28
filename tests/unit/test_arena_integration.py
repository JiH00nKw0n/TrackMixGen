"""Validate that both Arena tasks enter the actual generator and policy workflow."""
from types import SimpleNamespace as NS

import pytest

from tmg.config import compose, resolved
from tmg.core.chunks import plan_chunks
from tmg.generation.generators import make_generator
from tmg.generation.simulators.isaaclab_tasks.arena import configure_observations


@pytest.mark.parametrize('task,count,environment', [
    ('microwave_opening', 10, 'gr1_open_microwave'),
    ('bottle_placement_door_closing', 5, 'put_item_in_fridge_and_close_door'),
])
def test_arena_source_and_cli_select_the_official_task(tmp_path, monkeypatch, task, count, environment):
    monkeypatch.setenv('TMG_DATA_ROOT', str(tmp_path))
    monkeypatch.setenv('TMG_ARENA_SRC', '/native/arena')
    cfg = resolved(compose('generate', ['generator=dexmimicgen', f'task={task}', 'scene=ours',
                                       'method=none', 'generation=pool_isaac']))
    assert cfg['task']['num_sources'] == count
    gen = make_generator(cfg)
    spec = gen.prepare_chunk(plan_chunks(250, 250, 30000)[0], tmp_path/task)
    args = spec.extra['script_args']
    assert spec.extra['script'] == '/native/arena/isaaclab_arena/scripts/imitation_learning/generate_dataset.py'
    assert args[args.index('--device')+1] == 'cpu'
    assert args[-3:] == [environment, '--embodiment', 'gr1_pink']
    assert '--task' not in args and '--mimic' in args
    assert args[args.index('--seed')+1] == str(spec.chunk['seed'])
    policy_cfg = resolved(compose('train', ['generator=dexmimicgen', f'task={task}', 'observation=image']))
    assert policy_cfg['observation']['cameras'] == ['robot_pov_cam_rgb']
    assert gen.demo_layout()['image_key'] == '{cam}'


def test_same_camera_is_recorded_and_used_for_policy_without_duplicate_images():
    term = NS(params={'sensor_cfg': 'robot_pov_cam', 'normalize': False})
    cfg = NS(scene=NS(robot_pov_cam=NS(height=512, width=512)),
             observations=NS(policy=NS(), camera_obs=NS(robot_pov_cam_rgb=term)),
             recorders=NS(record_pre_step_flat_camera_observations='duplicate'))
    task = {'image_size': [128,128], 'cameras':['robot_pov_cam_rgb']}
    configure_observations(cfg, task)
    assert cfg.scene.robot_pov_cam.height == cfg.scene.robot_pov_cam.width == 128
    assert cfg.observations.policy.robot_pov_cam_rgb.params == term.params
    assert cfg.observations.policy.robot_pov_cam_rgb is not term
    assert cfg.recorders.record_pre_step_flat_camera_observations is None
    assert cfg.rerender_on_reset


def test_arena_runtime_uses_its_own_lab_and_package_paths(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    repo=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('native_runtime_arena',repo/'scripts/native_runtime.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    root=tmp_path/'native'; py=root/'envs/isaaclab-arena/bin/python'
    py.parent.mkdir(parents=True); py.touch()
    monkeypatch.setenv('TMG_NATIVE_ROOT',str(root))
    actual,env=mod.runtime_environment('isaaclab-arena',str(tmp_path/'data'))
    assert actual==py
    assert env['TMG_ISAACLAB_SRC']==str(root/'src/isaaclab-arena/IsaacLab')
    assert env['TMG_ARENA_SRC']==str(root/'src/isaaclab-arena/IsaacLab-Arena')


def test_preflight_checks_real_dataset_keys_and_rejects_invalid_actions(tmp_path):
    import importlib.util
    import h5py
    import numpy as np
    from pathlib import Path
    repo=Path(__file__).resolve().parents[2]
    spec=importlib.util.spec_from_file_location('arena_preflight',repo/'scripts/arena_preflight.py')
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    path=tmp_path/'dataset.hdf5'
    with h5py.File(path,'w') as f:
        d=f.create_group('data/demo_0')
        for key in ('actions','actions_pre_noise','actions_executed'):
            d.create_dataset(key,data=np.zeros((3,36),dtype=np.float32))
        o=d.create_group('obs'); o.create_dataset('robot_state',data=np.zeros((3,36)))
        o.create_dataset('robot_pov_cam_rgb',data=np.random.default_rng(1).integers(0,256,(3,128,128,3),dtype=np.uint8))
    assert mod.check_dataset(path,tmp_path)['nonblank_camera']
    with h5py.File(path,'r+') as f: f['data/demo_0/actions'][0,0]=np.nan
    with pytest.raises(AssertionError): mod.check_dataset(path,tmp_path)
