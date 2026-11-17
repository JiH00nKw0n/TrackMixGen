"""Read every annotated episode through the installed SoftMimicGen generator's input pool."""
from pathlib import Path
import json
from isaaclab.app import AppLauncher
launcher=AppLauncher(headless=True,device="cuda:0")
try:
    import gymnasium as gym
    import softmimicgen.envs
    from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
    from softmimicgen.datagen.datagen_info_pool import DataGenInfoPool
    out={}
    root=Path('/data')
    for task,env_name in [('rope','Isaac-Rope-Franka-IK-Rel-Mimic-v0'),('towel','Isaac-Fold-Towel-Franka-IK-Rel-Mimic-v0')]:
        path=root/'sources'/'softmimicgen'/task/'source.hdf5'
        cfg=parse_env_cfg(env_name,device='cuda:0',num_envs=1)
        # Parsing saved actions needs the native wrapper's gripper slice and its
        # configured EEF names; no simulator reset or new trajectory is needed.
        from importlib import import_module
        from types import SimpleNamespace
        module, name=gym.spec(env_name).entry_point.split(':')
        wrapper=getattr(import_module(module),name)
        class ReaderEnv:
            actions_to_gripper_actions=wrapper.actions_to_gripper_actions
        env=ReaderEnv()
        env.cfg=cfg
        pool=DataGenInfoPool(env,cfg,'cpu')
        pool.load_from_dataset_file(str(path))
        assert pool.num_datagen_infos==10
        bounds=pool.subtask_boundaries['robot0']
        # The installed file handler may enumerate keys lexicographically.
        expected=json.loads((root/(task+'_annotations.json')).read_text())
        assert bounds==[[(0,r['boundary']+1),(r['boundary']+1,r['steps'])] for r in expected],(bounds,expected)
        out[task]={'count':pool.num_datagen_infos,'bounds':bounds}
    (root/'reader_verified.json').write_text(json.dumps(out,indent=2))
finally:
    launcher.app.close()
