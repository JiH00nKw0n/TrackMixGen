"""Bridge the pinned official Arena GR1 tasks into the shared image-policy workflow.

Task geometry, actions, subtask generation and success functions remain upstream.
The official head camera is recorded at 128 square pixels in the policy group so
training and rollout consume precisely the same image and robot-state terms.
"""
from __future__ import annotations

import copy


def configure_observations(env_cfg, task):
    camera = env_cfg.scene.robot_pov_cam
    camera.height, camera.width = map(int, task['image_size'])
    group = env_cfg.observations.camera_obs
    for name in task['cameras']:
        setattr(env_cfg.observations.policy, name, copy.deepcopy(getattr(group, name)))
    env_cfg.observations.policy.concatenate_terms = False
    env_cfg.rerender_on_reset = True
    # The normal recorder already saves the policy group. Avoid a second copy.
    if env_cfg.recorders is not None and hasattr(env_cfg.recorders, 'record_pre_step_flat_camera_observations'):
        env_cfg.recorders.record_pre_step_flat_camera_observations = None
    return env_cfg


def build_registered(task, *, seed=202600):
    from isaaclab_arena_environments.cli import (
        get_isaaclab_arena_environments_cli_parser, get_arena_builder_from_cli,
    )
    parser = get_isaaclab_arena_environments_cli_parser()
    args = parser.parse_args([
        '--headless', '--enable_cameras', '--device', str(task['simulation_device']),
        '--num_envs', '1', '--seed', str(seed), '--placement_seed', str(seed), '--mimic',
        task['arena_environment'], *list(task.get('arena_args') or []),
    ])
    name, cfg = get_arena_builder_from_cli(args).build_registered()
    return name, configure_observations(cfg, task)
