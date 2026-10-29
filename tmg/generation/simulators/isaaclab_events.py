"""Reset-event functions that a scene file can add to an Isaac Lab environment (``apply_scene`` with ``add: true``).

They run inside the Isaac Sim process (generation chunk or evaluator) and follow Isaac Lab's event signature
``func(env, env_ids, **params)``. Isaac Lab's own ``isaaclab.envs.mdp`` functions (``reset_root_state_uniform`` for a
pose, ``reset_joints_by_offset`` for joints) are added the same way; this module holds what Isaac Lab lacks.
"""
from __future__ import annotations


def set_joint_percentage_uniform(env, env_ids, asset_cfg, joint_names, percentage_range):
    """Set one revolute joint of an Arena openable object to a random fraction of its range, per environment.

    Arena resets such a joint to a fixed fraction (``Openable.rotate_revolute_joint(percentage=...)``, the microwave
    door's 0.2); this term, listed after it, redraws the fraction uniformly in ``percentage_range`` so the door's
    start angle varies between episodes. Uses Arena's normalisation (``set_normalized_joint_position``), so 0 is
    closed and 1 fully open whatever the sign of the joint limits.

    Args:
        env: the Isaac Lab environment.
        env_ids: environment indices being reset (None = all).
        asset_cfg: ``SceneEntityCfg`` of the articulation (e.g. ``microwave``).
        joint_names: the one joint to set (e.g. ``["microjoint"]``).
        percentage_range: ``(lo, hi)`` fractions in [0, 1].
    """
    import torch
    from isaaclab_arena.utils.joint_utils import set_normalized_joint_position

    lo, hi = (float(v) for v in percentage_range)
    if not 0.0 <= lo <= hi <= 1.0:
        raise ValueError(f"percentage_range must satisfy 0 <= lo <= hi <= 1, got {percentage_range}")
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    asset_cfg.joint_names = list(joint_names)
    draws = lo + (hi - lo) * torch.rand(len(env_ids), device=env_ids.device if hasattr(env_ids, "device") else "cpu")
    for i in range(len(env_ids)):
        set_normalized_joint_position(env, asset_cfg, float(draws[i]), env_ids[i:i + 1])
