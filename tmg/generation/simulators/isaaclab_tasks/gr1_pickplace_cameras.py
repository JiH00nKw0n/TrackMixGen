"""GR1 pick-and-place mimic environment with two recorded cameras (``TMG-PickPlace-GR1T2-Images-Mimic-v0``).

Same task, physics, controller, subtasks and success condition as ``Isaac-PickPlace-GR1T2-Abs-Mimic-v0``
(``isaaclab_mimic.envs.pinocchio_envs.pickplace_gr1t2_mimic_env_cfg.PickPlaceGR1T2MimicEnvCfg``); only the
scene gains two pinhole cameras and the policy observations gain their RGB images, which the recorder then
stores as ``obs/front_image`` and ``obs/side_image`` (128x128x3 uint8). Camera placement follows the
2026-10 reproduction (front camera at (0.15, 1.8, 1.9) m looking at the table centre, side camera at
(-1.4, 1.5, 1.8) m). ``rerender_on_reset`` makes the first frame of an episode valid.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation

ENV_ID = "TMG-PickPlace-GR1T2-Images-Mimic-v0"
IMAGE_SIZE = (128, 128)
FRONT = ((0.15, 1.8, 1.9), (0.15, 0.5, 1.05))
SIDE = ((-1.4, 1.5, 1.8), (0.15, 0.5, 1.05))


def look_at_quat_wxyz(eye, target) -> tuple[float, float, float, float]:
    """Orientation (w, x, y, z) of a ROS-convention camera at ``eye`` looking at ``target``."""
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    xyzw = Rotation.from_matrix(np.column_stack([right, down, forward])).as_quat()
    return float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])


def register() -> str:
    """Build the camera-enabled config class and register the gym id; returns the id."""
    import isaaclab.sim as sim_utils
    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationTermCfg, SceneEntityCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass
    from isaaclab_mimic.envs.pinocchio_envs.pickplace_gr1t2_mimic_env_cfg import PickPlaceGR1T2MimicEnvCfg
    from isaaclab_tasks.manager_based.manipulation.pick_place.pickplace_gr1t2_env_cfg import (
        ObjectTableSceneCfg,
        ObservationsCfg,
    )

    def camera(name: str, eye, target) -> CameraCfg:
        return CameraCfg(prim_path="{ENV_REGEX_NS}/" + name, update_period=0.0, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1],
                         data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 3.0)),
                         offset=CameraCfg.OffsetCfg(pos=eye, rot=look_at_quat_wxyz(eye, target), convention="ros"))

    @configclass
    class ImageSceneCfg(ObjectTableSceneCfg):
        front_camera = camera("FrontCamera", *FRONT)
        side_camera = camera("SideCamera", *SIDE)

    @configclass
    class ImageObservationsCfg(ObservationsCfg):
        @configclass
        class ImagePolicyCfg(ObservationsCfg.PolicyCfg):
            front_image = ObservationTermCfg(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front_camera"), "normalize": False})
            side_image = ObservationTermCfg(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("side_camera"), "normalize": False})

        policy: ImagePolicyCfg = ImagePolicyCfg()

    @configclass
    class PickPlaceGR1ImagesCfg(PickPlaceGR1T2MimicEnvCfg):
        scene: ImageSceneCfg = ImageSceneCfg(num_envs=1, env_spacing=2.5)
        observations: ImageObservationsCfg = ImageObservationsCfg()
        image_obs_list = ["front_image", "side_image"]
        rerender_on_reset = True

    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="isaaclab_mimic.envs.pinocchio_envs.pickplace_gr1t2_mimic_env:PickPlaceGR1T2MimicEnv",
                     kwargs={"env_cfg_entry_point": PickPlaceGR1ImagesCfg}, disable_env_checker=True)
    return ENV_ID


register()
