"""GR1 pick-and-place mimic environment with two recorded cameras (``TMG-PickPlace-GR1T2-Images-Mimic-v0``), Isaac Lab 3.0.

Same task, physics, controller, subtasks and success condition as ``Isaac-PickPlace-GR1T2-Abs-Mimic-v0``
(``isaaclab_mimic.envs.pickplace_gr1t2_mimic_env_cfg.PickPlaceGR1T2MimicEnvCfg`` of the Isaac Lab 3.0 nested in
Isaac Lab-Arena, task config ``isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg``). Two changes:

- the official scene's teleoperation head camera (``robot_pov_cam``, 450x720, listed in ``image_obs_list``) is removed and
  two 128x128 pinhole cameras are added; their RGB images form an observation group ``camera_obs`` (``front_image``,
  ``side_image``), which Isaac Lab's recorder does not store (it keeps the ``policy`` group only), so the generated hdf5
  stays small and the AutoData runtime captures the frames per step, as it does for Arena's head camera;
- ``rerender_on_reset`` makes the first frame of an episode valid;
- the camera offset quaternion is given as (x, y, z, w), Isaac Lab 3.0's order.

Camera placement follows the 2026-10 reproduction (front camera at (0.15, 1.8, 1.9) m looking at the table centre, side
camera at (-1.4, 1.5, 1.8) m). Until 2026-10-29 the same cameras lived in the ``policy`` group of an Isaac Lab 2.x
environment and were recorded by the official recorder.
"""
from __future__ import annotations

import importlib.util

import numpy as np
from scipy.spatial.transform import Rotation

ENV_ID = "TMG-PickPlace-GR1T2-Images-Mimic-v0"
IMAGE_SIZE = (128, 128)
FRONT = ((0.15, 1.8, 1.9), (0.15, 0.5, 1.05))
SIDE = ((-1.4, 1.5, 1.8), (0.15, 0.5, 1.05))
CAMERA_GROUP = "camera_obs"


def look_at_quat_xyzw(eye, target) -> tuple[float, float, float, float]:
    """Orientation (x, y, z, w) of a ROS-convention camera at ``eye`` looking at ``target``.

    Isaac Lab 3.0's ``CameraCfg.OffsetCfg.rot`` is (x, y, z, w); Isaac Lab 2.x took (w, x, y, z), and passing that order
    here points the cameras at the dome light (uniform white frames, found 2026-10-29).
    """
    eye, target = np.asarray(eye, dtype=float), np.asarray(target, dtype=float)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    xyzw = Rotation.from_matrix(np.column_stack([right, down, forward])).as_quat()
    return float(xyzw[0]), float(xyzw[1]), float(xyzw[2]), float(xyzw[3])


def register() -> str:
    """Build the camera-enabled config class and register the gym id; returns the id."""
    import gymnasium as gym
    import isaaclab.sim as sim_utils
    from isaaclab.envs import mdp
    from isaaclab.managers import ObservationGroupCfg, ObservationTermCfg, SceneEntityCfg
    from isaaclab.sensors import CameraCfg
    from isaaclab.utils import configclass
    from isaaclab_mimic.envs.pickplace_gr1t2_mimic_env_cfg import PickPlaceGR1T2MimicEnvCfg
    from isaaclab_tasks.contrib.pick_place.pickplace_gr1t2_env_cfg import ObservationsCfg, PickPlaceGR1T2SceneCfg

    def camera(name: str, eye, target) -> CameraCfg:
        return CameraCfg(prim_path="{ENV_REGEX_NS}/" + name, update_period=0.0, height=IMAGE_SIZE[0], width=IMAGE_SIZE[1],
                         data_types=["rgb"], spawn=sim_utils.PinholeCameraCfg(focal_length=18.15, clipping_range=(0.1, 3.0)),
                         offset=CameraCfg.OffsetCfg(pos=eye, rot=look_at_quat_xyzw(eye, target), convention="ros"))

    @configclass
    class ImageSceneCfg(PickPlaceGR1T2SceneCfg):
        robot_pov_cam = None                      # the 450x720 teleoperation camera is not rendered
        front_camera = camera("FrontCamera", *FRONT)
        side_camera = camera("SideCamera", *SIDE)

    @configclass
    class CameraObsCfg(ObservationGroupCfg):
        front_image = ObservationTermCfg(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("front_camera"), "normalize": False})
        side_image = ObservationTermCfg(func=mdp.image, params={"sensor_cfg": SceneEntityCfg("side_camera"), "normalize": False})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class ImageObservationsCfg(ObservationsCfg):      # the base policy group (no image term) plus the camera group
        camera_obs: CameraObsCfg = CameraObsCfg()

    @configclass
    class PickPlaceGR1ImagesCfg(PickPlaceGR1T2MimicEnvCfg):
        scene: ImageSceneCfg = ImageSceneCfg(num_envs=1, env_spacing=2.5, replicate_physics=True)
        observations: ImageObservationsCfg = ImageObservationsCfg()
        rerender_on_reset = True

        def __post_init__(self):
            super().__post_init__()
            self.image_obs_list = []              # nothing for AutoData's setup_env_config to strip from the policy group

    if ENV_ID not in gym.registry:
        gym.register(id=ENV_ID, entry_point="isaaclab_mimic.envs.pickplace_gr1t2_mimic_env:PickPlaceGR1T2MimicEnv",
                     kwargs={"env_cfg_entry_point": PickPlaceGR1ImagesCfg}, disable_env_checker=True)
    return ENV_ID


if importlib.util.find_spec("isaaclab") is not None:      # importing the module registers the environment inside Isaac Sim;
    register()                                              # elsewhere (unit tests) only the helpers are available
