"""Test fixture: the official ``CPPolicy.solve_constraint`` of CP-Gen (kevin-thankyou-lin/cpgen @ 1e7c1fc4,
demo_aug/generate.py), copied verbatim with the same indentation so ``inspect.getsource`` gives the text the
generator config pins by hash. The module-level names the body uses are stubbed below (``quat2axisangle``,
``mat2quat``, ``PlanningResult`` and the post-constraint helpers as in CP-Gen); tests replace
``compute_X_W_eef_des``, ``find_best_transform`` and ``optimize_robot_configuration`` and pass fake configurations.
Do not edit the method text.
"""
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

Constraint = Any


@dataclass
class PlanningResult:
    actions: List[np.ndarray]
    success: bool
    robot_configurations: Optional[List[np.ndarray]] = None


def quat2axisangle(quat):
    """x, y, z, w quaternion to axis-angle (CP-Gen's function)."""
    quat = np.array(quat, dtype=float)
    quat[3] = min(max(quat[3], -1.0), 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def mat2quat(rmat):
    """Rotation matrix to w, x, y, z quaternion (CP-Gen's function)."""
    return np.roll(R.from_matrix(rmat).as_quat(), 1)


def quat2mat(quaternion):
    """x, y, z, w quaternion to rotation matrix (same convention as CP-Gen's function)."""
    return R.from_quat(np.asarray(quaternion, dtype=float)).as_matrix()


def generate_lift_actions(last_action, retract_height=0.04, retract_steps=5):
    return [np.array(last_action, dtype=float) + np.r_[0, 0, retract_height * (i + 1) / retract_steps, 0, 0, 0, 0]
            for i in range(retract_steps)]


def generate_open_gripper_actions(last_action, open_gripper_steps=10):
    return [np.r_[np.asarray(last_action, dtype=float)[:6], -1] for _ in range(open_gripper_steps)]


def generate_close_gripper_actions(last_action, open_gripper_steps=10):
    return [np.r_[np.asarray(last_action, dtype=float)[:6], 1] for _ in range(open_gripper_steps)]


def compute_X_W_eef_des(*args, **kwargs):          # replaced by the tests
    raise NotImplementedError


def find_best_transform(*args, **kwargs):          # replaced by the tests
    raise NotImplementedError


def optimize_robot_configuration(*args, **kwargs):  # replaced by the tests
    raise NotImplementedError


def optimize_robot_configuration_kp(*args, **kwargs):
    raise NotImplementedError


def get_scale_factor(*args, **kwargs):
    return 1.0


def transform_keypoints(points, pose):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    return (np.c_[points, np.ones(len(points))] @ np.asarray(pose, dtype=float).T)


def set_mocap_pos_and_update_viewer(*args, **kwargs):
    raise NotImplementedError


def render_image(*args, **kwargs):
    raise NotImplementedError


def visualize_robot_configuration_and_keypoints(*args, **kwargs):
    raise NotImplementedError


class CPPolicy:
    def solve_constraint(
        self,
        constraint: Constraint,
        obs: Dict[str, Any],
        dummy_actions: bool = False,
        copy_constraint_actions: bool = True,
        ik_type: Literal["kpts_to_robot_q", "kpts_to_eef_to_q"] = "kpts_to_eef_to_q",
        fallback_to_kpts_to_robot_q: bool = True,
        solve_first_n_steps: Optional[List[int]] = None,
        # currently, no optimization for kpts_to_eef
    ) -> PlanningResult:
        """Generate a plan based on the updated constraint info.
        Args:
            solve_first_n_steps: first n steps for solve constraint for.
                Defaults to None, meaning all steps. Useful if only solving for e.g. first step.
            ik_type: type of IK solver to use.
                - kpts_to_robot_q: [single stage] solve IK directly from keypoints to robot q.
                - kpts_to_eef_to_q: [two stage] solve IK from keypoints to eef and then from eef to robot q.
                    In practice, currently, doing eef to q directly. TODO(klin): add kpts to eef.
            fallback_to_kpts_to_robot_q: if ik_type="kpts_to_eef_to_q" fails, fallback to "kpts_to_robot_q".
        """
        # Implement the logic for solving the constraint and generating a plan
        if dummy_actions:
            # stay at the current eef pose
            actions = np.zeros((30, 7))
            target_pos = obs["robot0_eef_pos"]
            actions[:, :3] = target_pos
            target_quat_xyzw = obs["robot0_eef_quat_site"]
            target_axis_angle = quat2axisangle(target_quat_xyzw)
            actions[:, 3:6] = target_axis_angle
            DUMMY_GRIPPER_ACTION = 0.1
            actions[:, 6] = DUMMY_GRIPPER_ACTION  # gripper action
            actions = actions  # + np.random.normal(0, 0.1, actions.shape)

        if copy_constraint_actions:
            actions = constraint.src_action

        if self.viz_robot_kpts:
            # record env's current state
            state = self.env.env.get_state()
            state.pop("model")

        # use the current gripper obj transform to attach the object to the gripper
        self._update_robot_configuration(
            self.robot_configuration,
            eef_configuration=self.eef_configuration,
            env=self.env,
            obs=obs,
            constraint=constraint,
        )

        qs = []
        actions = []  # assume action is eef_link_frame
        # let's use the original pose and get the updated pose for now?

        # solve optimization problem to get actions
        prev_q = self.robot_configuration.q[self.robot_configuration.robot_idxs]
        # two actuated objects case --- can directly use their original poses to get keypoints
        # one actuated object (either arm or object) case: use the non-actuated object as the anchor
        #   if arm: the optimize hand-pose + gripper config first
        #   if object: optimize object pose, then set the hand+gripper pose accordingly -- the key is knowing what to optimize in robot_configuration
        #       in both cases, need to 1) detect/remove joint between object and robot 2) optimize over the pose
        #       simplification --- can always optimize over robot eef pose (and optionally gripper qpos) for both arm/object being actuated I believe, until doing in hand manipulation
        obs["env"] = self.env
        X_obj0_transf, X_obj1_transf, X_eef_transf = None, None, None
        X_obj0_transf, X_obj1_transf, X_eef_transf = find_best_transform(
            constraint, obs, self.robot_configuration, self.eef_configuration
        )
        solve_first_n_steps: int = (
            solve_first_n_steps
            if solve_first_n_steps is not None
            else len(constraint.timesteps)
        )
        assert (
            solve_first_n_steps <= len(constraint.timesteps)
        ), "constraint_steps should be less than or equal to the number of timesteps in the constraint"
        for t_idx in range(solve_first_n_steps):
            # # how to add in symmetry here? if doing discree option per symmetry, would get exponential explosion
            if constraint.constraint_type == "robot-object":
                link_to_kpts = {
                    link_name: keypoints[t_idx]
                    for link_name, keypoints in constraint.keypoints_robot_link_frame.items()
                }
                # Get scaled keypoints
                task_relev_obj = constraint.obj_names[0]
                # Get the scale factor of the demo object and the current object
                demo_obj_geoms_size: Dict[str, np.ndarray] = (
                    constraint.src_obj_geoms_size[task_relev_obj][t_idx]
                )
                current_obj_geoms_size: Dict[str, np.ndarray] = obs[
                    f"{task_relev_obj}_geoms_size"
                ]
                # Use the helper function to get the scale factor
                scale_factor = (
                    get_scale_factor(demo_obj_geoms_size, current_obj_geoms_size)
                    if not self.ignore_scaling
                    else 1.0
                )
                # Scale the original keypoints to the current object's scale
                P_obj_target = (
                    constraint.keypoints_obj_frame[task_relev_obj][t_idx] * scale_factor
                )
                P_W_target = transform_keypoints(
                    P_obj_target, obs[f"{constraint.obj_names[0]}_pose"] @ X_obj0_transf
                )[:, :3]
            elif constraint.constraint_type == "object-object":
                # only keep keypoints attached to the kinematically 'actuated' object
                link_to_kpts = {
                    link_name: keypoints[t_idx]
                    for link_name, keypoints in constraint.keypoints_obj_frame.items()
                    if link_name in constraint.obj_to_parent_attachment_frame
                    and constraint.obj_to_parent_attachment_frame[link_name] is not None
                }
                # get object keypoints in world frame: will be used in optimization q* = argmin_q ||P_W_goal - P_W_robot_kp(q)||^2
                P_obj_target = np.array(
                    constraint.keypoints_obj_frame[constraint.obj_names[1]][t_idx]
                )
                P_W_target = transform_keypoints(
                    P_obj_target, obs[f"{constraint.obj_names[1]}_pose"] @ X_obj1_transf
                )[:, :3]  # s

                use_weld_heuristic_action: bool = True
                if use_weld_heuristic_action:
                    constraint.src_gripper_action[t_idx] = (
                        1  # hack for now to test if we needed to match actions rather than states
                    )

            flip_eef = not np.allclose(X_eef_transf, np.eye(4), rtol=1e-5, atol=1e-8)
            self.robot_configuration.set_keypoints(link_to_kpts, flip_eef=flip_eef)
            self.eef_configuration.set_keypoints(link_to_kpts, flip_eef=flip_eef)

            if self.viz_robot_kpts:
                # set mocap keypoints to the target keypoints
                set_mocap_pos_and_update_viewer(
                    obs["env"],
                    self.robot_configuration.get_keypoints(),
                    update_viewer=False,
                )
                render_image(
                    obs["env"].env.env.sim.model._model,
                    obs["env"].env.env.sim.data._data,
                    image_prefix="kpts-rob",
                )
                set_mocap_pos_and_update_viewer(
                    obs["env"], P_W_target, update_viewer=False
                )
                set_mocap_pos_and_update_viewer(
                    obs["env"], P_W_target, update_viewer=False
                )
                render_image(
                    obs["env"].env.env.sim.model._model,
                    obs["env"].env.env.sim.data._data,
                    image_prefix="kpts-obj-1",
                )
                set_mocap_pos_and_update_viewer(
                    obs["env"], P_W_target[2:], update_viewer=False
                )
                render_image(
                    obs["env"].env.env.sim.model._model,
                    obs["env"].env.env.sim.data._data,
                    image_prefix="kpts-obj-2",
                )
            # potentially merge eef des computation w/ joint q computation
            X_W_eef_des = compute_X_W_eef_des(
                constraint,
                obs,
                self.robot_configuration,
                self.eef_configuration,
                X_obj0_transf=X_obj0_transf,
                X_obj1_transf=X_obj1_transf,
                X_eef_transf=X_eef_transf,
                src_t_idx=t_idx,
                ignore_scaling=self.ignore_scaling,
            )
            q_eef_config = np.zeros(7 + 2)
            q_eef_config[:3] = X_W_eef_des[:3, 3]
            q_eef_config[3:7] = mat2quat(X_W_eef_des[:3, :3])
            q_eef_config[7:] = constraint.src_obs["robot0_gripper_qpos"][t_idx]
            self.eef_configuration.update(q_eef_config)
            # get the action
            action = np.zeros(7)
            action[:3] = X_W_eef_des[:3, 3]
            action[3:6] = quat2axisangle(np.roll(mat2quat(X_W_eef_des[:3, :3]), -1))
            action[6] = constraint.src_gripper_action[t_idx].item()
            optimized_q = optimize_robot_configuration(
                self.robot_configuration,
                self.eef_configuration,
                retract_q=np.array(
                    [
                        0,
                        np.pi / 16.0,
                        0.00,
                        -np.pi / 2.0 - np.pi / 3.0,
                        0.00,
                        np.pi - 0.2,
                        np.pi / 4,
                        0,
                        0,
                    ]
                ),
                update_gripper_qpos=False,
                retract_q_weights=np.array([2, 2, 1, 1, 1, 0.1, 0.1, 0, 0]),
                # prefer the 'elbow' up pose: domain knowledge to avoid singularities
                start_q=prev_q,
                eef_interp_mink_mp=self.eef_interp_mink_motion_planer,
                optimization_type="scipy" if t_idx == 0 else "scipy",
                verbose=self.verbose,
            )

            if optimized_q is None:
                if self.verbose:
                    logging.info(
                        "Optimization failed. Usually occurs when a previous stage wasn't executed correctly or when. "
                        "the optimization problem is ill-posed due to e.g. kinematics."
                    )
                if fallback_to_kpts_to_robot_q:
                    logging.info("Falling back to ik_type=kpts_to_robot_q.")
                else:
                    return PlanningResult(
                        actions=None,
                        robot_configurations=None,
                        success=False,
                    )
            else:
                optimized_q[7:] = constraint.src_obs["robot0_gripper_qpos"][
                    t_idx
                ]  # hack for now to test if we needed to match actions rather than states
            if (
                optimized_q is None and fallback_to_kpts_to_robot_q
            ) or ik_type == "kpts_to_robot_q":
                logging.info(f"Falling back to IK on keypoints for t_idx: {t_idx}")
                # TODO: set up IK optimization for eef pose. For now, directly provide eef pose for curobo's expected frame
                start = time.time()
                if t_idx == 0:
                    # need to get the keypoint of the SquareNut to the peg
                    prev_q = optimize_robot_configuration_kp(
                        self.robot_configuration,
                        P_W_target,
                        prev_q,
                        max_iter=self.max_iter + 100,
                        verbose=True,
                    )
                    prev_q = optimize_robot_configuration_kp(
                        self.robot_configuration,
                        P_W_target,
                        prev_q,
                        max_iter=self.max_iter + 100,
                        verbose=True,
                    )
                optimized_q = optimize_robot_configuration_kp(
                    self.robot_configuration,
                    P_W_target,
                    prev_q,
                    max_iter=self.max_iter,
                    verbose=True,
                    global_opt=False,
                )
                logging.info(
                    f"Optimization t_idx: {t_idx} took {time.time() - start} seconds"
                )

            if self.viz_robot_kpts:
                set_mocap_pos_and_update_viewer(self.env, P_W_target)
                visualize_robot_configuration_and_keypoints(
                    self.robot_configuration,
                    self.env,
                    optimized_q,
                    constraint=constraint,
                )

            qs.append(optimized_q)

            self.robot_configuration.update(optimized_q)
            # get the eef pose from the robot configuration
            eef_pose = self.robot_configuration.get_transform_frame_to_world(
                "gripper0_right_grip_site", "site"
            ).as_matrix()
            # update action to work w/ expected format
            action = np.zeros(7)
            action[:3] = eef_pose[:3, 3]
            # TODO(klin): use the transformed eef pose as the action if no scaling etc actions.
            action[3:6] = quat2axisangle(R.from_matrix(eef_pose[:3, :3]).as_quat())
            action[6] = constraint.src_gripper_action[
                t_idx
            ].item()  # TODO: do better than copying the gripper action
            actions.append(action)

        if self.viz_robot_kpts:
            # reset the env to the original state
            self.env.env.reset_to(
                state
            )  # ideally these methods are inside self.env itself
            self.env.env.env.sim.forward()
            self.env.env.env.viewer.update()

        for behavior in constraint.post_constraint_behavior:  # heuristic
            if behavior == "lift":
                actions.extend(generate_lift_actions(actions[-1]))
            elif behavior == "open_gripper":
                actions.extend(generate_open_gripper_actions(actions[-1]))
            elif behavior == "close_gripper":
                actions.extend(generate_close_gripper_actions(actions[-1]))
            # TODO also update qs based on the extra actions

        actions = [np.array(a) for a in actions]
        return PlanningResult(
            actions=actions,
            robot_configurations=qs,
            success=True,
        )
