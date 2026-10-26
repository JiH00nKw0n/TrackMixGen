"""Test fixture: the official ``DemoGenerator.generate_demo`` of CP-Gen (kevin-thankyou-lin/cpgen @ 1e7c1fc4,
demo_aug/generate.py), copied verbatim with the same indentation so ``inspect.getsource`` gives the text the
generator config pins by hash. The names the body uses are stubbed here; tests replace ``self.env`` and
``self.policy`` with fakes. Do not edit the method text.
"""
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class SystemNoiseConfig:
    constraint_segment_noise_magnitude: float = 0.0
    motion_segment_noise_magnitude: float = 0.0


class CPEnvRobomimic:            # isinstance target only; the fakes are not instances of it
    pass


class Constraint:                # constructed when a new constraint segment starts; keeps the fields
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.timesteps = []


def load_demos(*args, **kwargs):
    raise NotImplementedError


def update_fixed_joint_objects_in_xml(model, xml):
    return xml


def is_env_state_close(env, constraint):
    return False


def create_passive_viewer(*args, **kwargs):
    pass


def save_images(*args, **kwargs):
    pass


def anonymize_model_file_paths(xml):
    return xml


class DemoGenerator:
    use_reset_near_constraint = False
    use_reset_to_state = False
    anonymize_model_file_paths = False

    def generate_demo(
        self,
        gen_single_stage_only: bool = False,
        store_single_stage_only: bool = False,
        store_single_stage_max_extra_steps: int = 0,
        system_noise_cfg: SystemNoiseConfig = field(default_factory=SystemNoiseConfig)
    ) -> Dict[str, Any]:
        obss, actions, dones, rewards, infos, states = [], [], [], [], [], []
        constraint_sequence: List[Constraint] = []
        timestep = 0
        obs = self.env.reset()
        state = self.env.get_state()
        if self.use_reset_near_constraint:
            self.policy.current_constraint_sequence_idx = random.randint(
                0, len(self.policy.constraint_sequences) - 1
            )  # randomly choose constraint sequence from which to choose a constraint
            # maybe the source constraint method selection can be here
            if self.constraint_selection_method == "random":
                self.policy.current_constraint_idx = random.randint(
                    0, len(self.policy.current_constraint_sequence) - 1
                )
            elif self.constraint_selection_method == "first":
                self.policy.current_constraint_idx = 0
            elif self.constraint_selection_method == "second":
                self.policy.current_constraint_idx = 1
            elif self.constraint_selection_method == "third":
                self.policy.current_constraint_idx = 2
            elif self.constraint_selection_method == "last":
                self.policy.current_constraint_idx = (
                    len(self.policy.current_constraint_sequence) - 1
                )

            obs = self.reset_near_constraint(
                self.policy.current_constraint,
                pos_bounds=(
                    (
                        -self.policy.current_constraint.reset_near_constraint_pos_bound,
                        self.policy.current_constraint.reset_near_constraint_pos_bound,
                    ),
                    (
                        -self.policy.current_constraint.reset_near_constraint_pos_bound,
                        self.policy.current_constraint.reset_near_constraint_pos_bound,
                    ),
                    (
                        -self.policy.current_constraint.reset_near_constraint_pos_bound,
                        self.policy.current_constraint.reset_near_constraint_pos_bound,
                    ),
                ),
                max_rot_angle=np.pi / 4,
                verbose=self.policy.verbose,
                randomize_gripper=True,
            )
            state["states"] = self.env.get_state()["states"]
        elif self.use_reset_to_state:
            demos = load_demos(self.reset_state_demo_path, self.reset_state_demo_idx)
            state["states"] = demos[0].states[0]
            self.env.env.reset_to(state)
            obs = self.env.get_observation()

        model = state[
            "model"
        ]  # temp: use original model file to avoid get_xml() issues
        states.append(state["states"])
        obss.append(obs)
        success = False
        timestep = 0
        failure_type = None
        while True:
            action_dct = self.policy.get_action(obs)
            action = action_dct["action"]
            if action is None:
                failure_type = action_dct["failure_type"]
                if failure_type is not None:  # failured
                    logging.info(f"policy.get_action failed with type: {failure_type}")
                    if len(rewards) > 0:
                        rewards[-1] = 0
                    else:
                        rewards.append(0)
                    break
            if self.policy.done:
                break

            if self.policy.new_constraint:
                if isinstance(self.env, CPEnvRobomimic):
                    model = update_fixed_joint_objects_in_xml(
                        self.env.env.env.sim.model, model
                    )
                curr_constraint = Constraint(
                    obj_names=self.policy.current_constraint.obj_names,
                    obj_to_parent_attachment_frame=self.policy.current_constraint.obj_to_parent_attachment_frame,
                    post_constraint_behavior=self.policy.current_constraint.post_constraint_behavior,
                    constraint_type=self.policy.current_constraint.constraint_type,
                    keypoints_obj_frame_annotation=self.policy.current_constraint.keypoints_obj_frame_annotation,
                    keypoints_robot_link_frame_annotation=self.policy.current_constraint.keypoints_robot_link_frame_annotation,
                    src_model_file=model,
                    reset_near_random_constraint_state=self.policy.current_constraint.reset_near_random_constraint_state,
                )
                curr_constraint.timesteps = list(
                    range(
                        timestep + self.policy.curr_motion_plan_steps,
                        timestep
                        + self.policy.curr_motion_plan_steps
                        + self.policy.curr_constraint_steps,
                    )
                )
                constraint_sequence.append(curr_constraint)

            action_to_take = np.array(action)

            # goal: simulate errors that robot might see during test time
            actions_left = len(self.policy.action_queue)
            n_constraint_segment_actions = self.policy.curr_constraint_steps
            in_constraint_segment = actions_left < n_constraint_segment_actions
            is_last_constraint_segment = (
                self.policy.current_constraint_idx
                == len(self.policy.current_constraint_sequence) - 1
            )
            if in_constraint_segment and not is_last_constraint_segment:
                action_to_take += np.random.uniform(-1, 1, action.shape) * np.array(
                    system_noise_cfg.constraint_segment_noise_magnitude
                )
            elif not in_constraint_segment:
                action_to_take += np.random.uniform(-1, 1, action.shape) * np.array(
                    system_noise_cfg.motion_segment_noise_magnitude
                )
            # mightn't be the best great to add noise in constraint segment --- however, needed for stuff like grasping?
            # another strategy is to make the guy 'see' incorrect poses and try to solve for the constraint preservation
            # in this way it's more correlated but still centered?
            # issue: still want the 'correct' actions to train on ... we'll want to ignore the add-noise script in this case
            # issue with the above approach is it may lead to bias on the failure cases ...

            obs, reward, done, info = self.env.step(action_to_take)
            timestep += 1
            obss.append(obs)
            state = self.env.get_state()
            states.append(state["states"])
            actions.append(action)
            rewards.append(reward)

            if self.policy.check_constraint and gen_single_stage_only:
                if is_env_state_close(self.env, self.policy.current_constraint):
                    # TODO(klin): check if state is close enough to original state ...
                    # check constraint means flipped to new constraint
                    rewards[-1] = 1  # hardcode success to keep single stage
                break
            if done:
                logging.info("Episode ended before success.")
                # currently unused, but need rewards
                break

            if self.policy.check_constraint:
                # TODO: move this out of demo generator loop if possible?
                obs_lst: List[np.ndarray] = [
                    obs_dct.get("agentview_image", None) for obs_dct in obss
                ]
                if self.policy.verbose:
                    # visualize the robot configurations
                    create_passive_viewer(
                        self.policy.robot_configuration.model,
                        self.policy.robot_configuration.data,
                        show_left_ui=False,
                        show_right_ui=False,
                    )
                # TODO: fix the obs please
                if obs_lst[0] is not None:
                    save_images(
                        obs_lst, "datasets/generated/images-new-full/", save_as_mp4=True
                    )
        self.policy.reset()

        if len(actions) == 0:
            logging.warning(
                "Using dummy action for failed demo. TODO: reproduce failure for diagnosis."
            )
            actions.append(np.zeros(7))

        success = (rewards[-1] == 1) if rewards else False

        # quirk of mujoco based envs that directly model the mjmodel
        if isinstance(self.env, CPEnvRobomimic):
            model = update_fixed_joint_objects_in_xml(self.env.env.env.sim.model, model)

        if store_single_stage_only and len(constraint_sequence) > 0:
            assert (
                store_single_stage_max_extra_steps >= 0
            ), "store_single_stage_max_extra_steps must be non-negative"
            constraint_last_tstep = constraint_sequence[0].timesteps[-1]
            max_steps = min(
                constraint_last_tstep + store_single_stage_max_extra_steps, len(obss)
            )
            obss = obss[:max_steps]
            actions = actions[:max_steps]
            dones = dones[:max_steps]
            rewards = rewards[:max_steps]
            infos = infos[:max_steps]
            states = states[:max_steps]
            constraint_sequence = [constraint_sequence[0]]

        if self.anonymize_model_file_paths:
            model = anonymize_model_file_paths(model)

        # Save the episode states/actions
        episode_data: Dict[str, Any] = {
            "observations": obss,
            "actions": np.array(actions),
            "dones": dones,
            "rewards": rewards,
            "infos": infos,
            "states": np.array(states)[: len(actions)],
            "model_file": model,  # temp: use original model file to avoid get_xml() issues
            "success": success,
            "failure_type": failure_type,
            "constraint_sequence": constraint_sequence,
        }
        return episode_data
