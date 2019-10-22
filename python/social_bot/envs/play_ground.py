# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A simple enviroment for an agent play on a play ground
"""
import os
import time
import math
import numpy as np
import random
import json
import PIL
import gin
import gym
from gym import spaces
from absl import logging
from abc import abstractmethod
from collections import OrderedDict

import social_bot
import social_bot.pygazebo as gazebo
from social_bot import teacher
from social_bot import teacher_tasks
from social_bot.envs.gazebo_base import GazeboEnvBase
from social_bot.teacher import TaskGroup
from social_bot.teacher import TeacherAction
from social_bot.teacher_tasks import GoalWithDistractionTask, ICubAuxiliaryTask, KickingBallTask


@gin.configurable
class PlayGround(GazeboEnvBase):
    """
    This envionment support agent type of pr2_noplugin, pioneer2dx_noplugin,
    turtlebot, icub, and kuka youbot for now. Note that for the models without
    camera sensor like icub (without hands), you can not use image as observation.

    Joints of the agent are controllable by force or pid controller,

    The observation space is a numpy array or a dict with keys 'image',
    'states', 'sentence', depends on the configuration.
    If without language and internal_states, observation is a numpy array:
        pure image (use_image_observation=True)
        pure low-dimentional states (use_image_observation=False)
    Otherwise observation is a dict, it could be:
        image and internal states (part of the low-dimentional states)
        image and language sequence
        image, internal states and language sequence
        pure low-dimensional states and language sequence

    The objects are rearranged each time the environment is reseted.

    Agent will receive a reward provided by the teacher. The goal's position
    can also be controlled by the teacher.

    """

    def __init__(self,
                 agent_type='pioneer2dx_noplugin',
                 world_name="play_ground.world",
                 task=GoalWithDistractionTask,
                 secondary_task=None,
                 with_language=False,
                 use_image_observation=False,
                 image_with_internal_states=False,
                 world_time_precision=None,
                 step_time=0.1,
                 port=None,
                 action_cost=0.0,
                 resized_image_size=(64, 64),
                 vocab_sequence_length=20):
        """
        Args:
            agent_type (string): Select the agent robot, supporting pr2_noplugin,
                pioneer2dx_noplugin, turtlebot, youbot_noplugin and icub_with_hands for now
                note that 'agent_type' should be the same str as the model's name
            world_name (string): Select the world file, e.g., empty.world, play_ground.world, 
                grocery_ground.world
            task, secondary_task (teacher.Task): the teacher task, like GoalTask, 
                GoalWithDistractionTask, KickingBallTask, etc.
            with_language (bool): The observation will be a dict with an extra sentence
            use_image_observation (bool): Use image, or use low-dimentional states as
                observation. Poses in the states observation are in world coordinate
            image_with_internal_states (bool): If true, the agent's self internal states
                i.e., joint position and velocities would be available together with image.
                Only affect if use_image_observation is true
            world_time_precision (float|None): if not none, the time precision of
                simulator, i.e., the max_step_size defined in the agent cfg file, will be
                override. e.g., '0.002' for a 2ms sim step
            step_time (float): the peroid of one step of the environment.
                step_time / world_time_precision is how many simulator substeps during one
                environment step. for some complex agent like icub, using a step_time of
                0.05 is more faster to converage
            port: Gazebo port, need to specify when run multiple environment in parallel
            action_cost (float): Add an extra action cost to reward, which helps to train
                an energy/forces efficency policy or reduce unnecessary movements
            resized_image_size (None|tuple): If None, use the original image size
                from the camera. Otherwise, the original image will be resized
                to (width, height)
            vocab_sequence_length (int): the length if encoded sequence
        """

        self._action_cost = action_cost
        self._with_language = with_language
        self._use_image_obs = use_image_observation
        self._image_with_internal_states = self._use_image_obs and image_with_internal_states
        self._resized_image_size = resized_image_size
        self._substep_time = world_time_precision

        # Load agent and world file
        with open(
                os.path.join(social_bot.get_model_dir(), "agent_cfg.json"),
                'r') as cfg_file:
            agent_cfgs = json.load(cfg_file)
        agent_cfg = agent_cfgs[agent_type]
        wd_path = os.path.join(social_bot.get_world_dir(), world_name)
        with open(wd_path, 'r+') as world_file:
            world_string = self._insert_agent_to_world_file(
                world_file, agent_type)
        if world_time_precision is None:
            world_time_precision = agent_cfg['max_sim_step_time']
        sub_steps = int(round(step_time / world_time_precision))
        self._sub_steps = sub_steps
        sim_time_cfg = [
            "//physics//max_step_size=" + str(world_time_precision)
        ]

        super().__init__(
            world_string=world_string, world_config=sim_time_cfg, port=port)

        # Setup teacher and tasks
        self._teacher = teacher.Teacher(task_groups_exclusive=False)
        main_task = task(step_time=step_time)
        task_group = TaskGroup()
        task_group.add_task(main_task)
        self._teacher.add_task_group(task_group)
        if secondary_task != None:
            task_2 = secondary_task(step_time=step_time)
            task_group_2 = TaskGroup()
            task_group_2.add_task(task_2)
            self._teacher.add_task_group(task_group_2)
        self._teacher._build_vocab_from_tasks()
        self._seq_length = vocab_sequence_length
        if self._teacher.vocab_size:
            self._sentence_space = gym.spaces.MultiDiscrete(
                [self._teacher.vocab_size] * self._seq_length)
        self._world.step(20)
        self._agent = self._world.get_agent()
        for task_group in self._teacher.get_task_groups():
            for task in task_group.get_tasks():
                task.setup(self._world, agent_type)

        # Setup action space
        logging.debug(self._world.info())
        self._agent_joints = agent_cfg['control_joints']
        joint_states = list(
            map(lambda s: self._agent.get_joint_state(s), self._agent_joints))
        self._joints_limits = list(
            map(lambda s: s.get_effort_limits()[0], joint_states))
        if agent_cfg['use_pid']:
            for joint_index in range(len(self._agent_joints)):
                self._agent.set_pid_controller(
                    self._agent_joints[joint_index],
                    'velocity',
                    p=0.02,
                    d=0.00001,
                    max_force=self._joints_limits[joint_index])
            self._agent_control_range = agent_cfg['pid_control_limit']
        else:
            self._agent_control_range = np.array(self._joints_limits)
        logging.debug("joints to control: %s" % self._agent_joints)
        self._control_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=[len(self._agent_joints)],
            dtype=np.float32)
        if self._with_language:
            self.action_space = gym.spaces.Dict(
                control=self._control_space, sentence=self._sentence_space)
        else:
            self.action_space = self._control_space

        # Setup observation space
        self._agent_camera = agent_cfg['camera_sensor']
        self.reset()
        obs_sample = self._get_observation_with_sentence("hello")
        if self._with_language or self._image_with_internal_states:
            self.observation_space = self._construct_dict_space(
                obs_sample, self._teacher.vocab_size)
        elif self._use_image_obs:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=obs_sample.shape, dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_sample.shape,
                dtype=np.float32)

    def reset(self):
        """
        Args:
            None
        Returns:
            Observaion of the first step
        """
        self._collision_cnt = 0
        self._cum_reward = 0.0
        self._steps_in_this_episode = 0
        self._world.reset()
        self._teacher.reset(self._agent, self._world)
        # The first call of "teach() after "done" will reset the task
        teacher_action = self._teacher.teach("")
        # Give an intilal random pose offset by take random action
        actions = self._control_space.sample()
        controls = dict(
            zip(self._agent_joints, self._agent_control_range * actions))
        self._agent.take_action(controls)
        self._world.step(self._sub_steps)
        obs = self._get_observation_with_sentence(teacher_action.sentence)
        return obs

    def step(self, action):
        """
        Args:
            action (dict|int): If with_language, action is a dictionary
                    with key "control" and "sentence".
                    action['control'] is a vector whose dimention is
                    len(_joint_names). action['sentence'] is a sentence sequence.
                    If not with_language, it is an int for the action id.
        Returns:
            If with_language, it is a dictionary with key 'data' and 'sentence'
            If not with_language, it is a numpy.array or image for observation
        """
        if self._with_language:
            sentence = action.get('sentence', None)
            if type(sentence) != str:
                sentence = self._teacher.sequence_to_sentence(sentence)
            action_ctrl = action['control']
        else:
            sentence = ''
            action_ctrl = action
        controls = np.clip(action_ctrl, -1.0, 1.0) * self._agent_control_range
        controls = dict(zip(self._agent_joints, controls))
        self._agent.take_action(controls)
        self._world.step(self._sub_steps)
        teacher_action = self._teacher.teach(sentence)
        obs = self._get_observation_with_sentence(teacher_action.sentence)
        self._steps_in_this_episode += 1
        ctrl_cost = np.sum(np.square(action_ctrl)) / action_ctrl.shape[0]
        reward = teacher_action.reward - self._action_cost * ctrl_cost
        self._cum_reward += reward
        if teacher_action.done:
            logging.debug("episode ends at cum reward:" +
                          str(self._cum_reward))
        return obs, reward, teacher_action.done, {}

    def _insert_agent_to_world_file(self, world_file, model):
        content = world_file.read()
        insert_pos = content.find("<!-- AGENT-INSERTION-POINT -->")
        assert insert_pos != -1, "Can not found insertion point in world file"
        content = list(content)
        insert_str = "<include> <uri>model://" + model + "</uri> </include>\n"
        content.insert(insert_pos, insert_str)
        return "".join(content)

    def _get_camera_observation(self):
        image = np.array(
            self._agent.get_camera_observation(self._agent_camera), copy=False)
        if self._resized_image_size:
            image = PIL.Image.fromarray(image).resize(self._resized_image_size,
                                                      PIL.Image.ANTIALIAS)
            image = np.array(image, copy=False)
        return image

    def _get_low_dim_full_states(self):
        task_specific_ob = self._teacher.get_task_pecific_observation()
        agent_pose = np.array(self._agent.get_pose()).flatten()
        agent_vel = np.array(self._agent.get_velocities()).flatten()
        internal_states = self._get_internal_states(self._agent,
                                                    self._agent_joints)
        obs = np.concatenate(
            (task_specific_ob, agent_pose, agent_vel, internal_states), axis=0)
        return obs

    def _create_observation_dict(self, sentence_raw):
        obs = OrderedDict()
        if self._use_image_obs:
            obs['image'] = self._get_camera_observation()
            if self._image_with_internal_states:
                obs['states'] = self._get_internal_states(
                    self._agent, self._agent_joints)
        else:
            obs['states'] = self._get_low_dim_full_states()
        if self._with_language:
            obs['sentence'] = self._teacher.sentence_to_sequence(
                sentence_raw, self._seq_length)
        return obs

    def _get_observation_with_sentence(self, sentence_raw):
        if self._image_with_internal_states or self._with_language:
            # observation is an OrderedDict
            obs = self._create_observation_dict(sentence_raw)
        elif self._use_image_obs:  # observation is pure image
            obs = self._get_camera_observation()
        else:  # observation is pure low-dimentional states
            obs = self._get_low_dim_full_states()
        return obs


def main():
    """
    Simple testing of this environment.
    """
    import matplotlib.pyplot as plt
    import time
    with_language = True
    use_image_obs = False
    image_with_internal_states = True
    fig = None
    env = PlayGround(
        with_language=with_language,
        use_image_observation=use_image_obs,
        image_with_internal_states=image_with_internal_states,
        agent_type='pioneer2dx_noplugin',
        task=GoalWithDistractionTask)
    env.render()
    step_cnt = 0
    last_done_time = time.time()
    while True:
        actions = env._control_space.sample()
        if with_language:
            actions = dict(control=actions, sentence="hello")
        obs, _, done, _ = env.step(actions)
        step_cnt += 1
        if with_language and (env._steps_in_this_episode == 1 or done):
            seq = obs["sentence"]
            logging.info("sentence_seq: " + str(seq))
            logging.info("sentence_raw: " +
                         env._teacher.sequence_to_sentence(seq))
        if use_image_obs:
            if with_language or image_with_internal_states:
                obs = obs['image']
            if fig is None:
                fig = plt.imshow(obs)
            else:
                fig.set_data(obs)
            plt.pause(0.00001)
        if done:
            env.reset()
            step_per_sec = step_cnt / (time.time() - last_done_time)
            logging.info("step per second: " + str(step_per_sec))
            step_cnt = 0
            last_done_time = time.time()


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main()
