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
A simple enviroment for an agent playing on the ground
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

import social_bot
import social_bot.pygazebo as gazebo
from social_bot.gazebo_agent import GazeboAgent
from social_bot import teacher
from social_bot import tasks
from social_bot.envs.gazebo_base import GazeboEnvBase
from social_bot.teacher import TaskGroup
from social_bot.teacher import TeacherAction
from social_bot.tasks import GoalTask, ICubAuxiliaryTask, KickingBallTask, Reaching3D, PickAndPlace


@gin.configurable
class PlayGround(GazeboEnvBase):
    """
    This envionment support agent type of pr2_noplugin, pioneer2dx_noplugin,
    turtlebot, icub, and kuka youbot for now. 

    Joints of the agent are controllable by force or pid controller,

    The observation space is a numpy array or a dict with keys 'image',
    'states', 'sentence', depends on the configuration. Note that for the
    models without camera sensor like icub (without hands), you can not
    use image as observation.

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
                 tasks=[GoalTask],
                 with_language=False,
                 with_agent_language=False,
                 use_image_observation=False,
                 image_with_internal_states=False,
                 max_steps=200,
                 world_time_precision=None,
                 step_time=0.1,
                 real_time_update_rate=0,
                 port=None,
                 action_cost=0.0,
                 resized_image_size=(64, 64),
                 vocab_sequence_length=20):
        """
        Args:
            agent_type (string): Select the agent robot, supporting pr2_noplugin,
                pioneer2dx_noplugin, turtlebot, youbot_noplugin and icub_with_hands for now
                note that 'agent_type' should be exactly the same string as the model's
                name at the beginning of model's sdf file
            world_name (string): Select the world file, e.g., empty.world, play_ground.world, 
                grocery_ground.world
            tasks (list): a list of teacher.Task, e.g., GoalTask, KickingBallTask
            with_language (bool): The observation will be a dict with an extra sentence
            with_agent_language (bool): Include agent sentence in action space.
            use_image_observation (bool): Use image, or use low-dimentional states as
                observation. Poses in the states observation are in world coordinate
            max_steps (int): episode will end in so many steps, will be passed to the tasks
            image_with_internal_states (bool): If true, the agent's self internal states
                i.e., joint position and velocities would be available together with image.
                Only affect if use_image_observation is true
            world_time_precision (float|None): this parameter depends on the agent. 
                if not none, the default time precision of simulator, i.e., the max_step_size
                defined in the agent cfg file, will be override. Note that pr2 and iCub
                requires a max_step_size <= 0.001, otherwise cannot train a successful policy.
            step_time (float): the peroid of one step() function of the environment in simulation.
                step_time is rounded to multiples of world_time_precision
                step_time / world_time_precision is how many simulator substeps during one
                environment step. for the tasks need higher control frequency (such as the 
                tasks need walking by 2 legs), using a smaller step_time like 0.05 is better.
                experiments show that iCub can not learn how to walk in a 0.1 step_time
            real_time_update_rate (int): max update rate per second. There is no limit if
                this is set to 0 as default in the world file. If 1:1 real time is prefered
                (like playing or recording video), it should be set to 1.0/world_time_precision.
            port: Gazebo port, need to specify when run multiple environment in parallel
            action_cost (float): Add an extra action cost to reward, which helps to train
                an energy/forces efficency policy or reduce unnecessary movements
            resized_image_size (None|tuple): If None, use the original image size
                from the camera. Otherwise, the original image will be resized
                to (width, height)
            vocab_sequence_length (int): the length of encoded sequence
        """

        self._action_cost = action_cost
        self._with_language = with_language
        self._seq_length = vocab_sequence_length
        self._with_agent_language = with_language and with_agent_language
        self._use_image_obs = use_image_observation
        self._image_with_internal_states = self._use_image_obs and image_with_internal_states

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
        self._sub_steps = int(round(step_time / world_time_precision))
        self._step_time = world_time_precision * self._sub_steps
        sim_time_cfg = [
            "//physics//max_step_size=" + str(world_time_precision),
            "//physics//real_time_update_rate=" + str(real_time_update_rate)
        ]
        super().__init__(
            world_string=world_string, world_config=sim_time_cfg, port=port)
        logging.debug(self._world.info())

        # Setup agent
        self._agent = GazeboAgent(
            world=self._world,
            agent_type=agent_type,
            config=agent_cfg,
            with_language=with_language,
            with_agent_language=with_agent_language,
            vocab_sequence_length=self._seq_length,
            use_image_observation=use_image_observation,
            resized_image_size=resized_image_size,
            image_with_internal_states=image_with_internal_states)

        # Setup teacher and tasks
        self._teacher = teacher.Teacher(task_groups_exclusive=False)
        for task in tasks:
            task_group = TaskGroup()
            task_group.add_task(task(env=self, max_steps=max_steps))
            self._teacher.add_task_group(task_group)
        self._teacher._build_vocab_from_tasks()

        # Setup action space and observation space
        if self._teacher.vocab_size:
            sentence_space = gym.spaces.MultiDiscrete(
                [self._teacher.vocab_size] * self._seq_length)
        self._agent.set_sentence_space(sentence_space)
        self._control_space = self._agent.get_control_space()
        self.action_space = self._agent.get_action_space()
        self.reset()
        self.observation_space = self._agent.get_observation_space(
            self._teacher)

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
        self._world.step(self._sub_steps)
        # The first call of "teach() after "done" will reset the task
        teacher_action = self._teacher.teach("")
        obs = self._agent.get_observation(self._teacher,
                                          teacher_action.sentence)
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
        if self._with_agent_language and self._with_language:
            sentence = action.get('sentence', None)
            if type(sentence) != str:
                sentence = self._teacher.sequence_to_sentence(sentence)
            controls = action['control']
        else:
            sentence = ''
            controls = action
        self._agent.take_action(controls)
        self._world.step(self._sub_steps)
        teacher_action = self._teacher.teach(sentence)
        obs = self._agent.get_observation(self._teacher,
                                          teacher_action.sentence)
        self._steps_in_this_episode += 1
        ctrl_cost = np.sum(np.square(controls)) / controls.shape[0]
        reward = teacher_action.reward - self._action_cost * ctrl_cost
        self._cum_reward += reward
        if teacher_action.done:
            logging.debug("episode ends at cum reward:" +
                          str(self._cum_reward))
        return obs, reward, teacher_action.done, {}

    def get_step_time(self):
        """
        Args:
            None
        Returns:
            The time span of an environment step
        """
        return self._step_time

    def _insert_agent_to_world_file(self, world_file, model):
        content = world_file.read()
        insert_pos = content.find("<!-- AGENT-INSERTION-POINT -->")
        assert insert_pos != -1, "Can not found insertion point in world file"
        content = list(content)
        insert_str = "<include> <uri>model://" + model + "</uri> </include>\n"
        content.insert(insert_pos, insert_str)
        return "".join(content)


def main():
    """
    Simple testing of this environment.
    """
    import matplotlib.pyplot as plt
    import time
    with_language = True
    with_agent_language = False
    use_image_obs = False
    image_with_internal_states = True
    fig = None
    env = PlayGround(
        with_language=with_language,
        with_agent_language=with_agent_language,
        use_image_observation=use_image_obs,
        image_with_internal_states=image_with_internal_states,
        agent_type='youbot_noplugin',
        tasks=[PickAndPlace])
    env.render()
    step_cnt = 0
    last_done_time = time.time()
    while True:
        actions = env._control_space.sample()
        if with_agent_language:
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
