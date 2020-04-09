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
"""A simple enviroment for navigation."""

from collections import OrderedDict
import numpy as np
import os
import random
import time

from absl import logging
import gym
import gym.spaces
import gin

import social_bot
from social_bot import teacher
from social_bot.envs.gazebo_base import GazeboEnvBase
from social_bot.gazebo_agent import GazeboAgent
from social_bot.teacher import TeacherAction
from social_bot.teacher import DiscreteSequence
from social_bot.tasks import GoalTask
import social_bot.pygazebo as gazebo


@gin.configurable
class SimpleNavigation(GazeboEnvBase):
    """
    In this environment, the agent will receive reward 1 when it is close enough to the goal.
    If it is moving away from the goal too much or still not close to the goal after max_steps,
    it will get reward -1.

    The observation space is a numpy array or a dict with keys 'image', 'states', 'sentence'
    If without language and internal_states, observation is a numpy array contains the image
    Otherwise observation is a dict. Depends on the configuration, it could be :
        image and internal states (the states of agent joints)
        image and language sequence
        image, internal states and language sequence
    """

    # number of physics simulation steps per step(). Each step() corresponds to
    # a real time of NUM_SIMULATION_STEPS * max_step_size, where `max_step_size`
    # is defined in file pioneer2dx_camera.world
    NUM_SIMULATION_STEPS = 20

    def __init__(self,
                 with_language=False,
                 image_with_internal_states=False,
                 port=None,
                 resized_image_size=None):
        """Create SimpleNavigation environment.

        Args:
            with_language (bool): whether to generate language for observation
            image_with_internal_states (bool): If true, the agent's self internal
                states i.e., joint position and velocities would be available
                together with the image.
            port (int): TCP/IP port for the simulation server
            resized_image_size (None|tuple): If None, use the original image size
                from the camera. Otherwise, the original image will be resized
                to (width, height)
        """
        super(SimpleNavigation, self).__init__(
            world_file='pioneer2dx_camera.world', port=port)

        self._with_language = with_language
        self._image_with_internal_states = image_with_internal_states
        self.set_rendering_cam_pose('4 -4 3 0 0.4 2.3')
        self._seq_length = 20

        # Setup agent
        self._agent = GazeboAgent(
            world=self._world,
            agent_type='pioneer2dx_noplugin',
            with_language=with_language,
            vocab_sequence_length=self._seq_length,
            use_image_observation=True,
            resized_image_size=resized_image_size,
            image_with_internal_states=image_with_internal_states)

        # Setup teacher and tasks
        self._teacher = teacher.Teacher(task_groups_exclusive=False)
        task_group = teacher.TaskGroup()
        task = GoalTask(
            env=self,
            max_steps=120,
            goal_name="goal",
            fail_distance_thresh=0.5,
            distraction_list=[],
            random_range=2.7,
            polar_coord=False)
        task_group.add_task(task)
        self._teacher.add_task_group(task_group)
        self._sentence_space = DiscreteSequence(self._teacher.vocab_size,
                                                self._seq_length)

        # Setup action space and observation space
        self.reset()
        self._agent.set_sentence_space(self._sentence_space)
        self._control_space = self._agent.get_control_space()
        self._action_space = self._agent.get_action_space()
        self._observation_space = self._agent.get_observation_space(
            self._teacher)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def reward_range(self):
        return -1., 1.

    def step(self, action):
        """
        Args:
            action (dict|int): If with_language, action is a dictionary with key "control" and "sentence".
                    action['control'] is a vector whose dimension is
                    len(joints). action['sentence'] is a sentence sequence.
                    If not with_language, it is an int for the action id.
        Returns:
            If with_language, it is a dictionary with key 'obs' and 'sentence'
            If not with_language, it is a numpy.array for observation
        """
        if self._with_language:
            sentence = action.get('sentence', None)
            if type(sentence) != str:
                sentence = self._teacher.sequence_to_sentence(sentence)
            controls = action['control']
        else:
            sentence = ''
            controls = action
        self._agent.take_action(controls)
        self._world.step(self.NUM_SIMULATION_STEPS)
        teacher_action = self._teacher.teach(sentence)
        obs = self._agent.get_observation(self._teacher,
                                          teacher_action.sentence)
        return (obs, teacher_action.reward, teacher_action.done, {})

    def reset(self):
        self._teacher.reset(self._agent, self._world)
        teacher_action = self._teacher.teach("")
        self._world.step(self.NUM_SIMULATION_STEPS)
        obs = self._agent.get_observation(self._teacher,
                                          teacher_action.sentence)
        return obs


class SimpleNavigationDiscreteAction(SimpleNavigation):
    def __init__(self, port=None):
        super(SimpleNavigationDiscreteAction, self).__init__(port=port)
        self._action_space = gym.spaces.Discrete(25)

    def step(self, action):
        control = [0.05 * (action // 5) - 0.1, 0.05 * (action % 5) - 0.1]
        return super(SimpleNavigationDiscreteAction, self).step(control)


class SimpleNavigationLanguage(SimpleNavigation):
    def __init__(self, port=None):
        super(SimpleNavigationLanguage, self).__init__(
            with_language=True, port=port)


class SimpleNavigationSelfStatesLanguage(SimpleNavigation):
    def __init__(self, port=None):
        super(SimpleNavigationSelfStatesLanguage, self).__init__(
            with_language=True, image_with_internal_states=True, port=port)


def main():
    """
    Simple testing of this enviroenment.
    """
    import matplotlib.pyplot as plt
    env = SimpleNavigationSelfStatesLanguage()
    for _ in range(10000000):
        obs = env.reset()
        control = [random.random() * 0.2, random.random() * 0.2]
        plt.imshow(obs['image'])
        logging.info("Close the figure to continue")
        plt.show()
        fig = None
        while True:
            obs, reward, done, _ = env.step(
                dict(control=control, sentence="hello"))
            if fig is None:
                fig = plt.imshow(obs['image'])
            else:
                fig.set_data(obs['image'])
            plt.pause(0.00001)
            if done:
                logging.info("reward: " + str(reward) + "sent: " +
                             str(obs["sentence"]))
                break


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    main()
