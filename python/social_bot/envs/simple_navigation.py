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
A simple enviroment for navigation.
"""
from collections import OrderedDict
import gym
import gym.spaces
import logging
import numpy as np
import os
import random

import social_bot
from social_bot import teacher
from social_bot.teacher import TeacherAction
from social_bot.teacher import DiscreteSequence
from social_bot.teacher_tasks import GoalTask
import social_bot.pygazebo as gazebo

logger = logging.getLogger(__name__)


class SimpleNavigation(gym.Env):
    """
    In this environment, the agent will receive reward 1 when it is close enough to the goal.
    If it is moving away from the goal too much or still not close to the goal after max_steps,
    it will get reward -1.
    """

    def __init__(self, with_language=True, port=None):
        if port is None:
            port = 0
        gazebo.initialize(port=port)
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(),
                         "pioneer2dx_camera.world"))
        self._agent = self._world.get_agent()
        assert self._agent is not None
        logger.info("joint names: %s" % self._agent.get_joint_names())
        self._joint_names = self._agent.get_joint_names()
        self._teacher = teacher.Teacher(False)
        task_group = teacher.TaskGroup()
        task_group.add_task(GoalTask())
        self._teacher.add_task_group(task_group)
        self._with_language = with_language

        # get observation dimension
        image = self._agent.get_camera_observation("camera")
        image = np.array(image, copy=False)
        if with_language:
            self._observation_space = gym.spaces.Dict(
                image=gym.spaces.Box(
                    low=0, high=1, shape=image.shape, dtype=np.uint8),
                sentence=DiscreteSequence(256, 20))

            self._action_space = gym.spaces.Dict(
                control=gym.spaces.Box(
                    low=-0.2,
                    high=0.2,
                    shape=[len(self._joint_names)],
                    dtype=np.float32),
                sentence=DiscreteSequence(256, 20))
        else:
            self._observation_space = gym.spaces.Box(
                low=0, high=1, shape=image.shape, dtype=np.uint8)
            self._action_space = gym.spaces.Box(
                low=-0.2,
                high=0.2,
                shape=[len(self._joint_names)],
                dtype=np.float32)

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
                    action['control'] is a vector whose dimention is
                    len(_joint_names). action['sentence'] is a string.
                    If not with_language, it is an int for the action id.
        Returns:
            If with_language, it is a dictionary with key 'obs' and 'sentence'
            If not with_language, it is a numpy.array for observation
        """
        if self._with_language:
            sentence = action.get('sentence', None)
            controls = action['control']
        else:
            sentence = ''
            controls = action
        controls = dict(zip(self._joint_names, controls))
        teacher_action = self._teacher.teach(sentence)
        self._agent.take_action(controls)
        self._world.step(100)
        image = self._agent.get_camera_observation("camera")
        image = np.array(image, copy=False)
        if self._with_language:
            obs = OrderedDict(image=image, sentence=teacher_action.sentence)
        else:
            obs = image
        return (obs, teacher_action.reward, teacher_action.done, {})

    def reset(self):
        self._teacher.reset(self._agent, self._world)
        teacher_action = self._teacher.teach("")
        image = self._agent.get_camera_observation("camera")
        image = np.array(image, copy=False)
        if self._with_language:
            obs = OrderedDict(image=image, sentence=teacher_action.sentence)
        else:
            obs = image
        return obs


class SimpleNavigationNoLanguage(SimpleNavigation):
    def __init__(self, port=None):
        super(SimpleNavigationNoLanguage, self).__init__(
            with_language=False, port=port)


class SimpleNavigationNoLanguageDiscreteAction(SimpleNavigationNoLanguage):
    def __init__(self, port=None):
        super(SimpleNavigationNoLanguageDiscreteAction,
              self).__init__(port=port)
        self._action_space = gym.spaces.Discrete(25)

    def step(self, action):
        control = [0.05 * (action // 5) - 0.1, 0.05 * (action % 5) - 0.1, 0.]
        return super(SimpleNavigationNoLanguageDiscreteAction,
                     self).step(control)


def main():
    """
    Simple testing of this enviroenment.
    """
    env = SimpleNavigation()
    for _ in range(10000000):
        obs = env.reset()
        control = [random.random() * 0.2, random.random() * 0.2, 0]
        while True:
            obs, reward, done, info = env.step(
                dict(control=control, sentence="hello"))
            if done:
                logger.info("reward: " + str(reward) + "sent: " +
                            str(obs["sentence"]))
                break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
