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
A variety of teacher tasks.
"""

import math
import numpy as np
import os
import random

import social_bot
from social_bot import teacher
from social_bot.teacher import TeacherAction
import social_bot.pygazebo as gazebo

from absl import logging


class GoalTask(teacher.Task):
    """
    A simple teacher task to find a goal.
    For this task, the agent will receive reward 1 when it is close enough to the goal.
    If it is moving away from the goal too much or still not close to the goal after max_steps,
    it will get reward -1.
    """

    def __init__(self,
                 max_steps=500,
                 goal_name="goal",
                 success_distance_thresh=0.5,
                 fail_distance_thresh=0.5,
                 sparse_reward=True,
                 random_range=2.0):
        """
        Args:
            max_steps (int): episode will end if not reaching gaol in so many steps
            goal_name (string): name of the goal in the world
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            fail_distance_thresh (float): if the agent moves away from the goal more than this distance,
                it's considered a failure and is given reward -1
            sparse_reward (bool): if true, the reward is -1/0/1, otherwise the 0 case will be replaced
                with normalized distance the agent get closer to goal.
            random_range (float): the goal's random position range
        """
        super().__init__()
        self._goal_name = goal_name
        self._success_distance_thresh = success_distance_thresh
        self._fail_distance_thresh = fail_distance_thresh
        self._max_steps = max_steps
        self._sparse_reward = sparse_reward
        self._random_range = random_range
        self.task_vocab = ['hello', 'goal', 'well', 'done', 'failed', 'to']

    def run(self, agent, world):
        """
        Start a teaching episode for this task.
        Args:
            agent (pygazebo.Agent): the learning agent
            world (pygazebo.World): the simulation world
        """
        agent_sentence = yield
        agent.reset()
        goal = world.get_agent(self._goal_name)
        loc, dir = agent.get_pose()
        loc = np.array(loc)
        self._move_goal(goal, loc)
        steps_since_last_reward = 0
        while steps_since_last_reward < self._max_steps:
            steps_since_last_reward += 1
            loc, dir = agent.get_pose()
            goal_loc, _ = goal.get_pose()
            loc = np.array(loc)
            goal_loc = np.array(goal_loc)
            dist = np.linalg.norm(loc - goal_loc)
            # dir from get_pose is (roll, pitch, roll)
            dir = np.array([math.cos(dir[2]), math.sin(dir[2])])
            goal_dir = (goal_loc[0:2] - loc[0:2]) / dist
            dot = sum(dir * goal_dir)
            if dist < self._success_distance_thresh and dot > 0.707:
                # within 45 degrees of the agent direction
                logging.debug("loc: " + str(loc) + " goal: " +
                                str(goal_loc) + "dist: " + str(dist))
                agent_sentence = yield TeacherAction(
                    reward=1.0, sentence="well done", done=False)
                steps_since_last_reward = 0
                self._move_goal(goal, loc)
            elif dist > self._initial_dist + self._fail_distance_thresh:
                logging.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                              "dist: " + str(dist))
                yield TeacherAction(reward=-1.0, sentence="failed", done=True)
            else:
                reward = (self._prev_dist - dist) / self._initial_dist
                self._prev_dist = dist
                agent_sentence = yield TeacherAction(
                    reward=reward * (not self._sparse_reward),
                    sentence=self._goal_name)
        logging.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                      "dist: " + str(dist))
        yield TeacherAction(reward=-1.0, sentence="failed", done=True)

    def _move_goal(self, goal, agent_loc):
        range = self._random_range
        while True:
            loc = (random.random() * range - range / 2,
                   random.random() * range - range / 2, 0)
            self._initial_dist = np.linalg.norm(loc - agent_loc)
            if self._initial_dist > self._success_distance_thresh:
                break
        self._prev_dist = self._initial_dist
        goal.set_pose((loc, (0, 0, 0)))

    def get_goal_name(self):
        """
        Args:
            None
        Returns:
            Goal's name at this episode
        """
        return self._goal_name

    def set_goal_name(self, goal_name):
        """
        Args:
            Goal's name
        Returns:
            None
        """
        self._goal_name = goal_name

