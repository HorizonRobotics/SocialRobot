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

from collections import deque
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
                 random_range=2.0,
                 use_curriculum_training=False,
                 start_range=0,
                 increase_range_by_percent=50.,
                 reward_thresh_to_increase_range=0.4,
                 percent_full_range_in_curriculum=0.1,
                 max_reward_q_length=100):
        """
        Args:
            max_steps (int): episode will end if not reaching gaol in so many steps
            goal_name (string): name of the goal in the world
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            fail_distance_thresh (float): if the agent moves away from the goal more than this distance,
                it's considered a failure and is given reward -1
            random_range (float): the goal's random position range
            use_curriculum_training (bool): when true, use curriculum in goal task training
            start_range (float): for curriculum learning, the starting random_range to set the goal
            increase_range_by_percent (float): for curriculum learning, how much to increase random range
                every time agent reached the specified amount of reward.
            reward_thresh_to_increase_range (float): for curriculum learning, how much reward to reach
                before the teacher increases random range.
            percent_full_range_in_curriculum (float): if above 0, randomly throw in x% of training examples
                where random_range is the full range instead of the easier ones in the curriculum.
            max_reward_q_length (int): how many recent rewards to consider when estimating agent accuracy.
        """
        super().__init__()
        self._goal_name = goal_name
        self._success_distance_thresh = success_distance_thresh
        self._fail_distance_thresh = fail_distance_thresh
        self._max_steps = max_steps
        self._use_curriculum_training = use_curriculum_training
        self._start_range = start_range
        self._is_full_range_in_curriculum = False
        if self.should_use_curriculum_training():
            logging.info("Setting random_range to %f", self._start_range)
            self._orig_random_range = random_range
            self._random_range = start_range
            self._max_reward_q_length = max_reward_q_length
            self._q = deque(maxlen=max_reward_q_length)
            self._reward_thresh_to_increase_range = reward_thresh_to_increase_range
            self._increase_range_by_percent = increase_range_by_percent
            self._percent_full_range_in_curriculum = percent_full_range_in_curriculum
        else:
            self._random_range = random_range
        self.task_vocab = ['hello', 'goal', 'well', 'done', 'failed', 'to']

    def should_use_curriculum_training(self):
        return (self._use_curriculum_training and
            self._start_range >= self._success_distance_thresh * 1.2)

    def _push_reward_queue(self, value):
        if (not self.should_use_curriculum_training() or
            self._is_full_range_in_curriculum):
            return
        self._q.append(value)
        if (value > 0 and
            sum(self._q) >= self._max_reward_q_length *
                self._reward_thresh_to_increase_range):
            self._random_range *= 1. + self._increase_range_by_percent
            if self._random_range > self._orig_random_range:
                self._random_range = self._orig_random_range
            logging.info("Raising random_range to %f", self._random_range)
            self._q.clear()

    def get_random_range(self):
        return self._random_range

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
            if dist < self._success_distance_thresh:
                # dir from get_pose is (roll, pitch, roll)
                dir = np.array([math.cos(dir[2]), math.sin(dir[2])])
                goal_dir = (goal_loc[0:2] - loc[0:2]) / dist
                dot = sum(dir * goal_dir)
                if dot > 0.707:
                    self._push_reward_queue(1)
                    # within 45 degrees of the agent direction
                    logging.debug("loc: " + str(loc) + " goal: " +
                                  str(goal_loc) + "dist: " + str(dist))
                    agent_sentence = yield TeacherAction(
                        reward=1.0, sentence="well done", done=False)
                    steps_since_last_reward = 0
                    self._move_goal(goal, loc)
                else:
                    agent_sentence = yield TeacherAction()
            elif dist > self._initial_dist + self._fail_distance_thresh:
                self._push_reward_queue(0)
                logging.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                              "dist: " + str(dist))
                yield TeacherAction(reward=-1.0, sentence="failed", done=True)
            else:
                agent_sentence = yield TeacherAction(sentence=self._goal_name)
        logging.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                      "dist: " + str(dist))
        self._push_reward_queue(0)
        yield TeacherAction(reward=-1.0, sentence="failed", done=True)

    def _move_goal(self, goal, agent_loc):
        if (self.should_use_curriculum_training() and
            self._percent_full_range_in_curriculum > 0 and
            random.random() < self._percent_full_range_in_curriculum):
            range = self._orig_random_range
            self._is_full_range_in_curriculum = True
        else:
            range = self._random_range
            self._is_full_range_in_curriculum = False
        while True:
            loc = (random.random() * range - range / 2,
                   random.random() * range - range / 2, 0)
            self._initial_dist = np.linalg.norm(loc - agent_loc)
            if self._initial_dist > self._success_distance_thresh:
                break
        goal.reset()
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
        logging.debug('Setting Goal to %s', goal_name)
        self._goal_name = goal_name

