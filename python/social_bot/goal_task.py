# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
"""
A simple teacher task to find a goal.
"""
import logging
import math
import numpy as np
import os
import random

import social_bot
from social_bot import teacher
from social_bot.teacher import TeacherAction
import social_bot.pygazebo as gazebo

logger = logging.getLogger(__name__)


class GoalTask(teacher.Task):
    """
    For this task, the agent will receive reward 1 when it is close enough to the goal.
    If it is moving away from the goal too much or still not close to the goal after max_steps,
    it will get reward -1.
    """

    def __init__(self,
                 max_steps=500,
                 goal_name="goal",
                 success_distance_thresh=0.5,
                 fail_distance_thresh=0.5):
        """
        Args:
            max_steps (int): episode will end if not reaching gaol in so many steps
            goal_name (string): name of the goal in the world
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            fail_distance_thresh (float): if the agent moves away from the goal more than this distance,
                it's considered a failure and is givne reward -1
        """
        self._goal_name = goal_name
        self._success_distance_thresh = success_distance_thresh
        self._fail_distance_thresh = fail_distance_thresh
        self._max_steps = max_steps

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
                    # within 45 degrees of the agent direction
                    logger.debug("loc: " + str(loc) + " goal: " +
                                 str(goal_loc) + "dist: " + str(dist))
                    agent_sentence = yield TeacherAction(
                        reward=1.0, sentence="Well done!", done=False)
                    steps_since_last_reward = 0
                    self._move_goal(goal, loc)
                else:
                    agent_sentence = yield TeacherAction()
            elif dist > self._initial_dist + self._fail_distance_thresh:
                logger.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                             "dist: " + str(dist))
                yield TeacherAction(reward=-1.0, sentence="Failed", done=True)
            else:
                agent_sentence = yield TeacherAction()
        logger.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                     "dist: " + str(dist))
        yield TeacherAction(reward=-1.0, sentence="Failed", done=True)

    def _move_goal(self, goal, agent_loc):
        while True:
            loc = (random.random() * 2 - 1, random.random() * 2 - 1, 0)
            self._initial_dist = np.linalg.norm(loc - agent_loc)
            if self._initial_dist > 0.5:
                break
        goal.set_pose((loc, (0, 0, 0)))
