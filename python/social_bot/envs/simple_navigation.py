# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
"""
A simple enviroment for navigation.
"""
from collections import OrderedDict
import gym
import gym.spaces
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
        Arguments
            max_steps(int): episode will end if not reaching gaol in so many steps
            goal_name(string): name of the goal in the world
            success_distance_thresh(float): the goal is reached if it's within this distance to the agent
            fail_distance_thresh(float): if the agent moves away from the goal more than this distance,
                it's considered a failure and is givne reward -1
        """
        self._goal_name = goal_name
        self._success_distance_thresh = success_distance_thresh
        self._fail_distance_thresh = fail_distance_thresh
        self._max_steps = max_steps

    def run(self, agent, world):
        """
        Start a teaching episode for this task.
        Arguments
            agent(pygazebo.Agent): the learning agent 
            world(pygazebo.World): the simulation world
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


class DiscreteSequence(gym.Space):
    """
    gym.Space object for language sequence
    """

    def __init__(self, vocab_size, max_length):
        """
        Arguments
            vocab_size(int): number of different tokens
            max_length(int): maximal length of the sequence
        """
        super()
        self._vocab_size = vocab_size
        self._max_length = max_length
        self.dtype = np.int32
        self.shape = (max_length)


class SimpleNavigation(gym.Env):
    """
    In this environment, the agent will receive reward 1 when it is close enough to the goal.
    If it is moving away from the goal too much or still not close to the goal after max_steps,
    it will get reward -1.
    """

    def __init__(self, with_language=True):
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(),
                         "pioneer2dx_camera.world"))
        self._agent = self._world.get_agent()
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
            self._observation_space = image = gym.spaces.Box(
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
        Arguments
            action(dict|int): If with_language, action is a dictionary with key "control" and "sentence".
                    action['control'] is a vector whose dimention is
                    len(_joint_names). action['sentence'] is a string.
                    If not with_language, it is an int for the action id.
        Returns
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
    def __init__(self):
        super(SimpleNavigationNoLanguage, self).__init__(with_language=False)


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
    gazebo.initialize()
    main()
