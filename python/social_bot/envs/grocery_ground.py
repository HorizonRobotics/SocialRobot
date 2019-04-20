# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
"""
A simple enviroment for an agent play on a groceryground
"""
import os
import logging
import numpy as np
import math
import random
import PIL

import gym
from gym import spaces
from collections import OrderedDict

import social_bot
from social_bot import teacher
from social_bot.teacher import TeacherAction
from social_bot.envs import simple_navigation
from social_bot.envs.simple_navigation import GoalTask
from social_bot.envs.simple_navigation import DiscreteSequence
import social_bot.pygazebo as gazebo

logger = logging.getLogger(__name__)


class GroceryGroundGoalTask(GoalTask):
    """
    For this task, the agent will receive reward 1 when it is close enough to the goal.
    If it is moving away from the goal too much or still not close to the goal after max_steps,
    it will get reward -1.
    """

    def __init__(self,
                 max_steps=500,
                 goal_name="bookshelf",
                 success_distance_thresh=0.5,
                 fail_distance_thresh=3.0,
                 goal_random_range=20.0):
        """
        Args:
            max_steps (int): episode will end if not reaching gaol in so many steps
            goal_name (string): name of the goal in the world
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            fail_distance_thresh (float): if the agent moves away from the goal more than this distance,
                it's considered a failure and is givne reward -1
            goal_random_range (float): the goal's random position range
            
        """
        self._goal_random_range = goal_random_range
        super(GroceryGroundGoalTask, self).__init__(
            max_steps=max_steps,
            goal_name=goal_name,
            success_distance_thresh=success_distance_thresh,
            fail_distance_thresh=fail_distance_thresh)

    def _move_goal(self, goal, agent_loc):
        while True:
            loc = (random.random() * self._goal_random_range -
                   self._goal_random_range / 2,
                   random.random() * self._goal_random_range -
                   self._goal_random_range / 2, 0)
            self._initial_dist = np.linalg.norm(loc - agent_loc)
            if self._initial_dist > 0.5:
                break
        goal.set_pose((loc, (0, 0, 0)))


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


class GroceryGround(gym.Env):
    """
    The goal of this task is to train the agent to navigate to the objects given its
    name.

    Joints of caster and arms in the PR2 are controllable by force,
    the observations are the states of the world, including every model's position
    and rotation.

    In the version:
    * agent will receive the total contacts of the wall as the reward

    """

    def __init__(self,
                 max_steps=500,
                 with_language=False,
                 use_image_obs=False,
                 port=None):
        """
        Args:
            max_steps (int): episode will end when the agent exceeds the number of steps
            with_language (bool): the observation will be a dict with an extra sentence
            use_image_obs (bool): use image as observation, or use pose of the objects
            port: Gazebo port, need to specify when run multiple environment in parallel
        """
        if port is None:
            port = 0
        gazebo.initialize(port=port)
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(), "grocery_ground.world"))
        # self._world.info()

        self._agent = self._world.get_agent()
        self._pr2_joints = list(
            filter(
                lambda s: s.find('fl_caster_r_wheel_joint') != -1 or s.find('fr_caster_l_wheel_joint') != -1,
                self._agent.get_joint_names()))
        logger.info("joints to control: %s" % self._pr2_joints)
        self._objects_names = [
            'coke_can', 'coke_can_0', 'coke_can_1', 'coke_can_2', 'coke_can_3',
            'unit_sphere', 'unit_sphere_0', 'table', 'grey_wall', 'cube_20k',
            'caffe_table', 'bookshelf', 'car_wheel', 'car_wheel_0',
            'trash_can', 'plastic_cup', 'plastic_cup_0', 'plastic_cup_1',
            'plastic_cup_2', 'plastic_cup_3'
        ]
        self._goal_name = 'bookshelf'
        self._object_name = 'trash_can'

        self._teacher = teacher.Teacher(False)
        task_group = teacher.TaskGroup()
        task_group.add_task(GroceryGroundGoalTask())
        self._teacher.add_task_group(task_group)

        self._max_steps = max_steps
        self._with_language = with_language
        self._use_image_obs = use_image_obs
        self._resized_image_size = (84, 84)

        obs = self.reset()
        if self._use_image_obs:
            obs_data_space = gym.spaces.Box(
                low=0, high=1, shape=obs.shape, dtype=np.uint8)
        else:
            obs_data_space = gym.spaces.Box(
                low=-50, high=50, shape=obs.shape, dtype=np.float32)
        control_space = gym.spaces.Box(
            low=-20, high=20, shape=[len(self._pr2_joints)], dtype=np.float32)

        if self._with_language:
            self.observation_space = gym.spaces.Dict(
                data=obs_data_space, sentence=DiscreteSequence(128, 24))
            self._action_space = gym.spaces.Dict(
                control=control_space, sentence=DiscreteSequence(128, 24))
        else:
            self.observation_space = obs_data_space
            self.action_space = control_space

    def reset(self):
        """
        Reset the environment
        Args: None
        Returns: Observaion of the first step
        """
        self._world.reset()
        self._world.step(5)
        self._collision_cnt = 0
        self._cum_reward = 0.0
        self._steps_in_this_episode = 0
        self._prev_dist = self._get_object_distance()
        self._goal = self._world.get_agent(self._goal_name)
        self._teacher.reset(self._agent, self._world)
        teacher_action = self._teacher.teach("")
        if self._with_language:
            obs_data = self._get_observation()
            obs = OrderedDict(data=obs_data, sentence=teacher_action)
        else:
            obs = self._get_observation()
        return obs

    def _get_observation(self):
        if self._use_image_obs:
            obs_data = np.array(
                self._agent.get_camera_observation(
                    "default::pr2::pr2::head_tilt_link::head_mount_prosilica_link_sensor"
                ),
                copy=False)
            obs_data = PIL.Image.fromarray(rgb2gray(obs_data)).resize(
                self._resized_image_size, PIL.Image.ANTIALIAS)

            obs_data = np.reshape(
                np.array(obs_data),
                [self._resized_image_size[0], self._resized_image_size[1], 1])
        else:
            objects_poses = []
            objects_poses.append(self._goal.get_pose())
            objects_poses.append(self._agent.get_pose())
            obs_data = np.array(objects_poses).reshape(-1)
        return obs_data

    def _get_object_distance(self):
        object_loc, _ = self._world.get_agent(self._object_name).get_pose()
        agent_loc, _ = self._agent.get_pose()
        dist = np.linalg.norm(np.array(object_loc) - np.array(agent_loc))
        return dist

    def _compute_reward(self):
        dist = self._get_object_distance()
        diff = self._prev_dist - dist
        self._prev_dist = dist
        reward = diff
        return reward

    def step(self, action):
        """
        Args:
            action (dict|int): If with_language, action is a dictionary with key "control" and "sentence".
                    action['control'] is a vector whose dimention is
                    len(_joint_names). action['sentence'] is a string.
                    If not with_language, it is an int for the action id.
        Returns:
            If with_language, it is a dictionary with key 'data' and 'sentence'
            If not with_language, it is a numpy.array or image for observation
        """
        if self._with_language:
            sentence = action.get('sentence', None)
            controls = action['control']
        else:
            sentence = ''
            controls = action
        controls = dict(zip(self._pr2_joints, controls))
        teacher_action = self._teacher.teach(sentence)
        self._agent.take_action(controls)
        self._world.step(5)
        if self._with_language:
            obs_data = self._get_observation()
            obs = OrderedDict(data=obs_data, sentence=teacher_action.sentence)
        else:
            obs = self._get_observation()
        reward = self._compute_reward()
        self._steps_in_this_episode += 1
        done = self._steps_in_this_episode >= self._max_steps
        self._cum_reward += reward
        if done:
            logger.debug("episode ends at cum reward:" + str(self._cum_reward))
        if self._with_language:
            return obs, teacher_action.reward, teacher_action.done, {}
        else:
            return obs, reward, done, {}


def main():
    """
    Simple testing of this enviroenment.
    """
    env = GroceryGround(port=11345)
    while True:
        actions = 20 * np.random.randn(env.action_space.shape[0])
        obs, _, done, _ = env.step(actions)
        if done:
            env.reset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
