# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
import gym
import os
import logging
import numpy as np
import random
import math
import PIL

from gym import spaces
import social_bot
from social_bot import teacher
from social_bot.teacher import TeacherAction
import social_bot.pygazebo as gazebo
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)

class GroceryGround(gym.Env):
    """
    The goal of this task is to train the agent to stack the groceries to the wall.

    Joints of caster and arms in the PR2 are controllable by force,
    the observations are the states of the world, including every model's position
    and rotation.

    In the version:
    * agent will receive the total contacts of the wall as the reward

    """
    def __init__(self,
                 max_steps=200,
                 port=None):
        """
        Args:
            max_steps (int): episode will end when the agent exceeds the number of steps.
            port: Gazebo port, need to specify when run multiple environment in parallel
        """

        if port is None:
            port = 0
        gazebo.initialize(port=port)
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(), "grocery_ground.world"))
        self._world.info()

        self._agent = self._world.get_agent()
        self._pr2_joints = list(
            filter(lambda s: s.find('screw_joint') == -1 and
                             s.find('mount_joint') == -1 and
                             s.find('root_joint')  == -1 and
                             s.find('flex_joint')  == -1 and
                             s.find('lift_joint')  == -1 and
                             s.find('pan_joint')   == -1 and
                             s.find('tip_joint')   == -1,   self._agent.get_joint_names()))
        logger.info("joints to control: %s" % self._pr2_joints)
        self._objects_names = ['coke_can', 'coke_can_0', 'coke_can_1', 'coke_can_2', 
                                'coke_can_3', 'unit_sphere','unit_sphere_0', 'cube_20k', 
                                'car_wheel', 'first_2015_trash_can', 'plastic_cup', 
                                'plastic_cup_0', 'plastic_cup_1','plastic_cup_2',
                                'plastic_cup_3']
        self._goal_name = 'grey_wall'

        self._max_steps = max_steps
        obs = self.reset()
        self.observation_space = gym.spaces.Box(
            low=-50, high=50, shape=obs.shape, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-10, high=10, shape=[len(self._pr2_joints)], dtype=np.float32)

    def reset(self):
        self._world.reset()
        self._world.step(10)
        self._collision_cnt = 0
        self._cum_reward = 0.0
        self._steps_in_this_episode = 0
        self._goal = self._world.get_agent(self._goal_name)
        return self._get_observation()

    def _get_observation(self):
        objects_poses = []
        objects_poses.append(self._goal.get_pose())
        objects_poses.append(self._agent.get_pose())
        for object_name in self._objects_names:
            objects_poses.append(self._world.get_agent(object_name).get_pose())
        obs = np.array(objects_poses).reshape(-1)
        return obs

    def _compute_reward(self):
        # get collision_cnt of the goal
        collision_cnt = self._goal.get_collision_count("grey_wall_contact")
        if collision_cnt > 1:
            logger.debug("wall_collide_cnt:" + str(collision_cnt))
        # a simple rewad shaping
        goal_loc, _ = self._goal.get_pose()
        agent_loc, _ = self._agent.get_pose()
        dist = np.linalg.norm(np.array(goal_loc) - np.array(agent_loc))
        # compute reward
        reward = collision_cnt - dist
        return reward

    def step(self, action): 
        controls = dict(zip(self._pr2_joints, action))
        self._agent.take_action(controls)
        self._world.step(10)
        obs = self._get_observation()
        self._steps_in_this_episode += 1
        done = self._steps_in_this_episode >= self._max_steps
        reward = self._compute_reward()
        self._cum_reward += reward
        if done:
            logger.debug("episode ends at cum reward:" + str(self._cum_reward))
        return obs, reward, done, {}

    def run(self):
        while True:
            actions = 50*np.random.randn(len(self._pr2_joints))
            obs, _, done, _ = self.step(actions)
            if done:
                self.reset()


def main():
    env = GroceryGround()
    env.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()      