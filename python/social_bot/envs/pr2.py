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

def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


class Pr2Gripper(gym.Env):
    """
    The goal of this task is to train the agent to use its arm and fingers.

    All the joints in the right arm of PR2 are controllable by force,
    the observations are the image taken from head camera, the goal could be
    blocked by agent's own arm.

    The goal position will be randomly placed in front of the agent at start of each
    episode.

    In the version:
    * agent will receive +1 reward every timestep the goal lifts off the groud
    * agent will receive -0.01 reward every timestep
    * agent will receive 1.2 - finger_tip_to_goal distance at the end of episode

    currently, the agent only learns to use arm/hand to play bat to hit the goal
    off the ground.

    """
    def __init__(self,
                 goal_name='beer',
                 max_steps=100,
                 reward_shaping=True,
                 port=None):
        """
        Args:
            goal_name (string): name of the object to lift off ground
            max_steps (int): episode will end when the agent exceeds the number of steps.
            reward_shaping (boolean): whether it adds distance based reward shaping.
            port: Gazebo port, need to specify when run multiple environment in parallel
        """

        if port is None:
            port = 0
        gazebo.initialize(port=port)
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(), "pr2.world"))
        self._agent = self._world.get_agent()
        logger.info("joint names: %s" % self._agent.get_joint_names())

        self._all_joints = self._agent.get_joint_names()

        self._world.info()
        self._r_arm_joints = list(
            filter(lambda s: s.find('pr2::r_') != -1, self._all_joints))
        logger.info("joints in the right arm to control: %s" % self._r_arm_joints)

        self._goal_name = goal_name
        self._goal = self._world.get_agent(self._goal_name)
        self._max_steps = max_steps
        self._steps_in_this_episode = 0
        self._reward_shaping = reward_shaping
        self._resized_image_size = (84, 84)

        self._prev_dist = self._get_finger_tip_distance()
        self._collision_cnt = 0
        self._cum_reward = 0.0

        img = self._get_observation()
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=img.shape, dtype=np.uint8)
        self.action_space = gym.spaces.Box(
            low=-5, high=5, shape=[len(self._r_arm_joints)], dtype=np.float32)

    def reset(self):
        self._world.reset()
        # move camera to focus on ground. it is hacky
        joint_state = gazebo.JointState(1)
        joint_state.set_positions([1.4])
        joint_state.set_velocities([0.0])
        self._collision_cnt = 0
        self._agent.set_joint_state("pr2::pr2::head_tilt_joint", joint_state)
        self._world.step(200)

        self._goal = self._world.get_agent(self._goal_name)
        self._move_goal()
        self._steps_in_this_episode = 0
        self._cum_reward = 0.0
        self._prev_dist = self._get_finger_tip_distance()
        return self._get_observation()

    def _move_goal(self):
        loc = (0.6 + 0.3 * (random.random() - 1), 0.4 * (random.random() - 1),
               0)
        self._goal.set_pose((loc, (0, 0, 0)))

    def _get_observation(self):
        obs = np.array(
            self._agent.get_camera_observation(
                "default::pr2::pr2::head_tilt_link::head_mount_prosilica_link_sensor"
            ),
            copy=False)
        obs = PIL.Image.fromarray(rgb2gray(obs)).resize(
            self._resized_image_size, PIL.Image.ANTIALIAS)

        obs = np.reshape(np.array(obs),
                         [self._resized_image_size[0], self._resized_image_size[1], 1])
        return obs

    def _get_finger_tip_distance(self):
        loc, _ = self._goal.get_pose()
        goal_loc = np.array(loc)
        finger_tip_loc, _ = self._agent.get_link_pose(
            "pr2::pr2::r_gripper_l_finger_tip_link")

        dist = np.linalg.norm(np.array(finger_tip_loc) - np.array(goal_loc))
        return dist

    def step(self, action):
        controls = dict(zip(self._r_arm_joints, action))

        self._agent.take_action(controls)
        self._world.step(100)
        obs = self._get_observation()

        self._steps_in_this_episode += 1
        dist = self._get_finger_tip_distance()

        done = self._steps_in_this_episode >= self._max_steps

        reward = -0.01 if (not done) else (1.2 - dist)

        shape_reward = self._prev_dist - dist
        if self._reward_shaping:
            reward += shape_reward

        collision_cnt = self._goal.get_collision_count("beer_contact")
        if collision_cnt > 1:
            logger.debug("collide_cnt:" + str(collision_cnt))

        goal_loc, _ = self._goal.get_pose()

        goal_move_reward = (goal_loc[2] > 0.001)
        if goal_move_reward:
            logger.debug("beer lift! " + str(goal_loc))
            reward += 10

        self._cum_reward += reward
        if done:
            logger.debug("episode ends at dist: " + str(dist) +
                         " with cum reward:" + str(self._cum_reward))

        self._prev_dist = dist
        return obs, reward, done, {}

    def run(self):
        while True:
            actions = np.random.randn(len(self._r_arm_joints))
            obs, _, done, _ = self.step(actions)
            plt.imshow(obs[:, :, 0], cmap='gray')
            plt.pause(0.001)
            if done:
                self.reset()

    def render(self, mode='human'):
        return


def main():
    env = Pr2Gripper()
    env.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
