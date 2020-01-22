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
A simple enviroment for iCub robot walking task
"""
import gym
import os
from absl import logging
import numpy as np
import random

from gym import spaces
import social_bot
from social_bot.envs.gazebo_base import GazeboEnvBase
import social_bot.pygazebo as gazebo


class ICubWalk(GazeboEnvBase):
    """
    The goal of this task is to make the agent learn how to walk.
    Joints are controllable with force, or with target velocity if the internal
        pid controller is used.
    Observation is a single ndarray with internal joint states, and agent pose.
    Reward shaping:
        reward = not_fall_bonus + truncked_walk_velocity - ctrl_cost
    Examples/tain_icub_sac.py solves this taks in about 400K episodes.
    """

    def __init__(self, use_pid=False, obs_stack=True, sub_steps=50, port=None):
        """
        Args:
            use_pid (bool): use pid or direct force to control
            port: Gazebo port, need to specify when run multiple environment in parallel
            obs_stack (bool): Use staked multi step observation if True
            sub_steps (int): take how many simulator substeps during one gym step
        """
        super(ICubWalk, self).__init__(world_file='icub.world', port=port)
        self._sub_steps = sub_steps
        self._obs_stack = obs_stack
        self._agent = self._world.get_agent('icub')
        logging.debug(self._world.info())

        self._agent_joints = [
            'icub::icub::l_leg::l_hip_pitch',
            'icub::icub::l_leg::l_hip_roll',
            'icub::icub::l_leg::l_hip_yaw',
            'icub::icub::l_leg::l_knee',
            'icub::icub::l_leg::l_ankle_pitch',
            'icub::icub::l_leg::l_ankle_roll',
            'icub::icub::r_leg::r_hip_pitch',
            'icub::icub::r_leg::r_hip_roll',
            'icub::icub::r_leg::r_hip_yaw',
            'icub::icub::r_leg::r_knee',
            'icub::icub::r_leg::r_ankle_pitch',
            'icub::icub::r_leg::r_ankle_roll',
            'icub::icub::torso_yaw',
            'icub::icub::torso_roll',
            'icub::icub::torso_pitch',
            'icub::icub::l_shoulder_pitch',
            'icub::icub::l_shoulder_roll',
            'icub::icub::l_shoulder_yaw',
            'icub::icub::l_elbow',
            'icub::icub::neck_pitch',
            'icub::icub::neck_roll',
            'icub::icub::r_shoulder_pitch',
            'icub::icub::r_shoulder_roll',
            'icub::icub::r_shoulder_yaw',
            'icub::icub::r_elbow',
        ]

        joint_states = list(
            map(lambda s: self._agent.get_joint_state(s), self._agent_joints))
        self._joints_limits = list(
            map(lambda s: s.get_effort_limits()[0], joint_states))
        if use_pid:
            for joint_index in range(len(self._agent_joints)):
                self._agent.set_pid_controller(
                    self._agent_joints[joint_index],
                    'velocity',
                    p=0.02,
                    d=0.00001,
                    max_force=self._joints_limits[joint_index])
            self._agent_control_range = 3.0 * np.pi
        else:
            self._agent_control_range = np.array(self._joints_limits)

        logging.info("joints to control: %s" % self._agent_joints)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=[len(self._agent_joints)],
            dtype=np.float32)

        self.reset()
        obs = self._get_observation()
        if self._obs_stack:
            obs = np.concatenate((obs, obs), axis=0)
        self.observation_space = gym.spaces.Box(
            low=-100, high=100, shape=obs.shape, dtype=np.float32)

    def reset(self):
        """
        Args:
            None
        Returns:
            Observaion of the first step
        """
        self._world.reset()
        self._steps_in_this_episode = 0
        self._cum_reward = 0.0
        # Give an intilal random pose offset by take an random action
        actions = np.array(np.random.randn(self.action_space.shape[0]))
        controls = dict(
            zip(self._agent_joints, self._agent_control_range * actions))
        self._world.step(10)
        self._agent.take_action(controls)
        self._world.step(10)
        obs = self._get_observation()
        self._obs_prev = obs
        if self._obs_stack:
            obs = np.concatenate((obs, obs), axis=0)
        return obs

    def _check_contacts_to_ground(self, contacts_sensor):
        contacts = self._agent.get_collisions(contacts_sensor)
        for collision in contacts:
            if collision[1] == 'ground_plane::link::collision':
                return True
        return False

    def _get_observation(self):
        agent_pose = np.array(
            self._agent.get_link_pose('icub::icub::root_link')).flatten()
        chest_pose = np.array(
            self._agent.get_link_pose('icub::icub::chest')).flatten()
        l_foot_pose = np.array(
            self._agent.get_link_pose('icub::icub::l_leg::l_foot')).flatten()
        r_foot_pose = np.array(
            self._agent.get_link_pose('icub::icub::r_leg::r_foot')).flatten()
        average_pos = np.sum([
            agent_pose[0:3], chest_pose[0:3], l_foot_pose[0:3],
            r_foot_pose[0:3]
        ],
                             axis=0) / 4.0
        agent_poses = np.concatenate((average_pos, agent_pose, chest_pose,
                                      l_foot_pose, r_foot_pose))
        agent_vel = np.array(self._agent.get_velocities()).flatten()
        joint_pos = []
        joint_vel = []
        for joint_id in range(len(self._agent_joints)):
            joint_name = self._agent_joints[joint_id]
            joint_state = self._agent.get_joint_state(joint_name)
            joint_pos.append(joint_state.get_positions())
            joint_vel.append(joint_state.get_velocities())
        joint_pos = np.array(joint_pos).flatten()
        self.joint_pos = joint_pos
        joint_vel = np.array(joint_vel).flatten()
        foot_contacts = np.array([
            self._check_contacts_to_ground("l_foot_contact_sensor"),
            self._check_contacts_to_ground("r_foot_contact_sensor")
        ]).astype(np.float32)
        obs = np.concatenate(
            (agent_poses, agent_vel, joint_pos, joint_vel, foot_contacts),
            axis=0)
        return obs

    def step(self, action):
        """
        Args:
            action (np.array):  action is a vector whose dimention is
                    len(_joint_names).
        Returns:
            A numpy.array for observation
        """
        controls = action * self._agent_control_range
        controls = dict(zip(self._agent_joints, controls))
        self._agent.take_action(controls)
        self._world.step(self._sub_steps)
        obs = self._get_observation()
        walk_vel = (obs[0] - self._obs_prev[0]) * (1000.0 / self._sub_steps)
        stacked_obs = np.concatenate((obs, self._obs_prev), axis=0)
        self._obs_prev = obs
        torso_pose = np.array(
            self._agent.get_link_pose('icub::icub::chest')).flatten()
        action_cost = np.sum(np.square(action)) / action.shape[0]
        movement_cost = np.sum(np.abs(
            self.joint_pos)) / self.joint_pos.shape[0]
        ctrl_cost = action_cost + 0.5 * movement_cost
        reward = 1.0 + 6.0 * min(walk_vel, 1.0) - 1.0 * ctrl_cost
        self._cum_reward += reward
        self._steps_in_this_episode += 1
        done = torso_pose[2] < 0.58
        if done:
            logging.debug("episode ends at cum reward:" +
                          str(self._cum_reward) + ", step:" +
                          str(self._steps_in_this_episode))
        if self._obs_stack:
            obs = stacked_obs
        return obs, reward, done, {}


class ICubWalkPID(ICubWalk):
    def __init__(self, sub_steps=50, port=None):
        super(ICubWalkPID, self).__init__(
            use_pid=True, sub_steps=sub_steps, port=port)


def main():
    """
    Simple testing of this environment.
    """
    import matplotlib.pyplot as plt
    env = ICubWalkPID(sub_steps=50)
    while True:
        actions = np.array(env.action_space.sample())
        obs, _, done, _ = env.step(actions)
        plt.imshow(env.render('rgb_array'))
        plt.pause(0.00001)
        if done or env._steps_in_this_episode > 100:
            env.reset()


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    main()
