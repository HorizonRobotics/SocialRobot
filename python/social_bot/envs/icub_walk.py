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
A simple enviroment to train icub robot standing and walking
"""
import gym
import os
import logging
import numpy as np
import random

from gym import spaces
import social_bot
from social_bot.envs.gazebo_base import GazeboEnvBase
import social_bot.pygazebo as gazebo

logger = logging.getLogger(__name__)


class ICubWalk(GazeboEnvBase):
    """
    The goal of this task is to make the agent icub standing and walking.
    All the joints are controllable by pid or force.
    Observation is a single ndarray with internal joint states, and agent pose.
    Reward shaping:
        reward = not_fall_bonus + walk_distance - ctrl_cost
    This task is not solved yet. But examples/tain_icub_sac.py can help the agent
        stand much longer.
    """

    def __init__(self,
                 max_steps=120,
                 use_pid=False,
                 port=None):
        """
        Args:
            max_steps (int): episode will end when the agent exceeds the number of steps.
            use_pid (bool): use pid or direct force to control
            port: Gazebo port, need to specify when run multiple environment in parallel
        """
        super(ICubWalk, self).__init__(port=port)
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(), "icub.world"))
        self._agent = self._world.get_agent('icub')
        # to avoid different parallel simulation has the same randomness
        random.seed(port)

        self._all_joints = self._agent.get_joint_names()
        #logger.info("joint names: %s" % self._all_joints)
        self._world.info()

        self._max_steps = max_steps
        self._agent_control_range = 100.0
        self._agent_joints = [
            'icub::iCub::l_hip_pitch',
            'icub::iCub::l_hip_roll', 
            'icub::iCub::l_hip_yaw',
            'icub::iCub::l_knee',
            'icub::iCub::l_ankle_pitch',
            'icub::iCub::l_ankle_roll',
            'icub::iCub::r_hip_pitch',
            'icub::iCub::r_hip_roll',
            'icub::iCub::r_hip_yaw',
            'icub::iCub::r_knee',
            'icub::iCub::r_ankle_pitch',
            'icub::iCub::r_ankle_roll',
            'icub::iCub::torso_yaw',
            'icub::iCub::l_shoulder_pitch',
            'icub::iCub::l_shoulder_roll',
            'icub::iCub::l_shoulder_yaw',
            'icub::iCub::l_elbow',
            'icub::iCub::neck_pitch',
            'icub::iCub::neck_roll',
            'icub::iCub::r_shoulder_pitch',
            'icub::iCub::r_shoulder_roll',
            'icub::iCub::r_shoulder_yaw',
            'icub::iCub::r_elbow',
        ]
        logger.info("joints to control: %s" % self._agent_joints)
        if use_pid:
            for _joint in self._agent_joints:
                self._agent.set_pid_controller(_joint, 'velocity', p=60.0, d=5.0, max_force=100.0)

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=[len(self._agent_joints)],
            dtype=np.float32)
        
        self.reset()
        obs = self._get_observation()
        self.observation_space = gym.spaces.Box(
            low=-100, high=100, shape=obs.shape, dtype=np.float32)

    def reset(self):
        self._world.reset()
        self._world.step(20)
        self._steps_in_this_episode = 0
        self._cum_reward = 0.0
        # Give an intilal random pose offset by take an random action
        actions = np.array(np.random.randn(self.action_space.shape[0]))
        controls = dict(zip(self._agent_joints, self._agent_control_range * actions))
        self._agent.take_action(controls)
        self._world.step(20)
        obs = self._get_observation()
        return obs

    def _get_observation(self):
        agent_pose = np.array(self._agent.get_pose()).flatten()
        agent_vel = np.array(self._agent.get_velocities()).flatten()
        torso_pose = np.array(
            self._agent.get_link_pose('icub::iCub::chest')).flatten()
        joint_pos = []
        joint_vel = []
        for joint_id in range(len(self._agent_joints)):
            joint_name = self._agent_joints[joint_id]
            joint_state = self._agent.get_joint_state(joint_name)
            joint_pos.append(joint_state.get_positions())
            joint_vel.append(joint_state.get_velocities())
        joint_pos = np.array(joint_pos).flatten()
        joint_vel = np.array(joint_vel).flatten()
        obs = np.concatenate(
            (agent_pose, agent_vel, torso_pose, joint_pos, joint_vel), axis=0)
        obs = np.array(obs).reshape(-1)
        return obs

    def step(self, action):
        """
        Args:
            action (float):  action is a vector whose dimention is
                    len(_joint_names).
        Returns:
            A numpy.array for observation
        """
        controls = action * self._agent_control_range
        controls = dict(zip(self._agent_joints, controls))
        self._agent.take_action(controls)
        self._world.step(20)
        obs = self._get_observation()
        torso_pose = np.array(self._agent.get_link_pose('icub::iCub::chest')).flatten()
        ctrl_cost = np.sum(np.square(action))/action.shape[0]
        walk_distance = torso_pose[0]
        reward = 1.0 + 1e-1 * walk_distance - ctrl_cost
        self._cum_reward += reward
        self._steps_in_this_episode += 1
        fail = torso_pose[2] < 0.58
        done = self._steps_in_this_episode > self._max_steps or fail
        if done:
            logger.debug("episode ends at cum reward:" + str(self._cum_reward))
        return obs, reward, done, {}


class ICubWalkPID(ICubWalk):
    def __init__(self, max_steps=120, port=None):
        super(ICubWalkPID, self).__init__(
            use_pid=True, max_steps=max_steps, port=port)


def main():
    env = ICubWalkPID(max_steps=120)
    env.render()
    while True:
        actions = np.array(np.random.randn(env.action_space.shape[0]))
        obs, _, done, _ = env.step(actions)
        if done:
            env.reset()

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
