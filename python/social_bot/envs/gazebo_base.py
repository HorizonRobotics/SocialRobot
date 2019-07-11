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

import os
import random

import gym
import gin
import numpy as np

import social_bot.pygazebo as gazebo


@gin.configurable
class GazeboEnvBase(gym.Env):
    def __init__(self, port=None, quiet=False):
        if port is None:
            port = 0
        self._port = port
        # to avoid different parallel simulation has the same randomness
        random.seed(port)
        self._rendering_process = None
        gazebo.initialize(port=port, quiet=quiet)

    def render(self, mode='human'):
        """Render the environment.

        Args:
            mode (str): Currently only 'human' is supported.
        """
        if mode == 'human':
            if self._rendering_process is None:
                from subprocess import Popen
                if self._port != 0:
                    os.environ[
                        'GAZEBO_MASTER_URI'] = "localhost:%s" % self._port
                self._rendering_process = Popen(['gzclient'])
            return
        raise NotImplementedError(
            "rendering mode 'rgb_array' is not implemented.")

    def _get_internal_states(self, agent, agent_joints):
        joint_pos = []
        joint_vel = []
        for joint_id in range(len(agent_joints)):
            joint_name = agent_joints[joint_id]
            joint_state = agent.get_joint_state(joint_name)
            joint_pos.append(joint_state.get_positions())
            joint_vel.append(joint_state.get_velocities())
        joint_pos = np.array(joint_pos).flatten()
        joint_vel = np.array(joint_vel).flatten()
        # pos of continous joint could be huge, wrap the range to [-pi, pi)
        joint_pos = (joint_pos + np.pi) % (2 * np.pi) - np.pi
        internal_states = np.concatenate((joint_pos, joint_vel), axis=0)
        return internal_states

    def __del__(self):
        if self._rendering_process is not None:
            self._rendering_process.terminate()
