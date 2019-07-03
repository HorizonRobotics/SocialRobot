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

import social_bot.pygazebo as gazebo


class GazeboEnvBase(gym.Env):
    def __init__(self, port=None):
        if port is None:
            port = 0
        self._port = port
        # to avoid different parallel simulation has the same randomness
        random.seed(port)
        self._rendering_process = None
        gazebo.initialize(port=port)

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

    def __del__(self):
        if self._rendering_process is not None:
            self._rendering_process.terminate()
