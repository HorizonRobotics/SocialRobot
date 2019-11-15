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
import time
import random
import gin
import numpy as np
from absl import logging
import social_bot
import social_bot.pygazebo as gazebo


@gin.configurable
class GazeboAgent():
    """
    Class of agent for gazebo-based SocialRobot enviroments
    """

    def __init__(self,
                 world,
                 agent_type,
                 name=None,
                 config=None,
                 with_language=False):
        """
        Args:
             world_file (str|None): world file path
             world_string (str|None): world xml string content,
             world_config (list[str]): list of str config `key=value`
                see `_modify_world_xml` for details
             sim_time_precision (float): the time precision 
        """
        self._world = world
        self.type = agent_type
        if name == None:
            self.name = agent_type
        self.name = name
        self.config = config
        self.joints = config['control_joints']
        self.with_language = with_language
        self._agent = self._world.get_agent(agent_type)
        self.get_pose = self._agent.get_pose
        self.set_pose = self._agent.set_pose
        self.get_link_pose = self._agent.get_link_pose
        self.set_link_pose = self._agent.set_link_pose
        self.get_joint_state = self._agent.get_joint_state
        self.set_joint_state = self._agent.set_joint_state
        self.set_pid_controller = self._agent.set_pid_controller
        self.get_collisions = self._agent.get_collisions
        self.get_camera_observation = self._agent.get_camera_observation
        self.get_velocities = self._agent.get_velocities

        #self.control_space
        #self.action_space

        #self.set_joints()
        #self.get_camera_observation()

    def reset(self):
        """
        Reset the agent.
        """
        self._agent.reset()

    def take_action(self, controls):
        """
        Reset the agent.
        Args:
            the actions to be taken
        """
        self._agent.take_action(controls)
