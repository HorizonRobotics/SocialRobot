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
import json
import gin
import numpy as np
import PIL.Image
import gym
from absl import logging
import social_bot
import social_bot.pygazebo as gazebo


@gin.configurable
class GazeboAgent():
    """
    Class for the agent of gazebo-based SocialRobot enviroments
    """

    def __init__(self,
                 world,
                 agent_type,
                 name=None,
                 config=None,
                 use_image_observation=True,
                 resized_image_size=None,
                 image_with_internal_states=False,
                 with_language=False):
        """
        Args:
            world (pygazebo.World): the world
            agent_type (str): the agent_type, supporting pr2_noplugin,
                pioneer2dx_noplugin, turtlebot, youbot_noplugin and icub_with_hands for now
                note that 'agent_type' should be exactly the same string as the model's
                name at the beginning of model's sdf file
            config (dict): the configuarations for the agent
                see `agent_cfg.jason` for details
            use_image_observation (bool): Use image or not
            resized_image_size (None|tuple): If None, use the original image size
                from the camera. Otherwise, the original image will be resized
                to (width, height)
            image_with_internal_states (bool): If true, the agent's self internal states
                i.e., joint position and velocities would be available together with image.
                Only affect if use_image_observation is true
            with_language (bool): The observation will be a dict with an extra sentence
        """
        self._world = world
        self.type = agent_type
        if name == None:
            name = agent_type
        if config == None:
            # Load agent config file
            with open(
                    os.path.join(social_bot.get_model_dir(), "agent_cfg.json"),
                    'r') as cfg_file:
                agent_cfgs = json.load(cfg_file)
            config = agent_cfgs[agent_type]
        self.name = name
        self.config = config
        self._use_image_observation = use_image_observation
        self._resized_image_size = resized_image_size
        self._image_with_internal_states = image_with_internal_states
        self._with_language = with_language
        self._sentence_space = None

        self._agent = self._world.get_agent(agent_type)

        # Set the funtions from pygazebo.agent to Agent
        self.get_pose = self._agent.get_pose
        self.set_pose = self._agent.set_pose
        self.get_link_pose = self._agent.get_link_pose
        self.set_link_pose = self._agent.set_link_pose
        self.get_joint_state = self._agent.get_joint_state
        self.set_joint_state = self._agent.set_joint_state
        self.set_pid_controller = self._agent.set_pid_controller
        self.get_collisions = self._agent.get_collisions
        self.get_velocities = self._agent.get_velocities

        # Setup joints and sensors
        self.joints = config['control_joints']
        self._camera = config['camera_sensor']
        self.action_range = self.setup_joints(self._agent, self.joints, config)
        logging.debug("joints to control: %s" % self.joints)

    def reset(self):
        """
        Reset the agent.
        """
        self._agent.reset()

    def take_action(self, action):
        """
        Take actions.
        Args:
            the actions to be taken
        """
        controls = np.clip(action, -1.0, 1.0) * self.action_range
        controls_dict = dict(zip(self.joints, controls))
        self._agent.take_action(controls_dict)

    def get_camera_observation(self):
        """
        Get the camera image
        Returns:
            a numpy.array of the image
        """
        image = np.array(self._agent.get_camera_observation(self._camera), copy=False)
        if self._resized_image_size:
            image = PIL.Image.fromarray(image).resize(self._resized_image_size,
                                                      PIL.Image.ANTIALIAS)
            image = np.array(image, copy=False)
        return image

    def get_internal_states(self):
        """
        Get the internal joint states of the agent
        Returns:
            a numpy.array including joint positions and velocities
        """
        joint_pos = []
        joint_vel = []
        for joint_id in range(len(self.joints)):
            joint_name = self.joints[joint_id]
            joint_state = self._agent.get_joint_state(joint_name)
            joint_pos.append(joint_state.get_positions())
            joint_vel.append(joint_state.get_velocities())
        joint_pos = np.array(joint_pos).flatten()
        joint_vel = np.array(joint_vel).flatten()
        # pos of continous joint could be huge, wrap the range to [-pi, pi)
        joint_pos = (joint_pos + np.pi) % (2 * np.pi) - np.pi
        internal_states = np.concatenate((joint_pos, joint_vel), axis=0)
        return internal_states

    def get_control_space(self):
        """
        Get the pure controlling space without language.
        """
        control_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=[len(self.joints)], dtype=np.float32)
        return control_space

    def get_action_space(self):
        """
        Get the action space with optional language
        """
        control_space = self.get_control_space()
        if self._with_language:
            action_space = gym.spaces.Dict(
                control=control_space, sentence=self._sentence_space)
        else:
            action_space = control_space
        return action_space

    def get_observation_space(self, obs_sample):
        """
        Get the observation space with optional language
        """
        if self._with_language or self._image_with_internal_states:
            observation_space = self._construct_dict_space(obs_sample)
        elif self._use_image_observation:
            observation_space = gym.spaces.Box(
                low=0, high=255, shape=obs_sample.shape, dtype=np.uint8)
        else:
            observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_sample.shape,
                dtype=np.float32)
        return observation_space

    def set_sentence_space(self, sentence_space):
        """
        Set the sentence if with_languange is enabled
        This function should be called if with_language is enabled
        Args:
            sentence_space (gym.spaces): the space for sentence sequence
        """
        self._sentence_space = sentence_space

    def _construct_dict_space(self, obs_sample):
        """
        A helper function when gym.spaces.Dict is used as observation
        Args:
            obs_sample (dict) : a sample observation
        Returns:
            Return a gym.spaces.Dict with keys 'image', 'states', 'sentence'
            Possible situation:
                image with internal states
                image with language sentence
                image with both internal states and language sentence
                pure low-dimensional states with language sentence
        """
        ob_space_dict = dict()
        if 'image' in obs_sample.keys():
            ob_space_dict['image'] = gym.spaces.Box(
                low=0,
                high=255,
                shape=obs_sample['image'].shape,
                dtype=np.uint8)
        if 'states' in obs_sample.keys():
            ob_space_dict['states'] = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_sample['states'].shape,
                dtype=np.float32)
        if 'sentence' in obs_sample.keys():
            ob_space_dict['sentence'] = self._sentence_space
        ob_space = gym.spaces.Dict(ob_space_dict)
        return ob_space

    def setup_joints(self, agent, joints, agent_cfg):
        """
        Setup the joints acrroding to agent configuration.
        Args:
            agent (pygazebo.Agent): the agent
            joints (list of string): the name of joints
            agent_cfg (dict): the configuration
        """
        joint_states = list(map(lambda s: agent.get_joint_state(s), joints))
        joints_limits = list(
            map(lambda s: s.get_effort_limits()[0], joint_states))
        if agent_cfg['use_pid']:
            for joint_index in range(len(joints)):
                agent.set_pid_controller(
                    joint_name=joints[joint_index],
                    pid_control_type=agent_cfg['pid_type'][joint_index],
                    p=agent_cfg['pid'][joint_index][0],
                    i=agent_cfg['pid'][joint_index][1],
                    d=agent_cfg['pid'][joint_index][2],
                    max_force=joints_limits[joint_index])
            control_range = agent_cfg['pid_control_limit']
        else:
            control_range = np.array(joints_limits)
        return control_range
