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

import math
import os
import time
import random
import json
import gin
import numpy as np
import PIL.Image
from collections import OrderedDict
import gym
from absl import logging
import social_bot
import social_bot.pygazebo as gazebo


@gin.configurable
class GazeboAgent():
    """ Class for the agent of gazebo-based SocialRobot enviroments
    """

    def __init__(self,
                 world,
                 agent_type,
                 name=None,
                 config=None,
                 use_image_observation=True,
                 resized_image_size=None,
                 image_with_internal_states=False,
                 with_language=False,
                 with_agent_language=False,
                 vocab_sequence_length=20,
                 action_wrapper=None):
        """
        Args:
            world (pygazebo.World): the world
            agent_type (str): the agent_type, supporting pr2_noplugin,
                pioneer2dx_noplugin, turtlebot, youbot_noplugin and icub_with_hands for now
                note that 'agent_type' should be exactly the same string as the model's
                name at the beginning of model's sdf file
            name (str): the name of the agent in world
                if None it will be set the same as agent_type
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
            with_agent_language (bool): Include language in agent's action space
            vocab_sequence_length (int): the length of encoded sequence if with_language
            action_wrapper (None|class): Some times primitive joints is not wanted, e.g., has
                redundant dimensions or offset. If not None, this is used to transform the agent
                actions. See ActionWrapper of gazebo_agent.py for example.
        """
        self._world = world
        self.type = agent_type
        self._use_image_observation = use_image_observation
        self._resized_image_size = resized_image_size
        self._image_with_internal_states = image_with_internal_states
        self._with_language = with_language
        self._with_agent_language = with_agent_language
        self._vocab_sequence_length = vocab_sequence_length
        self._sentence_space = None

        if config == None:
            # Load agent configurations
            with open(
                    os.path.join(social_bot.get_model_dir(), "agent_cfg.json"),
                    'r') as cfg_file:
                agent_cfgs = json.load(cfg_file)
            config = agent_cfgs[agent_type]
        self.config = config
        joints = config['control_joints']
        if action_wrapper is not None:
            self._action_wrapper = action_wrapper()
            self._action_dim = self._action_wrapper.get_actions_dim()
        else:
            self._action_wrapper = None
            self._action_dim = len(joints)

        if name:
            # the agent is wrapped by a new name in world
            self.name = name
            self.joints = []
            for joint in joints:
                self.joints.append(name + '::' + joint)
        else:
            self.name = agent_type
            self.joints = joints
        self._agent = self._world.get_agent(self.name)

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
        self._camera = config['camera_sensor']
        self.action_range = self.setup_joints(self._agent, self.joints, config)
        logging.debug("joints to control: %s" % self.joints)

    def reset(self):
        """ Reset the agent. """
        self._agent.reset()

    def take_action(self, action):
        """ Take actions.
        
        Args:
            the actions to be taken.
        """
        if self._action_wrapper is not None:
            action = self._action_wrapper.wrap_actions(action)
        controls = np.clip(action, -1.0, 1.0) * self.action_range
        controls_dict = dict(zip(self.joints, controls))
        self._agent.take_action(controls_dict)

    def get_observation(self, teacher, sentence_raw="hello"):
        """ Get the observation of agent.

        Args:
            teacher (social_bot.Teacher): the teacher, used to get the task specific
                observations from teacher's taskgroups.
            sentence_raw (string): the sentence intened to sent to the Agent. This can
                be ignored if with_language is False.
        Returns:
            obs (dict |numpy.array): the return depends on the configurations: with
                language or not, use image or not, and image_with_internal_states or not.
                Possible situations:
                    low-dimensional full states
                    low-dimensional full states with language sentence
                    image from the camera of agent
                    image with internal states
                    image with language sentence
                    image with both internal states and language sentence
                Note that low-dimensional full states is defined in
                "Task.task_specific_observation()", which has all the infomation need
                for the task. While the internal states that used as a supplementary
                to image is form "Agent.get_internal_states()", which only contains
                self joint positions and velocities. Joint positions are wrapped with
                sin() and cos() to avoid the discontinuous point at 0 to 2*pi.
        """
        if self._image_with_internal_states or self._with_language:
            # observation is an OrderedDict
            obs = self._create_observation_dict(teacher, sentence_raw)
        elif self._use_image_observation:  # observation is pure image
            obs = self.get_camera_observation()
        else:  # observation is pure low-dimentional states
            obs = teacher.get_task_specific_observation(self)
        return obs

    def get_camera_observation(self):
        """ Get the camera image.

        Returns:
            a numpy.array of the image.
        """
        image = np.array(
            self._agent.get_camera_observation(self._camera), copy=False)
        if self._resized_image_size:
            image = PIL.Image.fromarray(image).resize(self._resized_image_size,
                                                      PIL.Image.ANTIALIAS)
            image = np.array(image, copy=False)
        return image

    def get_internal_states(self):
        """ Get the internal joint states of the agent.

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
        # pos of continous joint could be huge, wrap the range with sin and cos.
        joint_pos_sin = np.sin(joint_pos)
        joint_pos_cos = np.cos(joint_pos)
        internal_states = np.concatenate(
            (joint_pos_sin, joint_pos_cos, joint_vel), axis=0)
        return internal_states

    def get_control_space(self):
        """ Get the pure controlling space without language. """
        control_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=[self._action_dim], dtype=np.float32)
        return control_space

    def get_action_space(self):
        """ Get the action space with optional language. """
        control_space = self.get_control_space()
        if self._with_agent_language and self._with_language:
            action_space = gym.spaces.Dict(
                control=control_space, sentence=self._sentence_space)
        else:
            action_space = control_space
        return action_space

    def get_observation_space(self, teacher):
        """
        Get the observation space with optional language.

        Args:
            teacher (social_bot.Teacher): the teacher, used to get the task specific
                observations from teacher's taskgroups as a sample.
        """
        obs_sample = self.get_observation(teacher)
        if self._with_language or self._image_with_internal_states:
            # observation is a dictionary
            observation_space = self._construct_dict_space(obs_sample)
        elif self._use_image_observation:
            # observation is image
            observation_space = gym.spaces.Box(
                low=0, high=255, shape=obs_sample.shape, dtype=np.uint8)
        else:
            # observation is spare states
            observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_sample.shape,
                dtype=np.float32)
        return observation_space

    def set_sentence_space(self, sentence_space):
        """ Set the sentence if with_languange is enabled.

        Args:
            sentence_space (gym.spaces): the space for sentence sequence
        """
        self._sentence_space = sentence_space

    def _create_observation_dict(self, teacher, sentence_raw):
        obs = OrderedDict()
        if self._use_image_observation:
            obs['image'] = self.get_camera_observation()
            if self._image_with_internal_states:
                obs['states'] = self.get_internal_states()
        else:
            obs['states'] = teacher.get_task_specific_observation(self)
        if self._with_language:
            obs['sentence'] = teacher.sentence_to_sequence(
                sentence_raw, self._vocab_sequence_length)
        return obs

    def _construct_dict_space(self, obs_sample):
        """ A helper function when gym.spaces.Dict is used as observation.

        Args:
            obs_sample (numpy.array|dict) : a sample observation
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
        """  Setup the joints acrroding to agent configuration.

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

    def get_egocentric_cord_2d(self, x, y, agent_yaw):
        """ Get the egocentric coordinate from a global 2D x-y plane coordinate.

        This is achieved by rotating the global coordinates x, y by -agent_yaw.

        Args:
            x (float): x of global x-y plane coordinate
            y (float): y of global x-y plane coordinate
            agent_yaw (float): agent yaw (rotation in z-axis), in radian
        Returns:
            tuple of float, the position in the transformed coordinate
        """
        rotate = -agent_yaw
        rotated_x = x * np.cos(rotate) - y * np.sin(rotate)
        rotated_y = x * np.sin(rotate) + y * np.cos(rotate)
        return (rotated_x, rotated_y)

    def get_contacts(self, contacts_sensor, contact_collision):
        """ Get contacts to the link.

        Args:
            contacts_sensor(string): the name of contacts_sensor
            contact_collision(string): the collision to check contacts
        Returns:
            bool, there is contact or not
        """
        contacts = self.get_collisions(contacts_sensor)
        for collision in contacts:
            if collision[0] == contact_collision or collision[
                    1] == contact_collision:
                return True
        return False


class ActionWrapper():
    """ The action wrapper transform a new actions to primitive actions.

    The primitive actions (like the force/velocity/position of joints) may have redundant
    dimensions or offsets. By the action wrapper, we can transform the action to more
    efficency one. The sub class should define the new action space in _NEW_ACTION_LIST.
    """

    _NEW_ACTION_LIST = []

    def get_actions_dim(self):
        """ Get the dimension of the new action space
        """
        return len(self._NEW_ACTION_LIST)

    def wrap_actions(self, action):
        """ Wrap transformed actions to primitive actions.

        Args:
            action (nparray): the new action from policy network
        Returns:
            np.array, the primitive actions send to simulator
        """
        raise NotImplementedError("wrap_actions not implemented!")


@gin.configurable
class YoubotActionWrapper(ActionWrapper):
    """ This action wrapper transform a new actions to primitive actions.

    The new action space is the same as keyboard demostration interface, defined in _NEW_ACTION_LIST
    The primitive actions (the joints) please refer to social_bot/models/agent_cfg.json.
    """

    _NEW_ACTION_LIST = [
        'arm_joint_yaw', 'arm_joint_pitch', 'arm_joint_pitch_2', 'palm_joint',
        'gripper_finger_joint', 'wheel_speed', 'wheel_turning'
    ]

    def wrap_actions(self, action):
        """ Wrap transformed actions to primitive actions.

        Args:
            action (nparray): the new action from policy network
        Returns:
            np.array, the primitive actions send to simulator
        """
        action = dict(zip(self._NEW_ACTION_LIST, action))
        primitive_actions = [
            # arm joints
            action['arm_joint_yaw'],
            0.25 + action['arm_joint_pitch'] / 2,  # add pi/4 offset
            0.25 + action['arm_joint_pitch'] / 2,
            0.25 + action['arm_joint_pitch_2'],
            action['palm_joint'],
            # gripper joints
            action['gripper_finger_joint'],
            action['gripper_finger_joint'],
            # wheel joints
            action['wheel_speed'] + action['wheel_turning'],
            action['wheel_speed'] - action['wheel_turning']
        ]
        return primitive_actions
