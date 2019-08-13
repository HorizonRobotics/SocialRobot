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

import gym
import gin
import numpy as np

import social_bot.pygazebo as gazebo
from social_bot.teacher import DiscreteSequence


@gin.configurable
class GazeboEnvBase(gym.Env):
    """
    Base class for gazebo physics simulation
    These environments create scenes behave like normal Gym environments.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 30
    }

    def __init__(self, port=None, quiet=False):
        if port is None:
            port = 0
        self._port = port
        # to avoid different parallel simulation has the same randomness
        random.seed(port)
        self._rendering_process = None
        self._world = None
        self._rendering_camera = None
        # the default camera pose for rendering rgb_array, could be override
        self._rendering_cam_pose = "10 -10 6 0 0.4 2.4"
        gazebo.initialize(port=port, quiet=quiet)

    def render(self, mode='human'):
        """Render the environment.

        Args:
            mode (str): 'human' and 'rgb_array' is supported.
        """
        if mode == 'human':
            if self._rendering_process is None:
                from subprocess import Popen
                if self._port != 0:
                    os.environ[
                        'GAZEBO_MASTER_URI'] = "localhost:%s" % self._port
                self._rendering_process = Popen(['gzclient'])
            return
        if mode == 'rgb_array':
            if self._rendering_camera is None:
                render_camera_sdf = """
                <?xml version='1.0'?>
                <sdf version ='1.4'>
                <model name ='render_camera'>
                    <static>1</static>
                    <pose>%s</pose>
                    <link name="link">
                        <sensor name="camera" type="camera">
                            <camera>
                            <horizontal_fov>0.95</horizontal_fov>
                            <image>
                                <width>640</width>
                                <height>480</height>
                            </image>
                            <clip>
                                <near>0.1</near>
                                <far>100</far>
                            </clip>
                            </camera>
                            <always_on>1</always_on>
                            <update_rate>30</update_rate>
                            <visualize>true</visualize>
                        </sensor>
                    </link>
                </model>
                </sdf>
                """
                render_camera_sdf = render_camera_sdf % self._rendering_cam_pose
                self._world.insertModelFromSdfString(render_camera_sdf)
                time.sleep(0.2)
                self._world.step(20)
                self._rendering_camera = self._world.get_agent('render_camera')
            image = self._rendering_camera.get_camera_observation(
                "default::render_camera::link::camera")
            return np.array(image)

        raise NotImplementedError("rendering mode: " + mode +
                                  " is not implemented.")

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

    def _construct_dict_space(self, obs_sample, vocab_size):
        """
        Args:
            obs_sample (dict) : a sample observation
            vocab_size (int): the vocab size for the sentence sequence
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
            sentence_space = DiscreteSequence(vocab_size,
                                              len(obs_sample['sentence']))
            ob_space_dict['sentence'] = sentence_space
        ob_space = gym.spaces.Dict(ob_space_dict)
        return ob_space

    def __del__(self):
        if self._rendering_process is not None:
            self._rendering_process.terminate()
