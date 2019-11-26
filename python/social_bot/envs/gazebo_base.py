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
from lxml import etree

import gym
import gin
import numpy as np
from absl import logging
import social_bot
import social_bot.pygazebo as gazebo


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

    def __init__(self,
                 world_file=None,
                 world_string=None,
                 world_config=None,
                 sim_time_precision=0.001,
                 port=None,
                 quiet=False):
        """
        Args:
             world_file (str|None): world file path
             world_string (str|None): world xml string content,
             world_config (list[str]): list of str config `key=value`
                see `_modify_world_xml` for details
             sim_time_precision (float): the time precision of the simulator
             port (int): Gazebo port
             quiet (bool) Set quiet output
        """
        os.environ["GAZEBO_MODEL_DATABASE_URI"] = ""
        if port is None:
            port = 0
        self._port = port
        # to avoid different parallel simulation has the same randomness
        random.seed(port)
        self._rendering_process = None
        self._rendering_camera = None
        # the default camera pose for rendering rgb_array, could be override
        self._rendering_cam_pose = "10 -10 6 0 0.4 2.4"
        gazebo.initialize(port=port, quiet=quiet)

        if world_file:
            world_file_abs_path = os.path.join(social_bot.get_world_dir(),
                                               world_file)
            world_string = gazebo.world_sdf(world_file_abs_path)

        if world_config:
            world_string = _modify_world_xml(world_string, world_config)

        self._world = gazebo.new_world_from_string(world_string)

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
                <sdf version ='1.6'>
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

    def close(self):
        super().close()
        gazebo.close_without_model_base_fini()

    def insert_model(self, model, name=None, pose="0 0 0 0 0 0"):
        """
        Insert a model with a name into a specific position of the world
        Args:
            model (string): the name of the model in the model database
            name (string): the name of the model in the world.
                If not provided, it's the same as the model name.
            pose (string): the pose of the model, format is "x y z roll pitch yaw"
        """
        if name == None:
            name = model
        model_sdf = """
        <?xml version='1.0'?>
        <sdf version ='1.6'>
        <model name=""" + name + """>
            <include>
                <uri>model://""" + model + """</uri>
            </include>
            <pose frame=''>""" + pose + """</pose>
        </model>
        </sdf>
        """
        self._world.insertModelFromSdfString(model_sdf)
        # Sleep for a while waiting for gzserver to finish the inserting
        # operation. Or it may not be successfully inserted and report error.
        time.sleep(0.2)
        self._world.step(20)

    def insert_model_list(self, model_list):
        """
        Insert models into the world
        Args:
            model_list (list) : the list of the models
        """
        obj_num = len(model_list)
        for obj_id in range(obj_num):
            model_name = model_list[obj_id]
            # the way to construct the key needs to be in sync with pygazebo.cc
            # ModelListInfo()
            key = '"{}"'.format(model_name)
            if self._world.model_list_info().find(key) == -1:
                self._world.insertModelFile('model://' + model_name)
                logging.debug('model ' + model_name + ' inserted')
                time.sleep(0.2)
                self._world.step(20)

    def set_rendering_cam_pose(self, camera_pose):
        """
        Set the camera pose for rendering using rgb_array mode
        Args:
            camera_pose (string) : the camera pose, "x y z roll pitch yaw"
        """
        self._rendering_cam_pose = camera_pose

    def seed(self, seed=None):
        """Gym interface for setting random seed."""
        random.seed(seed)

    def __del__(self):
        if self._rendering_process is not None:
            self._rendering_process.terminate()


def _modify_world_xml(xml, modifications):
    """Modify world xml content
    eg:
    <sensor name="head_mount_sensor" type="camera">
          <visualize>0</visualize>
          <camera name="__default__">
            <horizontal_fov>0.994838</horizontal_fov>
            <image>
              <width>640</width>
              <height>480</height>
              <format>R8G8B8</format>
            </image>
          </camera>
    </sensor>

    1. set element value: ${selector}=${value}, if ${value} is empty,
    the element will be removed
    "//image/width=128"
    "//image/height=128"
    "//image/format=L8"

    <sensor name="head_mount_sensor" type="camera">
          <visualize>0</visualize>
          <camera name="__default__">
            <horizontal_fov>0.994838</horizontal_fov>
            <image>
              <width>128</width>
              <height>128</height>
              <format>L8</format>
            </image>
          </camera>
    </sensor>

    2. modify element attribute: ${selector}.${attr_name}=${value}
    "//sensor.name="sensor"
    "//sensor/camera.name="camera"

    <sensor name="sensor" type="camera">
          <visualize>0</visualize>
          <camera name="camera">
            <horizontal_fov>0.994838</horizontal_fov>
            <image>
              <width>640</width>
              <height>480</height>
              <format>R8G8B8</format>
            </image>
          </camera>
    </sensor>

    3. insert sub element: ${selector}<>${ele_name}=${value}
    "//sensor<>always_on=1"

    <sensor name="head_mount_sensor" type="camera">
          <always_on>1</always_on>
          <visualize>0</visualize>
          <camera name="__default__">
            <horizontal_fov>0.994838</horizontal_fov>
            <image>
              <width>640</width>
              <height>480</height>
              <format>R8G8B8</format>
            </image>
          </camera>
    </sensor>

    Args:
        xml (str):
        modifications (list[str]): list or `${selector}[${op}${name}]=${value}` strs
            where `selector` is used for locating the element in the xml content,
            it must be xpath selector, see 'https://lxml.de/xpathxslt.html' for details
            and `op`, `name` are optional. And op can be '.' that set attribute value of
            selected element or '<>' that create a sub element
    Returns (str):
        Return modified xml string
    """
    tree = etree.XML(xml)
    for config in modifications:
        i = config.rfind('=')
        key, value = config[:i].strip(), config[i + 1:].strip()

        i = key.rfind('<>')

        # add sub element
        if i != -1:
            key, sub_ele = key[:i].strip(), key[i + 2:].strip()
            logging.debug("Add element: %s %s %s", key, sub_ele, value)
            for ele in tree.xpath(key):
                etree.SubElement(ele, sub_ele).text = value
            continue

        i = key.rfind('.')

        # set attribute
        if i != -1:
            key, attr = key[:i].strip(), key[i + 1:].strip()
            logging.debug("Set attribute: %s %s %s", key, attr, value)
            for ele in tree.xpath(key):
                ele.set(attr, value)
            continue

        # set text value
        for ele in tree.xpath(key):
            if len(value) != 0:
                logging.debug("Set value: %s %s", key, value)
                ele.text = value
            else:
                logging.debug("Removing: %s", key)
                ele.getparent().remove(ele)

    xml = etree.tostring(tree, encoding='unicode')
    logging.debug(xml)
    return xml
