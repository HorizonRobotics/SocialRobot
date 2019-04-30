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
A simple enviroment for an agent play on a groceryground
"""
import os
import logging
import numpy as np
import random
import PIL

import gym
from gym import spaces
from collections import OrderedDict

import social_bot
from social_bot import teacher
from social_bot.teacher import TeacherAction
from social_bot.teacher import DiscreteSequence
from social_bot.goal_task import GoalTask
import social_bot.pygazebo as gazebo

logger = logging.getLogger(__name__)


class GroceryGroundGoalTask(GoalTask):
    """
    For this task, the agent will receive reward 1 when it is close enough to the goal.
    If it is moving away from the goal too much or still not close to the goal after max_steps,
    it will get reward -1.
    """

    def __init__(self,
                 max_steps=500,
                 goal_name="first_2015_trash_can",
                 success_distance_thresh=0.5,
                 fail_distance_thresh=3.0,
                 goal_random_range=10.0):
        """
        Args:
            max_steps (int): episode will end if not reaching goal in so many steps
            goal_name (string): name of the goal in the world
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            fail_distance_thresh (float): if the agent moves away from the goal more than this distance,
                it's considered a failure and is givne reward -1
            goal_random_range (float): the goal's random position range
            
        """
        super(GroceryGroundGoalTask, self).__init__(
            max_steps=max_steps,
            goal_name=goal_name,
            success_distance_thresh=success_distance_thresh,
            fail_distance_thresh=fail_distance_thresh,
            goal_random_range=goal_random_range)


class GroceryGround(gym.Env):
    """
    The goal of this task is to train the agent to navigate to the objects given its
    name.

    Joints of the agent are controllable by force,
    the observations are image or the states of the world, including every model's
    position and rotation.

    Agent will receive a reward provided by the teacher

    """

    def __init__(self,
                 with_language=False,
                 use_image_obs=False,
                 agent_type='pioneer2dx_noplugin',
                 port=None):
        """
        Args:
            with_language (bool): the observation will be a dict with an extra sentence
            use_image_obs (bool): use image as observation, or use pose of the objects
            agent_type (string): select the agent robot, supporting pr2_differential, 
                pioneer2dx_noplugin, turtlebot, and create for now
            port: Gazebo port, need to specify when run multiple environment in parallel
        """
        if port is None:
            port = 0
        gazebo.initialize(port=port)
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(), "grocery_ground.world"))
        self._object_types = [
            'coke_can', 'cube_20k', 'car_wheel', 'first_2015_trash_can',
            'plastic_cup', 'postbox', 'cabinet', 'beer', 'hammer'
        ]
        self._world.info()
        self._world.insertModelFile('model://' + agent_type)
        self._world.step(20)
        self._random_insert_objects()
        self._world.model_list_info()
        self._random_move_objects()

        # Specify joints and sensors for the robots
        control_joints = {
            'pr2_differential': [
                'pr2_differential::fl_caster_r_wheel_joint',
                'pr2_differential::fr_caster_l_wheel_joint'
            ],
            'pioneer2dx_noplugin': [
                'pioneer2dx_noplugin::left_wheel_hinge',
                'pioneer2dx_noplugin::right_wheel_hinge'
            ],
            'turtlebot': [
                'turtlebot::create::left_wheel',
                'turtlebot::create::right_wheel'
            ],
            'create': ['create::left_wheel', 'create::right_wheel'],
        }
        control_force = {
            'pr2_differential': 20,
            'pioneer2dx_noplugin': 2,
            'turtlebot': 1,
            'create': 0.5,
        }
        # Camera, TODO
        camera_sensor = {
            'pr2_differential':
            'default::pr2_differential::head_tilt_link::head_mount_prosilica_link_sensor',
            'pioneer2dx_noplugin':
            ' ',
            'turtlebot':
            ' ',
            'create':
            ' ',
        }

        self._agent = self._world.get_agent(agent_type)
        self._agent_joints = control_joints[agent_type]
        self._agent_control_force = control_force[agent_type]
        self._agent_camera = camera_sensor[agent_type]

        logger.info("joints to control: %s" % self._agent_joints)

        self._teacher = teacher.Teacher(False)
        task_group = teacher.TaskGroup()
        task_group.add_task(GroceryGroundGoalTask())
        self._teacher.add_task_group(task_group)

        self._with_language = with_language
        self._use_image_obs = use_image_obs
        self._resized_image_size = (84, 84, 3)

        obs = self.reset()
        if self._use_image_obs:
            obs_data_space = gym.spaces.Box(
                low=0, high=1, shape=obs.shape, dtype=np.uint8)
        else:
            obs_data_space = gym.spaces.Box(
                low=-50, high=50, shape=obs.shape, dtype=np.float32)

        control_space = gym.spaces.Box(
            low=-self._agent_control_force,
            high=self._agent_control_force,
            shape=[len(self._agent_joints)],
            dtype=np.float32)

        if self._with_language:
            self.observation_space = gym.spaces.Dict(
                data=obs_data_space, sentence=DiscreteSequence(128, 24))
            self._action_space = gym.spaces.Dict(
                control=control_space, sentence=DiscreteSequence(128, 24))
        else:
            self.observation_space = obs_data_space
            self.action_space = control_space

    def reset(self):
        """
        Args:
            None
        Returns:
            Observaion of the first step
        """
        self._world.reset()
        self._world.step(20)
        self._collision_cnt = 0
        self._cum_reward = 0.0
        self._steps_in_this_episode = 0
        self._teacher.reset(self._agent, self._world)
        self._random_move_objects()
        self._world.step(20)
        teacher_action = self._teacher.teach("")
        if self._with_language:
            obs_data = self._get_observation()
            obs = OrderedDict(data=obs_data, sentence=teacher_action)
        else:
            obs = self._get_observation()
        return obs

    def _get_observation(self):
        if self._use_image_obs:
            obs_data = np.array(
                self._agent.get_camera_observation(self._agent_camera),
                copy=False)
            obs_data = PIL.Image.fromarray(
                obs_data.resize(self._resized_image_size, PIL.Image.ANTIALIAS))
            obs_data = np.reshape(
                np.array(obs_data), [
                    self._resized_image_size[0], self._resized_image_size[1],
                    self._resized_image_size[2]
                ])
        else:
            objects_poses = []
            objects_poses.append(self._agent.get_pose())
            for obj_id in range(len(self._object_types)):
                model_name = self._object_types[obj_id]
                pose = self._world.get_model(model_name).get_pose()
                objects_poses.append(pose)
            obs_data = np.array(objects_poses).reshape(-1)
        return obs_data

    def step(self, action):
        """
        Args:
            action (dict|int): If with_language, action is a dictionary 
                    with key "control" and "sentence".
                    action['control'] is a vector whose dimention is
                    len(_joint_names). action['sentence'] is a string.
                    If not with_language, it is an int for the action id.
        Returns:
            If with_language, it is a dictionary with key 'data' and 'sentence'
            If not with_language, it is a numpy.array or image for observation
        """
        if self._with_language:
            sentence = action.get('sentence', None)
            controls = action['control']
        else:
            sentence = ''
            controls = action
        controls = dict(zip(self._agent_joints, controls))
        teacher_action = self._teacher.teach(sentence)
        self._agent.take_action(controls)
        self._world.step(10)
        if self._with_language:
            obs_data = self._get_observation()
            obs = OrderedDict(data=obs_data, sentence=teacher_action.sentence)
        else:
            obs = self._get_observation()
        self._steps_in_this_episode += 1
        self._cum_reward += teacher_action.reward
        if teacher_action.done:
            logger.debug("episode ends at cum reward:" + str(self._cum_reward))
        return obs, teacher_action.reward, teacher_action.done, {}

    def _random_insert_objects(self):
        for obj_id in range(len(self._object_types)):
            model_name = self._object_types[obj_id]
            self._world.insertModelFile('model://' + model_name)
            logger.debug('model ' + model_name + ' inserted')
            self._world.step(20)  # Avoid 'px!=0' error

    def _random_move_objects(self, random_range=10.0):
        for obj_id in range(len(self._object_types)):
            model_name = self._object_types[obj_id]
            loc = (random.random() * random_range - random_range / 2,
                   random.random() * random_range - random_range / 2, 0)
            pose = (loc, (0, 0, 0))
            self._world.get_model(model_name).set_pose(pose)


def main():
    """
    Simple testing of this enviroenment.
    """
    env = GroceryGround()
    while True:
        actions = env._agent_control_force * np.random.randn(
            env.action_space.shape[0])
        obs, _, done, _ = env.step(actions)
        if done:
            env.reset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
