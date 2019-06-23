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
import time
import logging
import numpy as np
import random
import PIL
import itertools
import hashlib

import gym
from gym import spaces
import gin
from collections import OrderedDict

import social_bot
from social_bot import teacher
from social_bot.envs.gazebo_base import GazeboEnvBase
from social_bot.teacher import TeacherAction
from social_bot.teacher import DiscreteSequence
from social_bot import teacher_tasks
import social_bot.pygazebo as gazebo

logger = logging.getLogger(__name__)

def gro2_encode_goal_name(name):
    return int(hashlib.md5(name.encode('utf-8')).hexdigest(),16)/(2 ** 127) - 1.0


class GroceryGroundGoalTask(teacher_tasks.GoalTask):
    """
    A simple teacher task to find a goal for env GroceryGround.
    """

    def __init__(self, reward_shaping=False, random_goal=False, **kwargs):
        """
        Args:
            reward_shaping (bool): if ture, use shaped reward accroding to distance rather than -1 and 1
            random_goal (bool): if ture, teacher will randomly select goal from the object list each episode
        """
        super(GroceryGroundGoalTask, self).__init__(**kwargs)
        self._reward_shaping = reward_shaping
        self._random_goal = random_goal
        self._goal_name = 'cube_20k'
        self._object_list = ['coke_can', 'table', 'bookshelf', 'cube_20k', 'car_wheel',
            'plastic_cup', 'beer', 'hammer'
        ]

    def get_object_list(self):
        """
        Args:
            None
        Returns:
            Object list defined by teacher task
        """
        return self._object_list

    def get_goal_name(self):
        """
        Args:
            None
        Returns:
            Goal's name at this episode
        """
        return self._goal_name

    def run(self, agent, world):
        """
        Start a teaching episode for this task.
        Args:
            agent (pygazebo.Agent): the learning agent 
            world (pygazebo.World): the simulation world
        """
        agent_sentence = yield
        agent.reset()
        if self._random_goal:
            random_id = random.randrange(len(self._object_list))
            self._goal_name = self._object_list[random_id]
        goal = world.get_agent(self._goal_name)
        loc, dir = agent.get_pose()
        loc = np.array(loc)
        self._move_goal(goal, loc)
        steps_since_last_reward = 0
        while steps_since_last_reward < self._max_steps:
            steps_since_last_reward += 1
            loc, dir = agent.get_pose()
            goal_loc, _ = goal.get_pose()
            loc = np.array(loc)
            goal_loc = np.array(goal_loc)
            dist = np.linalg.norm(loc - goal_loc)
            if dist < self._success_distance_thresh:
                logger.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                             "dist: " + str(dist))
                agent_sentence = yield TeacherAction(
                    reward=10.0, sentence="Well done!", done=True)
                steps_since_last_reward = 0
            else:
                if self._reward_shaping:
                    agent_sentence = yield TeacherAction(
                        reward=-dist / self._random_range,
                        sentence="Failed",
                        done=False)
                else:
                    agent_sentence = yield TeacherAction()
        yield TeacherAction(reward=0.0, sentence="Failed", done=True)


@gin.configurable
class GroceryGround(GazeboEnvBase):
    """
    The goal of this task is to train the agent to navigate to a fixed type of 
    object. The name of the object is provided in the constructor. In each 
    episode, the location of the goal object is randomly chosen

    The envionment support agent type of pr2_differential, pioneer2dx_noplugin, 
    turtlebot, and irobot create for now. Note that for the models without camera
    sensor (pioneer, create), you can not use image as observation.

    Joints of the agent are controllable by force,
    the observations are image or the internal states of the world, including the
    position and rotation in world coordinate.

    The objects are rearranged each time the environment is reseted.
    
    Agent will receive a reward provided by the teacher. The goal's position is
    can also be controlled by the teacher.

    """

    def __init__(self,
                 with_language=False,
                 use_image_obs=False,
                 agent_type='pioneer2dx_noplugin',
                 random_goal=False,
                 max_steps=200,
                 port=None,
                 resized_image_size=(64, 64),
                 data_format='channels_last'):
        """
        Args:
            with_language (bool): the observation will be a dict with an extra sentence
            use_image_obs (bool): use image, or use internal states as observation
                poses in internal states observation are in world coordinate
            agent_type (string): select the agent robot, supporting pr2_differential, 
                pioneer2dx_noplugin, turtlebot, and irobot create for now
            port: Gazebo port, need to specify when run multiple environment in parallel
            resized_image_size (None|tuple): If None, use the original image size
                from the camera. Otherwise, the original image will be resized
                to (width, height)
            data_format (str):  one of `channels_last` or `channels_first`.
                The ordering of the dimensions in the images.
                `channels_last` corresponds to images with shape
                `(height, width, channels)` while `channels_first` corresponds
                to images with shape `(channels, height, width)`.
        """
        super(GroceryGround, self).__init__(port=port)
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(), "grocery_ground.world"))

        self._teacher = teacher.Teacher(False)
        task_group = teacher.TaskGroup()
        self._teacher_task = GroceryGroundGoalTask(
            max_steps=max_steps,
            success_distance_thresh=0.5,
            fail_distance_thresh=3.0,
            reward_shaping=True,
            random_goal=random_goal,
            random_range=10.0)
        task_group.add_task(self._teacher_task)
        self._teacher.add_task_group(task_group)

        self._object_list = self._teacher_task.get_object_list()
        self._pos_list = list(itertools.product(range(-5, 5), range(-5, 5)))
        self._pos_list.remove((0, 0))
        self._world.info()
        self._world.insertModelFile('model://' + agent_type)
        self._world.step(20)
        time.sleep(0.1)  # Avoid 'px!=0' error
        self._random_insert_objects()
        self._world.model_list_info()
        self._world.info()

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
        control_limit = {
            'pr2_differential': 20,
            'pioneer2dx_noplugin': 20.0,
            'turtlebot': 0.5,
            'create': 0.5,
        }
        camera_sensor = {
            'pr2_differential':
            'default::pr2_differential::head_tilt_link::head_mount_prosilica_link_sensor',
            'pioneer2dx_noplugin':
            'default::pioneer2dx_noplugin::camera_link::camera',
            'turtlebot':
            'default::turtlebot::kinect::link::camera',
            'create':
            ' ',
        }

        self._agent = self._world.get_agent(agent_type)
        self._agent_joints = control_joints[agent_type]
        for _joint in self._agent_joints:
            self._agent.set_pid_controller(_joint, 'velocity', d=0.005)
        self._agent_control_range = control_limit[agent_type]
        self._agent_camera = camera_sensor[agent_type]

        logger.info("joints to control: %s" % self._agent_joints)

        self._with_language = with_language
        self._use_image_obs = use_image_obs
        assert data_format in ('channels_first', 'channels_last')
        self._data_format = data_format
        self._resized_image_size = resized_image_size

        self.reset()
        obs_data = self._get_observation()
        if self._use_image_obs:
            obs_data_space = gym.spaces.Box(
                low=0, high=255, shape=obs_data.shape, dtype=np.uint8)
        else:
            obs_data_space = gym.spaces.Box(
                low=-50, high=50, shape=obs_data.shape, dtype=np.float32)

        control_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=[len(self._agent_joints)],
            dtype=np.float32)

        if self._with_language:
            self.observation_space = gym.spaces.Dict(
                data=obs_data_space, sentence=DiscreteSequence(128, 24))
            self.action_space = gym.spaces.Dict(
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
        self._collision_cnt = 0
        self._cum_reward = 0.0
        self._steps_in_this_episode = 0
        self._agent.reset()
        self._random_move_objects()
        self._world.step(50)
        self._teacher.reset(self._agent, self._world)
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
            obs_data = np.array(obs_data)
            if self._resized_image_size:
                obs_data = PIL.Image.fromarray(obs_data).resize(
                    self._resized_image_size, PIL.Image.ANTIALIAS)
                obs_data = np.array(obs_data, copy=False)
            if self._data_format == "channels_first":
                obs_data = np.transpose(obs_data, [2, 0, 1])
        else:
            goal_name = self._teacher_task.get_goal_name()
            goal = self._world.get_model(goal_name)
            goal_pos = np.array(goal.get_pose()[0]).flatten()
            agent_pose = np.array(self._agent.get_pose()).flatten()
            agent_vel = np.array(self._agent.get_velocities()[0]).flatten()
            joint_vel = []
            for joint_id in range(len(self._agent_joints)):
                joint_name = self._agent_joints[joint_id]
                joint_state = self._agent.get_joint_state(joint_name)
                joint_vel.append(joint_state.get_velocities())
            joint_vel = np.array(joint_vel).flatten()
            encoded_goal_name = gro2_encode_goal_name(goal_name)
            encoded_goal_name = np.array(encoded_goal_name).reshape(-1)
            obs_data = np.concatenate(
                (goal_pos, agent_pose, agent_vel, joint_vel, encoded_goal_name),
                axis=0)
            obs_data = np.array(obs_data).reshape(-1)
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
            controls = action['control'] * self._agent_control_range
        else:
            sentence = ''
            controls = action * self._agent_control_range
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
        for obj_id in range(len(self._object_list)):
            model_name = self._object_list[obj_id]
            self._world.insertModelFile('model://' + model_name)
            logger.debug('model ' + model_name + ' inserted')
            self._world.step(10)
            # Sleep for a while waiting for Gazebo server to finish the inserting
            # operation. Or the model may not be completely inserted, boost will
            # throw 'px!=0' error when set_pose/get_pose of the model is called
            time.sleep(0.1)

    def _random_move_objects(self, random_range=10.0):
        obj_num = len(self._object_list)
        obj_pos_list = random.sample(self._pos_list, obj_num)
        for obj_id in range(obj_num):
            model_name = self._object_list[obj_id]
            loc = (obj_pos_list[obj_id][0], obj_pos_list[obj_id][1], 0)
            pose = (np.array(loc), (0, 0, 0))
            self._world.get_model(model_name).set_pose(pose)
            self._world.step(5)


def main():
    """
    Simple testing of this environment.
    """
    import matplotlib.pyplot as plt
    use_image_obs = False
    random_goal = True
    fig = None
    env = GroceryGround(use_image_obs=use_image_obs, random_goal=random_goal)
    env.render()
    while True:
        actions = np.array(np.random.randn(env.action_space.shape[0]))
        obs, _, done, _ = env.step(actions)
        if use_image_obs:
            if fig is None:
                fig = plt.imshow(obs)
            else:
                fig.set_data(obs)
            plt.pause(0.00001)
        if done:
            if random_goal:
                goalname = env._teacher_task.get_goal_name()
                logger.debug("goalname:" + goalname + 
                    ", encode to: " + str(gro2_encode_goal_name(goalname)))
                assert gro2_encode_goal_name(goalname) == obs[-1]
            env.reset()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
