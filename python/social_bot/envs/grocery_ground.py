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

import gym
from gym import spaces
from collections import OrderedDict

import social_bot
from social_bot import teacher
from social_bot.teacher import TeacherAction
from social_bot.teacher import DiscreteSequence
from social_bot import teacher_tasks
import social_bot.pygazebo as gazebo

logger = logging.getLogger(__name__)


class GroceryGroundGoalTask(teacher_tasks.GoalTask):
    """
    A simple teacher task to find a goal for env GroceryGround.
    """

    def __init__(self, reward_shaping=False, **kwargs):
        """
        Args:
            eward_shaping (bool): if ture, use shaped reward accroding to distance rather than -1 and 1
        """
        super(GroceryGroundGoalTask, self).__init__(**kwargs)
        self._reward_shaping = reward_shaping

    def run(self, agent, world):
        """
        Start a teaching episode for this task.
        Args:
            agent (pygazebo.Agent): the learning agent 
            world (pygazebo.World): the simulation world
        """
        agent_sentence = yield
        agent.reset()
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
                self._move_goal(goal, loc)
            else:
                if self._reward_shaping:
                    agent_sentence = yield TeacherAction(
                        reward=-0.1 * dist / self._random_range,
                        sentence="Failed",
                        done=False)
                else:
                    agent_sentence = yield TeacherAction()
        logger.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                     "dist: " + str(dist))
        yield TeacherAction(reward=-10.0, sentence="Failed", done=True)


class GroceryGround(gym.Env):
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
                 goal_name='table',
                 max_steps=160,     
                 port=None):
        """
        Args:
            with_language (bool): the observation will be a dict with an extra sentence
            use_image_obs (bool): use image, or use internal states as observation
                poses in internal states observation are in world coordinate
            agent_type (string): select the agent robot, supporting pr2_differential, 
                pioneer2dx_noplugin, turtlebot, and irobot create for now
            port: Gazebo port, need to specify when run multiple environment in parallel
        """
        if port is None:
            port = 0
        gazebo.initialize(port=port)
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(), "grocery_ground.world"))
        self._object_types = [
            'coke_can', 'cube_20k', 'car_wheel', 'plastic_cup', 'beer', 'hammer'
        ]
        self._pos_list=list(itertools.product(range(-5, 5), range(-5, 5)))
        self._pos_list.remove((0,0))
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
        control_force = {
            'pr2_differential': 20,
            'pioneer2dx_noplugin': 0.5,
            'turtlebot': 0.5,
            'create': 0.5,
        }
        camera_sensor = {
            'pr2_differential': 'default::pr2_differential::head_tilt_link::head_mount_prosilica_link_sensor',
            'pioneer2dx_noplugin': 'default::pioneer2dx_noplugin::camera_link::camera',
            'turtlebot': 'default::turtlebot::kinect::link::camera',
            'create': ' ',
        }

        self._agent = self._world.get_agent(agent_type)
        self._agent_joints = control_joints[agent_type]
        self._agent_control_force = control_force[agent_type]
        self._agent_camera = camera_sensor[agent_type]
        self._goal_name = goal_name
        self._goal = self._world.get_model(goal_name)

        logger.info("joints to control: %s" % self._agent_joints)

        self._teacher = teacher.Teacher(False)
        task_group = teacher.TaskGroup()
        self._teacher_task = GroceryGroundGoalTask(
            max_steps=max_steps,
            goal_name=self._goal_name,
            success_distance_thresh=0.5,
            fail_distance_thresh=3.0,
            reward_shaping=True,
            random_range=10.0)
        task_group.add_task(self._teacher_task)
        self._teacher.add_task_group(task_group)

        self._with_language = with_language
        self._use_image_obs = use_image_obs

        obs = self.reset()
        if self._use_image_obs:
            obs_data_space = gym.spaces.Box(
                low=0, high=255, shape=obs.shape, dtype=np.uint8)
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
        else:
            goal_pose = np.array(self._goal.get_pose()).flatten()
            agent_pose = np.array(self._agent.get_pose()).flatten()
            agent_vel = np.array(self._agent.get_velocities()).flatten()
            joint_vel = []
            for joint_id in range(len(self._agent_joints)):
                joint_name = self._agent_joints[joint_id]
                joint_state = self._agent.get_joint_state(joint_name)
                joint_vel.append(joint_state.get_velocities())
            joint_vel = np.array(joint_vel).flatten()
            obs_data = np.concatenate(
                (goal_pose, agent_pose, agent_vel, joint_vel), axis=0)
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
            self._world.step(10)
            # Sleep for a while waiting for Gazebo server to finish the inserting
            # operation. Or the model may not be completely inserted, boost will
            # throw 'px!=0' error when set_pose/get_pose of the model is called
            time.sleep(0.1)

    def _random_move_objects(self, random_range=10.0):
        obj_num = len(self._object_types)
        obj_pos_list = random.sample(self._pos_list, obj_num)
        for obj_id in range(obj_num):
            model_name = self._object_types[obj_id]
            loc = (obj_pos_list[obj_id][0], obj_pos_list[obj_id][1], 0)
            pose = (np.array(loc), (0, 0, 0))
            self._world.get_model(model_name).set_pose(pose)
            self._world.step(10)


def main():
    """
    Simple testing of this environment.
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
