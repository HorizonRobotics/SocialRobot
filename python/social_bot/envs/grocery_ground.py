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
import numpy as np
import random
import json
import PIL
import itertools
from absl import logging

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
        self._goal_name = 'coke_can_on_table'
        self._static_object_list = [
            'placing_table', 'plastic_cup_on_table', 'coke_can_on_table',
            'hammer_on_table', 'cafe_table'
        ]
        self._object_list = [
            'coke_can', 'table', 'bookshelf', 'car_wheel',
            'plastic_cup', 'beer', 'hammer'
        ]
        self.task_vocab = self.task_vocab + self._static_object_list + self._object_list

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
                logging.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                              "dist: " + str(dist))
                agent_sentence = yield TeacherAction(
                    reward=10.0, sentence="well done", done=True)
                steps_since_last_reward = 0
            else:
                if self._reward_shaping:
                    reward = -dist / self._random_range
                else:
                    reward = 0.0
                agent_sentence = yield TeacherAction(
                    reward=reward, sentence=self._goal_name, done=False)
        yield TeacherAction(
            reward=0.0, sentence="failed to " + self._goal_name, done=True)


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
                 random_goal=False,
                 agent_type='pioneer2dx_noplugin',
                 max_steps=120,
                 port=None,
                 resized_image_size=(64, 64),
                 data_format='channels_last'):
        """
        Args:
            with_language (bool): The observation will be a dict with an extra sentence
            use_image_obs (bool): Use image, or use internal states as observation
                poses in internal states observation are in world coordinate
            random_goal (bool): If ture, teacher will randomly select goal from the 
                object list each episode
            agent_type (string): Select the agent robot, supporting pr2_differential, 
                pioneer2dx_noplugin, turtlebot, irobot create and icub_with_hands for now
                note that 'agent_type' should be the same str as the model's name
            port: Gazebo port, need to specify when run multiple environment in parallel
            resized_image_size (None|tuple): If None, use the original image size
                from the camera. Otherwise, the original image will be resized
                to (width, height)
            data_format (str):  One of `channels_last` or `channels_first`.
                The ordering of the dimensions in the images.
                `channels_last` corresponds to images with shape
                `(height, width, channels)` while `channels_first` corresponds
                to images with shape `(channels, height, width)`.
        """
        super(GroceryGround, self).__init__(port=port)
        
        self._teacher = teacher.Teacher(task_groups_exclusive=False)
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
        self._teacher.build_vocab_from_tasks()

        self._object_list = self._teacher_task.get_object_list()
        self._pos_list = list(itertools.product(range(-5, 5), range(-5, 5)))
        self._pos_list.remove((0, 0))

        wf_path = os.path.join(social_bot.get_world_dir(), "grocery_ground.world")
        with open(wf_path, 'r+') as world_file:
            world_string = self._insert_agent_to_world_file(world_file, agent_type)
        self._world = gazebo.new_world_from_string(world_string)
        self._world.step(20)
        self._insert_objects(self._object_list)
        self._agent = self._world.get_agent()
        logging.debug(self._world.info())
        agent_cfgs = json.load(
            open(
                os.path.join(social_bot.get_model_dir(), "agent_cfg.json"),
                'r'))
        agent_cfg = agent_cfgs[agent_type]
        self._agent_joints = agent_cfg['control_joints']
        joint_states = list(
            map(lambda s: self._agent.get_joint_state(s), self._agent_joints))
        self._joints_limits = list(
            map(lambda s: s.get_effort_limits()[0], joint_states))
        if agent_cfg['use_pid']:
            for joint_index in range(len(self._agent_joints)):
                self._agent.set_pid_controller(
                    self._agent_joints[joint_index],
                    'velocity',
                    p=0.02,
                    d=0.00001,
                    max_force=self._joints_limits[joint_index])
            self._agent_control_range = agent_cfg['pid_control_limit']
        else:
            self._agent_control_range = np.array(self._joints_limits)
        self._agent_camera = agent_cfg['camera_sensor']

        logging.debug("joints to control: %s" % self._agent_joints)

        self._with_language = with_language
        self._use_image_obs = use_image_obs
        assert data_format in ('channels_first', 'channels_last')
        self._data_format = data_format
        self._resized_image_size = resized_image_size

        obs_data = self._get_observation()
        if self._use_image_obs:
            obs_data_space = gym.spaces.Box(
                low=0, high=255, shape=obs_data.shape, dtype=np.uint8)
        else:
            obs_data_space = gym.spaces.Box(
                low=-50, high=50, shape=obs_data.shape, dtype=np.float32)
        self._control_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=[len(self._agent_joints)],
            dtype=np.float32)
        if self._with_language:
            self._seq_length = 20
            sequence_space = DiscreteSequence(self._teacher.vocab_size,
                                              self._seq_length)
            self.observation_space = gym.spaces.Dict(
                data=obs_data_space, sequence=sequence_space)
            self.action_space = gym.spaces.Dict(
                control=self._control_space, sequence=sequence_space)
        else:
            self.observation_space = obs_data_space
            self.action_space = self._control_space
        self.reset()

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
        self._world.reset()
        self._teacher.reset(self._agent, self._world)
        self._random_move_objects()
        teacher_action = self._teacher.teach("")
        if self._with_language:
            obs_data = self._get_observation()
            seq = self._teacher.sentence_to_sequence(teacher_action.sentence,
                                                     self._seq_length)
            obs = OrderedDict(data=obs_data, sequence=seq)
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
            goal = self._world.get_model(self._teacher_task.get_goal_name())
            goal_pos = np.array(goal.get_pose()[0]).flatten()
            agent_pose = np.array(self._agent.get_pose()).flatten()
            agent_vel = np.array(self._agent.get_velocities()[0]).flatten()
            joint_vel = []
            for joint_id in range(len(self._agent_joints)):
                joint_name = self._agent_joints[joint_id]
                joint_state = self._agent.get_joint_state(joint_name)
                joint_vel.append(joint_state.get_velocities())
            joint_vel = np.array(joint_vel).flatten()
            obs_data = np.concatenate(
                (goal_pos, agent_pose, agent_vel, joint_vel), axis=0)
            obs_data = np.array(obs_data).reshape(-1)
        return obs_data

    def step(self, action):
        """
        Args:
            action (dict|int): If with_language, action is a dictionary 
                    with key "control" and "sequence".
                    action['control'] is a vector whose dimention is
                    len(_joint_names). action['sequence'] is a sentence.
                    If not with_language, it is an int for the action id.
        Returns:
            If with_language, it is a dictionary with key 'data' and 'sequence'
            If not with_language, it is a numpy.array or image for observation
        """
        if self._with_language:
            sequence = action.get('sequence', None)
            sentence = self._teacher.sequence_to_sentence(sequence)
            controls = action['control'] * self._agent_control_range
        else:
            sentence = ''
            controls = action * self._agent_control_range
        controls = dict(zip(self._agent_joints, controls))
        teacher_action = self._teacher.teach(sentence)
        self._agent.take_action(controls)
        self._world.step(50)
        if self._with_language:
            obs_data = self._get_observation()
            seq = self._teacher.sentence_to_sequence(teacher_action.sentence,
                                                     self._seq_length)
            obs = OrderedDict(data=obs_data, sequence=seq)
        else:
            obs = self._get_observation()
        self._steps_in_this_episode += 1
        self._cum_reward += teacher_action.reward
        if teacher_action.done:
            logging.debug("episode ends at cum reward:" +
                          str(self._cum_reward))
        return obs, teacher_action.reward, teacher_action.done, {}

    def _insert_agent_to_world_file(self, world_file, model):
        content = world_file.read()
        insert_pos = content.find("<!-- Static Objects -->")
        content = list(content)
        insert_str = "<include> <uri>model://"+model+"</uri> </include>\n"
        content.insert(insert_pos, insert_str)
        return "".join(content)

    def _insert_objects(self, object_list):
        for obj_id in range(len(object_list)):
            model_name = object_list[obj_id]
            self._world.insertModelFile('model://' + model_name)
            logging.debug('model ' + model_name + ' inserted')
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


def main():
    """
    Simple testing of this environment.
    """
    import matplotlib.pyplot as plt
    with_language = False
    use_image_obs = False
    random_goal = False
    fig = None
    env = GroceryGround(
        with_language=with_language,
        use_image_obs=use_image_obs,
        agent_type='icub_with_hands',
        random_goal=random_goal)
    env.render()
    while True:
        actions = np.array(np.random.randn(env._control_space.shape[0]))
        if with_language:
            seq = env._teacher.sentence_to_sequence("hello", env._seq_length)
            actions = dict(control=actions, sequence=seq)
        obs, _, done, _ = env.step(actions)
        if with_language and (env._steps_in_this_episode == 1 or done):
            seq = obs["sequence"]
            logging.info("sequence: " + str(seq))
            logging.info("sentence: " + env._teacher.sequence_to_sentence(seq))
        if use_image_obs:
            if fig is None:
                fig = plt.imshow(obs)
            else:
                fig.set_data(obs)
            plt.pause(0.00001)
        if done:
            env.reset()


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    main()
