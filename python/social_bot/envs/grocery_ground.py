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
        self._goal_name = 'cube_20k'
        self._object_list = [
            'coke_can', 'table', 'bookshelf', 'cube_20k', 'car_wheel',
            'plastic_cup', 'beer', 'hammer'
        ]
        self.task_vocab = self.task_vocab + self._object_list

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
    sensor (irobot create), you can not use image as observation.

    Joints of the agent are controllable by force or pid controller,

    The observation space is a numpy array or a dict with keys 'image', 
    'states', 'sequence', depends on the configuration.
    If without language and internal_states, observation is a numpy array:
        pure image (use_image_observation=True)
        pure low-dimentional states (use_image_observation=False)
    Otherwise observation is a dict, it could be:
        image with internal states (part of the low-dimentional states)
        image with language sequence
        image with both internal states and language sequence
        pure low-dimensional states with language sequence

    The objects are rearranged each time the environment is reseted.
    
    Agent will receive a reward provided by the teacher. The goal's position is
    can also be controlled by the teacher.

    """

    def __init__(self,
                 with_language=False,
                 use_image_observation=False,
                 image_with_internal_states=False,
                 random_goal=False,
                 use_pid=False,
                 agent_type='pioneer2dx_noplugin',
                 max_steps=160,
                 port=None,
                 resized_image_size=(64, 64),
                 data_format='channels_last'):
        """
        Args:
            with_language (bool): The observation will be a dict with an extra sentence
            use_image_observation (bool): Use image, or use low-dimentional states as 
                observation. Poses in the states observation are in world coordinate
            image_with_internal_states (bool): If true, the agent's self internal states
                i.e., joint position and velocities would be available together with image.
                Only affect if use_image_observation is true
            random_goal (bool): If ture, teacher will randomly select goal from the 
                object list each episode
            use_pid (bool): If ture, joints will be equipped with PID controller, action 
                is the target velocity. Otherwise joints are directly controlled by force.
            agent_type (string): Select the agent robot, supporting pr2_differential, 
                pioneer2dx_noplugin, turtlebot, and irobot create for now
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
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(), "grocery_ground.world"))

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
        self._seq_length = 20
        self._sequence_space = DiscreteSequence(self._teacher.vocab_size,
                                                self._seq_length)

        self._object_list = self._teacher_task.get_object_list()
        self._pos_list = list(itertools.product(range(-5, 5), range(-5, 5)))
        self._pos_list.remove((0, 0))
        self._world.info()
        self._world.insertModelFile('model://' + agent_type)
        self._world.step(20)
        time.sleep(0.1)  # Avoid 'px!=0' error
        self._random_insert_objects()
        logging.debug(self._world.model_list_info())
        logging.debug(self._world.info())
        agent_cfgs = json.load(
            open(
                os.path.join(social_bot.get_model_dir(), "agent_cfg.json"),
                'r'))
        agent_cfg = agent_cfgs[agent_type]
        self._agent = self._world.get_agent(agent_type)
        self._agent_joints = agent_cfg['control_joints']
        if use_pid:
            for _joint in self._agent_joints:
                self._agent.set_pid_controller(_joint, 'velocity', d=0.005)
            self._agent_control_range = 20.0
        else:
            self._agent_control_range = agent_cfg['control_limit']
        self._agent_camera = agent_cfg['camera_sensor']

        logging.debug("joints to control: %s" % self._agent_joints)

        self._with_language = with_language
        self._use_image_obs = use_image_observation
        self._image_with_internal_states = self._use_image_obs and image_with_internal_states
        assert data_format in ('channels_first', 'channels_last')
        self._data_format = data_format
        self._resized_image_size = resized_image_size

        self.reset()
        obs_sample = self._get_observation("hello")
        if self._with_language or self._image_with_internal_states:
            self.observation_space = self._construct_dict_space(
                obs_sample, self._teacher.vocab_size)
        elif self._use_image_obs:
            self.observation_space = gym.spaces.Box(
                low=0, high=255, shape=obs_sample.shape, dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=obs_sample.shape,
                dtype=np.float32)

        self._control_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=[len(self._agent_joints)],
            dtype=np.float32)
        if self._with_language:
            self.action_space = gym.spaces.Dict(
                control=self._control_space, sequence=self._sequence_space)
        else:
            self.action_space = self._control_space

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
        obs = self._get_observation("hello")
        return obs

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
        self._world.step(10)
        obs = self._get_observation(teacher_action.sentence)
        self._steps_in_this_episode += 1
        self._cum_reward += teacher_action.reward
        if teacher_action.done:
            logging.debug("episode ends at cum reward:" +
                          str(self._cum_reward))
        return obs, teacher_action.reward, teacher_action.done, {}

    def _get_camera_observation(self):
        img = np.array(
            self._agent.get_camera_observation(self._agent_camera), copy=False)
        if self._resized_image_size:
            img = PIL.Image.fromarray(img).resize(self._resized_image_size,
                                                  PIL.Image.ANTIALIAS)
            img = np.array(img, copy=False)
        if self._data_format == "channels_first":
            img = np.transpose(img, [2, 0, 1])
        return img

    def _get_low_dim_full_states(self):
        goal = self._world.get_model(self._teacher_task.get_goal_name())
        goal_pos = np.array(goal.get_pose()[0]).flatten()
        agent_pose = np.array(self._agent.get_pose()).flatten()
        agent_vel = np.array(self._agent.get_velocities()[0]).flatten()
        internal_states = self._get_internal_states(self._agent,
                                                    self._agent_joints)
        obs = np.concatenate(
            (goal_pos, agent_pose, agent_vel, internal_states), axis=0)
        return obs

    def _create_observation_dict(self, sentence):
        obs = OrderedDict()
        if self._use_image_obs:
            obs['image'] = self._get_camera_observation()
            if self._image_with_internal_states:
                obs['states'] = self._get_internal_states(
                    self._agent, self._agent_joints)
        else:
            obs['states'] = self._get_low_dim_full_states()
        if self._with_language:
            obs['sequence'] = self._teacher.sentence_to_sequence(
                sentence, self._seq_length)
        return obs

    def _get_observation(self, sentence):
        if self._image_with_internal_states or self._with_language:
            # observation is an OrderedDict
            obs = self._create_observation_dict(sentence)
        elif self._use_image_obs:  # observation is pure image
            obs = self._get_camera_observation()
        else:  # observation is pure low-dimentional states
            obs = self._get_low_dim_full_states()
        return obs

    def _random_insert_objects(self):
        for obj_id in range(len(self._object_list)):
            model_name = self._object_list[obj_id]
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
            self._world.step(5)


def main():
    """
    Simple testing of this environment.
    """
    import matplotlib.pyplot as plt
    with_language = True
    use_image_obs = True
    random_goal = True
    image_with_internal_states = True
    fig = None
    env = GroceryGround(
        with_language=with_language,
        use_image_observation=use_image_obs,
        image_with_internal_states=image_with_internal_states,
        random_goal=random_goal)
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
            if with_language or image_with_internal_states:
                obs = obs['image']
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
