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
from abc import abstractmethod
import math
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
from social_bot import teacher_tasks
from social_bot.envs.gazebo_base import GazeboEnvBase
from social_bot.teacher import TaskGroup
from social_bot.teacher import TeacherAction
from social_bot.teacher_tasks import GoalTask
import social_bot.pygazebo as gazebo


class GroceryGroundTaskBase(teacher.Task):
    """
    A base task for grocery ground environment.
    """

    def __init__(self):
        self._agent = None
        self._world = None
        self._agent_name = None

    def setup(self, world, agent_name):
        """
        Setting things up during the initialization
        """
        self._world = world
        self._agent = self._world.get_agent()
        self._agent_name = agent_name


@gin.configurable
class GroceryGroundGoalTask(GroceryGroundTaskBase, GoalTask):
    """
    A simple task to find a goal on grocery ground.
    The goal of this task is to train the agent to navigate to an object.
    The name of the object is provided by the teacher. In each
    episode, the location of the goal object is randomly chosen.
    """

    def __init__(self,
                 max_steps=500,
                 goal_name="ball",
                 success_distance_thresh=0.5,
                 fail_distance_thresh=3,
                 random_range=10.0,
                 random_goal=False,
                 sparse_reward=False,
                 reward_weight=1.0):
        """
        Args:
            max_steps (int): episode will end if not reaching goal in so many steps, typically should be
                higher than max_episode_steps when register to gym, so that return of last step could be
                handled correctly
            goal_name (string): name of the goal in the world
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            fail_distance_thresh (float): if the agent moves away from the goal more than this distance,
                it's considered a failure and is givne reward -1
            random_range (float): the goal's random position range
            sparse_reward (bool): if true, the reward is -1/0/1, otherwise the 0 case will be replaced
                with normalized distance the agent get closer to goal.
            random_goal (bool): if ture, teacher will randomly select goal from the object list each episode
        """
        assert goal_name is not None, "Goal name needs to be set, not None."
        GoalTask.__init__(
            self,
            max_steps=max_steps,
            goal_name=goal_name,
            success_distance_thresh=success_distance_thresh,
            fail_distance_thresh=fail_distance_thresh,
            sparse_reward=sparse_reward,
            random_range=random_range)
        GroceryGroundTaskBase.__init__(self)
        self._random_goal = random_goal
        self._objects_in_world = [
            'placing_table', 'plastic_cup_on_table', 'coke_can_on_table',
            'hammer_on_table', 'cafe_table', 'ball'
        ]
        self._objects_to_insert = [
            'coke_can', 'table', 'bookshelf', 'car_wheel', 'plastic_cup',
            'beer', 'hammer'
        ]
        logging.debug("goal_name %s, random_goal %d, fail_distance_thresh %f",
            self._goal_name, self._random_goal, fail_distance_thresh)
        self._pos_list = list(itertools.product(range(-5, 5), range(-5, 5)))
        self._pos_list.remove((0, 0))
        self.reward_weight = reward_weight
        self.task_vocab = self.task_vocab + self._objects_in_world + self._objects_to_insert

    def setup(self, world, agent_name):
        """
        Setting things up during the initialization
        """
        GroceryGroundTaskBase.setup(self, world, agent_name)
        self._insert_objects(self._objects_to_insert)

    def run(self, agent, world):
        self._random_move_objects()
        if self._random_goal:
            random_id = random.randrange(len(self._objects_to_insert))
            self.set_goal_name(self._objects_to_insert[random_id])
        yield from GoalTask.run(self, agent, world)

    def _insert_objects(self, object_list):
        obj_num = len(object_list)
        for obj_id in range(obj_num):
            model_name = object_list[obj_id]
            self._world.insertModelFile('model://' + model_name)
            logging.debug('model ' + model_name + ' inserted')
            self._world.step(20)
            # Sleep for a while waiting for Gazebo server to finish the inserting
            # operation. Or the model may not be completely inserted, boost will
            # throw 'px!=0' error when set_pose/get_pose of the model is called
            time.sleep(0.2)

    def _random_move_objects(self, random_range=10.0):
        obj_num = len(self._objects_to_insert)
        obj_pos_list = random.sample(self._pos_list, obj_num)
        for obj_id in range(obj_num):
            model_name = self._objects_to_insert[obj_id]
            loc = (obj_pos_list[obj_id][0], obj_pos_list[obj_id][1], 0)
            pose = (np.array(loc), (0, 0, 0))
            self._world.get_model(model_name).set_pose(pose)

    def task_specific_observation(self):
        """
        Args:
            None
        Returns:
            np.array of the extra observations should be added into the
            observation besides self states, for the non-image case
        """
        goal = self._world.get_model(self._goal_name)
        return np.array(goal.get_pose()[0]).flatten()


@gin.configurable
class ICubAuxiliaryTask(GroceryGroundTaskBase):
    """
    An auxiliary task spicified for iCub, to keep the agent from falling down
        and to encourage the agent walk
    """

    def __init__(self,
                 reward_weight=1.0,
                 step_time=0.05,
                 target=None,
                 agent_init_pos=(0, 0),
                 agent_pos_random_range=0):
        """
        Args:
            reward_weight (float): the weight of the reward, should be tuned
                accroding to reward range of other tasks 
            step_time (float): used to caculate speed of the agent
            target (string): this is the target icub should face towards, since
                you may want the agent interact with something
            agent_init_pos (tuple): the expected initial position of the agent
            pos_random_range (float): random range of the initial position
        """
        super().__init__()
        self.reward_weight = reward_weight
        self.task_vocab = ['icub']
        self._step_time = step_time
        self._target_name = target
        self._pre_agent_pos = np.array([0, 0, 0], dtype=np.float32)
        self._agent_init_pos = agent_init_pos
        self._random_range = agent_pos_random_range

    def setup(self, world, agent_name):
        """
        Setting things up during the initialization
        """
        super().setup(world, agent_name)
        if self._target_name:
            self._target = world.get_agent(self._target_name)
        with open (os.path.join(social_bot.get_model_dir(), "agent_cfg.json"),
                   'r') as cfg_file:
            agent_cfgs = json.load(cfg_file)
        self._joints = agent_cfgs[self._agent_name]['control_joints']

    def run(self, agent, world):
        """
        Start a teaching episode for this task.
        Args:
            agent (pygazebo.Agent): the learning agent 
            world (pygazebo.World): the simulation world
        """
        self._pre_agent_pos = self.get_icub_extra_obs(agent)[:3]
        agent_sentence = yield
        done = False
        # set icub random initial pose
        x = self._agent_init_pos[0] + random.random() * self._random_range
        y = self._agent_init_pos[1] + random.random() * self._random_range
        orient = (random.random() - 0.5) * np.pi
        agent.set_pose((np.array([x, y, 0.6]), np.array([0, 0, orient])))
        while not done:
            # reward for not falling (alive reward)
            agent_height = np.array(agent.get_link_pose('iCub::head'))[0][2]
            done = agent_height < 0.7  # fall down
            standing_reward = agent_height
            # movement cost, to avoid uncessary movements
            joint_pos = []
            for joint_name in self._joints:
                joint_state = self._agent.get_joint_state(joint_name)
                joint_pos.append(joint_state.get_positions())
            joint_pos = np.array(joint_pos).flatten()
            movement_cost = np.sum(np.abs(joint_pos)) / joint_pos.shape[0]
            # orientation cost, the agent should face towards the target
            if self._target_name:
                agent_pos = self.get_icub_extra_obs(agent)[:3]
                head_angle = self._get_angle_to_target(agent_pos, 'iCub::head')
                root_angle = self._get_angle_to_target(agent_pos, 'iCub::root_link')
                l_foot_angle = self._get_angle_to_target(
                    agent_pos, 'iCub::l_leg::l_foot', np.pi)
                r_foot_angle = self._get_angle_to_target(
                    agent_pos, 'iCub::r_leg::r_foot', np.pi)
                orient_cost = (np.abs(head_angle) + np.abs(root_angle) +
                            np.abs(l_foot_angle) + np.abs(r_foot_angle)) / 4
            else:
                orient_cost = 0
            # sum all
            reward = standing_reward - 0.5 * movement_cost - 0.2 * orient_cost
            agent_sentence = yield TeacherAction(reward=reward, done=done)

    @staticmethod
    def get_icub_extra_obs(icub_agent):
        """
        Get contacts_to_ground, pose of key ponit of icub and center of them.
        A static method, other task can use this to get additional icub info.
        Args:
            the agent
        Returns:
            np.array of the extra observations of icub, including average pos
        """

        def _get_contacts_to_ground(icub_agent, contacts_sensor):
            contacts = icub_agent.get_collisions(contacts_sensor)
            for collision in contacts:
                if collision[1] == 'ground_plane::link::collision':
                    return True
            return False

        root_pose = np.array(
            icub_agent.get_link_pose('iCub::root_link')).flatten()
        chest_pose = np.array(
            icub_agent.get_link_pose('iCub::chest')).flatten()
        l_foot_pose = np.array(
            icub_agent.get_link_pose('iCub::l_leg::l_foot')).flatten()
        r_foot_pose = np.array(
            icub_agent.get_link_pose('iCub::r_leg::r_foot')).flatten()
        foot_contacts = np.array([
            _get_contacts_to_ground(icub_agent, "l_foot_contact_sensor"),
            _get_contacts_to_ground(icub_agent, "r_foot_contact_sensor")
        ]).astype(np.float32)
        average_pos = np.sum([
            root_pose[0:3], chest_pose[0:3], l_foot_pose[0:3], r_foot_pose[0:3]
        ],
                             axis=0) / 4.0
        obs = np.concatenate((average_pos, root_pose, chest_pose, l_foot_pose,
                              r_foot_pose, foot_contacts))
        return obs

    def _get_angle_to_target(self, agent_pos, link_name, offset=0):
        """
        Get angle from a icub link, relative to target.
        Args:
            agent_pos (numpay array): the pos of agent
            link_name (string): link name of the agent
            offset (float): the yaw offset of link, for some links have initial internal rotation
        Returns:
            float, angle to target
        """
        yaw = self._agent.get_link_pose(link_name)[1][2]
        yaw = (yaw + offset) % (
            2 * np.pi
        ) - np.pi  # model icub has a globle built-in 180 degree rotation
        target_pos, _ = self._target.get_pose()
        walk_target_theta = np.arctan2(target_pos[1] - agent_pos[1],
                                       target_pos[0] - agent_pos[0])
        angle_to_target = walk_target_theta - yaw
        # wrap the range to [-pi, pi)
        angle_to_target = (angle_to_target + np.pi) % (2 * np.pi) - np.pi
        return angle_to_target

    def task_specific_observation(self):
        """
        Args:
            None
        Returns:
            np.array of the extra observations should be added into the
            observation besides self states, for the non-image case
        """
        icub_extra_obs = self.get_icub_extra_obs(self._agent)
        if self._target_name:
            agent_pos = icub_extra_obs[:3]
            agent_speed = (agent_pos - self._pre_agent_pos) / self._step_time
            self._pre_agent_pos = agent_pos
            yaw = self._agent.get_link_pose('iCub::root_link')[1][2]
            angle_to_target = self._get_angle_to_target(agent_pos,
                                                        'iCub::root_link')
            rot_minus_yaw = np.array([[np.cos(-yaw), -np.sin(-yaw), 0],
                                    [np.sin(-yaw), np.cos(-yaw), 0], [0, 0, 1]])
            vx, vy, vz = np.dot(rot_minus_yaw, agent_speed)  # rotate to agent view
            orientation_ob = np.array(
                [np.sin(angle_to_target),
                np.cos(angle_to_target), vx, vy, vz], dtype=np.float32)
            return np.concatenate([icub_extra_obs] + [orientation_ob])
        else:
            return icub_extra_obs


@gin.configurable
class GroceryGroundKickBallTask(GroceryGroundTaskBase, GoalTask):
    """
    A simple task to kick a ball to the goal. Simple reward shaping is used to
    guide the agent run to the ball first:
        Agent will receive 100 when succefully kick the ball into the goal.
        Agent will receive the speed of getting closer to the ball before touching the
            ball within 45 degrees of agent direction. The reward is trunked within
            parameter target_speed.
        Agent will receive negative normalized distance from ball to goal after
            touching the ball within the direction. An offset of "target_speed + 1" is
            included since touching the goal must be better than not touching.
    """

    def __init__(self,
                 max_steps=500,
                 goal_name="goal",
                 success_distance_thresh=0.5,
                 fail_distance_thresh=0.5,
                 random_range=5.0,
                 target_speed=2.0,
                 step_time=0.1,
                 reward_weight=1.0):
        """
        Args:
            max_steps (int): episode will end if not reaching goal in so many steps
            goal_name (string): name of the goal in the world
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            fail_distance_thresh (float): if the agent moves away from the goal more than this distance,
                it's considered a failure and is given reward -1
            random_range (float): the goal's random position range
            target_speed (float): the target speed runing to the ball. The agent will receive no more 
                higher reward when its speed is higher than target_speed.
            step_time (float): used to caculate speed of the agent
            reward_weight (float): the weight of the reward
        """
        GoalTask.__init__(
            self,
            max_steps=max_steps,
            goal_name=goal_name,
            fail_distance_thresh=fail_distance_thresh,
            random_range=random_range)
        GroceryGroundTaskBase.__init__(self)
        self._goal_name = 'goal'
        self._success_distance_thresh = success_distance_thresh,
        self._objects_in_world = [
            'placing_table', 'plastic_cup_on_table', 'coke_can_on_table',
            'hammer_on_table', 'cafe_table', 'ball'
        ]
        self._step_time = step_time
        self._target_speed = target_speed
        self.reward_weight = reward_weight
        self.task_vocab = self.task_vocab + self._objects_in_world

    def setup(self, world, agent_name):
        """
        Setting things up during the initialization
        """
        GroceryGroundTaskBase.setup(self, world, agent_name)
        goal_sdf = """
        <?xml version='1.0'?>
        <sdf version ='1.4'>
        <model name ='goal'>
            <static>1</static>
            <include>
                <uri>model://robocup_3Dsim_goal</uri>
            </include>
            <pose frame=''>-5.0 0 0 0 -0 3.14159265</pose>
        </model>
        </sdf>
        """
        self._world.insertModelFromSdfString(goal_sdf)
        time.sleep(0.2)
        self._world.step(20)

    def run(self, agent, world):
        """
        Start a teaching episode for this task.
        Args:
            agent (pygazebo.Agent): the learning agent
            world (pygazebo.World): the simulation world
        """
        agent_sentence = yield
        goal = world.get_agent(self._goal_name)
        ball = world.get_agent('ball')
        goal_loc, dir = goal.get_pose()
        self._move_goal(ball, np.array(goal_loc))
        agent_loc, dir = agent.get_pose()
        ball_loc, _ = ball.get_pose()
        prev_dist = np.linalg.norm(
            np.array(ball_loc)[:2] - np.array(agent_loc)[:2])
        init_goal_dist = np.linalg.norm(
            np.array(ball_loc)[:2] - np.array(goal_loc)[:2])
        steps = 0
        hitted_ball = False
        while steps < self._max_steps:
            steps += 1
            if not hitted_ball:
                agent_loc, dir = agent.get_pose()
                if self._agent_name.find('icub') != -1:
                    # For agent icub, we need to use the average pos here
                    agent_loc = ICubAuxiliaryTask.get_icub_extra_obs(self._agent)[:3]
                ball_loc, _ = ball.get_pose()
                dist = np.linalg.norm(
                    np.array(ball_loc)[:2] - np.array(agent_loc)[:2])
                # distance/step_time so that number is in m/s, trunk to target_speed
                progress_reward = min(self._target_speed,
                                      (prev_dist - dist) / self._step_time)
                prev_dist = dist
                if dist < 0.3:
                    dir = np.array([math.cos(dir[2]), math.sin(dir[2])])
                    goal_dir = (np.array(ball_loc[0:2]) - np.array(
                        agent_loc[0:2])) / dist
                    dot = sum(dir * goal_dir)
                    if dot > 0.707:
                        # within 45 degrees of the agent direction
                        hitted_ball = True
                agent_sentence = yield TeacherAction(reward=progress_reward)
            else:
                goal_loc, _ = goal.get_pose()
                ball_loc, _ = ball.get_pose()
                dist = np.linalg.norm(np.array(ball_loc)[:2] - np.array(goal_loc)[:2])
                if dist < self._success_distance_thresh:
                    agent_sentence = yield TeacherAction(
                        reward=100.0, sentence="well done", done=True)
                else:
                    agent_sentence = yield TeacherAction(
                        reward=self._target_speed + 3 - dist / init_goal_dist)
        yield TeacherAction(reward=-1.0, sentence="failed", done=True)

    def task_specific_observation(self):
        model_list = [
            'ball',
            'goal',
        ]
        model_poss = []
        model_vels = []
        for model_id in range(len(model_list)):
            model = self._world.get_model(model_list[model_id])
            model_poss.append(model.get_pose()[0])
            model_vels.append(model.get_velocities()[0])
        model_poss = np.array(model_poss).flatten()
        model_vels = np.array(model_vels).flatten()
        return np.concatenate((model_poss, model_vels), axis=0)


@gin.configurable
class GroceryGround(GazeboEnvBase):
    """
    The envionment support agent type of pr2_noplugin, pioneer2dx_noplugin,
    turtlebot, icub, and irobot create for now. Note that for the models without
    camera sensor (like irobot create), you can not use image as observation.

    Joints of the agent are controllable by force or pid controller,

    The observation space is a numpy array or a dict with keys 'image',
    'states', 'sentence', depends on the configuration.
    If without language and internal_states, observation is a numpy array:
        pure image (use_image_observation=True)
        pure low-dimentional states (use_image_observation=False)
    Otherwise observation is a dict, it could be:
        image and internal states (part of the low-dimentional states)
        image and language sequence
        image, internal states and language sequence
        pure low-dimensional states and language sequence

    The objects are rearranged each time the environment is reseted.

    Agent will receive a reward provided by the teacher. The goal's position
    can also be controlled by the teacher.

    """

    def __init__(self,
                 with_language=False,
                 use_image_observation=False,
                 image_with_internal_states=False,
                 task_name='goal',
                 agent_type='pioneer2dx_noplugin',
                 world_time_precision=None,
                 step_time=0.1,
                 port=None,
                 action_cost=0.0,
                 resized_image_size=(64, 64),
                 image_data_format='channels_last',
                 vocab_sequence_length=20):
        """
        Args:
            with_language (bool): The observation will be a dict with an extra sentence
            use_image_observation (bool): Use image, or use low-dimentional states as
                observation. Poses in the states observation are in world coordinate
            image_with_internal_states (bool): If true, the agent's self internal states
                i.e., joint position and velocities would be available together with image.
                Only affect if use_image_observation is true
            task_name (string): the teacher task, now there are 2 tasks,
                a simple goal task: 'goal'
                a simple kicking ball task: 'kickball'
            agent_type (string): Select the agent robot, supporting pr2_noplugin,
                pioneer2dx_noplugin, turtlebot, irobot create and icub_with_hands for now
                note that 'agent_type' should be the same str as the model's name
            world_time_precision (float|None): if not none, the time precision of
                simulator, i.e., the max_step_size defined in the agent cfg file, will be
                override. e.g., '0.002' for a 2ms sim step
            step_time (float): the peroid of one step of the environment.
                step_time / world_time_precision is how many simulator substeps during one
                environment step. for some complex agent, i.e., icub, using step_time of 0.05 is better
            port: Gazebo port, need to specify when run multiple environment in parallel
            action_cost (float): Add an extra action cost to reward, which helps to train
                an energy/forces efficency policy or reduce unnecessary movements
            resized_image_size (None|tuple): If None, use the original image size
                from the camera. Otherwise, the original image will be resized
                to (width, height)
            image_data_format (str):  One of `channels_last` or `channels_first`.
                The ordering of the dimensions in the images.
                `channels_last` corresponds to images with shape
                `(height, width, channels)` while `channels_first` corresponds
                to images with shape `(channels, height, width)`.
        """

        with open (os.path.join(social_bot.get_model_dir(), "agent_cfg.json"),
                   'r') as cfg_file:
            agent_cfgs = json.load(cfg_file)
        agent_cfg = agent_cfgs[agent_type]

        wf_path = os.path.join(social_bot.get_world_dir(),
                               "grocery_ground.world")
        with open(wf_path, 'r+') as world_file:
            world_string = self._insert_agent_to_world_file(
                world_file, agent_type)
        if world_time_precision is None:
            world_time_precision = agent_cfg['max_sim_step_time']
        sub_steps = int(round(step_time / world_time_precision))
        sim_time_cfg = ["//physics//max_step_size=" + str(world_time_precision)]

        super(GroceryGround, self).__init__(
            world_string=world_string, world_config=sim_time_cfg, port=port)

        self._teacher = teacher.Teacher(task_groups_exclusive=False)
        if task_name is None or task_name == 'goal':
            main_task = GroceryGroundGoalTask()
        elif task_name == 'kickball':
            main_task = GroceryGroundKickBallTask(step_time=step_time)
        else:
            logging.debug("upsupported task name: " + task_name)

        main_task_group = TaskGroup()
        main_task_group.add_task(main_task)
        self._teacher.add_task_group(main_task_group)
        if agent_type.find('icub') != -1:
            icub_aux_task_group = TaskGroup()
            icub_standing_task = ICubAuxiliaryTask(step_time=step_time)
            icub_aux_task_group.add_task(icub_standing_task)
            self._teacher.add_task_group(icub_aux_task_group)
        self._teacher._build_vocab_from_tasks()
        self._seq_length = vocab_sequence_length
        if self._teacher.vocab_size:
            # using MultiDiscrete instead of DiscreteSequence so gym
            # _spec_from_gym_space won't complain.
            self._sentence_space = gym.spaces.MultiDiscrete(
                [self._teacher.vocab_size] * self._seq_length)
        self._sub_steps = sub_steps

        self._world.step(20)
        self._agent = self._world.get_agent()
        for task_group in self._teacher.get_task_groups():
            for task in task_group.get_tasks():
                task.setup(self._world, agent_type)

        logging.debug(self._world.info())
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

        self._action_cost = action_cost
        self._with_language = with_language
        self._use_image_obs = use_image_observation
        self._image_with_internal_states = self._use_image_obs and image_with_internal_states
        assert image_data_format in ('channels_first', 'channels_last')
        self._data_format = image_data_format
        self._resized_image_size = resized_image_size
        self._substep_time = world_time_precision

        self._control_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=[len(self._agent_joints)],
            dtype=np.float32)
        if self._with_language:
            self.action_space = gym.spaces.Dict(
                control=self._control_space, sentence=self._sentence_space)
        else:
            self.action_space = self._control_space

        self.reset()
        obs_sample = self._get_observation_with_sentence("hello")
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
        # the first call of "teach() after "done" will reset the task
        teacher_action = self._teacher.teach("")
        # Give an intilal random pose offset by take random action
        actions = self._control_space.sample()
        controls = dict(
            zip(self._agent_joints, self._agent_control_range * actions))
        self._agent.take_action(controls)
        self._world.step(self._sub_steps)
        obs = self._get_observation_with_sentence(teacher_action.sentence)
        return obs

    def step(self, action):
        """
        Args:
            action (dict|int): If with_language, action is a dictionary
                    with key "control" and "sentence".
                    action['control'] is a vector whose dimention is
                    len(_joint_names). action['sentence'] is a sentence sequence.
                    If not with_language, it is an int for the action id.
        Returns:
            If with_language, it is a dictionary with key 'data' and 'sentence'
            If not with_language, it is a numpy.array or image for observation
        """
        if self._with_language:
            sentence = action.get('sentence', None)
            if type(sentence) != str:
                sentence = self._teacher.sequence_to_sentence(sentence)
            action_ctrl = action['control']
        else:
            sentence = ''
            action_ctrl = action
        controls = np.clip(action_ctrl, -1.0, 1.0) * self._agent_control_range
        controls = dict(zip(self._agent_joints, controls))
        self._agent.take_action(controls)
        self._world.step(self._sub_steps)
        teacher_action = self._teacher.teach(sentence)
        obs = self._get_observation_with_sentence(teacher_action.sentence)
        self._steps_in_this_episode += 1
        ctrl_cost = np.sum(np.square(action_ctrl)) / action_ctrl.shape[0]
        reward = teacher_action.reward - self._action_cost * ctrl_cost
        self._cum_reward += reward
        if teacher_action.done:
            logging.debug("episode ends at cum reward:" +
                          str(self._cum_reward))
        return obs, reward, teacher_action.done, {}

    def _insert_agent_to_world_file(self, world_file, model):
        content = world_file.read()
        insert_pos = content.find("<!-- AGENT-INSERTION-POINT -->")
        assert insert_pos != -1, "Can not found insertion point in world file"
        content = list(content)
        insert_str = "<include> <uri>model://" + model + "</uri> </include>\n"
        content.insert(insert_pos, insert_str)
        return "".join(content)

    def _get_camera_observation(self):
        image = np.array(
            self._agent.get_camera_observation(self._agent_camera), copy=False)
        if self._resized_image_size:
            image = PIL.Image.fromarray(image).resize(self._resized_image_size,
                                                      PIL.Image.ANTIALIAS)
            image = np.array(image, copy=False)
        if self._data_format == "channels_first":
            image = np.transpose(image, [2, 0, 1])
        return image

    def _get_low_dim_full_states(self):
        task_specific_ob = self._teacher.get_task_pecific_observation()
        agent_pose = np.array(self._agent.get_pose()).flatten()
        agent_vel = np.array(self._agent.get_velocities()).flatten()
        internal_states = self._get_internal_states(self._agent,
                                                    self._agent_joints)
        obs = np.concatenate(
            (task_specific_ob, agent_pose, agent_vel, internal_states), axis=0)
        return obs

    def _create_observation_dict(self, sentence_raw):
        obs = OrderedDict()
        if self._use_image_obs:
            obs['image'] = self._get_camera_observation()
            if self._image_with_internal_states:
                obs['states'] = self._get_internal_states(
                    self._agent, self._agent_joints)
        else:
            obs['states'] = self._get_low_dim_full_states()
        if self._with_language:
            obs['sentence'] = self._teacher.sentence_to_sequence(
                sentence_raw, self._seq_length)
        return obs

    def _get_observation_with_sentence(self, sentence_raw):
        if self._image_with_internal_states or self._with_language:
            # observation is an OrderedDict
            obs = self._create_observation_dict(sentence_raw)
        elif self._use_image_obs:  # observation is pure image
            obs = self._get_camera_observation()
        else:  # observation is pure low-dimentional states
            obs = self._get_low_dim_full_states()
        return obs


class GroceryGroundImage(GroceryGround):
    def __init__(self, port=None):
        super(GroceryGroundImage, self).__init__(
            use_image_observation=True,
            image_with_internal_states=False,
            with_language=False,
            port=port)


class GroceryGroundLanguage(GroceryGround):
    def __init__(self, port=None):
        super(GroceryGroundLanguage, self).__init__(
            use_image_observation=False,
            image_with_internal_states=False,
            with_language=True,
            port=port)


class GroceryGroundImageLanguage(GroceryGround):
    def __init__(self, port=None):
        super(GroceryGroundImageLanguage, self).__init__(
            use_image_observation=True,
            image_with_internal_states=False,
            with_language=True,
            port=port)


class GroceryGroundImageSelfStatesLanguage(GroceryGround):
    def __init__(self, port=None):
        super(GroceryGroundImageSelfStatesLanguage, self).__init__(
            use_image_observation=True,
            image_with_internal_states=True,
            with_language=True,
            port=port)


def main():
    """
    Simple testing of this environment.
    """
    import matplotlib.pyplot as plt
    import time
    with_language = True
    use_image_obs = False
    image_with_internal_states = True
    fig = None
    env = GroceryGround(
        with_language=with_language,
        use_image_observation=use_image_obs,
        image_with_internal_states=image_with_internal_states,
        agent_type='pr2_noplugin',
        task_name='goal')
    env.render()
    step_cnt = 0
    last_done_time = time.time()
    while True:
        actions = env._control_space.sample()
        if with_language:
            actions = dict(control=actions, sentence="hello")
        obs, _, done, _ = env.step(actions)
        step_cnt += 1
        if with_language and (env._steps_in_this_episode == 1 or done):
            seq = obs["sentence"]
            logging.info("sentence_seq: " + str(seq))
            logging.info("sentence_raw: " +
                         env._teacher.sequence_to_sentence(seq))
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
            step_per_sec = step_cnt / (time.time()-last_done_time)
            logging.info("step per second: " + str(step_per_sec))
            step_cnt = 0
            last_done_time = time.time()

if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main()
