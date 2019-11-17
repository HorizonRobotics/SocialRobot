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
A variety of teacher tasks.
"""

from collections import deque
from abc import abstractmethod
import math
import numpy as np
import os
import gin
import itertools
import time
import random
import json

import social_bot
from social_bot import teacher
from social_bot.teacher import TeacherAction
import social_bot.pygazebo as gazebo

from absl import logging


class Task(object):
    """Base class for Task.

    A Task is for teaching a single task.
    """

    def __init__(self, env, max_steps=200, reward_weight=1.0):
        """
        Setting things up during the initialization

        Args:
            env (social_bot.GazeboEnvBase): an instance of Gym Environment
            reward_weight(float): the weight of reward for caculating final_reward in teacher.teach()
        Returns:
            None
        """
        self._env = env
        self._world = env._world
        self._agent = env._agent
        self._max_steps = max_steps
        self.reward_weight = reward_weight
        self.task_vocab = ['hello', 'well', 'done', 'failed', 'to']

    @abstractmethod
    def run(self):
        """
        run() use yield to generate TeacherAction
        Structure of run():
        ```python
        def run(self):
          ...
          # agent_sentence is provided by Teacher using send() in TaskGroup.teach()
          agent_sentence = yield  # the first yielded value is ignored
          ...
          # TeacherAction will be passed to Teacher as the return value of send() in TaskGroup.teach()
          agent_sentence = yield TeacherAction(...)
          ...
          agent_sentence = yield TeacherAction(...)
          ...
          yield TeacherAction(done=True)
        ```

        Returns:
            None
        """
        pass

    def task_specific_observation(self, agent):
        """
        Args:
            agent (Agent): the agent
        Returns:
            np.array, the extra observations should be added into the observation
            besides original observation from the environment. This can be overide
            by the sub task
        """
        return np.array([])
    
    def set_agent(self, agent):
        """
        The agent can be override by this function. 
        This might be useful when multi agents share the same task or embodied teacher.
        Args:
            agent (Agent): the agent
        """
        self._agent = agent

    def _get_states_of_model_list(self, model_list, including_rotation=False):
        """
        Get the poses and velocities for the models
        Args:
            model_list (list): a list of model names
            including_rotation (bool): if Ture, the rotation of objects (in roll pitch yaw) will be included.
        Returns:
            np.array, the poses and velocities of the models
        """
        model_poss = []
        model_vels = []
        for model_id in range(len(model_list)):
            model = self._world.get_model(model_list[model_id])
            model_poss.append(model.get_pose()[0])
            if including_rotation:
                model_poss.append(model.get_pose()[1])
            model_vels.append(model.get_velocities()[0])
        model_poss = np.array(model_poss).flatten()
        model_vels = np.array(model_vels).flatten()
        return np.concatenate((model_poss, model_vels), axis=0)



@gin.configurable
class GoalTask(Task):
    """
    A simple teacher task to find a goal.
    For this task, the agent will receive reward 1 when it is close enough to the goal.
    If it is moving away from the goal too much or still not close to the goal after max_steps,
    it will get reward -1.
    """

    def __init__(self,
                 env,
                 max_steps,
                 goal_name="ball",
                 distraction_list=[
                     'coke_can', 'table', 'car_wheel', 'plastic_cup', 'beer'
                 ],
                 success_distance_thresh=0.5,
                 fail_distance_thresh=2.0,
                 distraction_penalty_distance_thresh=0,
                 distraction_penalty=0.5,
                 sparse_reward=True,
                 random_range=5.0,
                 random_goal=False,
                 use_curriculum_training=False,
                 start_range=0,
                 increase_range_by_percent=50.,
                 reward_thresh_to_increase_range=0.4,
                 percent_full_range_in_curriculum=0.1,
                 max_reward_q_length=100,
                 reward_weight=1.0):
        """
        Args:
            env (gym.Env): an instance of Environment
            max_steps (int): episode will end if not reaching gaol in so many steps
            goal_name (string): name of the goal in the world
            distraction_list (list of string): a list of model. the model shoud be in gazebo database
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            fail_distance_thresh (float): if the agent moves away from the goal more than this distance,
                it's considered a failure and is given reward -1
            distraction_penalty_distance_thresh (float): if positive, penalize agent getting too close
                to distraction objects (objects that are not the goal itself)
            distraction_penalty (float): positive float of how much to penalize getting too close to
                distraction objects
            sparse_reward (bool): if true, the reward is -1/0/1, otherwise the 0 case will be replaced
                with normalized distance the agent get closer to goal.
            random_range (float): the goal's random position range
            random_goal (bool): if ture, teacher will randomly select goal from the object list each episode
            use_curriculum_training (bool): when true, use curriculum in goal task training
            start_range (float): for curriculum learning, the starting random_range to set the goal
            increase_range_by_percent (float): for curriculum learning, how much to increase random range
                every time agent reached the specified amount of reward.
            reward_thresh_to_increase_range (float): for curriculum learning, how much reward to reach
                before the teacher increases random range.
            percent_full_range_in_curriculum (float): if above 0, randomly throw in x% of training examples
                where random_range is the full range instead of the easier ones in the curriculum.
            max_reward_q_length (int): how many recent rewards to consider when estimating agent accuracy.
            reward_weight (float): the weight of the reward, is used in multi-task case
        """
        super().__init__(
            env=env, max_steps=max_steps, reward_weight=reward_weight)
        self._goal_name = goal_name
        self._success_distance_thresh = success_distance_thresh
        self._fail_distance_thresh = fail_distance_thresh
        self._distraction_penalty_distance_thresh = distraction_penalty_distance_thresh
        self._distraction_penalty = distraction_penalty
        self._sparse_reward = sparse_reward
        self._use_curriculum_training = use_curriculum_training
        self._start_range = start_range
        self._is_full_range_in_curriculum = False
        self._random_goal = random_goal
        self._distraction_list = distraction_list
        self._object_list = distraction_list
        if goal_name not in distraction_list:
            self._object_list.append(goal_name)
        self._goals = self._object_list
        self._pos_list = list(itertools.product(range(-5, 5), range(-5, 5)))
        self._pos_list.remove((0, 0))
        if self.should_use_curriculum_training():
            self._orig_random_range = random_range
            self._random_range = start_range
            self._max_reward_q_length = max_reward_q_length
            self._q = deque(maxlen=max_reward_q_length)
            self._reward_thresh_to_increase_range = reward_thresh_to_increase_range
            self._increase_range_by_percent = increase_range_by_percent
            self._percent_full_range_in_curriculum = percent_full_range_in_curriculum
            logging.info("start_range %f, reward_thresh_to_increase_range %f",
                         self._start_range,
                         self._reward_thresh_to_increase_range)
        else:
            self._random_range = random_range
        self.task_vocab += self._object_list
        self._env.insert_model_list(self._object_list)

    def should_use_curriculum_training(self):
        return (self._use_curriculum_training
                and self._start_range >= self._success_distance_thresh * 1.2)

    def _push_reward_queue(self, value):
        if (not self.should_use_curriculum_training()
                or self._is_full_range_in_curriculum):
            return
        self._q.append(value)
        if (value > 0 and len(self._q) == self._max_reward_q_length
                and sum(self._q) >= self._max_reward_q_length *
                self._reward_thresh_to_increase_range):
            self._random_range *= 1. + self._increase_range_by_percent
            if self._random_range > self._orig_random_range:
                self._random_range = self._orig_random_range
            logging.info("Raising random_range to %f", self._random_range)
            self._q.clear()

    def get_random_range(self):
        return self._random_range

    def run(self):
        """
        Start a teaching episode for this task.
        """
        agent_sentence = yield
        self._agent.reset()
        loc, dir = self._agent.get_pose()
        loc = np.array(loc)
        self._random_move_objects()
        if self._random_goal:
            random_id = random.randrange(len(self._goals))
            self.set_goal_name(self._goals[random_id])
        goal = self._world.get_model(self._goal_name)
        self._move_goal(goal, loc)
        steps_since_last_reward = 0
        while steps_since_last_reward < self._max_steps:
            steps_since_last_reward += 1
            loc, dir = self._agent.get_pose()
            if self._agent.type.find('icub') != -1:
                # For agent icub, we need to use the average pos here
                loc = ICubAuxiliaryTask.get_icub_extra_obs(self._agent)[:3]
            goal_loc, _ = goal.get_pose()
            loc = np.array(loc)
            goal_loc = np.array(goal_loc)
            dist = np.linalg.norm(loc - goal_loc)
            # dir from get_pose is (roll, pitch, roll)
            dir = np.array([math.cos(dir[2]), math.sin(dir[2])])
            goal_dir = (goal_loc[0:2] - loc[0:2]) / dist
            dot = sum(dir * goal_dir)

            distraction_penalty = 0
            if self._distraction_penalty_distance_thresh > 0 and self._distraction_list:
                for obj_name in self._distraction_list:
                    obj = self._world.get_model(obj_name)
                    if obj:
                        obj_loc, obj_dir = obj.get_pose()
                        obj_loc = np.array(obj_loc)
                        distraction_dist = np.linalg.norm(loc - obj_loc)
                        if distraction_dist < self._distraction_penalty_distance_thresh:
                            distraction_penalty += self._distraction_penalty

            if dist < self._success_distance_thresh and dot > 0.707:
                # within 45 degrees of the agent direction
                reward = 1.0 - distraction_penalty
                self._push_reward_queue(reward)
                logging.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                              "dist: " + str(dist))
                agent_sentence = yield TeacherAction(
                    reward=reward, sentence="well done", done=False)
                steps_since_last_reward = 0
                self._move_goal(goal, loc)
            elif dist > self._initial_dist + self._fail_distance_thresh:
                reward = -1.0 - distraction_penalty
                self._push_reward_queue(0)
                logging.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                              "dist: " + str(dist))
                yield TeacherAction(
                    reward=reward, sentence="failed", done=True)
            else:
                if self._sparse_reward:
                    reward = 0
                else:
                    reward = (self._prev_dist - dist) / self._initial_dist
                reward = reward - distraction_penalty
                self._push_reward_queue(reward)
                self._prev_dist = dist
                agent_sentence = yield TeacherAction(
                    reward=reward, sentence=self._goal_name)
        logging.debug("loc: " + str(loc) + " goal: " + str(goal_loc) +
                      "dist: " + str(dist))
        self._push_reward_queue(0)
        yield TeacherAction(reward=-1.0, sentence="failed", done=True)

    def _move_goal(self, goal, agent_loc):
        if (self.should_use_curriculum_training()
                and self._percent_full_range_in_curriculum > 0
                and random.random() < self._percent_full_range_in_curriculum):
            range = self._orig_random_range
            self._is_full_range_in_curriculum = True
        else:
            range = self._random_range
            self._is_full_range_in_curriculum = False
        while True:
            loc = (random.random() * range - range / 2,
                   random.random() * range - range / 2, 0)
            self._initial_dist = np.linalg.norm(loc - agent_loc)
            if self._initial_dist > self._success_distance_thresh:
                break
        self._prev_dist = self._initial_dist
        goal.reset()
        goal.set_pose((loc, (0, 0, 0)))

    def _random_move_objects(self, random_range=10.0):
        obj_num = len(self._object_list)
        obj_pos_list = random.sample(self._pos_list, obj_num)
        for obj_id in range(obj_num):
            model_name = self._object_list[obj_id]
            loc = (obj_pos_list[obj_id][0], obj_pos_list[obj_id][1], 0)
            pose = (np.array(loc), (0, 0, 0))
            self._world.get_model(model_name).set_pose(pose)

    def get_goal_name(self):
        """
        Args:
            None
        Returns:
            Goal's name at this episode
        """
        return self._goal_name

    def set_goal_name(self, goal_name):
        """
        Args:
            Goal's name
        Returns:
            None
        """
        logging.debug('Setting Goal to %s', goal_name)
        self._goal_name = goal_name

    def task_specific_observation(self, agent):
        """
        Args:
            agent (Agent): the agent
        Returns:
            np.array of the extra observations should be added into the
            observation besides self states, for the non-image case
        """
        goal = self._world.get_model(self._goal_name)
        return np.array(goal.get_pose()[0]).flatten()


@gin.configurable
class ICubAuxiliaryTask(Task):
    """
    An auxiliary task spicified for iCub, to keep the agent from falling down
        and to encourage the agent walk
    """

    def __init__(self,
                 env,
                 max_steps,
                 target=None,
                 agent_init_pos=(0, 0),
                 agent_pos_random_range=0,
                 reward_weight=1.0):
        """
        Args:
            env (gym.Env): an instance of Environment
            max_steps (int): episode will end in so many steps
            reward_weight (float): the weight of the reward, should be tuned
                accroding to reward range of other tasks 
            target (string): this is the target icub should face towards, since
                you may want the agent interact with something
            agent_init_pos (tuple): the expected initial position of the agent
            pos_random_range (float): random range of the initial position
        """
        super().__init__(
            env=env, max_steps=max_steps, reward_weight=reward_weight)
        self.task_vocab = ['icub']
        self._target_name = target
        self._pre_agent_pos = np.array([0, 0, 0], dtype=np.float32)
        self._agent_init_pos = agent_init_pos
        self._random_range = agent_pos_random_range
        if self._target_name:
            self._target = self._world.get_model(self._target_name)
        with open(
                os.path.join(social_bot.get_model_dir(), "agent_cfg.json"),
                'r') as cfg_file:
            agent_cfgs = json.load(cfg_file)
        self._joints = agent_cfgs[self._agent.type]['control_joints']

    def run(self):
        """
        Start a teaching episode for this task.
        """
        self._pre_agent_pos = self.get_icub_extra_obs(self._agent)[:3]
        agent_sentence = yield
        done = False
        # set icub random initial pose
        x = self._agent_init_pos[0] + random.random() * self._random_range
        y = self._agent_init_pos[1] + random.random() * self._random_range
        orient = (random.random() - 0.5) * np.pi
        if self._target_name and random.randint(0, 1) == 0:
            # a trick from roboschool humanoid flag run, important to learn to steer
            pos = np.array([x, y, 0.6])
            orient = self._get_angle_to_target(
                pos, self._agent.type + '::root_link', np.pi)
        self._agent.set_pose((np.array([x, y, 0.6]), np.array([0, 0, orient])))
        while not done:
            # reward for not falling (alive reward)
            agent_height = np.array(
                self._agent.get_link_pose(self._agent.type + '::head'))[0][2]
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
                agent_pos = self.get_icub_extra_obs(self._agent)[:3]
                head_angle = self._get_angle_to_target(
                    agent_pos, self._agent.type + '::head')
                root_angle = self._get_angle_to_target(
                    agent_pos, self._agent.type + '::root_link')
                l_foot_angle = self._get_angle_to_target(
                    agent_pos, self._agent.type + '::l_leg::l_foot', np.pi)
                r_foot_angle = self._get_angle_to_target(
                    agent_pos, self._agent.type + '::r_leg::r_foot', np.pi)
                orient_cost = (np.abs(head_angle) + np.abs(root_angle) +
                               np.abs(l_foot_angle) + np.abs(r_foot_angle)) / 4
            else:
                orient_cost = 0
            # sum all
            reward = standing_reward - 0.5 * movement_cost - 0.2 * orient_cost
            agent_sentence = yield TeacherAction(reward=reward, done=done)

    @staticmethod
    def get_icub_extra_obs(agent):
        """
        Get contacts_to_ground, pose of key ponit of icub and center of them.
        A static method, other task can use this to get additional icub info.
        Args:
            the agent
        Returns:
            np.array of the extra observations of icub, including average pos
        """

        def _get_contacts_to_ground(agent, contacts_sensor):
            contacts = agent.get_collisions(contacts_sensor)
            for collision in contacts:
                if collision[1] == 'ground_plane::link::collision':
                    return True
            return False

        root_pose = np.array(
            agent.get_link_pose(agent.name + '::root_link')).flatten()
        chest_pose = np.array(
            agent.get_link_pose(agent.name + '::chest')).flatten()
        l_foot_pose = np.array(
            agent.get_link_pose(agent.name +
                                     '::l_leg::l_foot')).flatten()
        r_foot_pose = np.array(
            agent.get_link_pose(agent.name +
                                     '::r_leg::r_foot')).flatten()
        foot_contacts = np.array([
            _get_contacts_to_ground(agent, "l_foot_contact_sensor"),
            _get_contacts_to_ground(agent, "r_foot_contact_sensor")
        ]).astype(np.float32)
        average_pos = np.sum([
            root_pose[0:3], chest_pose[0:3], l_foot_pose[0:3], r_foot_pose[0:3]
        ],
                             axis=0) / 4.0
        obs = np.concatenate((average_pos, root_pose, chest_pose, l_foot_pose,
                              r_foot_pose, foot_contacts))
        return obs

    def _get_angle_to_target(self, aegnt, agent_pos, link_name, offset=0):
        """
        Get angle from a icub link, relative to target.
        Args:
            agent (Agent): the agent
            agent_pos (numpay array): the pos of agent
            link_name (string): link name of the agent
            offset (float): the yaw offset of link, for some links have initial internal rotation
        Returns:
            float, angle to target
        """
        yaw = aegnt.get_link_pose(link_name)[1][2]
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

    def task_specific_observation(self, agent):
        """
        Args:
            agent (Agent): the agent
        Returns:
            np.array of the extra observations should be added into the
            observation besides self states, for the non-image case
        """
        icub_extra_obs = self.get_icub_extra_obs(agent)
        if self._target_name:
            agent_pos = icub_extra_obs[:3]
            # TODO: be compatible for calling multiple times in one env step
            agent_speed = (
                agent_pos - self._pre_agent_pos) / self._env.get_step_time()
            self._pre_agent_pos = agent_pos
            yaw = agent.get_link_pose(agent.type + '::root_link')[1][2]
            angle_to_target = self._get_angle_to_target(
                agent, agent_pos, agent.type + '::root_link')
            rot_minus_yaw = np.array([[np.cos(-yaw), -np.sin(-yaw), 0],
                                      [np.sin(-yaw),
                                       np.cos(-yaw), 0], [0, 0, 1]])
            vx, vy, vz = np.dot(rot_minus_yaw,
                                agent_speed)  # rotate to agent view
            orientation_ob = np.array(
                [np.sin(angle_to_target),
                 np.cos(angle_to_target), vx, vy, vz],
                dtype=np.float32)
            return np.concatenate([icub_extra_obs] + [orientation_ob])
        else:
            return icub_extra_obs


@gin.configurable
class KickingBallTask(Task):
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
                 env,
                 max_steps,
                 goal_name="goal",
                 success_distance_thresh=0.5,
                 random_range=4.0,
                 target_speed=2.0,
                 reward_weight=1.0):
        """
        Args:
            env (gym.Env): an instance of Environment
            max_steps (int): episode will end if not reaching goal in so many steps
            goal_name (string): name of the goal in the world
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            random_range (float): the goal's random position range
            target_speed (float): the target speed runing to the ball. The agent will receive no more 
                higher reward when its speed is higher than target_speed.
            reward_weight (float): the weight of the reward
        """
        super().__init__(
            env=env, max_steps=max_steps, reward_weight=reward_weight)
        self._goal_name = goal_name
        self._random_range = random_range
        self._success_distance_thresh = success_distance_thresh
        self._target_speed = target_speed
        self._env.insert_model(
            model="robocup_3Dsim_goal",
            name="goal",
            pose="-5.0 0 0 0 -0 3.14159265")
        self._env.insert_model(model="ball", pose="1.50 1.5 0.2 0 -0 0")

    def run(self):
        """
        Start a teaching episode for this task.
        """
        agent_sentence = yield
        goal = self._world.get_model(self._goal_name)
        ball = self._world.get_model('ball')
        goal_loc, dir = goal.get_pose()
        self._move_ball(ball, np.array(goal_loc))
        agent_loc, dir = self._agent.get_pose()
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
                agent_loc, dir = self._agent.get_pose()
                if self._agent.type.find('icub') != -1:
                    # For agent icub, we need to use the average pos here
                    agent_loc = ICubAuxiliaryTask.get_icub_extra_obs(self._agent)[:3]
                ball_loc, _ = ball.get_pose()
                dist = np.linalg.norm(
                    np.array(ball_loc)[:2] - np.array(agent_loc)[:2])
                # trunk progress_reward to target_speed
                progress_reward = min(
                    self._target_speed,
                    (prev_dist - dist) / self._env.get_step_time())
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
                dist = np.linalg.norm(
                    np.array(ball_loc)[:2] - np.array(goal_loc)[:2])
                if dist < self._success_distance_thresh:
                    agent_sentence = yield TeacherAction(
                        reward=100.0, sentence="well done", done=True)
                else:
                    agent_sentence = yield TeacherAction(
                        reward=self._target_speed + 3 - dist / init_goal_dist)
        yield TeacherAction(reward=-1.0, sentence="failed", done=True)

    def task_specific_observation(self, agent):
        """
        Args:
            agent (Agent): the agent
        Returns:
            np.array, the extra observations should be added into the observation
        """
        return self._get_states_of_model_list(['ball', 'goal'])

    def _move_ball(self, ball, goal_loc):
        range = self._random_range
        while True:
            loc = (random.random() * range - range / 2,
                   random.random() * range - range / 2, 0)
            if np.linalg.norm(loc - goal_loc) > self._success_distance_thresh:
                break
        ball.set_pose((loc, (0, 0, 0)))
