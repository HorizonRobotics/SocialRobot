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

import math
import numpy as np
import operator
import os
import gin
import itertools
import random
import json
from collections import deque, OrderedDict
from abc import abstractmethod
from absl import logging
import social_bot
from social_bot.teacher import TeacherAction


class Task(object):
    """Base class for Task.

    A Task is for teaching a single task.
    """

    compatible_agents = [
        'pioneer2dx_noplugin',
        'pr2_noplugin',
        'icub',
        'icub_with_hands',
        'youbot_noplugin',
    ]

    def __init__(self, env, max_steps=200, reward_weight=1.0):
        """
        Setting things up during the initialization.

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
        """ run() use yield to generate TeacherAction.

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
            A generator of TeacherAction
        """
        pass

    def task_specific_observation(self, agent):
        """
        The extra infomation needed by the task if sparse states are used.

        This can be overridden by the sub task. Note that this is only for the
        case "Agent._use_image_observation" is False. For image case, the
        image form camera of agent is used. For case of image with internal
        states, Agent.get_internal_states() is used, which only returns
        self joint positions and velocities.

        Args:
            agent (GazeboAgent): the agent
        Returns:
            np.array, the observations of the task for non-image case
        """
        return np.array([])

    def set_agent(self, agent):
        """ Set the agent of task.

        The agent can be overridden by this function. This might be useful when multi
        agents share the same task or embodied teacher.
        Args:
            agent (GazeboAgent): the agent
        """
        self._agent = agent

    def _get_states_of_model_list(self,
                                  model_list,
                                  including_velocity=True,
                                  including_rotation=False):
        """ Get the poses and velocities from a model list.

        Args:
            model_list (list): a list of model names
            including_velocity (bool): if Ture, the velocity of objects will be included.
            including_rotation (bool): if Ture, the rotation of objects (in roll pitch yaw) will be included.
        Returns:
            np.array, the poses and velocities of the models
        """
        model_states = []
        for model_id in range(len(model_list)):
            model = self._world.get_model(model_list[model_id])
            model_states.append(model.get_pose()[0])
            if including_rotation:
                model_states.append(model.get_pose()[1])
            if including_velocity:
                model_states.append(model.get_velocities()[0])
        model_states = np.array(model_states).flatten()
        return model_states

    def _random_move_object(self,
                            target,
                            random_range,
                            center_pos=np.array([0, 0]),
                            min_distance=0,
                            height=0):
        """ Move an object to a random position.

        Args:
            target (pyagzebo.Model): the target to move
            random_range (float): the range of the new position
            center_pos (numpy.array): the center coordinates (x, y) of the random range
            min_distance (float): the new position will not be closer than this distance
            height (float): height offset
        Returns:
            np.array, the new position
        """
        r = random.uniform(min_distance, random_range)
        theta = random.random() * 2 * np.pi
        loc = (center_pos[0] + r * np.cos(theta),
               center_pos[1] + r * np.sin(theta), height)
        target.set_pose((loc, (0, 0, 0)))
        return np.array(loc)


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
                 goal_name="pioneer2dx_noplugin_ghost",
                 distraction_list=[
                     'coke_can', 'table', 'car_wheel', 'plastic_cup', 'beer'
                 ],
                 goal_conditioned=False,
                 speed_goal=False,
                 speed_goal_limit=1.6,
                 pose_goal=False,
                 yaw_and_speed=False,
                 use_aux_achieved=False,
                 xy_only_aux=False,
                 multi_dim_reward=False,
                 end_on_hitting_distraction=False,
                 end_episode_after_success=False,
                 reset_time_limit_on_success=True,
                 chain_task_rate=0,
                 success_distance_thresh=0.5,
                 fail_distance_thresh=2.0,
                 distraction_penalty_distance_thresh=0,
                 distraction_penalty=0.5,
                 random_agent_position=False,
                 random_agent_orientation=False,
                 sparse_reward=True,
                 random_range=5.0,
                 min_distance=0,
                 polar_coord=True,
                 random_goal=False,
                 use_curriculum_training=False,
                 curriculum_distractions=True,
                 curriculum_target_angle=False,
                 switch_goal_within_episode=False,
                 start_range=0.0,
                 increase_range_by_percent=50.,
                 reward_thresh_to_increase_range=0.4,
                 percent_full_range_in_curriculum=0.1,
                 max_reward_q_length=100,
                 reward_weight=1.0,
                 move_goal_during_episode=True,
                 success_with_angle_requirement=True,
                 additional_observation_list=[],
                 use_egocentric_states=False,
                 egocentric_perception_range=0):
        """
        Args:
            env (gym.Env): an instance of Environment
            max_steps (int): episode will end if not reaching gaol in so many steps
            goal_name (string): name of the goal in the world
            distraction_list (list of string): a list of model. the model shoud be in gazebo database
            goal_conditioned (bool): if True, each step has -1 reward, unless at goal state, which gives 0.
            speed_goal (bool): if True, use speed pose etc. together with position as part of goal.
            speed_goal_limit (float): randomly sample speed goal in the range: -limit to +limit.
            pose_goal (bool): When speed_goal is True, if pose_goal is True, speed and everything else is put
                into observations, instead of aux_achieved, so only pose is in aux_achieved.
            yaw_and_speed (bool): When speed_goal is True, and not pose_goal, if yaw_and_speed is True,
                will include yaw and x-y speed into goal, nothing else.
            use_aux_achieved (bool): if True, pull out speed, pose dimensions into a separate
                field: aux_achieved.  Only valid when goal_conditioned is True.
            xy_only_aux (bool): exclude irrelevant dimensions (z-axis movements) from
                aux_achieved field.
            multi_dim_reward (bool): if True, separate goal reward and distraction penalty into two dimensions.
            end_episode_after_success (bool): if True, the episode will end once the goal is reached. A True value of this
                flag will overwrite the effects of flags ``switch_goal_within_episode`` and ``move_goal_during_episode``.
            end_on_hitting_distraction (bool): whether to end episode on hitting distraction
            reset_time_limit_on_success (bool): if not ending after success, if hit success before time limit,
                reset clock to 0.
            chain_task_rate (float): if positive, with this much probability, the current task will be chained together
                with the next task: after one tasks finishes, episode doesn't end, but keeps onto the second goal/task.
                If the first task isn't achieved within max_steps, episode ends.  When the first task is achieved,
                step_type is MID and no goal reward is given.  Reward is given only after achieving all chained tasks.
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            fail_distance_thresh (float): if the agent moves away from the goal more than this distance,
                it's considered a failure and is given reward -1
            distraction_penalty_distance_thresh (float): if positive, penalize agent getting too close
                to distraction objects (objects that include the goal itself, as approaching goal without
                facing it is considered hitting a distraction)
            distraction_penalty (float): positive float of how much to penalize getting too close to
                distraction objects
            random_agent_position (bool): whether randomize the position of the agent at beginning of episode.
            random_agent_orientation (bool): whether randomize the orientation (yaw) of the agent at the beginning of an
                episode.
            sparse_reward (bool): if true, the reward is -1/0/1, otherwise the 0 case will be replaced
                with normalized distance the agent get closer to goal.
            random_range (float): the goal's random position range
            min_distance (float): the goal must be this minimum distance away from avoided locations.  If this is smaller
                than the success_distance_thresh, then success_distance_thresh is used instead.
            polar_coord (bool): use cartesian coordinates in random_range, otherwise, use polar coord.
            random_goal (bool): if True, teacher will randomly select goal from the object list each episode
            use_curriculum_training (bool): when true, use curriculum in goal task training
            curriculum_distractions (bool): move distractions according to curriculum as well
            curriculum_target_angle (bool): enlarge angle to target when initializing target according
                to curriculum.  Only when all angles are satisfied does curriculum try to increase distance.
                Uses range of 0-360 degrees, starting from 60 with increments of 20.
            switch_goal_within_episode (bool): if random_goal and this are both true, goal will be re-picked
                within episode every time target is reached, besides picking after whole episode ends.
            start_range (float): for curriculum learning, the starting random_range to set the goal
            increase_range_by_percent (float): for curriculum learning, how much to increase random range
                every time agent reached the specified amount of reward.
            reward_thresh_to_increase_range (float): for curriculum learning, how much reward to reach
                before the teacher increases random range.
            percent_full_range_in_curriculum (float): if above 0, randomly throw in x% of training examples
                where random_range is the full range instead of the easier ones in the curriculum.
            max_reward_q_length (int): how many recent rewards to consider when estimating agent accuracy.
            reward_weight (float): the weight of the reward, is used in multi-task case
            move_goal_during_episode (bool): if True, the goal will be moved during episode, when it has been achieved
            success_with_angle_requirement: if True then calculate the reward considering the angular requirement
            additional_observation_list: a list of additonal objects to be added
            use_egocentric_states (bool): For the non-image observation case, use the states transformed to
                egocentric coordinate, e.g., agent's egocentric distance and direction to goal
            egocentric_perception_range (float): the max range in degree to limit the agent's observation.
                E.g. 60 means object is only visible when it's within +/-60 degrees in front of the agent's
                direction (yaw).
        """
        self._max_play_ground_size = 5  # play ground will be (-5, 5) for both x and y axes.
        # TODO: Remove the default grey walls in the play ground world file,
        # and insert them according to the max_play_ground_size.
        # The wall should be lower, and adjustable in length.  Add a custom model for that.
        super().__init__(
            env=env, max_steps=max_steps, reward_weight=reward_weight)
        self._goal_name = goal_name
        self._goal_conditioned = goal_conditioned
        self._speed_goal = speed_goal
        self._speed_goal_limit = speed_goal_limit
        self._pose_goal = pose_goal
        self._yaw_and_speed = yaw_and_speed
        self._use_aux_achieved = use_aux_achieved
        self._xy_only_aux = xy_only_aux
        self._multi_dim_reward = multi_dim_reward
        self.end_on_hitting_distraction = end_on_hitting_distraction
        self._end_episode_after_success = end_episode_after_success
        self._reset_time_limit_on_success = reset_time_limit_on_success
        self._chain_task_rate = chain_task_rate
        self._success_distance_thresh = success_distance_thresh
        self._fail_distance_thresh = fail_distance_thresh
        self._distraction_penalty_distance_thresh = distraction_penalty_distance_thresh
        if distraction_penalty_distance_thresh > 0:
            assert distraction_penalty_distance_thresh < success_distance_thresh
        self._distraction_penalty = distraction_penalty
        self._sparse_reward = sparse_reward
        self._random_agent_position = random_agent_position
        self._random_agent_orientation = random_agent_orientation
        self._use_curriculum_training = use_curriculum_training
        self._curriculum_distractions = curriculum_distractions
        self._curriculum_target_angle = curriculum_target_angle
        self._switch_goal_within_episode = switch_goal_within_episode
        if curriculum_target_angle:
            self._random_angle = 60
        self._start_range = start_range
        self._is_full_range_in_curriculum = False
        self._random_goal = random_goal
        if random_goal and goal_name not in distraction_list:
            distraction_list.append(goal_name)
        self._distraction_list = distraction_list
        self._object_list = distraction_list.copy()
        if goal_name and goal_name not in self._object_list:
            self._object_list.append(goal_name)
        self._goals = self._object_list
        self._move_goal_during_episode = move_goal_during_episode
        self._success_with_angle_requirement = success_with_angle_requirement
        if not additional_observation_list:
            additional_observation_list = self._object_list
        else:
            self._object_list.extend(additional_observation_list)
        self._additional_observation_list = additional_observation_list
        self._pos_list = list(
            itertools.product(
                range(-self._max_play_ground_size, self._max_play_ground_size),
                range(-self._max_play_ground_size,
                      self._max_play_ground_size)))
        self._pos_list.remove((0, 0))
        self._polar_coord = polar_coord
        self._use_egocentric_states = use_egocentric_states
        self._egocentric_perception_range = egocentric_perception_range
        if self.should_use_curriculum_training():
            self._orig_random_range = random_range
            self._random_range = start_range
            self._max_reward_q_length = max_reward_q_length
            self._q = deque(maxlen=max_reward_q_length)
            self._reward_thresh_to_increase_range = reward_thresh_to_increase_range
            self._increase_range_by_percent = increase_range_by_percent
            self._percent_full_range_in_curriculum = percent_full_range_in_curriculum
            angle_str = ""
            if curriculum_target_angle:
                angle_str = ", start_angle {}".format(self._random_angle)
            logging.info(
                "Env %d: start_range %f%s, reward_thresh_to_increase_range %f",
                self._env._port, self._start_range, angle_str,
                self._reward_thresh_to_increase_range)
        else:
            self._random_range = random_range
        if min_distance < self._success_distance_thresh:
            min_distance = self._success_distance_thresh
        self._min_distance = min_distance
        self._goal_dist = 0.
        obs_format = "image or full_state"
        obs_relative = "ego"
        if use_egocentric_states:
            obs_format = "full_state"
        else:
            obs_relative = "absolute"
        logging.info("Observations: {}, {}.".format(obs_format, obs_relative))
        if not use_egocentric_states:
            logging.info(
                "Dims: 0-5: agent's velocity and angular " +
                "velocity, 6-11: agent's position and pose, 12-13: goal x, y" +
                ", all distractions' x, y coordinates.")
        self.task_vocab += self._object_list
        self._env.insert_model_list(self._object_list)

    def should_use_curriculum_training(self):
        return (self._use_curriculum_training
                and self._start_range >= self._success_distance_thresh * 1.2)

    def _push_reward_queue(self, value):
        if (not self.should_use_curriculum_training()
            ) or self._is_full_range_in_curriculum:
            return
        self._q.append(value)
        if (value > 0 and len(self._q) == self._max_reward_q_length
                and sum(self._q) >= self._max_reward_q_length *
                self._reward_thresh_to_increase_range):
            if self._curriculum_target_angle:
                self._random_angle += 20
                logging.info("Env %d: Raising random_angle to %d",
                             self._env._port, self._random_angle)
            if (not self._curriculum_target_angle or self._random_angle > 360):
                self._random_angle = 60
                new_range = min((1. + self._increase_range_by_percent) *
                                self._random_range, self._orig_random_range)
                if self._random_range < self._orig_random_range:
                    logging.info("Env %d: Raising random_range to %f",
                                 self._env._port, new_range)
                self._random_range = new_range
            self._q.clear()

    def get_random_range(self):
        return self._random_range

    def pick_goal(self):
        if self._random_goal:
            random_id = random.randrange(len(self._goals))
            self.set_goal_name(self._goals[random_id])

    def _get_agent_loc(self):
        loc, agent_dir = self._agent.get_pose()
        if self._agent.type.find('icub') != -1:
            # For agent icub, we need to use the average pos here
            loc = ICubAuxiliaryTask.get_icub_extra_obs(self._agent)[:3]
        loc = np.array(loc)
        return loc, agent_dir

    def _prepare_teacher_action(self,
                                reward,
                                sentence,
                                done,
                                success=False,
                                rewards=None):
        goal_dist = 0.
        if rewards is not None:
            rewards = rewards.astype(np.float32)
        # only output when episode done, otherwise, reward accumulator in tensorboard
        # may miss it.
        if done:
            goal_dist = self._goal_dist
            # clear self._goal_dist so it is only output once
            self._goal_dist = 0.
        return TeacherAction(
            reward=reward,
            sentence=sentence,
            done=done,
            success=success,
            goal_range=goal_dist,
            rewards=rewards)

    def _within_angle(self, dot):
        return (not self._success_with_angle_requirement) or dot > 0.707

    def _get_agent_aux_dims(self, agent_pose=None, agent_vel=None):
        if agent_pose is None:
            agent_pose = np.array(self._agent.get_pose()).flatten()
        if agent_vel is None:
            agent_vel = np.array(self._agent.get_velocities()).flatten()
        # 0, 1, 2: vel; 3, 4, 5: angular vel; 6: z position; 7, 8, 9: roll pitch yaw
        return np.concatenate((agent_vel, agent_pose[2:]), axis=0)

    def _get_goal_dist(self, goal):
        loc, agent_dir = self._get_agent_loc()
        goal_loc, _ = goal.get_pose()
        goal_loc = np.array(goal_loc)
        _goal_loc = goal_loc.copy()
        _loc = np.array(loc).copy()
        if self._speed_goal:
            _goal_loc = np.concatenate((_goal_loc, self._aux_desired), axis=0)
            aux = self._get_agent_aux_dims()
            if self._pose_goal:
                aux = aux[-1:]  # yaw
            elif self._yaw_and_speed:
                aux = np.concatenate((aux[:2], aux[-1:]), axis=0)
            _loc = np.concatenate((_loc, aux), axis=0)

        dist = np.linalg.norm(_loc - _goal_loc)
        # dir from get_pose is (roll, pitch, yaw)
        dir = np.array([math.cos(agent_dir[2]), math.sin(agent_dir[2])])
        goal_dir = (goal_loc[0:2] - loc[0:2]) / dist
        dot = sum(dir * goal_dir)
        return dist, dot, loc, agent_dir

    def run(self):
        """ Start a teaching episode for this task. """
        agent_sentence = yield
        self._agent.reset()
        if self._random_agent_orientation or self._random_agent_position:
            loc, agent_dir = self._agent.get_pose()
            if self._random_agent_position:
                loc = (self._max_play_ground_size * (1 - 2 * random.random()),
                       self._max_play_ground_size * (1 - 2 * random.random()),
                       loc[2])
            if self._random_agent_orientation:
                agent_dir = (agent_dir[0], agent_dir[1],
                             2 * math.pi * random.random())
            self._agent.set_pose((loc, agent_dir))
        self._random_move_objects()
        self.pick_goal()
        goal = self._world.get_model(self._goal_name)
        done = False
        gen = self._run_one_goal(goal)
        while not done:
            action = next(gen)
            if (action.done and action.success and self._chain_task_rate > 0
                    and random.random() < self._chain_task_rate):
                action.done = False
                gen = self._run_one_goal(goal, move_distractions=False)
            done = action.done
            yield action

    def _run_one_goal(self, goal, move_distractions=True):
        """Generator function of task feedback."""
        a_loc, a_dir = self._get_agent_loc()
        self._move_goal(goal, a_loc, a_dir, move_distractions)
        steps_since_last_reward = 0
        prev_min_dist_to_distraction = 100
        rewards = None  # reward array in multi_dim_reward case
        while steps_since_last_reward < self._max_steps:
            steps_since_last_reward += 1
            dist, dot, loc, agent_dir = self._get_goal_dist(goal)
            distraction_penalty, prev_min_dist_to_distraction = (
                self._get_distraction_penalty(loc, dot,
                                              prev_min_dist_to_distraction))
            # TODO(Le): compare achieved goal with desired goal if task is
            # goal conditioned?
            if dist < self._success_distance_thresh and self._within_angle(
                    dot):
                # within 45 degrees of the agent direction
                reward = 1.0 - distraction_penalty
                self._push_reward_queue(max(reward, 0))
                if self._goal_conditioned:
                    reward -= 1.
                    if self._multi_dim_reward:
                        rewards = np.array([0, -distraction_penalty])
                else:
                    if self._multi_dim_reward:
                        rewards = np.array([1, -distraction_penalty])
                logging.debug("yielding reward: " + str(reward))
                logging.debug("at location: %s, aux: %s", a_loc.astype('|S5'),
                              self._get_agent_aux_dims().astype('|S5'))
                done = self._end_episode_after_success
                agent_sentence = yield self._prepare_teacher_action(
                    reward=reward,
                    sentence="well done",
                    done=done,
                    success=True,
                    rewards=rewards)
                if self._reset_time_limit_on_success:
                    steps_since_last_reward = 0
                if self._switch_goal_within_episode:
                    self.pick_goal()
                    goal = self._world.get_agent(self._goal_name)
                if self._move_goal_during_episode:
                    self._agent.reset()
                    loc, agent_dir = self._get_agent_loc()
                    self._move_goal(goal, loc, agent_dir)
            elif dist > self._initial_dist + self._fail_distance_thresh:
                reward = -1.0 - distraction_penalty
                self._push_reward_queue(0)
                logging.debug(
                    "yielding reward: {}, farther than {} from goal".format(
                        str(reward), str(self._fail_distance_thresh)))
                if self._multi_dim_reward:
                    rewards = np.array([-1, -distraction_penalty])
                yield self._prepare_teacher_action(
                    reward=reward,
                    sentence="failed",
                    done=True,
                    rewards=rewards)
            else:
                if self._sparse_reward:
                    reward = 0
                    if self._goal_conditioned:
                        reward = -1
                else:
                    reward = (self._prev_dist - dist) / self._initial_dist
                if self._multi_dim_reward:
                    rewards = np.array([reward, -distraction_penalty])
                reward = reward - distraction_penalty
                done = False
                if distraction_penalty > 0:
                    logging.debug("yielding reward: " + str(reward))
                    self._push_reward_queue(0)
                    done = self.end_on_hitting_distraction
                self._prev_dist = dist
                agent_sentence = yield self._prepare_teacher_action(
                    reward=reward,
                    sentence=self._goal_name,
                    done=done,
                    rewards=rewards)
        reward = -1.0
        dist, dot, loc, agent_dir = self._get_goal_dist(goal)
        distraction_penalty, prev_min_dist_to_distraction = (
            self._get_distraction_penalty(loc, dot,
                                          prev_min_dist_to_distraction))
        # TODO(Le): compare achieved goal with desired goal if task is
        # goal conditioned?
        success = False
        if dist < self._success_distance_thresh and self._within_angle(dot):
            success = True
            reward = 1.0 - distraction_penalty
            self._push_reward_queue(max(reward, 0))
            if self._goal_conditioned:
                reward -= 1.
                if self._multi_dim_reward:
                    rewards = np.array([0, -distraction_penalty])
            else:
                if self._multi_dim_reward:
                    rewards = np.array([1, -distraction_penalty])
        else:
            self._push_reward_queue(0)
            logging.debug("took more than {} steps".format(
                str(self._max_steps)))
        agent_vel = np.array(self._agent.get_velocities()).flatten()

        def _str(arr):
            res = ["["]
            res.extend(["{:.2f}".format(e) for e in arr])
            res.append("]")
            return " ".join(res)

        logging.debug("yielding reward: %s at\nloc %s pose %s vel %s",
                      str(reward), _str(loc), _str(agent_dir), _str(agent_vel))
        if self.should_use_curriculum_training():
            logging.debug("reward queue len: {}, sum: {}".format(
                str(len(self._q)), str(sum(self._q))))
        yield self._prepare_teacher_action(
            reward=reward,
            sentence="failed",
            done=True,
            success=success,
            rewards=rewards)

    def _get_distraction_penalty(self, agent_loc, dot,
                                 prev_min_dist_to_distraction):
        """
        Calculate penalty for hitting/getting close to distraction objects
        """
        distraction_penalty = 0
        if (self._distraction_penalty_distance_thresh > 0
                and self._distraction_list):
            curr_min_dist = 100
            for obj_name in self._distraction_list:
                obj = self._world.get_model(obj_name)
                if not obj:
                    continue
                obj_loc, _ = obj.get_pose()
                obj_loc = np.array(obj_loc)
                distraction_dist = np.linalg.norm(agent_loc - obj_loc)
                if (distraction_dist >=
                        self._distraction_penalty_distance_thresh):
                    continue
                if obj_name == self._goal_name and self._within_angle(dot):
                    continue  # correctly getting to goal, no penalty
                if distraction_dist < curr_min_dist:
                    curr_min_dist = distraction_dist
                if (prev_min_dist_to_distraction >
                        self._distraction_penalty_distance_thresh):
                    logging.debug("hitting object: " + obj_name)
                    distraction_penalty += self._distraction_penalty
            prev_min_dist_to_distraction = curr_min_dist
        return distraction_penalty, prev_min_dist_to_distraction

    def _move_goal(self, goal, agent_loc, agent_dir, move_distractions=True):
        """
        Move goal as well as all distraction objects to a random location.
        """
        avoid_locations = [agent_loc]
        if not move_distractions:
            for item in self._distraction_list:
                if item is not self._goal_name:
                    obj = self._world.get_model(item)
                    if obj:
                        distraction_loc = obj.get_pose()[0]
                        avoid_locations.append(distraction_loc)
        if self._speed_goal:
            MAX_SPEED = self._speed_goal_limit * 2
            xspeed = (0.5 - random.random()) * MAX_SPEED
            yspeed = (0.5 - random.random()) * MAX_SPEED
            yawspeed = (0.5 - random.random()) * MAX_SPEED
            yaw = np.arctan2(
                yspeed, xspeed)  # numpy.arctan2 is defined as arctan(x0/x1)
            # 0, 1, 2: vel; 3, 4, 5: angular vel; 6: z position; 7, 8, 9: roll pitch yaw
            self._aux_desired = np.array(
                [xspeed, yspeed, 0, 0, 0, yawspeed, 0, 0, 0, yaw])
            if self._pose_goal:
                self._aux_desired = np.array([yaw])
            elif self._yaw_and_speed:
                self._aux_desired = np.array([xspeed, yspeed, yaw])
        loc, dist = self._move_obj(
            obj=goal,
            agent_loc=agent_loc,
            agent_dir=agent_dir,
            is_goal=True,
            avoid_locations=avoid_locations,
            name="goal")
        self._goal_dist += dist
        avoid_locations.append(loc)
        distractions = OrderedDict()
        if move_distractions:
            for item in self._distraction_list:
                if item is not self._goal_name:
                    distractions[item] = 1
            if len(distractions) and self._curriculum_distractions:
                for item, _ in distractions.items():
                    distraction = self._world.get_agent(item)
                    loc, _ = self._move_obj(
                        obj=distraction,
                        agent_loc=agent_loc,
                        agent_dir=agent_dir,
                        is_goal=False,
                        avoid_locations=avoid_locations,
                        name=item)
                    avoid_locations.append(loc)

    def _move_obj(self,
                  obj,
                  agent_loc,
                  agent_dir,
                  is_goal=True,
                  avoid_locations=[],
                  close_to_agent=False,
                  name="Unspecified"):
        if (self.should_use_curriculum_training()
                and self._percent_full_range_in_curriculum > 0
                and random.random() < self._percent_full_range_in_curriculum):
            range = self._orig_random_range
            self._is_full_range_in_curriculum = is_goal
        else:
            range = self._random_range
            self._is_full_range_in_curriculum = False
        attempts = 0
        dist = range
        _min_distance = self._min_distance
        _succ_thd = self._success_distance_thresh
        if close_to_agent:
            _min_distance = 0.3
            _succ_thd = _min_distance
        while True:
            attempts += 1
            dist = random.random() * (range - _succ_thd) + _succ_thd
            if self._curriculum_target_angle:
                angle_range = self._random_angle
            else:
                angle_range = 360
            angle = math.radians(
                math.degrees(agent_dir[2]) + random.random() * angle_range -
                angle_range / 2)
            loc = (dist * math.cos(angle), dist * math.sin(angle),
                   0) + agent_loc

            if not self._polar_coord:
                loc = np.asarray((random.random() * range - range / 2,
                                  random.random() * range - range / 2, 0))

            satisfied = True
            if (abs(loc[0]) > self._max_play_ground_size or abs(loc[1]) >
                    self._max_play_ground_size):  # not within walls
                satisfied = False
            for avoid_loc in avoid_locations:
                ddist = np.linalg.norm(loc - avoid_loc)
                if ddist < _min_distance:
                    satisfied = False
                    break
            if satisfied or attempts % 100 == 0 or attempts > 300:
                if not satisfied:
                    if attempts <= 300:
                        _min_distance /= 2.
                        logging.warning(
                            "Took {} times to find satisfying {} "
                            "location. reducing _min_dist to {}".format(
                                str(attempts), name, str(_min_distance)))
                        continue
                    else:
                        logging.warning(
                            "Took forever to find satisfying " +
                            "{} location. " +
                            "agent_loc: {}, range: {}, _min_dist: {}, max_size"
                            ": {}.".format(name, str(agent_loc), str(range),
                                           str(_min_distance),
                                           str(self._max_play_ground_size)))
                if is_goal:
                    self._initial_dist = np.linalg.norm(loc - agent_loc)
                break
        if is_goal:
            self._prev_dist = self._initial_dist
        obj.reset()
        yaw = 0
        if is_goal and self._speed_goal:
            yaw = self._aux_desired[-1]
        obj.set_pose((loc, (0, 0, yaw)))
        return loc, dist

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

    def generate_goal_conditioned_obs(self, agent):
        flat_obs = self.task_specific_observation(agent)
        obs = OrderedDict()
        agent_pose = np.array(agent.get_pose()).flatten()
        agent_vel = np.array(agent.get_velocities()).flatten()
        goal = self._world.get_model(self._goal_name)
        goal_pose = np.array(goal.get_pose()[0]).flatten()
        obs['observation'] = flat_obs
        obs['achieved_goal'] = agent_pose[0:2]
        obs['desired_goal'] = goal_pose[0:2]
        aux = self._get_agent_aux_dims(agent_pose, agent_vel)
        if self._speed_goal:
            obs['observation'] = flat_obs[14:]
            if self._pose_goal:
                obs['observation'] = np.concatenate(
                    (obs['observation'], aux[:-1]), axis=0)
                aux = aux[-1:]  # yaw
            elif self._yaw_and_speed:
                obs['observation'] = np.concatenate(
                    (obs['observation'], aux[2:-1]), axis=0)
                aux = np.concatenate((aux[:2], aux[-1:]), axis=0)
            obs['achieved_goal'] = np.concatenate((obs['achieved_goal'], aux),
                                                  axis=0)
            obs['desired_goal'] = np.concatenate(
                (obs['desired_goal'], self._aux_desired), axis=0)
        elif self._use_aux_achieved:
            # distraction objects' x, y coordinates
            obs['observation'] = flat_obs[14:]
            obs['aux_achieved'] = aux
            # if self._xy_only_aux:
            #     # agent speed: 2: z-speed; 3, 4: angular velocities; 5: yaw-vel,
            #     # agent pose: 2: z; 3, 4, 5: roll pitch yaw.
            #     obs['observation'] = np.concatenate(
            #         (agent_vel[2:5], agent_pose[2:5], flat_obs[14:]), axis=0)
            #     obs['aux_achieved'] = np.concatenate(
            #         (agent_vel[0:2], np.expand_dims(agent_vel[5], 0),
            #          np.expand_dims(agent_pose[5], 0)),
            #         axis=0)
        return obs

    def task_specific_observation(self, agent):
        """
        Args:
            agent (GazeboAgent): the agent
        Returns:
            np.array, the observations of the task for non-image case
        """
        goal = self._world.get_model(self._goal_name)
        goal_first = not agent._with_language
        if goal_first:  # put goal first
            pose = np.array(goal.get_pose()[0]).flatten()
        else:  # has language input, don't put goal first
            pose = None

        for name in self._additional_observation_list:
            if goal_first and name == self._goal_name:
                continue
            obj = self._world.get_model(name)
            obj_pos = np.array(obj.get_pose()[0]).flatten()
            if pose is None:
                pose = obj_pos
            else:
                pose = np.concatenate((pose, obj_pos), axis=0)

        agent_pose = np.array(agent.get_pose()).flatten()
        yaw = agent_pose[5]
        # adds egocentric velocity input
        vx, vy, vz, a1, a2, a3 = np.array(agent.get_velocities()).flatten()
        if self._use_egocentric_states:
            rvx, rvy = agent.get_egocentric_cord_2d(vx, vy, yaw)
        else:
            rvx, rvy = vx, vy
        obs = [rvx, rvy, vz, a1, a2, a3] + list(agent_pose)
        # adds objects' (goal's as well as distractions') egocentric
        # coordinates to observation
        while len(pose) > 1:
            x = pose[0]
            y = pose[1]
            if self._use_egocentric_states:
                x = pose[0] - agent_pose[0]
                y = pose[1] - agent_pose[1]
                rotated_x, rotated_y = agent.get_egocentric_cord_2d(x, y, yaw)
            else:
                rotated_x, rotated_y = x, y
            if (self._use_egocentric_states
                    and self._egocentric_perception_range > 0):
                dist = math.sqrt(rotated_x * rotated_x + rotated_y * rotated_y)
                rotated_x /= dist
                rotated_y /= dist
                magnitude = 1. / dist
                if rotated_x < np.cos(
                        self._egocentric_perception_range / 180. * np.pi):
                    rotated_x = 0.
                    rotated_y = 0.
                    magnitude = 0.
                obs.extend([rotated_x, rotated_y, magnitude])
            else:
                obs.extend([rotated_x, rotated_y])
            pose = pose[3:]
        obs = np.array(obs)

        return obs


@gin.configurable
class PushReachTask(GoalTask):
    def __init__(self,
                 env,
                 max_steps,
                 push_reach=False,
                 obj_names=['wood_cube_30cm_without_offset'],
                 goal_names=['goal_indicator'],
                 distraction_list=['car_wheel'],
                 multi_dim_reward=True,
                 close_to_agent=False,
                 target_relative_to_obj=False,
                 use_obj_pose=False):
        """A Push or Push and Reach task.

        We utilize some of the curriculum, distraction obj handling logic in GoalTask.

        Args:
            push_reach (bool): if False, push only, goal positions are for objects to achieve; otherwise,
                the first goal position is for the agent to achieve.
            obj_names (list of string): when not empty, it's the names of the objects to be moved.
            goal_names (list of string): when not empty, these goal objects indicate the goal locations for
                the objects in obj_names.
            close_to_agent (bool): whether to initialize object to be pushed closer to the agent.
            target_relative_to_obj (bool): initialize target position relative to object vs agent location.
            use_obj_pose (bool): include object auxiliary dimensions as input.
        """
        self._push_reach = push_reach
        if push_reach:
            goal_names.append('target_ball')
        self._obj_names = obj_names
        self._goal_names = goal_names
        self._multi_dim_reward = multi_dim_reward
        self._target_relative_to_obj = target_relative_to_obj
        self._close_to_agent = close_to_agent
        self._use_obj_pose = use_obj_pose
        if not push_reach:
            assert len(obj_names) == len(goal_names)
        else:
            assert len(obj_names) == len(goal_names) - 1
        super().__init__(
            env,
            max_steps,
            goal_conditioned=True,
            use_aux_achieved=True,
            multi_dim_reward=multi_dim_reward,
            end_episode_after_success=True,
            goal_name=goal_names[0],
            distraction_list=distraction_list,
            additional_observation_list=obj_names + goal_names[1:])

    def _random_move_objects(self, random_range=10.0):
        pass

    def _move_objs(self,
                   obj_names,
                   ap,
                   ad,
                   avoids=[],
                   record_goal_dist=False,
                   close_to_agent=False,
                   target_relative_to_obj=False):
        """Move objects according to agent location."""
        obj_positions = avoids
        if target_relative_to_obj:
            assert len(avoids) == len(obj_names), "num objects != num targets"
            orig_obj_pos = avoids.copy()

        if record_goal_dist:
            self._goal_dist = 0
        n = 0
        for obj_name in obj_names:
            obj = self._world.get_agent(obj_name)
            start_pos = ap
            _avoids = obj_positions
            if target_relative_to_obj:
                start_pos = orig_obj_pos[n]
                _avoids = _avoids + [ap]
            p, dist = self._move_obj(
                obj,
                start_pos,
                ad,
                avoid_locations=obj_positions,
                name=obj_name,
                close_to_agent=close_to_agent)
            if record_goal_dist:
                self._goal_dist += dist
            n += 1
            obj_positions.append(p)
        if record_goal_dist:
            self._goal_dist /= n
        return obj_positions

    def _move_goals(self, ap, ad, obj_locs=[]):
        """Move goal locations according to object locations."""
        # TODO(lezhao): copy obj_locs into avoids and put ap in avoids,
        # then find goal location based on each obj location.
        return self._move_objs(
            self._goal_names,
            ap,
            ad,
            obj_locs,
            record_goal_dist=True,
            target_relative_to_obj=self._target_relative_to_obj)

    def _move_distractions(self, ap, ad, avoids=[]):
        return self._move_objs(self._distraction_list, ap, ad, avoids)

    def _get_obj_loc_pose(self, obj_names):
        res_pos = np.array([])
        res_aux = np.array([])
        for obj_name in obj_names:
            obj = self._world.get_agent(obj_name)
            obj_pos, obj_dir = obj.get_pose()
            # vel = np.array(obj.get_velocities()).flatten()
            res_pos = np.concatenate((res_pos, obj_pos[0:2]))
            res_aux = np.concatenate((res_aux, obj_pos[2:], obj_dir))  # , vel
        return res_pos, res_aux

    def _get_achieved(self):
        achieved_loc = np.array([])
        aux_achieved = np.array(self._agent.get_velocities()).flatten()
        # print("gazebo port: ", self._env._port)
        # print("gazebo aux_ach: ", aux_achieved, flush=True)
        ap, ad = self._get_agent_loc()
        if not self._push_reach:
            aux_achieved = np.concatenate((aux_achieved, ap, ad))
        else:
            achieved_loc = np.concatenate((achieved_loc, ap[0:2]))
            aux_achieved = np.concatenate((aux_achieved, ap[2:], ad))
        ach_poss, aux_achs = self._get_obj_loc_pose(self._obj_names)
        achieved_loc = np.concatenate((achieved_loc, ach_poss))
        if self._use_obj_pose:
            aux_achieved = np.concatenate((aux_achieved, aux_achs))
        return achieved_loc, aux_achieved

    def _get_desired_goal(self):
        return self._get_obj_loc_pose(self._goal_names)[0]

    def generate_goal_conditioned_obs(self, agent):
        ach, aux_ach = self._get_achieved()
        obs = OrderedDict()
        obs['observation'] = self._get_obj_loc_pose(self._distraction_list)[0]
        obs['achieved_goal'] = ach
        obs['desired_goal'] = self._get_desired_goal()
        obs['aux_achieved'] = aux_ach
        return obs

    def _compute_reward(self):
        def l2_dist_close_reward_fn(achieved_goal, goal):
            return 0 if np.linalg.norm(
                achieved_goal - goal) < self._success_distance_thresh else -1

        return l2_dist_close_reward_fn(self._get_achieved()[0],
                                       self._get_desired_goal())

    def _produce_rewards(self, prev_min_dist_to_distraction):
        ap, ad = self._get_agent_loc()
        distraction_penalty, prev_min_dist_to_distraction = (
            self._get_distraction_penalty(ap, 1.,
                                          prev_min_dist_to_distraction))
        reward = self._compute_reward()
        done = reward >= 0
        if done:
            distraction_penalty = 0
            av = np.array(self._agent.get_velocities()).flatten()
            logging.debug("yielding reward: " + str(reward))
            logging.debug("at location: %s, aux: %s, vel: %s",
                          ap.astype('|S5'),
                          np.array(ad).astype('|S5'), av.astype('|S5'))
        reward += distraction_penalty
        rewards = None
        if self._multi_dim_reward:
            rewards = np.array([reward, -distraction_penalty])
        return self._prepare_teacher_action(
            reward=reward,
            sentence="well done",
            done=done,
            success=done,
            rewards=rewards), prev_min_dist_to_distraction

    def _run_one_goal(self, goal=None, move_distractions=None):
        # The two params are not used here.
        ap, ad = self._get_agent_loc()
        obj_positions = self._move_objs(
            self._obj_names, ap, ad, close_to_agent=self._close_to_agent)
        avoids = self._move_goals(ap, ad, obj_positions)
        avoids = self._move_distractions(ap, ad, avoids)
        avoids.clear()
        steps_since_last_reward = 0
        prev_min_dist_to_distraction = 100
        while steps_since_last_reward < self._max_steps:
            steps_since_last_reward += 1
            (agent_sentence, prev_min_dist_to_distraction
             ) = self._produce_rewards(prev_min_dist_to_distraction)
            yield agent_sentence
        (agent_sentence, prev_min_dist_to_distraction
         ) = self._produce_rewards(prev_min_dist_to_distraction)
        agent_sentence.done = True
        yield agent_sentence


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
        """ Start a teaching episode for this task. """
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
                self._agent, pos, self._agent.type + '::root_link', np.pi)
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
                    self._agent, agent_pos, self._agent.type + '::head')
                root_angle = self._get_angle_to_target(
                    self._agent, agent_pos, self._agent.type + '::root_link')
                l_foot_angle = self._get_angle_to_target(
                    self._agent, agent_pos,
                    self._agent.type + '::l_leg::l_foot', np.pi)
                r_foot_angle = self._get_angle_to_target(
                    self._agent, agent_pos,
                    self._agent.type + '::r_leg::r_foot', np.pi)
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
        root_pose = np.array(
            agent.get_link_pose(agent.name + '::root_link')).flatten()
        chest_pose = np.array(
            agent.get_link_pose(agent.name + '::chest')).flatten()
        l_foot_pose = np.array(
            agent.get_link_pose(agent.name + '::l_leg::l_foot')).flatten()
        r_foot_pose = np.array(
            agent.get_link_pose(agent.name + '::r_leg::r_foot')).flatten()
        foot_contacts = np.array([
            agent.get_contacts("l_foot_contact_sensor",
                               'ground_plane::link::collision'),
            agent.get_contacts("r_foot_contact_sensor",
                               'ground_plane::link::collision')
        ]).astype(np.float32)
        average_pos = np.sum([
            root_pose[0:3], chest_pose[0:3], l_foot_pose[0:3], r_foot_pose[0:3]
        ],
                             axis=0) / 4.0
        obs = np.concatenate((average_pos, root_pose, chest_pose, l_foot_pose,
                              r_foot_pose, foot_contacts))
        return obs

    def _get_angle_to_target(self, aegnt, agent_pos, link_name, offset=0):
        """ Get angle from a icub link, relative to target.

        Args:
            agent (GazeboAgent): the agent
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
            agent (GazeboAgent): the agent
        Returns:
            np.array, the observations of the task for non-image case
        """
        icub_extra_obs = self.get_icub_extra_obs(agent)
        if self._target_name:
            agent_pos = icub_extra_obs[:3]
            # TODO: be compatible with calling multiple times in one env step
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
    A simple task to kick a ball so that it rolls into the goal. An
    optional reward shaping can be used to guide the agent run to the ball first:
        Agent will receive 100 when succefully kick the ball into the goal.
        Agent will receive the speed of getting closer to the ball before touching the
            ball within 45 degrees of agent direction. The reward is trunked within
            parameter target_speed.
        Agent will receive negative normalized distance from ball to goal center
            after touching the ball within the direction. An offset of
            "target_speed + 1" is included since touching the ball must be better
            than not touching.

    If no reward shaping, then the agent will only get -1/0/1 rewards.
    """

    def __init__(self,
                 env,
                 max_steps,
                 goal_distance=5.0,
                 random_range=4.0,
                 target_speed=2.0,
                 reward_weight=1.0,
                 sparse_reward=False):
        """
        Args:
            env (gym.Env): an instance of Environment
            max_steps (int): episode will end if the task is not achieved in so
                many steps
            goal_distance (float): the distance from the goal to the ball on
                average. A smaller distance makes the kicking task easier.
            random_range (float): the ball's random position range
            target_speed (float): the target speed runing to the ball. The agent will receive no more
                higher reward when its speed is higher than target_speed.
            reward_weight (float): the weight of the reward
            sparse_reward (bool): if True, the agent will only get -1/0/1 rewards.
        """
        super().__init__(
            env=env, max_steps=max_steps, reward_weight=reward_weight)
        self._random_range = random_range
        self._target_speed = target_speed
        self._sparse_reward = sparse_reward
        self._goal_distance = goal_distance
        # By looking up the 'robocup_3Dsim_goal' model file:
        self._goal_width = 2.1
        self._goal_post_radius = 0.05
        self._env.insert_model(
            model="robocup_3Dsim_goal",
            name="goal",
            pose="-%s 0 0 0 -0 3.14159265" % goal_distance)
        self._env.insert_model(model="ball", pose="1.50 1.5 0.2 0 -0 0")

    def run(self):
        """ Start a teaching episode for this task. """
        agent_sentence = yield
        goal = self._world.get_model("goal")
        ball = self._world.get_model("ball")
        goal_loc, dir = goal.get_pose()
        self._move_ball(ball)
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
            if not hitted_ball and not self._sparse_reward:
                agent_loc, dir = self._agent.get_pose()
                if self._agent.type.find('icub') != -1:
                    # For agent icub, we need to use the average pos here
                    agent_loc = ICubAuxiliaryTask.get_icub_extra_obs(
                        self._agent)[:3]
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
                    ball_dir = (np.array(ball_loc[0:2]) - np.array(
                        agent_loc[0:2])) / dist
                    dot = sum(dir * ball_dir)
                    if dot > 0.707:
                        # within 45 degrees of the agent direction
                        hitted_ball = True
                agent_sentence = yield TeacherAction(reward=progress_reward)
            else:
                goal_loc, _ = goal.get_pose()
                ball_loc, _ = ball.get_pose()
                dist = np.linalg.norm(
                    np.array(ball_loc)[:2] - np.array(goal_loc)[:2])
                if self._in_the_goal(ball_loc):
                    if self._sparse_reward:
                        reward = 1.
                    else:
                        reward = 100.
                    agent_sentence = yield TeacherAction(
                        reward=reward,
                        sentence="well done",
                        done=True,
                        success=True)
                else:
                    if self._sparse_reward:
                        reward = 0.
                    else:
                        reward = self._target_speed + 3 - dist / init_goal_dist
                    agent_sentence = yield TeacherAction(reward=reward)
        yield TeacherAction(reward=-1.0, sentence="failed", done=True)

    def task_specific_observation(self, agent):
        """
        Args:
            agent (GazeboAgent): the agent
        Returns:
            np.array, the observations of the task for non-image case
        """
        obj_poses = self._get_states_of_model_list(['ball', 'goal'])
        agent_pose = np.array(agent.get_pose()).flatten()
        agent_vel = np.array(agent.get_velocities()).flatten()
        joints_states = agent.get_internal_states()
        obs = np.concatenate((obj_poses, agent_pose, agent_vel, joints_states),
                             axis=0)
        return obs

    def _in_the_goal(self, ball_loc):
        pass_goal_line = (ball_loc[0] < -self._goal_distance)
        half_width = self._goal_width / 2 - self._goal_post_radius  # =1.0
        within_goal = (half_width > ball_loc[1] > -half_width)
        return (pass_goal_line and within_goal)

    def _move_ball(self, ball):
        range = self._random_range
        while True:
            loc = (random.random() * range - range / 2,
                   random.random() * range - range / 2, 0)
            if not self._in_the_goal(loc):
                break
        ball.set_pose((loc, (0, 0, 0)))


@gin.configurable
class Reaching3D(Task):
    """
    A task to reach a random 3D position with the end effector of a robot arm.
    An optional distance based reward shaping can be used.
    This task is only compatible with Agent kuka_lwr_4plus.
    """

    compatible_agents = ['kuka_lwr_4plus']

    def __init__(self,
                 env,
                 max_steps,
                 random_range=0.65,
                 success_distance_thresh=0.1,
                 reward_shaping=True,
                 reward_weight=1.0):
        """
        Args:
            env (gym.Env): an instance of Environment
            max_steps (int): episode will end if not reaching goal in so many steps
            random_range (float): the goal's random position range
            success_distance_thresh (float): the goal is reached if it's within this distance to the agent
            reward_shaping (bool): if false, the reward is -1/0/1, otherwise the 0 case will be replaced
                with negative distance to goal.
            reward_weight (float): the weight of the reward
        """
        super().__init__(
            env=env, max_steps=max_steps, reward_weight=reward_weight)
        assert self._agent.type in self.compatible_agents, "Reaching3D Task only support kuka_lwr_4plus for now"
        self._reaching_link = '::lwr_arm_6_link'
        self._random_range = random_range
        self._success_distance_thresh = success_distance_thresh
        self._reward_shaping = reward_shaping
        self._env.insert_model(model="goal_indicator")
        self._goal = self._world.get_model('goal_indicator')

    def run(self):
        """ Start a teaching episode for this task. """
        agent_sentence = yield
        goal_loc, _ = self._goal.get_pose()
        reaching_loc, _ = self._agent.get_link_pose(self._agent.type +
                                                    self._reaching_link)
        self._move_goal(self._goal, np.array(reaching_loc))
        steps = 0
        while steps < self._max_steps:
            steps += 1
            reaching_loc, _ = self._agent.get_link_pose(self._agent.type +
                                                        self._reaching_link)
            goal_loc, _ = self._goal.get_pose()
            dist = np.linalg.norm(np.array(goal_loc) - np.array(reaching_loc))
            if dist < self._success_distance_thresh:
                agent_sentence = yield TeacherAction(
                    reward=1.0, sentence="well done", done=True)
            else:
                reward = (-dist) if self._reward_shaping else 0
                agent_sentence = yield TeacherAction(reward=reward, done=False)
        yield TeacherAction(reward=-1.0, sentence="failed", done=True)

    def _move_goal(self, goal, agent_loc):
        while True:
            r = 0.15 + random.random() * self._random_range
            theta = random.random() * 2 * np.pi
            phi = (random.random() - 0.5) * np.pi
            loc = (r * np.sin(phi) * np.cos(theta),
                   r * np.sin(phi) * np.sin(theta), 0.2 + np.cos(phi))
            if np.linalg.norm(loc - agent_loc) > self._success_distance_thresh:
                break
        goal.set_pose((loc, (0, 0, 0)))

    def task_specific_observation(self, agent):
        """
        Args:
            agent (GazeboAgent): the agent
        Returns:
            np.array, the observations of the task for non-image case
        """
        goal_loc, _ = self._goal.get_pose()
        reaching_loc, _ = agent.get_link_pose(self._agent.type +
                                              self._reaching_link)
        joints_states = agent.get_internal_states()
        obs = np.concatenate((goal_loc, reaching_loc, joints_states), axis=0)
        return obs


@gin.configurable
class PickAndPlace(Task):
    """
    A task to grip an object (a wood cube), move and then place it to the target position.
    A simple reward shaping can be used to guide the agent to grip cube and move to the position:
        If object is not being gripped, the reward is the gripper contacts, wether object is off the
            ground, and negative distance between object and gripper
        If being gripped, an extra truncked negative distance from object to goal is added.
        If suceesfully placed, a reward of 100 is given.
    This task is only compatible with Agent youbot_noplugin.
    """

    compatible_agents = ['youbot_noplugin']

    def __init__(self,
                 env,
                 max_steps,
                 object_random_range=0.6,
                 place_to_random_range=0.6,
                 min_distance=0.3,
                 object_half_height=0.025,
                 success_distance_thresh=0.05,
                 reward_shaping=False,
                 reward_weight=1.0):
        """
        Args:
            env (gym.Env): an instance of Environment
            max_steps (int): episode will end if not complet the task in so many steps, recommend to be 150
                for agent youbot_noplugin and object 5cm cube
            object_random_range (float): the object's random position range to the agent
            place_to_random_range (float): the range of target placing position to the object
            min_distance (float): the min_distance of the placing position to the object
            object_half_height (float): Note that model for stacking task should be of no offset inside the model.
                This means an initial pose of 0 height makes half of the obejct underground. This specifies the
                initial height of object's center, e.g, half of the edge length of a cube, or radius of a ball.
            success_distance_thresh (float): consider success if the target is within this distance to the
                goal position
            reward_shaping (bool): if false, the reward is -1/0/1, otherwise the 0 case will be replaced
                with shapped reward.
            reward_weight (float): the weight of the reward
        """
        super().__init__(
            env=env, max_steps=max_steps, reward_weight=reward_weight)
        assert self._agent.type in self.compatible_agents, "PickAndPlace Task \
            only support youbot_noplugin for now"

        self._palm_link = 'youbot_noplugin::gripper_palm_link'
        self._finger_link_l = 'youbot_noplugin::gripper_finger_link_l'
        self._finger_link_r = 'youbot_noplugin::gripper_finger_link_r'
        self._object_name = 'wood_cube_5cm_without_offset'
        self._object_collision_name = 'wood_cube_5cm_without_offset::link::collision'
        self._object_half_height = object_half_height
        self._object_random_range = object_random_range
        self._place_to_random_range = place_to_random_range
        self._min_distance = min_distance
        self._success_distance_thresh = success_distance_thresh
        self._reward_shaping = reward_shaping
        self._env.insert_model_list([self._object_name, 'goal_indicator'])
        self._goal = self._world.get_model('goal_indicator')
        self._object = self._world.get_model(self._object_name)

    def run(self):
        """ Start a teaching episode for this task. """
        agent_sentence = yield
        obj_pos = self._random_move_object(
            target=self._object,
            random_range=self._object_random_range,
            center_pos=np.array([0, 0]),
            min_distance=self._min_distance,
            height=self._object_half_height)
        goal_pos = self._random_move_object(
            target=self._goal,
            random_range=self._place_to_random_range,
            center_pos=obj_pos[:2],
            min_distance=self._min_distance,
            height=self._object_half_height)
        steps = 0
        while steps < self._max_steps:
            steps += 1
            # get positions
            obj_pos, _ = self._object.get_pose()
            obj_height = obj_pos[2]
            finger_l_pos, _ = self._agent.get_link_pose(self._finger_link_l)
            finger_r_pos, _ = self._agent.get_link_pose(self._finger_link_r)
            finger_pos = (
                np.array(finger_l_pos) + np.array(finger_r_pos)) / 2.0
            # get contacts
            l_contact = self._agent.get_contacts('finger_cnta_l',
                                                 self._object_collision_name)
            r_contact = self._agent.get_contacts('finger_cnta_r',
                                                 self._object_collision_name)
            # check distance and contacts
            obj_dist = np.linalg.norm(np.array(obj_pos) - goal_pos)
            obj_dist_xy = np.linalg.norm(np.array(obj_pos)[:2] - goal_pos[:2])
            dist_z = abs(obj_height - goal_pos[2])
            palm_dist = np.linalg.norm(
                np.array(obj_pos) - np.array(finger_pos))
            obj_lifted = obj_height / self._object_half_height - 1.0
            gripping_feature = 0.25 * l_contact + 0.25 * r_contact + min(
                obj_lifted, 0.5)  # encourge to lift the object by obj_height
            gripping = (gripping_feature > 0.99)
            # success condition, minus an offset of object height on z-axis
            if gripping and obj_dist_xy < self._success_distance_thresh and (
                    dist_z - self._object_half_height <
                    self._success_distance_thresh):
                logging.debug("object has been successfuly placed")
                reward = 200.0 if self._reward_shaping else 1.0
                agent_sentence = yield TeacherAction(
                    reward=reward,
                    sentence="well done",
                    done=True,
                    success=True)
            else:
                shaped_reward = max(
                    2.0 - obj_dist / self._place_to_random_range,
                    1.0) if gripping else (gripping_feature - palm_dist)
                reward = shaped_reward if self._reward_shaping else 0
                agent_sentence = yield TeacherAction(reward=reward, done=False)
        yield TeacherAction(reward=-1.0, sentence="failed", done=True)

    def task_specific_observation(self, agent):
        """
        Args:
            agent (GazeboAgent): the agent
        Returns:
            np.array, the observations of the task for non-image case
        """
        # Use 3 position of the links to uniquely determine the 6 + 2 DoF gripper
        finger_l_pos, _ = agent.get_link_pose(self._finger_link_l)
        finger_r_pos, _ = agent.get_link_pose(self._finger_link_r)
        palm_pos, _ = agent.get_link_pose(self._palm_link)
        # goal pos and object pos
        goal_pos, _ = self._goal.get_pose()
        obj_pos, obj_rot = self._object.get_pose()
        # contacts
        finger_contacts = np.array([
            agent.get_contacts('finger_cnta_l', self._object_collision_name),
            agent.get_contacts('finger_cnta_r', self._object_collision_name)
        ]).astype(np.float32)
        # agent self states
        agent_pose = np.array(agent.get_pose()).flatten()
        joints_states = agent.get_internal_states()
        obs = np.array(
            [goal_pos, obj_pos, obj_rot, finger_l_pos, finger_r_pos,
             palm_pos]).flatten()
        return np.concatenate(
            (obs, finger_contacts, agent_pose, joints_states), axis=0)


@gin.configurable
class Stack(Task):
    """
    A task to stack several wood cubes. The agent need to grasp the cube and
        stack it one by one, until all of them are stacked together.
    The number of cubes can be configured by objects_num. Distribution of
        cubes' initial position is configured by average_distance_max,
        average_distance_min and objects_range.
    Success condition is that all objects are stacked, and the gripper of agent
        leave the cubes (no contacts to cubes) for 5 time steps.
    The agent will receive a reward of 1 when success if reward shaping is not
    used. If reward shaping is used, the reward is the stacking number plus:
        if not gripping, negative distance to the closest obj not being stacked
        if gripping, distance to closet stacking candidate, i.e, (x, y,
            target_height). target_height is (stacked_num + 1) * object_height,
            plus a margin of 0.55 * object_height
    This task is only compatible with Agent youbot_noplugin.
    """

    compatible_agents = ['youbot_noplugin']

    def __init__(self,
                 env,
                 max_steps,
                 average_distance_max=0.5,
                 average_distance_min=0.3,
                 objects_num=3,
                 objects_range=0.25,
                 object_half_height=0.025,
                 success_distance_thresh=0.03,
                 reward_shaping=True,
                 reward_weight=1.0):
        """
        Args:
            env (gym.Env): an instance of Environment.
            max_steps (int): episode will end if not complet the task in so
                many steps.
            average_distance_max (float): the max distance from the agent to
                the center of the objects' initial position distribution
            average_distance_min (float):  the min distance from the agent to
                the center of the objects' initial position distribution
            objects_num (int): the number of objects to stack.
            objects_range (float): the range of objects around center position.
            object_half_height (float): Note that model for stacking task
                should be of no offset inside the model. This means an initial
                pose of 0 height makes half of the obejct underground. This
                specifies the initial height of object's center, e.g, half of
                the edge length of a cube, or radius of a ball.
            success_distance_thresh (float): consider success if the objects'
                x-y plance distance is within this threshold.
            reward_shaping (bool): if false, the reward is -1/0/1, otherwise
                the shapeed reward will be used.
            reward_weight (float): the weight of the reward.
        """
        super().__init__(
            env=env, max_steps=max_steps, reward_weight=reward_weight)
        assert self._agent.type in self.compatible_agents, "Stack task only \
            support youbot_noplugin for now"

        self._reward_shaping = reward_shaping
        self._palm_link = 'youbot_noplugin::gripper_palm_link'
        self._finger_link_l = 'youbot_noplugin::gripper_finger_link_l'
        self._finger_link_r = 'youbot_noplugin::gripper_finger_link_r'
        self._object_collision_name = '::wood_cube_5cm_without_offset::link::collision'
        self._object_half_height = object_half_height
        self._avg_distance_max = average_distance_max
        self._avg_distance_min = average_distance_min
        self._objects_num = objects_num
        self._objects_range = objects_range
        self._success_distance_thresh = success_distance_thresh
        self._object_names = []
        self._objects = []
        for obj_index in range(objects_num):
            name = 'wood_cube_' + str(obj_index)
            self._object_names.append(name)
            self._env.insert_model(
                model='wood_cube_5cm_without_offset', name=name)
            self._objects.append(self._world.get_model(name))

    def run(self):
        """ Start a teaching episode for this task. """
        agent_sentence = yield
        # randomly move objects
        r = random.uniform(self._avg_distance_min, self._avg_distance_max)
        theta = random.random() * 2 * np.pi
        stacking_pos = np.array([r * np.cos(theta), r * np.sin(theta)])
        for obj_index in range(self._objects_num):
            self._random_move_object(
                target=self._objects[obj_index],
                random_range=self._objects_range,
                center_pos=stacking_pos,
                min_distance=0,
                height=self._object_half_height)
        steps = 0
        succ_cnt = 0
        while steps < self._max_steps:
            steps += 1
            # get gripper pos
            finger_l_pos, _ = self._agent.get_link_pose(self._finger_link_l)
            finger_r_pos, _ = self._agent.get_link_pose(self._finger_link_r)
            finger_pos = (
                np.array(finger_l_pos) + np.array(finger_r_pos)) / 2.0
            # get object's position and contacts
            obj_positions = []
            l_contacts = []
            r_contacts = []
            for obj_index in range(self._objects_num):
                obj_pos, _ = self._objects[obj_index].get_pose()
                obj_positions.append(obj_pos)
                l_contacts.append(1.0 * self._agent.get_contacts(
                    'finger_cnta_l', self._object_names[obj_index] +
                    self._object_collision_name))
                r_contacts.append(1.0 * self._agent.get_contacts(
                    'finger_cnta_r', self._object_names[obj_index] +
                    self._object_collision_name))
            # convert to ndarray
            l_contacts = np.array(l_contacts)
            r_contacts = np.array(r_contacts)
            contacts = l_contacts + r_contacts
            obj_positions = np.array(obj_positions)
            obj_positions_xy = obj_positions[:, :2]
            obj_heights = obj_positions[:, 2]
            # get the objects in different stacking states
            obj_list = np.arange(self._objects_num)
            stacked_candidates = np.where(
                (contacts == 0) *
                (obj_heights / self._object_half_height > 1.5)
            )[0]  # off the ground and not being grasped, considerd as being stacked
            stacked_pos = obj_positions[stacked_candidates]
            top_index = None
            bottom_obj = None
            stacked_obj_num = 0
            while (len(stacked_pos) > 0):
                # find the highest object of the stack
                top_index = np.argmax(stacked_pos[:, 2])
                # find the bottom one within self._success_distance_thresh
                bottom_obj = np.where(
                    (obj_heights - self._object_half_height < 0.01) *
                    (np.linalg.norm(
                        obj_positions_xy - stacked_pos[top_index][:2], axis=1)
                     < self._success_distance_thresh))[0]
                if (len(bottom_obj) == 0):
                    # can not find an object below, for some reason the object
                    # is in the air without being grasped or stacked
                    stacked_pos = np.delete(stacked_pos, top_index, axis=0)
                else:
                    # get the stacked object list in which object is
                    # within success_distance_thresh and without contacts
                    stacked_obj_num = len(
                        np.where((contacts == 0) * (np.linalg.norm(
                            obj_positions_xy -
                            obj_positions_xy[bottom_obj[0]][:2],
                            axis=1) < self._success_distance_thresh))[0]) - 1
                    break
            # check success condition and give returns
            # if reward shaping is used, the reward is the stacking number plus:
            #   if not gripping, - distance to the closest obj not being stacked
            #   if gripping, distance to closet stacking candidate:
            #       (x, y, target_height)
            #       target_height: (stacked_num + 1) * object_height,
            #           plus a margin 0.55 * object_height
            # being_grasped: contacts are True and off the ground
            target_height_by_half_obj_height = 3.1 + stacked_obj_num * 2.0
            grasped_obj_index = np.where(
                (l_contacts * r_contacts) *
                (obj_heights / self._object_half_height > 2.0))[0]
            # success flag: all objects are stacked and no contacts to gripper
            succ_flag = (stacked_obj_num == self._objects_num -
                         1) and np.sum(contacts) == 0
            succ_cnt = succ_cnt + 1 if succ_flag else 0
            # give returns
            if succ_cnt >= 5:  # successfully stacked and leave the objects for 5 steps
                logging.debug("object has been successfuly placed")
                reward = 200.0 * self._objects_num if self._reward_shaping else 1.0
                agent_sentence = yield TeacherAction(
                    reward=reward, sentence="well done", done=True)
            elif len(grasped_obj_index) == 0:  # nothing is being grasped
                if stacked_obj_num == 0:
                    unstacked_obj_list = obj_list
                else:
                    unstacked_obj_list = np.where(
                        np.linalg.norm(
                            obj_positions_xy -
                            obj_positions_xy[bottom_obj[0]][:2],
                            axis=1) >= self._success_distance_thresh)[0]
                if len(unstacked_obj_list) == 0:
                    # all are stacked, this can hapen during the last steps before success
                    stage_reward = 0.5
                else:
                    closest_obj_in_unstacked = np.argmin(
                        np.linalg.norm(
                            obj_positions[unstacked_obj_list] - finger_pos,
                            axis=1))
                    closest_obj = unstacked_obj_list[closest_obj_in_unstacked]
                    distance_to_closest_obj = np.linalg.norm(
                        obj_positions[closest_obj] - finger_pos)
                    lifted = obj_heights[
                        closest_obj] / self._object_half_height - 1.0
                    stage_reward = (0.5 * contacts[closest_obj] + max(
                        1.0 - distance_to_closest_obj / self._avg_distance_max,
                        0) + min(lifted, 1.0)) / 3.0
                reward = stacked_obj_num + 0.5 * stage_reward if self._reward_shaping else 0
                agent_sentence = yield TeacherAction(reward=reward, done=False)
            else:  # an object is being grasped
                if stacked_obj_num == 0:  # any target on the ground is fine, prefer the closest one
                    target_list = np.delete(obj_list, grasped_obj_index[0])
                    target_id = np.argmin(
                        np.linalg.norm(
                            obj_positions[target_list] -
                            obj_positions[grasped_obj_index[0]],
                            axis=1))
                    target_pos = obj_positions[target_list][target_id]
                else:
                    target_id = bottom_obj[0]
                    target_pos = obj_positions[target_id]
                dist_xy = np.linalg.norm(
                    obj_positions[grasped_obj_index[0]][:2] - target_pos[:2])
                dist_z = abs((obj_positions[grasped_obj_index[0]][2] /
                              self._object_half_height) /
                             target_height_by_half_obj_height - 1.0)
                stage_reward = 1.0 - min(
                    dist_xy / self._objects_range + dist_z, 2.0) / 2.0
                reward = stacked_obj_num + 0.5 + 0.5 * stage_reward if self._reward_shaping else 0
                agent_sentence = yield TeacherAction(reward=reward, done=False)

        yield TeacherAction(reward=-1.0, sentence="failed", done=True)

    def task_specific_observation(self, agent):
        """
        Args:
            agent (GazeboAgent): the agent
        Returns:
            np.array, the observations of the task for non-image case
        """
        # object poses and contacts
        obj_poses = []
        l_contacts = []
        r_contacts = []
        for obj_index in range(self._objects_num):
            # get object's position
            obj_pos, obj_rot = self._objects[obj_index].get_pose()
            obj_poses.append(obj_pos)
            obj_poses.append(obj_rot)
            # get contacts
            l_contacts.append(1.0 * self._agent.get_contacts(
                'finger_cnta_l',
                self._object_names[obj_index] + self._object_collision_name))
            r_contacts.append(1.0 * self._agent.get_contacts(
                'finger_cnta_r',
                self._object_names[obj_index] + self._object_collision_name))
        obj_poses = np.array(obj_poses).flatten()
        l_contacts = np.array(l_contacts)
        r_contacts = np.array(r_contacts)
        contacts = l_contacts + r_contacts
        # Use 3 points to uniquely determine the 6 + 2 DoF gripper
        finger_l_pos, _ = agent.get_link_pose(self._finger_link_l)
        finger_r_pos, _ = agent.get_link_pose(self._finger_link_r)
        palm_pos, _ = agent.get_link_pose(self._palm_link)
        gripper_states = np.array([finger_l_pos, finger_r_pos,
                                   palm_pos]).flatten()
        # agent self states
        agent_pose = np.array(agent.get_pose()).flatten()
        joints_states = agent.get_internal_states()
        return np.concatenate(
            (agent_pose, joints_states, gripper_states, contacts, obj_poses),
            axis=0)
