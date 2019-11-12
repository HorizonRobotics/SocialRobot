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
A simple enviroment for an agent play on the ground
"""
import numpy as np
import gin
from absl import logging

from social_bot import teacher
from social_bot import tasks
from social_bot.tasks import GoalTask, ICubAuxiliaryTask, KickingBallTask
from play_ground import PlayGround


@gin.configurable
class EmbodiedTeacher(PlayGround):
    """
    This envionment is the playgound with an embodied teacher in it.
    It supports demonstrations by teacher's movements. The demonstrations
    could be generated by an existing teacher policy or by a keyboard control
    interface for human.

    Existing teacher policy can be obtained by training on the same task
    using PlayGround.

    Demonstrations from human is through keyboard. Note that you should keep the
    terminal window on the forefront to capture the key being pressed.
    Some tricks are used to make the keyboard controlling a little bit more
    friendly. Move the agent around by key "WASD" and open or close gripper by
    key "E", and control the robot arm(if there is) by "IJKL".
    """

    def __init__(self,
                 agent_type='youbot_noplugin',
                 tasks=[GoalTask],
                 with_language=False,
                 use_image_observation=False,
                 image_with_internal_states=False,
                 world_time_precision=None,
                 step_time=0.1,
                 real_time_update_rate=0,
                 port=None,
                 action_cost=0.0,
                 resized_image_size=(64, 64),
                 vocab_sequence_length=20,
                 demo_by_human=False):
        """
        Args:
            agent_type (string): Select the agent robot, supporting pr2_noplugin,
                pioneer2dx_noplugin, turtlebot, youbot_noplugin for now
                note that 'agent_type' should be the same str as the model's name
            tasks (list): a list of teacher.Task, e.g., GoalTask, KickingBallTask
            with_language (bool): The observation will be a dict with an extra sentence
            use_image_observation (bool): Use image, or use low-dimentional states as
                observation. Poses in the states observation are in world coordinate
            image_with_internal_states (bool): If true, the agent's self internal states
                i.e., joint position and velocities would be available together with image.
                Only affect if use_image_observation is true
            world_time_precision (float|None): this parameter depends on the agent. 
                if not none, the default time precision of simulator, i.e., the max_step_size
                defined in the agent cfg file, will be override. Note that pr2 and iCub
                requires a max_step_size <= 0.001, otherwise cannot train a successful policy.
            step_time (float): the peroid of one step() function of the environment in simulation.
                step_time is rounded to multiples of world_time_precision
                step_time / world_time_precision is how many simulator substeps during one
                environment step. for the tasks need higher control frequency (such as the 
                tasks need walking by 2 legs), using a smaller step_time like 0.05 is better.
                experiments show that iCub can not learn how to walk in a 0.1 step_time
            real_time_update_rate (int): max update_rate per seconds. there is no limit if
                this is set to 0. if 1:1 real time is prefered(like playing or recording video),
                this should be set to 1.0/world_time_precision.
            port: Gazebo port, need to specify when run multiple environment in parallel
            action_cost (float): Add an extra action cost to reward, which helps to train
                an energy/forces efficency policy or reduce unnecessary movements
            resized_image_size (None|tuple): If None, use the original image size
                from the camera. Otherwise, the original image will be resized
                to (width, height)
            vocab_sequence_length (int): the length of encoded sequence
            demo_by_human (bool): demo by human or by existing teacher policy
        """
        self._demo_by_human = demo_by_human
        if self._demo_by_human:
            from social_bot.keybo_control import KeyboardControl
            self.step = self._step_with_human_demo
            self._keybo = KeyboardControl()
            real_time_update_rate = 500  # run "gz physics -u" to override
        else:  # demo by teacher policy
            self.step = self._step_with_teacher_policy

        super().__init__(
            agent_type=agent_type,
            world_name="play_ground.world",
            tasks=tasks,
            with_language=with_language,
            use_image_observation=use_image_observation,
            image_with_internal_states=image_with_internal_states,
            world_time_precision=world_time_precision,
            step_time=step_time,
            real_time_update_rate=real_time_update_rate,
            port=port,
            action_cost=action_cost,
            resized_image_size=resized_image_size,
            vocab_sequence_length=vocab_sequence_length)

        # insert teacher model
        self.insert_model(
            model=agent_type, name="teacher", pose="0 -2 0 0 0 0")
        # set up teacher joints
        self._teacher_joints = []
        for joint in self._agent_joints:
            self._teacher_joints.append("teacher::" + joint)
        self._teacher_embodied = self._world.get_agent('teacher')

    def reset(self):
        if self._demo_by_human:
            self._keybo.reset()
        obs = super().reset()
        if self._demo_by_human or self._is_training_for_teacher_policy:
            return obs
        else:
            return obs, obs

    def _step_with_teacher_policy(self, teacher_action, agent_action):
        obs, reward, done, _ = self._step_with_teacher_action(
            teacher_action, agent_action)
        teacher_obs = self._get_teacher_obs()
        return teacher_obs, obs, reward, done, {}

    def _step_with_human_demo(self, agent_action):
        teacher_action = self._keybo.get_agent_actions(self._agent_type)
        return self._step_with_teacher_action(teacher_action, agent_action)

    def _step_with_teacher_action(self, teacher_action, agent_action):
        if self._with_language:
            sentence = agent_action.get('sentence', None)
            if type(sentence) != str:
                sentence = self._teacher.sequence_to_sentence(sentence)
            controls = agent_action['control']
        else:
            sentence = ''
            controls = agent_action
        self._take_action(self._agent, self._agent_joints, controls)
        self._take_action(self._teacher_embodied, self._teacher_joints,
                          teacher_action)
        self._world.step(self._sub_steps)
        teacher_action = self._teacher.teach(sentence)
        obs = self._get_observation_with_sentence(teacher_action.sentence)
        self._steps_in_this_episode += 1
        ctrl_cost = np.sum(np.square(controls)) / controls.shape[0]
        reward = teacher_action.reward - self._action_cost * ctrl_cost
        self._cum_reward += reward
        if teacher_action.done:
            logging.debug("episode ends at cum reward:" +
                          str(self._cum_reward))
        return obs, reward, teacher_action.done, {}

    def _take_action(self, agent, joints, action):
        controls = np.clip(action, -1.0, 1.0) * self._agent_control_range
        controls_dict = dict(zip(joints, controls))
        agent.take_action(controls_dict)

    def _get_teacher_obs(self):
        agent = self._agent
        self._agent = self._teacher_embodied
        for group in self._teacher._task_groups:
            for task in group._tasks:
                task._agent = self._teacher_embodied
        teacher_obs = self._get_low_dim_full_states()
        self._agent = agent
        for group in self._teacher._task_groups:
            for task in group._tasks:
                task._agent = agent
        return teacher_obs


def main():
    """
    Simple testing of this environment.
    """
    import matplotlib.pyplot as plt
    import time
    with_language = False
    use_image_obs = False
    image_with_internal_states = True
    fig = None
    demo_by_human = True
    env = EmbodiedTeacher(
        with_language=with_language,
        use_image_observation=use_image_obs,
        image_with_internal_states=image_with_internal_states,
        tasks=[GoalTask],
        demo_by_human=demo_by_human)
    env.render()
    step_cnt = 0
    last_done_time = time.time()
    while True:
        actions = env._control_space.sample()
        teacher_actions = env._control_space.sample()
        if with_language:
            actions = dict(control=actions, sentence="hello")
            teacher_actions = dict(control=teacher_actions, sentence="hello")
        if demo_by_human:
            obs, _, done, _ = env.step(actions)
        else:
            teacher_obs, obs, _, done, _ = env.step(teacher_actions, actions)
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
            step_per_sec = step_cnt / (time.time() - last_done_time)
            logging.info("step per second: " + str(step_per_sec))
            step_cnt = 0
            last_done_time = time.time()


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main()
