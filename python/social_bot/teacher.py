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

from abc import abstractmethod
import numpy as np
import random
import gym
from gym import spaces


class DiscreteSequence(gym.Space):
    """
    Gym.Space object for encoded language sequence
    This class also provid helper functions to convert sequence and sentence
        to each other by BPEmb: https://nlp.h-its.org/bpemb/
    """
    def __init__(self, vocab_size=1000, max_length=20):
        """
        Args:
            vocab_size (int): number of different tokens
                support 1000, 3000, 5000, 10000, 25000 for now
            max_length (int): maximal length of the sequence
        """
        from bpemb import BPEmb
        super()
        self._vocab_size = vocab_size
        self._max_length = max_length
        self.dtype = np.int32
        self.shape = (max_length)
        self._bpemb_en = BPEmb(lang="en", vs=vocab_size)

    def from_sentence(self, sentence):
        """
        Get subword discrete sequence from a string.
        Args:
            sentence (string): the sentence 
        Returns:
            sequence (numpy array): the discrete sequence
        """
        sequence = np.array(self._bpemb_en.encode_ids(sentence))
        return np.pad(sequence,
                      (0, self._max_length - sequence.shape[0]),
                      'constant')

    def to_sentence(self, sequence):
        """
        Get sentence from discrete sequence.
        Args:
            sequence (numpy array): the discrete sequence 
        Returns:
            sentence (string): the sentence 
        """
        return self._bpemb_en.decode_ids(sequence)

    def to_bpemb_embedding(self, sequence):
        """
        Get pre-trained subword embeddings by bpemb.
        Args:
            sequence (numpy array): the discrete sequence 
        Returns:
            pre-trained embedding by bpemb (numpy array) 
        """
        return self._bpemb_en.vectors(sequence)


class TeacherAction(object):
    def __init__(self, reward=0.0, sentence="", done=False, is_idle=False):
        """
        Arguments
            done: end of an episode if true
        """
        self.reward = reward
        self.sentence = sentence
        self.done = done
        self.is_idle = is_idle


class Task(object):
    @abstractmethod
    def run(self):
        """
        run() use yield to generate TeacherAction
        Structure of run():

        def run(self, agent, world):
          ...
          # agent_sentence is provided by Teacher using send() in TaskGroup.run_stage()
          agent_sentence = yield  # the first yielded value is ignored
          ...
          # TeacherAction will be passed to Teacher as the return value of send() in TaskGroup.run_stage()
          agent_sentence = yield TeacherAction(...)
          ...
          agent_sentence = yield TeacherAction(...)
          ...
          yield TeacherAction(done=True)
        """
        pass


class TaskGroup(object):
    _tasks = []
    _current_task = None
    _agent = None
    _world = None
    _is_idle = True

    def add_task(self, task):
        """
        Add a task to the group
        Arguments:
            task: an instance of Task
        """
        self._tasks.append(task)

    def run_stage(self, agent_sentence):
        task = self._get_current_task()
        try:
            # teacher_action is the value yielded in task
            teacher_action = task.send(agent_sentence)
            self._is_idle = teacher_action.is_idle
            if teacher_action.done:
                task.close()
                self._current_task = None
                self._is_idle = True
        except StopIteration:
            task.close()
            self._current_task = None
            self._is_idle = True
            teacher_action = TeacherAction()

        return teacher_action

    def is_idle(self):
        return self._is_idle

    def reset(self, agent, world):
        self._agent = agent
        self._world = world
        if self._current_task is not None:
            self._current_task.close()
            self._current_task = None

    def _get_current_task(self):
        if self._current_task is None:
            tid = random.randint(0, len(self._tasks) - 1)
            self._current_task = self._tasks[tid].run(self._agent, self._world)
            # run() will execute until the first yield. We ignore the first yielded value.
            self._current_task.send(None)
        return self._current_task


class Teacher(object):
    """
    A teacher has several task groups. At each step
    * task_groups_exclusive is True
    Only one task group will run at the same time. After the active become idle,
    another one will be chosen randomly.
    * task_groups_exclusive is False
    All the task groups run concurrently. The reward are sum together. The first
    nonempty sentence will be used. If one of the action has done=True, the
    resulted done will be True.
    """
    _task_groups = []
    _weights = []
    _task_groups_exclusive = True

    def __init__(self, task_groups_exclusive=True):
        self._task_groups_exclusive = task_groups_exclusive

    def add_task_group(self, task_group, weight=1):
        self._task_groups.append(task_group)
        self._weights.append(weight)

    def reset(self, agent, world):
        for g in self._task_groups:
            g.reset(agent, world)
        self._switch_task_group()

    def _switch_task_group(self):
        self._current_task_group = np.random.choice(
            self._task_groups, p=np.array(self._weights) / sum(self._weights))

    def teach(self, agent_sentence):
        if self._task_groups_exclusive:
            if self._current_task_group.is_idle():
                self._switch_task_group()
            return self._current_task_group.run_stage(agent_sentence)
        else:
            final_sentence = ''
            final_reward = 0.
            done = False
            active_group_id = -1
            # run all groups in parallel
            for i, g in enumerate(self._task_groups):
                teacher_action = g.run_stage(agent_sentence)
                if teacher_action.done:
                    done = True
                final_reward += teacher_action.reward
                if not final_sentence:
                    final_sentence = teacher_action.sentence
                    active_group_id = i
            if active_group_id != -1:
                g = self._task_groups.pop(active_group_id)
                self._task_groups.insert(0, g)
            return TeacherAction(final_reward, final_sentence, done)
