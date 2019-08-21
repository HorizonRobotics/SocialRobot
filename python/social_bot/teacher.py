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
from absl import logging
from gym import spaces


class DiscreteSequence(gym.Space):
    """
    gym.Space object for language sequence
    """

    def __init__(self, vocab_size, max_length):
        """
        Args:
            vocab_size (int): number of different tokens
            max_length (int): maximal length of the sequence
        """
        super()
        self._vocab_size = vocab_size
        self._max_length = max_length
        self.dtype = np.int32
        self.shape = (max_length)


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
    def __init__(self):
        self._tasks = []
        self._current_task = None
        self._agent = None
        self._world = None
        self._is_idle = True

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

    def get_tasks(self):
        return self._tasks

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
    vocab_size = 0

    def __init__(self, task_groups_exclusive=True):
        self._task_groups_exclusive = task_groups_exclusive

    def add_task_group(self, task_group, weight=1):
        self._task_groups.append(task_group)
        self._weights.append(weight)

    def get_task_group(self):
        return self._task_groups

    def build_vocab_from_tasks(self):
        # Initialize vocab with '0' by index 0, which is used for padding
        vocab_list = [
            0,
        ]
        for g in self._task_groups:
            for t in g._tasks:
                vocab_list = vocab_list + t.task_vocab
        # Remove repeated words and convert to dict
        self._vocab_list = sorted(set(vocab_list), key=vocab_list.index)
        self.vocab_size = len(self._vocab_list)
        self._vocab_dict = dict(
            zip(self._vocab_list, list(range(0, self.vocab_size))))

    def sentence_to_sequence(self, sentence, max_sequence_length):
        word_list = sentence.split()
        for word in word_list:
            assert word in self._vocab_dict.keys(), \
                "Word is out of vocab: " + word + \
                ", during encoding sentence to sequence"
        sequence = list(map(lambda x: self._vocab_dict[x], word_list))
        padding_num = max_sequence_length - len(sequence)
        assert padding_num >= 0, "Sequence " + str(sequence) + \
            " exceed max_sequence_length: " + str(max_sequence_length) + \
            ", consider to increase the max_sequence_length"
        return np.pad(sequence, (0, padding_num), 'constant')

    def sequence_to_sentence(self, sequence):
        for seq_index in range(len(sequence)):
            assert sequence[seq_index] < self.vocab_size, \
                "Unknown word id: " + str(sequence[seq_index]) + \
                ", during decoding sequence to sentence"
            if sequence[seq_index] == 0: break
        word_list = list(
            map(lambda x: self._vocab_list[x], sequence[:seq_index]))
        return " ".join(word_list)

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
