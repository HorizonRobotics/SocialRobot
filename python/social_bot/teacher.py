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
"""Teacher framework."""

import numpy as np
import random
import gym
from absl import logging


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
        super().__init__(shape=(max_length, ), dtype=np.int32)
        self._vocab_size = vocab_size
        self._max_length = max_length


class TeacherAction(object):
    def __init__(self, reward=0.0, sentence="", done=False, is_idle=False,
                 success=False):
        """
        Args:
            done: end of an episode if true
            success: if the episode is successful or not
        """
        self.reward = reward
        self.sentence = sentence
        self.done = done
        self.is_idle = is_idle
        self.success = success


class TaskGroup(object):
    """A group of tasks.

    Each task group consists of one or more tasks. Within one task group, one
    task can run at one time. A random task is chosen after the current task is
    finished.
    """

    def __init__(self):
        self._tasks = []
        self._current_tid = None
        self._current_task = None
        self._current_reward_weight = 1.0
        self._agent = None
        self._world = None
        self._is_idle = True

    def add_task(self, task):
        """Add a task to the group.

        Args:
            task (Task): an instance of Task
        Returns:
            None
        """
        self._tasks.append(task)

    def teach(self, agent_sentence):
        """Generate TeacherAction.

        Args:
            agent_sentence (str): sentence from the agent
        Returns:
            TeacherAction
        """
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
        """Reset the task group.

        Current task will be closed and a random new one will be chosen.

        Args:
            agent (GazeboAgent): the learning agent in the world
            world (pygazebo.World): the world containing the agent
        Returns:
            None
        """
        self._agent = agent
        self._world = world
        if self._current_task is not None:
            self._current_task.close()
            self._current_task = None

    # This function only returns a generator function.
    # To get the task object use self._tasks[self._current_tid]
    def _get_current_task(self):
        if self._current_task is None:
            tid = random.randint(0, len(self._tasks) - 1)
            self._current_tid = tid
            self._current_task = self._tasks[tid].run()
            self._current_reward_weight = self._tasks[tid].reward_weight
            # This send will cause self._current_task to execute until the first
            # yield. We ignore the first yielded value.
            self._current_task.send(None)
        return self._current_task

    def get_current_reward_weight(self):
        """Get reward weight for current task of the group

        Args:
            None
        Returns:
            float, the reward weight of current task
        """
        return self._current_reward_weight

    def get_tasks(self):
        """Get current tasks in the group.

        Args:
            None
        Returns:
            list, a list of current tasks in the group
        """
        return self._tasks


class Teacher(object):
    """Teacher is for teaching the agent.

    It is responsible for:
    1. Giving reward
    2. Arranging the environment
    3. Generating sentences
    4. Interpreting sentences from the agent

    A teacher has several task groups. At each step
    * If task_groups_exclusive is True
      Only one task group will run at the same time. After the active become
      idle, another one will be chosen randomly.
    * If task_groups_exclusive is False
      All the task groups run concurrently. The reward are sum together. The
      first nonempty sentence will be used. If one of the action has done=True,
      the resulted done will be True.

    Each task group consists of one or more tasks. Within one task group, one
    task can run at one time. A random task is chosen after the current task is
    finished.
    """

    def __init__(self, task_groups_exclusive=True):
        """Create a Teacher instance.

        Args:
            task_groups_exclusive (bool): If True, only one task group is active
                at one time. Otherwise, multiple task groups run concurrently.
        """
        self._task_groups_exclusive = task_groups_exclusive
        self._vocab_list = None
        self._task_groups = []
        self._weights = []
        self.vocab_size = 0

    def add_task_group(self, task_group, weight=1):
        """Add a task group to teacher.

        Args:
            task_group (TaskGroup): TaskGroup to be added
            weight (float): In task_groups_exclusive=True mode, the probability
                of a TaskGroup being chosen is proportional to this value.
        Returns:
            None
        """
        self._task_groups.append(task_group)
        self._weights.append(weight)

    def get_task_groups(self):
        """Get current task groups of teacher.

        Args:
            None
        Returns:
            list, a list of current task group
        """
        return self._task_groups

    def get_task_specific_observation(self, agent):
        """Get the task specific observation of all the tasks added to the teacher

        Args:
            agent (GazeboAgent): the agent
        Returns:
            numpy.array, the specific observation for all the tasks added
        """
        task_specific_ob = np.array([])
        for task_group in self.get_task_groups():
            for task in task_group.get_tasks():
                task_specific_ob = np.append(
                    task_specific_ob, task.task_specific_observation(agent))
        return task_specific_ob

    def _build_vocab_from_tasks(self):
        """Build vocabulary table."""
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
        """Convert sentence string to numpy integer sequence.

        Args:
            sentence (str): string for the sentence. Note the currently, the
                tokenization is case-sensitive. For example, "This" and "this"
                are treated as word.
            max_sequence_length (int): The length of the generated numpy array.
                If number of words in sentence is smaller than this value, 0 is
                padded at the end.
        Returns:
            numpy.array
        """
        if self._vocab_list is None:
            self._build_vocab_from_tasks()
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
        """Convert integer sequence to str based on vocabulary table.

        Values after the first 0 in the sequence are ignored. In the generated
        string, words are separated by space ' '.

        Args:
            sequence (int[]): integer sequence
        Returns:
            str
        """
        if self._vocab_list is None:
            self._build_vocab_from_tasks()
        for seq_index in range(len(sequence)):
            assert sequence[seq_index] < self.vocab_size, \
                "Unknown word id: " + str(sequence[seq_index]) + \
                ", during decoding sequence to sentence"
            if sequence[seq_index] == 0:
                break
        word_list = list(
            map(lambda x: self._vocab_list[x], sequence[:seq_index]))
        return " ".join(word_list)

    def reset(self, agent, world):
        """Reset teacher.

        All the task group will be reset, that is, current task in each task
        group is closed and a random new one will be chosen.

        Args:
            agent (GazeboAgent): the learning agent in the world
            world (pygazebo.World): the world containing the agent
        Returns:
            None
        """
        for g in self._task_groups:
            g.reset(agent, world)
        self._switch_task_group()

    def _switch_task_group(self):
        self._current_task_group = np.random.choice(
            self._task_groups, p=np.array(self._weights) / sum(self._weights))

    def teach(self, agent_sentence):
        """Generate TeacherAction.

        Args:
            agent_sentence (str): sentence from the agent
        Returns:
            TeacherAction
        """
        return_action = None
        if self._task_groups_exclusive:
            if self._current_task_group.is_idle():
                self._switch_task_group()
            return_action = self._current_task_group.teach(agent_sentence)
        else:
            final_sentence = ''
            final_reward = 0.
            done = False
            active_group_id = -1
            success = False
            # run all groups in parallel
            for i, g in enumerate(self._task_groups):
                teacher_action = g.teach(agent_sentence)
                if teacher_action.done:
                    done = True
                if teacher_action.success:
                    success = True
                weight = g.get_current_reward_weight()
                final_reward += weight * teacher_action.reward
                if not final_sentence:
                    final_sentence = teacher_action.sentence
                    active_group_id = i
            if active_group_id != -1:
                g = self._task_groups.pop(active_group_id)
                self._task_groups.insert(0, g)
            return_action = TeacherAction(final_reward, final_sentence, done,
                                          success=success)
        return return_action
