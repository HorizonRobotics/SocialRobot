# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import numpy as np
import random
from collections import deque
from .segment_tree import SumSegmentTree, MinSegmentTree
""" Circular array based implementatioon of queue."""


class CircularBuffer(object):
    def __init__(self, maxlen, default_value):
        self._buffer = [default_value] * maxlen
        self._buffer_size = maxlen
        self._current_idx = 0
        self._num_samples = 0

    def _idx_to_location(self, idx):
        idx += self._current_idx
        if idx >= self._buffer_size:
            idx -= self._buffer_size
        return idx

    def _location_to_idx(self, loc):
        loc -= self._current_idx
        if loc < 0:
            loc += self._buffer_size
        return loc

    def __getitem__(self, index):
        return self._buffer[self._idx_to_location(index)]

    def __setitem__(self, index, e):
        self._buffer[self._idx_to_location(index)] = e

    def __len__(self):
        """Return the current size of internal memory."""
        return self._num_samples

    def append(self, e):
        if self._num_samples == self._buffer_size:
            self._current_idx += 1
            if self._current_idx >= self._buffer_size:
                self._current_idx = 0
        else:
            self._num_samples += 1

        loc = self._idx_to_location(self._num_samples - 1)
        self._buffer[loc] = e


class ReplayBuffer(CircularBuffer):
    """Replay buffer for storing experience tuples."""

    def __init__(self, buffer_size, history_length, future_length):
        """
        Get samples from history, it is gauranteed that for each sample,
        its previous history_length samples are not the end of an episode (i.e. done == False)
        and its next future_length samples are valid
        """
        super(ReplayBuffer, self).__init__(buffer_size, None)
        self._history_length = history_length
        self._future_length = future_length
        self._previous_done = 0  # how long ago is the previous done
        # number of experience whose do_not_sample is False
        self._num_valid_experiences = 0
        self._do_not_sample_flags = CircularBuffer(buffer_size, False)

    def get_experience(self, index):
        return self[index]

    @property
    def initial_priority(self):
        return 1.0

    def add_experience(self, experience):
        """
        Add a new experience to memory. experience is an object which has 'done' attribute
        indicating the end of an episode.
        """
        if (len(self) == self._buffer_size
                and not self._do_not_sample_flags[0]):
            self._num_valid_experiences -= 1

        if self._previous_done < self._history_length:
            priority = 0
            self._do_not_sample_flags.append(True)
        else:
            priority = None
            self._do_not_sample_flags.append(False)
            self._num_valid_experiences += 1
        self._add_sample(priority)

        self.append(experience)
        if experience.done:
            self._previous_done = 0
        else:
            self._previous_done += 1

    def num_valid_experiences(self):
        return self._num_valid_experiences

    def get_sample_indices(self, num):
        """
        Randomly sample a batch of experiences from memory.
        Return
           indices: list of the indices of the samples
           is_weights: list of the importance weights of the sample
        """
        assert self.num_valid_experiences() > 0, "Not enough experiences yet."

        indices = []
        is_weights = []
        while len(indices) < num:
            idx = random.randint(self._history_length,
                                 len(self) - self._future_length - 1)
            weight = 1.
            if self._do_not_sample_flags[idx]:
                continue
            indices.append(idx)
            is_weights.append(weight)

        return indices, is_weights

    def get_sample_features(self, num, f_make_sample):
        """
        Arguments
          num: the number of samples
          f_make_sample: a function to generate features, let k = history_length + future_length + 1
             the signature of f_make_sample is:
             def f_make_sample(e1, e2, ..., ek):
               return tuple of features with the dimension of the axis 0 equal to 1
        Returns
          tuple of (batch_features, indices, is_weights)
          batch_features: a tuple of numpy arrays with the dimension of axis 0 as num
          indices: the index of the selected samples
          is_weights: a numpy array of the importance sampling weights
        """

        indices, is_weights = self.get_sample_indices(num)
        batch_features = []
        for idx in indices:
            exps = [
                self[i] for i in range(idx - self._history_length, idx +
                                       self._future_length + 1)
            ]
            features = f_make_sample(*exps)
            if not batch_features:
                for _ in range(len(features)):
                    batch_features.append([])
            for feature, batch_feature in zip(features, batch_features):
                batch_feature.append(feature)
        for i, batch_feature in enumerate(batch_features):
            batch_features[i] = np.vstack(batch_feature)

        return batch_features, indices, np.vstack(is_weights).astype(
            np.float32)

    def _add_sample(self, priority=None):
        """
        called before self._buffer is appended with the new experience
        """
        return

    def update_priority(self, indices, priorities):
        """
        Update the priorities of the samples for prioritized replay
        """
        return


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Reply buffer which samples experience with probability proportional to their priorities
    """

    def __init__(self, buffer_size, history_length, future_length):
        super(PrioritizedReplayBuffer,
              self).__init__(buffer_size, history_length, future_length)
        self._sum_tree = SumSegmentTree(buffer_size)
        self._min_tree = MinSegmentTree(buffer_size)
        for i in range(buffer_size):
            self._min_tree[i] = 1.0
        self._current_idx = 0
        self._max_priority = 1.0

    @property
    def initial_priority(self):
        return self._max_priority

    def get_priority(self, idx):
        idx = self._idx_to_location(idx)
        return self._sum_tree[idx]

    def update_priority(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            idx = self._idx_to_location(idx)
            self._sum_tree[idx] = priority
            self._min_tree[idx] = priority
            self._max_priority = max(self._max_priority, priority)

    def get_sample_indices(self, num):
        indices = []
        weights = []
        total = self._sum_tree.summary()
        min_idx = self._history_length
        max_idx = len(self) - self._future_length - 1
        min_priority = self._min_tree.summary()
        while len(indices) < num:
            n_remaining = num - len(indices)
            for i in range(n_remaining):
                r = total * (i + random.random()) / n_remaining
                idx = self._sum_tree.find_sum_bound(r)
                priority = self._sum_tree[idx]
                idx = self._location_to_idx(idx)
                if idx < min_idx or idx > max_idx or self._do_not_sample_flags[
                        idx]:
                    continue
                indices.append(idx)
                weights.append(min_priority / priority)
        return indices, weights

    def _add_sample(self, priority=None):
        if priority is None:
            priority = self._max_priority
        idx = self._idx_to_location(self._num_samples - 1)
        self._sum_tree[idx] = priority
        if priority > 0:
            self._min_tree[idx] = priority
