# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import operator


class SegmentTree(object):
    """
    Data structure to allow efficient calculate the summary statistics over a
    segment of elements.
    See https://en.wikipedia.org/wiki/Segment_tree for detail.
    """

    def __init__(self, capacity, op):
        """
        Arguments
          capacity: the number of elements this tree holds
          op: a binary operator.
        """
        self._vals = [0] * (2 * capacity)
        self._capacity = capacity
        self._op = op
        self._leftest_leaf = 1
        while self._leftest_leaf < capacity:
            self._leftest_leaf *= 2

    def __setitem__(self, idx, val):
        op = self._op
        vals = self._vals
        idx = self._index_to_leaf(idx)
        vals[idx] = val
        idx //= 2
        while idx >= 1:
            vals[idx] = op(vals[2 * idx], vals[2 * idx + 1])
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._vals[self._index_to_leaf(idx)]

    def summary(self):
        return self._vals[1]

    def _index_to_leaf(self, idx):
        """
        Make sure idx=0 is the leftest leaf.
        """
        idx += self._leftest_leaf
        if idx >= 2 * self._capacity:
            idx -= self._capacity
        return idx

    def _leaf_to_index(self, leaf):
        idx = leaf - self._leftest_leaf
        if idx < 0:
            idx += self._capacity
        return idx


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(capacity, operator.add)

    def find_sum_bound(self, x):
        """
        return value is the minimum index such that x < self[0] + ... + self[idx] 
        Arguments
          x: a value smaller than self.summary()
        """
        idx = 1
        capacity = self._capacity
        vals = self._vals
        while idx < capacity:
            idx *= 2
            if x >= vals[idx]:
                x -= vals[idx]
                idx += 1
        return self._leaf_to_index(idx)


class MaxSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MaxSegmentTree, self).__init__(capacity, max)


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(capacity, min)
