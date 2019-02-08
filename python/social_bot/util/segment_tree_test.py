# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import unittest
import random
from segment_tree import SumSegmentTree, MaxSegmentTree


class TestSegmentTree(unittest.TestCase):
    def test_max_tree(self):
        for size in [1, 2, 3, 4, 7, 8, 9, 15, 16, 128]:
            tree = MaxSegmentTree(size)
            vals = [0] * size
            for _ in range(1000):
                i = random.randint(0, size - 1)
                v = random.randint(0, 1000000)
                vals[i] = v
                tree[i] = v
                self.assertEqual(tree.summary(), max(vals))
            for i in range(size):
                self.assertEqual(tree[i], vals[i])

    def test_sum_tree(self):
        for size in [3, 1, 2, 3, 4, 7, 8, 9, 15, 16, 128]:
            tree = SumSegmentTree(size)
            vals = [0] * size
            for _ in range(10000000):
                i = random.randint(0, size - 1)
                v = random.randint(0, 1000000)
                vals[i] = v
                tree[i] = v
                self.assertEqual(tree.summary(), sum(vals))
            s = 0
            for i in range(size):
                for j in range(16):
                    v = s + j / 16. * vals[i]
                    self.assertEqual(tree.find_sum_bound(v), i)
                s += vals[i]


if __name__ == '__main__':
    unittest.main()
