# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import unittest
import random
from social_bot.util.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from collections import namedtuple

Experience = namedtuple("Experience", field_names=["state", "action", "done"])


class TestReplayBuffer(unittest.TestCase):
    def test_replay_buffer(self):
        buf_len = 14
        replay_buffer = PrioritizedReplayBuffer(14, 3, 2)
        dones = [
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 1
        ]
        valid_flag = [
            0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1,
            1, 1
        ]
        for i in range(len(dones)):
            exp = Experience(random.random(), i, dones[i])
            replay_buffer.add_experience(exp)
            if i + len(dones) < buf_len:
                end = i + 1
                begin = max(end - buf_len, 0)
                self.assertEqual(replay_buffer.num_valid_experiences(),
                                 sum(valid_flag[begin:end]))
        features, indices, is_weights = \
            replay_buffer.get_sample_features(100, self.make_sample)
        for idx in indices:
            self.assertGreater(idx, 2)
            self.assertLess(idx, buf_len - 2)
        replay_buffer.update_priority(range(buf_len), [0] * buf_len)

        # verify the sample distribution is correct
        p1 = 8.
        p2 = 15.
        p3 = 9.
        n = 10000
        replay_buffer.update_priority([5, 6, 11], [p1, p2, p3])
        p = p1 + p2 + p3
        p1 /= p
        p2 /= p
        p3 /= p
        indices, is_weights = replay_buffer.get_sample_indices(n)
        self.assertEqual(len(indices), n)
        n1 = sum([x == 5 for x in indices])
        n2 = sum([x == 6 for x in indices])
        n3 = sum([x == 11 for x in indices])
        print("Expectation: ", n * p1, n * p2, n * p3)
        print("Actual: ", n1, n2, n3)
        self.assertLess(abs(n1 - p1 * n), 1)
        self.assertLess(abs(n2 - p2 * n), 1)
        self.assertLess(abs(n3 - p3 * n), 1)
        self.assertEqual(n1 + n2 + n3, n)

    def make_sample(self, e0, e1, e2, e3, e4, e5):
        return e3.state, e3.action, e3.done


if __name__ == '__main__':
    unittest.main()
