# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import unittest
import multiprocessing
from multiprocessing import Process, Value
from simple_navigation import SimpleNavigation
import random
import os


class Agent(Process):
    def __init__(self, agent_id):
        super(Agent, self).__init__()
        self.ok = Value('i', 0)
        self.agent_id = agent_id

    def run(self):
        port = os.environ.get('PYGAZEBO_PORT', 11345)
        env = SimpleNavigation(port=port + self.agent_id + 1)
        env.reset()
        for _ in range(1000):
            control = [
                random.random() * 0.2 - 0.1,
                random.random() * 0.2 - 0.1, 0
            ]
            obs, reward, done, info = env.step(
                dict(control=control, sentence="hello"))
            if done:
                env.reset()
        self.ok.value = 1


class TestMultiProcess(unittest.TestCase):
    def test_multiprocessing(self):
        env = SimpleNavigation()
        env.reset()
        agents = [Agent(i) for i in range(2)]
        for agent in agents:
            agent.start()
        for agent in agents:
            agent.join()
        for agent in agents:
            self.assertTrue(agent.ok.value)


if __name__ == '__main__':
    # we use spawn to make sure pygazebo has a clean state in each subprocess
    multiprocessing.set_start_method('spawn')
    unittest.main()
