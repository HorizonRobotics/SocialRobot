# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import gym
import random
import social_bot
import logging
import time
import psutil
import os


def main():
    env = gym.make("SocialBot-SimpleNavigationLanguage-v0")
    steps = 0
    t0 = time.time()
    proc = psutil.Process(os.getpid())
    logging.info(" mem=%dM" % (proc.memory_info().rss // 1e6))
    for _ in range(10000000):
        obs = env.reset()
        control = [random.random() * 0.2, random.random() * 0.2]
        while True:
            obs, reward, done, info = env.step(
                dict(control=control, sentence="hello"))
            steps += 1
            if done:
                logging.info("reward: " + str(reward) + "sent: " +
                             str(obs["sentence"]))
                break
        logging.info("steps=%s" % steps +
                     " frame_rate=%s" % (steps / (time.time() - t0)) +
                     " mem=%dM" % (proc.memory_info().rss // 1e6))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
