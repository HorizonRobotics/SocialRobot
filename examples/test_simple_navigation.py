# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import gym
import random
import social_bot
import logging


def main():
    env = gym.make("SocialBot-SimpleNavigation-v0")
    for _ in range(10000000):
        obs = env.reset()
        control = [random.random() * 0.2, random.random() * 0.2, 0]
        while True:
            obs, reward, done, info = env.step(
                dict(control=control, sentence="hello"))
            if done:
                logging.info("reward: " + str(reward) + "sent: " +
                             str(obs["sentence"]))
                break


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
