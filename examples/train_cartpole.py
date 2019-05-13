# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import gym
import social_bot

from baselines import ppo2
from baselines import deepq
from baselines import ddpg
from baselines.run import main as M
import numpy as np


def ppo2():
    M('--alg=ppo2 --env=SocialBot-CartPole-v0 --num_timesteps=1e6 --save_path=~/models/cartpole'
      .split(' '))


def cem(num_timesteps=200, n_samples=400, top_frac=0.2, smooth_alpha=0.9):
    """
    cross entropy model for policy search.
    https://www.aaai.org/Papers/ICML/2003/ICML03-068.pdf
    https://people.smp.uq.edu.au/DirkKroese/ps/CEopt.pdf
    """

    env = gym.make('SocialBot-CartPole-v0')

    def actions(obs, theta):
        # no biase of theta[-1] here
        return np.clip(
            np.array(obs).dot(theta[:-1]), env.action_space.low,
            env.action_space.high)

    mean = np.random.randn(env.observation_space.shape[0] + 1)
    var = np.square(np.ones_like(mean) * 1)

    for it in range(num_timesteps):
        theta_samples = np.transpose(
            np.array([
                np.random.normal(u, np.sqrt(o), n_samples)
                for u, o in zip(mean, var)
            ]))
        top_n = int(np.round(top_frac * n_samples))
        rewards_sample = np.array([0.0] * n_samples)

        for its in range(n_samples):
            obs = env.reset()
            total_rewards = 0

            done = False
            while not done:
                a = actions(obs, theta_samples[its])
                (obs, r, done, _) = env.step(a)
                total_rewards += r
            rewards_sample[its] = total_rewards
        print("Iteration {}. Episode Reward: {}".format(
            it, rewards_sample.mean(axis=0)))

        top_idxs = rewards_sample.argsort()[::-1][:top_n]
        top_theta_samples = theta_samples[top_idxs]

        #inject noise to support initial exploring
        v = max(5 - it / 10, 0)
        top_mean = top_theta_samples.mean(axis=0)
        top_var = top_theta_samples.var(axis=0) + v

        mean = smooth_alpha * top_mean + (1.0 - smooth_alpha) * mean
        var = smooth_alpha * top_var + (1.0 - smooth_alpha) * var
        print("mean: ", mean, ", var: ", var)


if __name__ == '__main__':
    ppo2()
