# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import gym
import social_bot

import logging
from baselines import ppo2
from baselines import deepq
from baselines import ddpg
from baselines.run import main as M
import tensorflow as tf
import numpy as np


# please refer to repo: github.com/HorizonRobotics/alf for a successful training setup for PR2 gripper
# task using PPO with tf_agents with 60 parallel environments with about
# 120 frames per sec

def ppo2():
    M('--alg=ppo2 --env=SocialBot-Pr2Gripper-v0 --num_timesteps=1e6 --network=cnn --save_path=~/models/pr2.model'
      .split(' '))

def ppo2_mlp():
    M('--alg=ppo2 --env=SocialBot-Pr2Gripper-v0 --num_timesteps=2e6 --network=lstm --save_path=~/models/pr2_lstm.model --num_env 1 --nminibatches=1 --ent_coef=0.01'
      .split(' '))


def ppo2_lstm():
    M('--alg=ppo2 --env=SocialBot-Pr2Gripper-v0 --num_timesteps=2e6 --network=cnn_lstm --save_path=~/models/pr2_lstm.model --num_env 1 --nminibatches=1 --ent_coef=0.01'
      .split(' '))


def eval_ppo2_lstm():
    M('--alg=ppo2 --env=SocialBot-Pr2Gripper-v0 --num_timesteps=0 --network=cnn_lstm --load_path=~/models/pr2_lstm.model --play --num_env 1 --nminibatches=1 --ent_coef=0.01'
      .split(' '))


def eval_ppo2():
    M('--alg=ppo2 --env=SocialBot-Pr2Gripper-v0 --num_timesteps=0 --network=cnn --load_path=~/models/pr2.model --play'
      .split(' '))


if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    logging.basicConfig(level=logging.DEBUG)
    ppo2_mlp()
