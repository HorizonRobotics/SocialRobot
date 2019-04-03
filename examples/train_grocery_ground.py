# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import gym
import social_bot

import logging
from baselines import ppo2
from baselines import deepq
from baselines import ddpg
from baselines.run import main as M
import numpy as np

def ppo2_mlp():
    M('--alg=ddpg --env=SocialBot-GroceryGround-v0 --num_timesteps=2e7 --network=mlp --save_path=~/models/grocery_mlp.model --num_env 1 --num_hidden=128 --num_layers=4'
      .split(' '))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    ppo2_mlp()
