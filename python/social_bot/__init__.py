# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import os
from gym.envs.registration import register

register(
    id='SocialBot-SimpleNavigation-v0',
    entry_point='social_bot.envs:SimpleNavigation',
)


def get_world_dir():
    return os.path.join(os.path.dirname(__file__), 'worlds')


def get_model_dir():
    return os.path.join(os.path.dirname(__file__), 'models')


os.environ['GAZEBO_MODEL_PATH'] = get_model_dir() + ':' + os.environ.get(
    'GAZEBO_MODEL_PATH', '')

from . import pygazebo
pygazebo.initialize()
