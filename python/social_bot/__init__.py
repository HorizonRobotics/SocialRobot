# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import os
from gym.envs.registration import register

register(
    id='SocialBot-SimpleNavigation-v0',
    entry_point='social_bot.envs:SimpleNavigation',
)

register(
    id='SocialBot-SimpleNavigationDiscreteAction-v0',
    entry_point='social_bot.envs:SimpleNavigationDiscreteAction',
)

register(
    id='SocialBot-SimpleNavigationLanguage-v0',
    entry_point='social_bot.envs:SimpleNavigationLanguage',
)

register(
    id='SocialBot-SimpleNavigationSelfStatesLanguage-v0',
    entry_point='social_bot.envs:SimpleNavigationSelfStatesLanguage',
)

register(
    id='SocialBot-CartPole-v0',
    entry_point='social_bot.envs:CartPole',
)

register(
    id='SocialBot-Pr2Gripper-v0',
    entry_point='social_bot.envs:Pr2Gripper',
)

register(
    id='SocialBot-GroceryGround-v0',
    entry_point='social_bot.envs:GroceryGround',
    max_episode_steps=200,
)

register(
    id='SocialBot-GroceryGroundImage-v0',
    entry_point='social_bot.envs:GroceryGroundImage',
    max_episode_steps=200,
)

register(
    id='SocialBot-GroceryGroundImage-v1',
    entry_point='social_bot.envs:GroceryGroundImage',
    max_episode_steps=1e8,
)

register(
    id='SocialBot-GroceryGroundLanguage-v0',
    entry_point='social_bot.envs:GroceryGroundLanguage',
    max_episode_steps=200,
)

register(
    id='SocialBot-GroceryGroundImageLanguage-v0',
    entry_point='social_bot.envs:GroceryGroundImageLanguage',
    max_episode_steps=200,
)

register(
    id='SocialBot-GroceryGroundImageSelfStatesLanguage-v0',
    entry_point='social_bot.envs:GroceryGroundImageSelfStatesLanguage',
    max_episode_steps=200,
)

register(
    id='SocialBot-ICubWalk-v0',
    entry_point='social_bot.envs:ICubWalk',
    max_episode_steps=200,
)

register(
    id='SocialBot-ICubWalkPID-v0',
    entry_point='social_bot.envs:ICubWalkPID',
    max_episode_steps=200,
)

def get_world_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'worlds')


def get_model_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')


os.environ['GAZEBO_MODEL_PATH'] = get_model_dir() + ':' + os.environ.get(
    'GAZEBO_MODEL_PATH', '')

from . import pygazebo
