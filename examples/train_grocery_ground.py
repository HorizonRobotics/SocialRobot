# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl import logging
import tensorflow as tf
import social_bot
from alf.environments import suite_socialbot
from tf_agents.agents.ddpg.examples.v2.train_eval import train_eval


def train_with_ddpg():
    tf.compat.v1.enable_v2_behavior()
    logging.set_verbosity(logging.INFO)
    train_eval(
        root_dir='~/tmp/ddpg/SocialBot-GroceryGround',
        env_name='SocialBot-GroceryGround-v0',
        env_load_fn=suite_socialbot.load,
        num_iterations=20000000,
        actor_fc_layers=(192, 192, 128),
        critic_obs_fc_layers=(192, ),
        critic_action_fc_layers=(128, ),
        critic_joint_fc_layers=(192, 128),
        # Params for collect
        initial_collect_steps=1200,
        collect_steps_per_iteration=1,
        num_parallel_environments=1,
        replay_buffer_capacity=100000)


if __name__ == '__main__':
    train_with_ddpg()
