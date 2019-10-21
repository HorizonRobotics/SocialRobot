# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import random
import os
import time
import json
import social_bot
from absl import logging
from grocery_ground import GroceryGround


class TestGroceryGround(unittest.TestCase):
    def test_grocery(self):
        with_language = True
        agents = [
            'pioneer2dx_noplugin', 'pr2_noplugin', 'icub', 'icub_with_hands',
            'youbot_noplugin'
        ]
        tasks = ['goal', 'kickball']
        with open (os.path.join(social_bot.get_model_dir(), "agent_cfg.json"), 'r') as cfg_file:
            agent_cfgs = json.load(cfg_file)
        for agent_type in agents:
            for task_name in tasks:
                for use_image_obs in [True, False]:
                    agent_cfg = agent_cfgs[agent_type]
                    if agent_cfg['camera_sensor'] == '' and use_image_obs:
                        continue
                    logging.info("Testing Case: Agent " + agent_type +
                                 ", Task " + task_name + ", UseImage: " +
                                 str(use_image_obs))
                    env = GroceryGround(
                        with_language=with_language,
                        use_image_observation=use_image_obs,
                        image_with_internal_states=True,
                        agent_type=agent_type,
                        task_name=task_name)
                    step_cnt = 0
                    last_done_time = time.time()
                    while step_cnt < 500 and (time.time() - last_done_time) < 10:
                        actions = env._control_space.sample()
                        if with_language:
                            actions = dict(control=actions, sentence="hello")
                        env.step(actions)
                        step_cnt += 1
                    step_per_sec = step_cnt / (time.time() - last_done_time)
                    logging.info("Test Passed, FPS: " + str(step_per_sec))
                    env.close()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    unittest.main()
