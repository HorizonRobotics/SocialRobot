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
from collections import OrderedDict
import social_bot
import social_bot.pygazebo as gazebo
from absl import logging
from play_ground import PlayGround
from embodied_teacher import EmbodiedTeacher
from social_bot.tasks import GoalTask, KickingBallTask


class TestEmbodiedTeacher(unittest.TestCase):
    def test_embodied_teacher(self):
        agents = ['youbot_noplugin', 'pioneer2dx_noplugin', 'pr2_noplugin']
        with open(
                os.path.join(social_bot.get_model_dir(), "agent_cfg.json"),
                'r') as cfg_file:
            agent_cfgs = json.load(cfg_file)
        for agent_type in agents:
            for use_image_obs in [True, False]:
                agent_cfg = agent_cfgs[agent_type]
                test_tasks = [GoalTask, KickingBallTask]
                if agent_cfg['camera_sensor'] == '' and use_image_obs:
                    continue
                logging.info("Testing Case: Agent " + agent_type + ", Task " +
                             str(test_tasks) + ", UseImage: " +
                             str(use_image_obs))
                env = EmbodiedTeacher(
                    with_language=True,
                    use_image_observation=use_image_obs,
                    image_with_internal_states=True,
                    agent_type=agent_type,
                    tasks=test_tasks,
                    demo_by_human=False)
                step_cnt = 0
                last_done_time = time.time()
                while step_cnt < 100 and (time.time() - last_done_time) < 5:
                    actions = env._control_space.sample()
                    actions = dict(control=actions, sentence="hello")
                    teacher_actions = env._teacher_control_space.sample()
                    actions = OrderedDict(
                        learner=actions, teacher=teacher_actions)
                    env.step(actions)
                    step_cnt += 1
                step_per_sec = step_cnt / (time.time() - last_done_time)
                logging.info("Test Passed, FPS: " + str(step_per_sec))
                env.close()
                gazebo.close()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    unittest.main()
