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
import social_bot
import social_bot.pygazebo as gazebo
from absl import logging
from pr2 import Pr2Gripper
from pr2 import PR2_WORLD_SETTING


class TestPr2(unittest.TestCase):
    def test_pr2(self):
        for use_internal_states_only in [True, False]:
            env = Pr2Gripper(
                world_config=PR2_WORLD_SETTING + [
                    "//sensor[@type='camera']<>visualize=true",
                    "//camera//format=R8G8B8",
                ],
                max_steps=100,
                use_internal_states_only=use_internal_states_only)

            env.run(render=False)
            env.close()
            gazebo.close()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    unittest.main()
