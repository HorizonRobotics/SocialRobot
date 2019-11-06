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

import sys, tty, termios
import select
import numpy as np
from absl import logging


# Get command from keyboard
class KeyboardControl:
    def __init__(self):
        self._decay = 0.9
        self._gripper_movements = [0, 0, 0]
        self._gripper_open = True
        self._speed = 0
        self._turning = 0

    def reset(self):
        self._gripper_movements = [0, 0, 0]
        self._gripper_open = True
        self._speed = 0
        self._turning = 0

    def _getch(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno(), termios.TCSANOW)
            if select.select([sys.stdin], [], [], 0.0)[0]:
                ch = sys.stdin.read(1)
            else:
                ch = None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

    def get_control(self, agent_type, agent):
        ch = self._getch()
        self._speed *= self._decay
        self._turning *= self._decay
        self._gripper_movements = [0, 0, 0]
        # movemnts
        if ch == "w":
            self._speed = self._speed + 0.1
        elif ch == "s":
            self._speed = self._speed - 0.1
        elif ch == "a":
            self._turning = self._turning - 0.1
        elif ch == "d":
            self._turning = self._turning + 0.1
        # gripper pose
        elif ch == "j":
            self._gripper_movements[0] = 0.01
        elif ch == "l":
            self._gripper_movements[1] = 0.01
        elif ch == "k":
            self._gripper_movements[0] = 0.01
        elif ch == "i":
            self._gripper_movements[1] = 0.01
        # gripper finger
        elif ch == "e":
            self._gripper_open = True
        elif ch == "r":
            self._gripper_open = False

        return self.convert_to_action(agent_type, agent)

    def convert_to_action(self, agent_type, agent):
        if agent_type == 'pioneer2dx_noplugin' or agent_type == 'turtlebot':
            actions = self.gen_pioneer2dx_action()
        elif agent_type == 'youbot_noplugin':
            actions = self.gen_youbot_action()
            pose = agent.get_link_pose('gripper_palm_link')
            pos = np.array(pose[0]) + np.array(self._gripper_movements)
            rot = np.array(pose[1])
            agent.set_link_pose('gripper_palm_link', (pos, rot))
        elif agent_type == 'pr2_noplugin':
            actions = self.gen_pr2_action()
        else:
            actions = []
            print("agent type not implement yet: " + agent_type)
        print(actions)
        return actions

    def gen_pioneer2dx_action(self):
        left_wheel_joint = self._speed + self._turning
        right_wheel_joint = self._speed - self._turning
        actions = [left_wheel_joint, right_wheel_joint]
        return actions

    def gen_youbot_action(self):
        wheel_joint_bl = self._speed + self._turning
        wheel_joint_br = self._speed - self._turning
        wheel_joint_fl = self._speed + self._turning
        wheel_joint_fr = self._speed - self._turning
        if self._gripper_open:
            gripper_joint = 0.5
        else:
            gripper_joint = -0.5
        actions = [
            0, 0, 0, 0, 0, 0,  # arm_joints
            0, gripper_joint, gripper_joint,  # palm joint and gripper joints
            wheel_joint_bl, wheel_joint_br, wheel_joint_fl, wheel_joint_fr
        ]
        return actions

    def gen_pr2_action(self):
        wheel_joint_bl = self._speed + self._turning
        wheel_joint_br = self._speed - self._turning
        wheel_joint_fl = self._speed + self._turning
        wheel_joint_fr = self._speed - self._turning
        actions = [
            wheel_joint_fl, wheel_joint_fl, wheel_joint_fr, wheel_joint_fr,
            wheel_joint_bl, wheel_joint_bl, wheel_joint_br, wheel_joint_br
        ]
        return actions


def main():
    """
    Simple testing of this environment.
    """
    import matplotlib.pyplot as plt
    import time
    import social_bot
    from social_bot.envs.play_ground import PlayGround
    from social_bot.tasks import GoalTask, KickingBallTask, ICubAuxiliaryTask
    with_language = False
    use_image_obs = False
    fig = None
    agent_type='youbot_noplugin' # support pioneer2dx_noplugin or youbot_noplugin
    env = PlayGround(
        with_language=with_language,
        use_image_observation=use_image_obs,
        image_with_internal_states=False,
        agent_type=agent_type,
        real_time_update_rate = 500,
        tasks=[GoalTask])
    env.render()
    keybo = KeyboardControl()
    step_cnt = 0
    last_done_time = time.time()
    while True:
        actions = np.array(keybo.get_control(agent_type, env._agent))
        if with_language:
            actions = dict(control=actions, sentence="hello")
        obs, _, done, _ = env.step(actions)
        step_cnt += 1
        if with_language and (env._steps_in_this_episode == 1 or done):
            seq = obs["sentence"]
            logging.info("sentence_seq: " + str(seq))
            logging.info("sentence_raw: " +
                         env._teacher.sequence_to_sentence(seq))
        if use_image_obs:
            if with_language:
                obs = obs['image']
            if fig is None:
                fig = plt.imshow(obs)
            else:
                fig.set_data(obs)
            plt.pause(0.00001)
        if step_cnt > 300:
            env.reset()
            keybo.reset()
            step_per_sec = step_cnt / (time.time() - last_done_time)
            logging.info("step per second: " + str(step_per_sec))
            step_cnt = 0
            last_done_time = time.time()


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main()
