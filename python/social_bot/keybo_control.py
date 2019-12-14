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
from pymouse import PyMouse
from pykeyboard import PyKeyboardEvent


class KeyboardControl(PyKeyboardEvent):
    """
    This class is used to generate demonstrations from human through keyboard.
    Note that you should keep the terminal window on the fore-front to capture
    the key being pressed.
    Some tricks are used to make the keyboard controlling a little bit more
    friendly. Move the agent around by key "WASD" and open or close gripper by
    key "E", and control the robot arm(if there is) by "IJKL".
    """

    def __init__(self):
        super().__init__(capture=False)
        self._mouse = PyMouse()
        x, y = self._mouse.screen_size()
        self._x_center, self._y_center = x / 2.0, y / 2.0
        self._speed = 0
        self._turning = 0
        self._gripper_pos = [0, 0, 0]
        self._gripper_open = True
        self._wheel_step = 0.5
        self._speed_decay = 0.9
        self._turning_decay = 0.6
        self.start()

    def reset(self):
        self._gripper_pos = [0, 0, 0]
        self._gripper_open = True
        self._speed = 0
        self._turning = 0

    def _get_mouse_pos(self):
        """ Get the mouse position normalized to (-1, 1)
        """
        x, y = self._mouse.position()
        x, y = x / self._x_center - 1.0, y / self._y_center - 1.0
        return x, y

    def tap(self, keycode, character, press):
        """ Keyboard event handler. argument 'press' 

        Args:
            keycode(int): the key code
            character(string): the key name
            press(bool): True if the key was depressed and False if the key was released.
        """
        if not press:
            return
        if character == "w":
            self._speed = 0 if self._speed < -0.01 else self._speed + self._wheel_step
        elif character == "s":
            self._speed = 0 if self._speed > 0.01 else self._speed - self._wheel_step
        elif character == "a":
            self._turning = 0 if self._turning > 0.01 else self._turning - self._wheel_step
        elif character == "d":
            self._turning = 0 if self._turning < -0.01 else self._turning + self._wheel_step
        # gripper finger
        elif character == "e":
            self._gripper_open = not self._gripper_open
        # set step size
        elif character == "+":
            self._wheel_step *= 1.5
        elif character == "-":
            self._wheel_step *= 0.7

    def get_agent_actions(self, agent_type):
        """
        Args:
            agent_type(sting): the agent type
        Returns:
            actions generated by the keyboard accroding to agent type
        """
        # decay the speed
        self._speed *= self._speed_decay
        self._turning *= self._turning_decay
        # get gripper pos
        mouse_x, mouse_y = self._get_mouse_pos()
        self._gripper_pos[0] = mouse_x
        self._gripper_pos[1] = mouse_y

        return self._convert_to_agent_action(agent_type)

    def _convert_to_agent_action(self, agent_type):
        if agent_type == 'pioneer2dx_noplugin' or agent_type == 'turtlebot':
            actions = self._to_diff_drive_action()
        elif agent_type == 'youbot_noplugin':
            actions = self._to_youbot_action()
        elif agent_type == 'pr2_noplugin':
            actions = self._to_pr2_action()
        elif agent_type == 'kuka_lwr_4plus':
            actions = self._to_lwr4_action()
        else:
            actions = []
            logging.info("agent type not supported yet: " + agent_type)
        return actions

    def _to_diff_drive_action(self):
        left_wheel_joint = self._speed + self._turning
        right_wheel_joint = self._speed - self._turning
        actions = [left_wheel_joint, right_wheel_joint]
        return actions

    def _to_youbot_action(self):
        wheel_joint_fl = self._speed + self._turning
        wheel_joint_fr = self._speed - self._turning
        if self._gripper_open:
            gripper_joint = 0.5
        else:
            gripper_joint = -0.5
        actions = [
            # arm joints
            0,
            self._gripper_pos[0] + 0.5,
            0.3,
            self._gripper_pos[1] - 0.1,
            0.2,
            0,
            # palm joint and gripper joints
            0,
            gripper_joint,
            gripper_joint,
            # wheel joints
            wheel_joint_fl,
            wheel_joint_fr
        ]
        return actions

    def _to_lwr4_action(self):
        actions = [
            # arm joints
            self._speed,
            self._turning,
            self._gripper_pos[0],
            self._gripper_pos[1],
            self._gripper_pos[2]
        ]
        return actions

    def _to_pr2_action(self):
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
    Simple testing of KeyboardControl class.
    """
    import matplotlib.pyplot as plt
    import time
    from social_bot.envs.play_ground import PlayGround
    from social_bot.tasks import GoalTask, KickingBallTask, ICubAuxiliaryTask, Reaching3D, PickAndPlace
    use_image_obs = True
    fig = None
    agent_type = 'youbot_noplugin'
    env = PlayGround(
        with_language=False,
        use_image_observation=use_image_obs,
        image_with_internal_states=False,
        agent_type=agent_type,
        max_steps=1000,
        step_time=0.05,
        real_time_update_rate=500,
        resized_image_size=(128, 128),
        tasks=[PickAndPlace])
    env.render()
    keybo = KeyboardControl()
    step_cnt = 0
    last_done_time = time.time()
    while True:
        actions = np.array(keybo.get_agent_actions(agent_type))
        obs, _, done, _ = env.step(actions)
        step_cnt += 1
        if use_image_obs:
            if fig is None:
                fig = plt.imshow(obs)
            else:
                fig.set_data(obs)
            plt.pause(0.00001)
        if done:
            env.reset()
            keybo.reset()
            step_per_sec = step_cnt / (time.time() - last_done_time)
            logging.info("step per second: " + str(step_per_sec))
            step_cnt = 0
            last_done_time = time.time()


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)
    main()
