
import sys, tty, termios
import select
import numpy as np

# Get command from keyboard
class KeyboardControl:
    def __init__(self):
        self._decay = 0.95
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
            self._speed = self._speed+0.1
        elif ch == "s":
            self._speed = self._speed-0.1
        elif ch == "a":
            self._turning = self._turning-0.1
        elif ch == "d":
            self._turning = self._turning+0.1
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
        left_wheel_joint = self._speed+self._turning
        right_wheel_joint = self._speed-self._turning
        actions = [left_wheel_joint, right_wheel_joint]
        return actions

    def gen_youbot_action(self):
        wheel_joint_bl = self._speed+self._turning
        wheel_joint_br = self._speed-self._turning
        wheel_joint_fl = self._speed+self._turning
        wheel_joint_fr = self._speed-self._turning
        if self._gripper_open:
            gripper_joint = 0.5
        else:
            gripper_joint = -0.5
        actions = [0,0,0,0,0,0, # arm_joints
            0,gripper_joint,gripper_joint, # gripper palm joint and finger joints
            wheel_joint_bl,wheel_joint_br,wheel_joint_fl,wheel_joint_fr]
        return actions

    def gen_pr2_action(self):
        wheel_joint_bl = self._speed+self._turning
        wheel_joint_br = self._speed-self._turning
        wheel_joint_fl = self._speed+self._turning
        wheel_joint_fr = self._speed-self._turning
        actions = [
            wheel_joint_fl, wheel_joint_fl, wheel_joint_fr,wheel_joint_fr,
            wheel_joint_bl,wheel_joint_bl,wheel_joint_br,wheel_joint_br]
        return actions