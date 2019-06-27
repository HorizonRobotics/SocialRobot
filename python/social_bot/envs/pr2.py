# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
import gym
import os
import logging
import numpy as np
import random
import math
import PIL

from gym import spaces
import social_bot
from social_bot import teacher
from social_bot.envs.gazebo_base import GazeboEnvBase
from social_bot.teacher import TeacherAction
import social_bot.pygazebo as gazebo
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class Pr2Gripper(GazeboEnvBase):
    """
    The goal of this task is to train the agent to use its arm and fingers.

    All the joints in the right arm of PR2 are controllable by force,
    the observations are the image taken from head camera, the goal could be
    blocked by agent's own arm.

    The goal position will be randomly placed in front of the agent at start of each
    episode.

    Observations for agent:
    * when use_internal_states_only is True, observation will be a single ndarray with
      internal joint states, object pose, finger tip pose, finger tip touch sensor output,
      and finger tip distances to the target object.
    * when use_internal_states_only is False, observation will be a tuple. The first element is
      6 channel image concatenate of reshaped images from two cameras at head of PR2,
      the second element includes only internal joint states and finger tip sensor output.

    In the version:
    * agent will receive small reward to get close to the goal if reward_shaping is True
    * agent will receive small reward to open/close gripper if reward_shaping is True
    * agent will receive 0.5 if either left finger tip or right finger tip touches the goal
    * agent will receive 1 if both left finger and right finger touches the goal
    * agent will receive 1.0 + 50 * lift if goal is lifted 1cm or high over the table.

    This task could be considered solved if agent could achieve average reward >300
    per episode. Using TF_Agents/PPO, we could reach that after ~5M environment steps with
    use_internal_states_only

    joints to control (with effort limits):
          r_shoulder_pan_joint:30.0
          r_shoulder_lift_joint:30.0
          r_upper_arm_roll_joint:30.0
          r_elbow_flex_joint:30.0
          r_forearm_roll_joint:30.0
          r_wrist_flex_joint:10.0
          r_wrist_roll_joint:10.0
          r_gripper_joint:100.0


    contact sensors:
           r_gripper_l_finger_tip_contact_sensor
           r_gripper_r_finger_tip_contact_sensor

    camera sensor:
           head_mount_prosilica_link_sensor
    """

    def __init__(self,
                 goal_name='beer',
                 max_steps=100,
                 reward_shaping=True,
                 motion_loss=0.0000,
                 use_internal_states_only=True,
                 port=None):
        """
        Args:
            goal_name (string): name of the object to lift off ground
            max_steps (int): episode will end when the agent exceeds the number of steps.
            reward_shaping (boolean): whether it adds distance based reward shaping.
            motion_loss (float): if not zero, it will add -motion_loss * || V_{all_joints} ||^2 in
                 the reward when episode ends.
            use_internal_states_only (boolean): whether to only use internal states (joint positions
                  and velocities) and goal positions, and not use camera sensors.
            port: Gazebo port, need to specify when run multiple environment in parallel
        """
        super(Pr2Gripper, self).__init__(port=port)
        self._world = gazebo.new_world_from_file(
            os.path.join(social_bot.get_world_dir(), "pr2.world"))
        self._agent = self._world.get_agent()
        #logger.info("joint names: %s" % self._agent.get_joint_names())

        self._all_joints = self._agent.get_joint_names()

        # to avoid different parallel simulation has the same randomness
        random.seed(port)

        # passive joints are joints that could move but have no actuators, are only indirectly
        # controled by the motion of active joints.
        # Though in the simulation, we could "control" through API, we chose to be more realistic.
        # from https://github.com/PR2/pr2_mechanism/blob/melodic-devel/pr2_mechanism_model/pr2.urdf
        passive_joints = set([
            "r_gripper_l_finger_joint", "r_gripper_r_finger_joint",
            "r_gripper_r_finger_tip_joint", "r_gripper_l_finger_tip_joint",
            "r_gripper_r_parallel_root_joint",
            "r_gripper_l_parallel_root_joint"
        ])

        # PR2 right arm has 7 DOF, gripper has 1 DoF, as specified on p18/p26 on PR2 manual
        # https://www.clearpathrobotics.com/wp-content/uploads/2014/08/pr2_manual_r321.pdf
        # we exclude rest of the joints
        unused_joints = set([
            "r_gripper_motor_slider_joint", "r_gripper_motor_screw_joint",
            "r_gripper_r_screw_screw_joint", "r_gripper_l_screw_screw_joint",
            "r_gripper_r_parallel_tip_joint", "r_gripper_l_parallel_tip_joint"
        ])

        unused_joints = passive_joints.union(unused_joints)

        self._r_arm_joints = list(
            filter(
                lambda s: s.find('pr2::r_') != -1 and s.split("::")[-1] not in unused_joints,
                self._all_joints))
        logger.info(
            "joints in the right arm to control: %s" % self._r_arm_joints)

        joint_states = list(
            map(lambda s: self._agent.get_joint_state(s), self._r_arm_joints))

        self._r_arm_joints_limits = list(
            map(lambda s: s.get_effort_limits()[0], joint_states))

        logger.info('\n'.join(
            map(lambda s: str(s[0]) + ":" + str(s[1]),
                zip(self._r_arm_joints, self._r_arm_joints_limits))))

        self._goal_name = goal_name
        self._goal = self._world.get_agent(self._goal_name)
        self._max_steps = max_steps
        self._steps_in_this_episode = 0
        self._reward_shaping = reward_shaping
        self._resized_image_size = (128, 128)  #(84, 84)
        self._use_internal_states_only = use_internal_states_only
        self._cum_reward = 0.0
        self._motion_loss = motion_loss

        # a hack to work around gripper not open initially, might not need now.
        self._gripper_reward_dir = 1
        self._gripper_upper_limit = 0.07
        self._gripper_lower_limit = 0.01

        # whether to move head cameras and gripper when episode starts.
        # for image based observation, we have to tilt the head down first to look at table
        self._adjust_position_at_start = not use_internal_states_only

        obs = self._get_observation()
        self._prev_dist = self._get_finger_tip_distance()
        self._prev_gripper_pos = self._get_gripper_pos()

        if self._use_internal_states_only:
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)
        else:
            self.observation_space = gym.spaces.Tuple([
                gym.spaces.Box(
                    low=0, high=1.0, shape=obs[0].shape, dtype=np.float32),
                gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=obs[1].shape,
                    dtype=np.float32)
            ])
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=[len(self._r_arm_joints)], dtype=np.float32)

    def reset(self):
        self._world.reset()
        self._move_goal()

        if self._adjust_position_at_start:
            # move camera to focus on ground. it is hacky
            joint_state = gazebo.JointState(1)
            joint_state.set_positions([1.2])
            joint_state.set_velocities([0.0])
            self._agent.set_joint_state("pr2::pr2::head_tilt_joint",
                                        joint_state)

            self._agent.take_action(dict({"pr2::pr2::r_gripper_joint": 5}))

        self._world.step(200)
        self._goal = self._world.get_agent(self._goal_name)

        obs = self._get_observation()

        self._table_height = self._goal_pose[0][2]
        #logger.debug("table height:" + str(self._table_height))
        self._steps_in_this_episode = 0
        self._cum_reward = 0.0

        self._prev_dist = self._get_finger_tip_distance()
        self._prev_gripper_pos = self._get_gripper_pos()
        self._gripper_reward_dir = 1
        return obs

    def _move_goal(self):
        loc = (0.76 + 0.14 * (random.random() - 1),
               0.1 * (random.random() - 1.0), 0.43)
        self._goal.set_pose((loc, (0, 0, 0)))

    def _get_observation(self):
        def get_camera_observation(camera_name):
            img = np.array(
                self._agent.get_camera_observation(camera_name), copy=False)
            img = PIL.Image.fromarray(img).resize(
                self._resized_image_size,
                PIL.Image.ANTIALIAS)  #.convert('L') if only grey images

            return np.array(img).astype(np.float32) / 255.

        self._l_touch = self._finger_tip_contact(
            "r_gripper_l_finger_tip_contact_sensor",
            "beer::link::box_collision")
        self._r_touch = self._finger_tip_contact(
            "r_gripper_r_finger_tip_contact_sensor",
            "beer::link::box_collision")
        self._goal_pose = self._goal.get_pose()
        self._l_finger_pose = self._agent.get_link_pose(
            "pr2::pr2::r_gripper_l_finger_tip_link")
        self._r_finger_pose = self._agent.get_link_pose(
            "pr2::pr2::r_gripper_r_finger_tip_link")

        joint_states = [
            self._agent.get_joint_state(joint) for joint in self._r_arm_joints
        ]
        joint_positions = [s.get_positions()[0] for s in joint_states]
        joint_velocities = [s.get_velocities()[0] for s in joint_states]

        if self._use_internal_states_only:
            loc, loc_a = self._goal_pose
            l_finger_tip_loc, l_finger_tip_loc_a = self._l_finger_pose
            r_finger_tip_loc, r_finger_tip_loc_a = self._r_finger_pose

            l_dist = np.linalg.norm(np.array(l_finger_tip_loc) - np.array(loc))
            r_dist = np.linalg.norm(np.array(r_finger_tip_loc) - np.array(loc))
            obs = np.concatenate(
                (np.array(joint_positions), np.array(joint_velocities),
                 np.array(loc), np.array(loc_a), np.array(l_finger_tip_loc),
                 np.array(l_finger_tip_loc_a), np.array(r_finger_tip_loc),
                 np.array(r_finger_tip_loc_a), np.array([l_dist, r_dist]),
                 np.array([self._l_touch, self._r_touch]).astype(np.float32)),
                0)
        else:
            states = np.concatenate(
                (np.array(joint_positions), np.array(joint_velocities),
                 np.array([self._l_touch, self._r_touch]).astype(np.float32)),
                0)

            img = get_camera_observation(
                "default::pr2::pr2::head_tilt_link::wide_stereo_gazebo_l_stereo_camera_sensor"
            )
            img2 = get_camera_observation(
                "default::pr2::pr2::head_tilt_link::wide_stereo_gazebo_r_stereo_camera_sensor"
            )
            obs = (np.concatenate((img, img2), axis=-1), states)

        return obs

    def _finger_tip_contact(self, finger_tip_sensor, goal_link):
        def check_goal_collisions(collisions, goal_link):
            for collision in collisions:
                if goal_link == collision[0] or goal_link == collision[1]:
                    return True
            return False

        finger_tip_collisions = self._agent.get_collisions(finger_tip_sensor)
        return check_goal_collisions(finger_tip_collisions, goal_link)

    def _get_finger_tip_distance(self):
        goal_loc = np.array(self._goal_pose[0])
        l_finger_tip_loc = np.array(self._l_finger_pose[0])
        r_finger_tip_loc = np.array(self._r_finger_pose[0])

        finger_tip_center = 0.5 * (l_finger_tip_loc + r_finger_tip_loc)
        dist = np.linalg.norm(np.array(finger_tip_center) - np.array(goal_loc))
        return dist

    # gripper_joint is 1 DOF prismatic joint, the joint state represent the
    # the parallel distance between two finger tips. see PR2 manual p18.
    def _get_gripper_pos(self):
        state = self._agent.get_joint_state("pr2::pr2::r_gripper_joint")
        return state.get_positions()[0]

    def step(self, actions):
        def trunc_scale(a, limit):
            return min(max(a, -1.0), 1.0) * limit

        scaled_actions = [trunc_scale(x, self._r_arm_joints_limits[i]) \
                          for i,x in enumerate(actions)]

        # final actions used to control each joints are:
        # joint_limit_{i} *  \pi (state)_{i}
        actions = scaled_actions

        controls = dict(zip(self._r_arm_joints, actions))

        self._agent.take_action(controls)
        self._world.step(100)

        obs = self._get_observation()

        self._steps_in_this_episode += 1
        dist = self._get_finger_tip_distance()
        gripper_pos = self._get_gripper_pos()

        done = self._steps_in_this_episode >= self._max_steps

        reward = 0 if (not done) else 1.0 - np.tanh(5.0 * dist)

        dist_reward = self._prev_dist - dist

        # this is reward to encourage the agent to open and close the gripper.
        # if gripper is closed (< _gripper_lower_limit), agent will receive small reward
        # to open the gripper.
        # if the gripper is wide open (> _gripper_upper_limit), agent will receive small
        # reward to close the gripper.
        pos_reward = self._gripper_reward_dir * (
            gripper_pos - self._prev_gripper_pos)

        if self._reward_shaping:
            reward += dist_reward + pos_reward

        delta_reward = 0

        if self._l_touch:
            logger.info("l finger touch!")
            delta_reward += 0.5

        if self._r_touch:
            logger.info("r finger touch!")
            delta_reward += 0.5

        if self._l_touch and self._r_touch:
            logger.info("both touch!")
            goal_loc = self._goal_pose[0]

            # lifting reward
            elevation = goal_loc[2] - self._table_height
            lift = min(max(elevation - 0.01, 0), 0.2)

            if lift > 0:
                logger.info("beer lift! " + str(lift))
                delta_reward += (1.0 + 50 * lift)

        if delta_reward > 0:
            reward += delta_reward
        else:
            reward += -0.01  #if no positive reward, penalize it a bit to speed up

        if self._motion_loss > 0.0:
            v2s = 0.0
            for joint in self._r_arm_joints:
                v = self._agent.get_joint_state(joint).get_velocities()[0]
                v2s += v * v
            reward += -1.0 * self._motion_loss * v2s

        self._cum_reward += reward
        if done:
            logger.info("episode ends at dist: " + str(dist) + "|" +
                        str(gripper_pos) + " with cum reward:" +
                        str(self._cum_reward))

        if self._gripper_reward_dir == 1 and gripper_pos > self._gripper_upper_limit:
            self._gripper_reward_dir = -1
        elif self._gripper_reward_dir == -1 and gripper_pos < self._gripper_lower_limit:
            self._gripper_reward_dir = 1
        self._prev_dist = dist
        self._prev_gripper_pos = gripper_pos
        return obs, reward, done, {}

    def run(self):
        self.reset()
        self._world.info()
        self._max_steps = 1  # To dbg initial setup only
        r_gripper_index = -1
        for i in range(len(self._r_arm_joints)):
            if self._r_arm_joints[i].find("pr2::pr2::r_gripper_joint") != -1:
                r_gripper_index = i
                break
        reward = 0.0
        while True:
            actions = np.random.randn(len(self._r_arm_joints))
            #actions = np.zeros(len(self._r_arm_joints))
            #actions[r_gripper_index] = np.random.uniform() - 0.5
            obs, r, done, _ = self.step(actions * self._gripper_reward_dir)
            reward += r

            if not self._use_internal_states_only:
                fig = plt.figure()
                fig.add_subplot(1, 2, 1)
                plt.imshow(obs[0][:, :, :3])
                fig.add_subplot(1, 2, 2)
                plt.imshow(obs[0][:, :, 3:])
                plt.show()
            if done:
                logger.info("episode reward:" + str(reward))
                self.reset()
                reward = 0.0

    def render(self, mode='human'):
        return


def main():
    env = Pr2Gripper(max_steps=100, use_internal_states_only=False)
    env.run()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
