# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
import gym
import os
import numpy as np
import random
import math
from absl import logging

from gym import spaces
import social_bot
from social_bot import teacher
from social_bot.envs.gazebo_base import GazeboEnvBase
from social_bot.teacher import TeacherAction
import social_bot.pygazebo as gazebo


class CartPole(GazeboEnvBase):
    """
    This environment simulates the classic cartpole in the pygazebo environment.

    Observation:
        Type: Box(4)
        Num	Observation                 Min         Max
        0	Cart Position              -10.0          10.0
        1	Cart Velocity              -Inf           Inf
        2	Pole Angle                 -36°           36°
        3	Pole Velocity At Joint     -Inf           Inf

    Actions:
        Type:   Box(1)
        Representing force acting on the cart.

    Reward:
        Reward is 1 for every step taken, including the termination step

    Starting State:
        Pole Angle will be set to small angle between +/- noise radian

    Episode Termination:
        Pole Angle is more than ±18°
        Cart Position is more than ±5

    Enviroment Settings:
        x_threshold:      terminational position for cart
        theta_threshold:  terminational angles for the pole
        noise:            noise levels added into observation and
                          initial postion

    """

    def __init__(self,
                 x_threshold=5,
                 theta_threshold=0.314,
                 noise=0.01,
                 port=None):
        super(CartPole, self).__init__(world_file="cartpole.world", port=port)
        self._agent = self._world.get_agent()
        logging.debug("joint names: %s" % self._agent.get_joint_names())
        self._x_threshold = x_threshold
        self._theta_threshold = theta_threshold
        high = np.array([
            self._x_threshold * 2,
            np.finfo(np.float32).max, self._theta_threshold * 2,
            np.finfo(np.float32).max
        ])
        self.noise = noise
        self.action_space = spaces.Box(-1, 1, shape=(1, ), dtype='float32')
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self._world.info()

    def _get_state(self):
        s1 = self._agent.get_joint_state("cartpole::cartpole::slider_to_cart")
        s2 = self._agent.get_joint_state("cartpole::cartpole::cart_to_pole")

        x, x_dot = s1.get_positions()[0], s1.get_velocities()[0]
        theta, theta_dot = s2.get_positions()[0], s2.get_velocities()[0]
        state = np.array([x, x_dot, theta, theta_dot])
        state += np.random.normal(0, self.noise, len(state))
        return state

    def step(self, action):
        """
          action is a single float number representing the force
          acting upon the cart

          observation is (cart position, cart speed, pole angle, pole angular veolocity)
        """
        self._world.step(20)
        state = self._get_state()
        self._agent.take_action({"cartpole::cartpole::slider_to_cart": action})
        done = math.fabs(state[0]) > self._x_threshold or math.fabs(
            state[2]) > self._theta_threshold
        return state, 1.0, done, {}

    def reset(self):
        """
        Set cartpole states back to original, and also add random perturbation
        """
        self._world.reset()

        joint_state = gazebo.JointState(1)
        joint_state.set_positions([self.noise * random.random()])
        joint_state.set_velocities([0.0])

        self._agent.set_joint_state("cartpole::cartpole::cart_to_pole",
                                    joint_state)
        return self._get_state()


def main():
    env = CartPole()
    env.reset()
    env._world.info()
    for _ in range(100):
        env.reset()
        total_rewards = 0
        while True:
            obs, reward, done, info = env.step((random.random() - 0.5) * 0.5)
            total_rewards += reward
            env.render()
            if done:
                print("total reward " + str(total_rewards))
                break


if __name__ == "__main__":
    logging.set_verbosity(logging.DEBUG)
    main()
