# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.
import random
import social_bot.pygazebo as bot
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

import time
import psutil
import os

enable_camera = True

bot.initialize()

if enable_camera:
    world = bot.new_world_from_file("./turtlebot_camera.world")
else:
    world = bot.new_world_from_file("./turtlebot.world")

world.info()
agent = world.get_agent()
print(agent.get_joint_names())

sleep(1)
steps = 0
t0 = time.time()
interval = 100
fig = None
for i in range(10000000):
    steps += 1
    agent.take_action({
        "turtlebot::turtlebot::wheel_left_joint":
        random.random() * 0.2 - 0.1,
        "turtlebot::turtlebot::wheel_right_joint":
        random.random() * 0.2 - 0.1
    })
    world.step(interval)
    pose = agent.get_pose()

    if enable_camera:
        obs = agent.get_camera_observation(
            "default::turtlebot::camera::link::camera")        
        npdata = np.array(obs, copy=False)
        """
        if fig is None:
            fig = plt.imshow(npdata)
        else:
            fig.set_data(npdata)
        plt.pause(0.00001)
        """
    if (i+1) % interval == 0:
        print("steps=%s" % interval +
                     " frame_rate=%s" % (interval / (time.time() - t0)))
        t0 = time.time()
