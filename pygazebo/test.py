# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import random
import social_bot as bot
import numpy as np
import matplotlib.pyplot as plt

bot.initialize()
world = bot.new_world_from_file("../worlds/pioneer2dx_camera.world")

box_sdf = """
<?xml version='1.0'?>
<sdf version ='1.4'>
  <model name ='%s'>
    <pose>%s</pose>
    <link name ='link'>
      <pose>0 0 .5 0 0 0</pose>
      <collision name ='collision'>
        <geometry>
          <box><size>%s</size></box>
        </geometry>
      </collision>
      <visual name ='visual'>
        <geometry>
          <box><size>%s</size></box>
        </geometry>
  <material>
    <ambient>1 0 0 1</ambient>
    <diffuse>1 0 0 1</diffuse>
    <specular>0.1 0.1 0.1 1</specular>
    <emissive>0 0 0 0</emissive>
  </material>
      </visual>
    </link>
  </model>
</sdf>
"""

for idx in range(10):
    pose = "%d %d 0 0 0 0" % (random.randint(1, 20) - 10,
                              random.randint(1, 20) - 10)
    size = "%f %f %f" % (random.randint(1, 6) * 0.1, random.randint(1, 4) *
                         0.1, random.randint(1, 8) * 0.1)
    sdf_str = box_sdf % ("box%d" % idx, pose, size, size)
    world.insertModelFromSdfString(sdf_str)

agent = world.get_agent()
agent.get_joint_names()

world.info()

for i in range(10000000):
    # observation = agent.sense()
    # add reward and text to observation
    # action = model.compute_action(observation)
    # agent.take_action(action)
    agent.take_action({
        "pioneer2dx::pioneer2dx::right_wheel_hinge":
        random.random() * 0.1,
        "pioneer2dx::pioneer2dx::left_wheel_hinge":
        random.random() * 0.2
    })
    len = random.randint(100, 500)
    for i in range(len):
        world.step()
        pose = agent.get_pose()
    agent.set_pose(((5, random.random() * 5, 0), (0, 0, 0)))
    print(pose)

    if i % 10 == 1:
        obs = agent.get_camera_observation(
            "default::pioneer2dx::camera::link::camera")
        npdata = np.array(obs, copy=False)
        plt.imshow(npdata)
        plt.show()
