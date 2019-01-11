# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import social_bot as bot
import random

bot.initialize()
world = bot.new_world_from_file("empty.world")

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
      </visual>
    </link>
  </model>
</sdf>
"""

for idx in xrange(20):
    pose = "%d %d 0 0 0 0" % (random.randint(1,10) - 5, random.randint(1, 10) - 5)
    size = "%f %f %f" % (random.randint(1, 4) * 0.1, random.randint(1, 4) * 0.1, random.randint(1, 4) * 0.1)
    sdf_str = box_sdf % ("box%d" % idx, pose, size, size)
    world.insertModelFromSdfString(sdf_str)

while True:
    world.step()
