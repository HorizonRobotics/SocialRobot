# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import social_bot as bot
bot.initialize()

world = bot.new_world_from_file("pioneer2dx_camera.world")

agent = world.get_agent()
agent.get_joint_names()
agent.take_action({"pioneer2dx::pioneer2dx::right_wheel_hinge": 10.})

for i in range(1000000):
    # observation = agent.sense()
    # add reward and text to observation
    # action = model.compute_action(observation)
    # agent.take_action(action)
    world.step()
