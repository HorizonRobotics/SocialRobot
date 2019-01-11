# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import social_bot as bot
bot.initialize()
world = bot.new_world_from_file("animated_box.world")

# agent = world.get_agetn(agent_name)

for i in range(100):
    # observation = agent.sense()
    # add reward and text to observation
    # action = model.compute_action(observation)
    # agent.take_action(action)
    world.step()
    raw_input("Press Enter to step")
