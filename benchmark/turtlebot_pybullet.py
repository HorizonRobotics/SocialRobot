# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

import pybullet as p
import time
import random
import numpy as np
import os
import matplotlib.pyplot as plt

# enable display
enable_gui = True
# enable camera rendering
enable_camera = True
# show image or not
show_image = enable_camera & False
# use open gl or cpu for image rendering
enable_open_gl_rendering = enable_camera & True
# use reduced coordinate method Featherstone Articulated Body Algorithm
#  or use MaximalCoordinates (btRigidBody)
use_maximal_coordinates = True

# use LCP DANTZIG solver or default Sequential Impulse Constraint solver
use_lcp_dantzig_solver = True
# substeps
num_substeps = 100
# enable GUI rendering
disable_gui_rendering = enable_gui & True


def get_image(cam_pos, cam_orientation):
    """
    Arguments
        cam_pos: camera position
        cam_orientation: camera orientation in quaternion
    """
    width = 160
    height = 120
    fov = 90
    aspect = width / height
    near = 0.001
    far = 5

    if use_maximal_coordinates:
        # cam_orientation has problem when enable bt_rigid_body,
        # looking at 0.0, 0.0, 0.0 instead
        # this does not affect performance
        cam_pos_offset = cam_pos + np.array([0.0, 0.0, 0.3])
        target_pos = np.array([0.0, 0.0, 0.0])
    else:
        # camera pos, look at, camera up direction
        rot_matrix = p.getMatrixFromQuaternion(cam_orientation)
        # offset to base pos
        cam_pos_offset = cam_pos + np.dot(
            np.array(rot_matrix).reshape(3, 3), np.array([0.1, 0.0, 0.3]))
        target_pos = cam_pos_offset + np.dot(
            np.array(rot_matrix).reshape(3, 3), np.array([-1.0, 0.0, 0.0]))
    # compute view matrix
    view_matrix = p.computeViewMatrix(cam_pos_offset, target_pos, [0, 0, 1])
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get depth values using the OpenGL renderer
    if enable_open_gl_rendering:
        w, h, rgb, depth, seg = p.getCameraImage(
            width,
            height,
            view_matrix,
            projection_matrix,
            shadow=True,
            renderer=p.ER_BULLET_HARDWARE_OPENGL)
    else:
        w, h, rgb, depth, seg = p.getCameraImage(
            width,
            height,
            view_matrix,
            projection_matrix,
            shadow=True,
            renderer=p.ER_TINY_RENDERER)

    # depth_buffer = np.reshape(images[3], [width, height])
    # depth = far * near / (far - (far - near) * depth_buffer)
    # seg = np.reshape(images[4],[width,height])*1./255.
    return rgb


if enable_gui:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT)  # without GUI

p.setAdditionalSearchPath("./robot_df")

offset = [0, 0, 0]
robot_path = "./robot_df/turtlebot/model.urdf"
plane_path = "./plane.urdf"
if use_maximal_coordinates:
    turtle_bot = p.loadURDF(robot_path, offset, useMaximalCoordinates=1)
    plane = p.loadURDF(plane_path, useMaximalCoordinates=1)
else:
    turtle_bot = p.loadURDF(robot_path, offset)
    plane = p.loadURDF(plane_path)

p.setTimeStep(0.001 * num_substeps)
p.setPhysicsEngineParameter(numSubSteps=num_substeps)
# -1 to disable max number of cmd per 1ms
p.setPhysicsEngineParameter(maxNumCmdPer1ms=-1)
if disable_gui_rendering:
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 0)

p.setGravity(0, 0, -9.8)
p.setRealTimeSimulation(0)

# physic engine parameter and constraint solver
if use_lcp_dantzig_solver:
    p.setPhysicsEngineParameter(
        constraintSolverType=p.CONSTRAINT_SOLVER_LCP_DANTZIG)

# print testing info
print(p.getPhysicsEngineParameters())
for j in range(p.getNumJoints(turtle_bot)):
    print(p.getJointInfo(turtle_bot, j))

steps = 0
t0 = time.time()
interval = 100
fig = None
while (1):
    steps += 1

    p.setJointMotorControl2(
        turtle_bot,
        0,
        p.VELOCITY_CONTROL,
        targetVelocity=random.random() * -1.0,
        force=60)
    p.setJointMotorControl2(
        turtle_bot,
        1,
        p.VELOCITY_CONTROL,
        targetVelocity=random.random() * -2.0,
        force=60)

    p.stepSimulation()

    if enable_camera:
        position, orientation = p.getBasePositionAndOrientation(turtle_bot)
        rgb = get_image(np.array(position), orientation)
        if show_image:
            if fig is None:
                fig = plt.imshow(rgb)
            else:
                fig.set_data(rgb)
            plt.pause(0.00001)
    if (steps + 1) % interval == 0:
        print("steps=%s" % interval +
              " frame_rate=%s" % (interval / (time.time() - t0)))
        t0 = time.time()
