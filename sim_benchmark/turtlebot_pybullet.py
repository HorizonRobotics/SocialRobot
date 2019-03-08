import pybullet as p
import time
import random
import numpy as np
from time import sleep
import os
import matplotlib.pyplot as plt

# enable display
enable_gui = True
# enable camera rendering 
enable_camera = True
# use open gl or cpu for image rendering
enable_open_gl_rendering = True
# use reduced coordinate method Featherstone Articulated Body Algorithm (ABA, btMultiBody in Bullet 2.x)
#  or use MaximalCoordinates (btRigidBody)
enable_bt_rigid_body = False


def get_image(cam_pos, cam_orientation):
    width = 320
    height = 240
    fov = 90
    aspect = width / height
    near = 0.001
    far = 5

    # camera pos, look at, camera up direction
    rot_matrix = p.getMatrixFromQuaternion(cam_orientation)
    # offset to base pos
    cam_pos = cam_pos + np.dot(np.array(rot_matrix).reshape(3,3),np.array([0.1, 0.0, 0.3]))
    # looking at
    target_pos =cam_pos + np.dot(np.array(rot_matrix).reshape(3,3),np.array([-1.0, 0.0, 0.0]))
    # compute view matrix
    view_matrix = p.computeViewMatrix(cam_pos, target_pos, [0, 0, 1])
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    # Get depth values using the OpenGL renderer
    if enable_open_gl_rendering:
        w, h, rgb, depth, seg = p.getCameraImage(width, height, view_matrix, projection_matrix, shadow=True,renderer=p.ER_BULLET_HARDWARE_OPENGL)
    else:
        w, h, rgb, depth, seg = p.getCameraImage(width, height, view_matrix, projection_matrix, shadow=True, renderer=p.ER_TINY_RENDERER)

    # depth_buffer = np.reshape(images[3], [width, height])
    # depth = far * near / (far - (far - near) * depth_buffer)
    # seg = np.reshape(images[4],[width,height])*1./255.
    return rgb


if enable_gui:
    p.connect(p.GUI)
else:
    p.connect(p.DIRECT) # without GUI

p.setAdditionalSearchPath("./robot_df")

offset = [0,0,0]
robot_path = "./robot_df/turtlebot/model.urdf"
plane_path = "./plane.urdf"
if enable_bt_rigid_body:
    # Experimental. By default, the joints in the URDF file are created using the reduced coordinate method: 
    #     the joints are simulated using the Featherstone Articulated Body Algorithm (ABA, btMultiBody in 
    #     Bullet 2.x). The useMaximalCoordinates option will create a 6 degree of freedom rigid body for
    #     each link, and constraints between those rigid bodies are used to model joints.
    turtle = p.loadURDF(robot_path, offset, useMaximalCoordinates = 1)
    plane = p.loadURDF(plane_path, useMaximalCoordinates = 1)
else:
    turtle = p.loadURDF(robot_path, offset)
    plane = p.loadURDF(plane_path)

p.setTimeStep(0.001)
p.setGravity(0,0,-9.8)
p.setRealTimeSimulation(0)

# physic engine parameter and constraint solver:
# p.setPhysicsEngineParameter(constraintSolverType=p.CONSTRAINT_SOLVER_LCP_DANTZIG, globalCFM = 0.000001)
# p.setPhysicsEngineParameter(solverResidualThreshold=0.01, numSolverIterations=200)

sleep(1)
steps = 0
t0 = time.time()

interval = 100
fig = None
while (1):
    steps += 1

    for i in range(interval):
        p.setJointMotorControl2(turtle, 0, p.VELOCITY_CONTROL, targetVelocity=random.random() * -4.0,force=100)
        p.setJointMotorControl2(turtle, 1, p.VELOCITY_CONTROL, targetVelocity=random.random() * -8.0,force=100)
        p.stepSimulation()
    
    if enable_camera:
        position, orientation = p.getBasePositionAndOrientation(turtle)
        rgb = get_image(np.array(position), orientation)
        """
        if fig is None:
            fig = plt.imshow(rgb)
        else:
            fig.set_data(rgb)
        plt.pause(0.00001)
        """
    if (steps + 1) % interval == 0:
        print("steps=%s" % interval +
                     " frame_rate=%s" % (interval / (time.time() - t0)))
        t0 = time.time()
