# A Simple Performance Benchmark for pyGazebo and pyBullet

A simple benchmark for pygazebo(use bullet/ode as physic engine) and pybullet simulator.

The scene is a turtlebot with 2 randomly moving wheels and a camera sensor. 

Pybullet incompatible sensors and plugins were removed from the turtlebot urdf file. The model file contains 2 continuous joints and 30 fixed joints.


# Test Result ( Core i7-6700HQ @ 2.6Ghz, GTX970M)
Note there are 100 substeps per one step

```
Turtlebot pyBullet without camera sensor (use_maximal_coordinates = True):
268 steps/second

Turtlebot pyBullet with camera sensor (use_maximal_coordinates = False, image rendered by OpenGL, disbale GUI rendering) :
320x240 - 65 steps/second

Turtlebot pyGazebo(bullet) without camera sensor:
185 steps/second

Turtlebot pyGazebo(bullet) with camera sensor (320x240):
104 steps/second

Turtlebot pyGazebo(ode) without camera sensor:
225 steps/second

Turtlebot pyGazebo(ode) with camera sensor (320x240):
122 steps/second
```


# Known Problems

In turtlebot_pybullet case:

When use_maximal_coordinates is enabled, camera orientation is not correct, and connecting direct without GUI fails. So the case of pyBullet with camera sensor is evaluated with use_maximal_coordinates disabled, which could affect the performance.


# Install of PyBullet

pip install pybullet

See also the [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3)

# About the Differences of the Simulators
Note that this is just a simple evaluation about their performace in default configuration, other evaluative dimension like accurancy is not included. Even bullet engine is used for gazebo, there are still some differences of pygazebo and pybullet implementation.

One difference is that how contacts are updated. Gazebo uses a builtin Contacts Manager rather than API of the physic engine. Another major difference is the method joints are simulated with. Gazebo uses Maximal Coordinates method (from class btRigidBody). btRigidBody is used to simulate single 6-degree of freedom moving objects, derived from btCollisionObject class in gazebo. While by pyBullet default (use_maximal_coordinates=False) the joints are simulated using the Featherstone Articulated Body Algorithm (btMultiBody introduced ABA in Bullet 2.x). btMultiBody is an alternative representation of a rigid body hierarchy using generalized (or reduced) coordinates, using the articulated body algorithm, and is potentially more accurate for complex tree hierarchy of joints.
