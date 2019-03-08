# Sim Benchmark

A simple benchmark for pygazebo and pybullet simulator.


The scene is a turtlebot with 2 randomly moving wheels and a camera sensor. 

Pybullet incompatible sensors and plugins were removed from the turtlebot urdf file. The model file contains 2 continuous joints and 30 fixed joints.

# Test Result  ( Core i9-7960X@2.8Ghz, RTX 2080 Ti)
Note there are 100 substeps per one step

Turtlebot pyBullet without camera sensor:
11.5 steps/second

Turtlebot pyBullet with camera sensor (image rendered by OpenGL) :
15 steps/second

Turtlebot pyBullet with camera sensor (image rendered by CPU) :
14.50 steps/second


Turtlebot pyGazebo without camera sensor:
175 steps/second

Turtlebot pyGazebo with camera sensor:
98 steps/second



# btMultiBody and btRigidBody

pyGazebo use Maximal Coordinates method (btRigidBody). btRigidBody is used to simulate single 6-degree of freedom moving objects. btRigidBody is derived from btCollisionObject, so it inherits its world transform, friction and restitution
and adds linear and angular velocity. btTypedConstraint is the base class for rigid body constraints, including btHingeConstraint, btPoint2PointConstraint, btConeTwistConstraint,btSliderConstraint and btGeneric6DOFconstraint. btDiscreteDynamicsWorld is UserCollisionAlgorithm btCollisionWorld, and is a container for rigid bodies and constraints. It provides the stepSimulation to proceed.

By pyBullet default, the joints in the URDF file are created using the reduced coordinate method: the joints are simulated using the Featherstone Articulated Body Algorithm (ABA, btMultiBody in Bullet 2.x). btMultiBody is an alternative representation of a rigid body hierarchy using generalized (or reduced) coordinates, using the articulated body algorithm, as discussed by Roy Featherstone. The tree hierarchy starts with a fixed or floating base and child bodies, also called links, are connected by joints: 1-DOF revolute joint (similar to the btHingeConstraint for btRigidBody), 1-DOF prismatic joint (similar to btSliderConstraint).


# Install of PyBullet

pip install pybullet

See also the [PyBullet Quickstart Guide](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3)

# Note
Change the mesh file path to absoulte path in model.urdf file at robot_df/turtlebot/
Pybullet working with relative path , but gazebo not.

