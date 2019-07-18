# SocialBot

A python environment for developing interactive learning agent with language communication ability.

## Install dependency
SocialBot is built on top of [Gazebo simulator](http://gazebosim.org). You need to install Gazebo first using the following command:
```bash
curl -sSL http://get.gazebosim.org | sh
```
If you already have a gazebo in your system, please make sure its version is greater than 9.6. You can check gazebo version by running `gazebo --version`. SocialRobot had been tested with Gazebo 9.6 and Gazebo 10.0.

You also need to add the models in this repp to GAZEBO_MODEL_PATH
```bash
export GAZEBO_MODEL_PATH=REPO_ROOT/python/social_bot/models:$GAZEBO_MODEL_PATH
```

You also need to install the following packages:
```bash
apt install python3-tk
```

## To compile
```bash
cd REPO_ROOT
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make -j
cd REPO_ROOT
pip3 install -e .
```

## To run test
#### [Simple Navigation Test](examples/test_simple_navigation.py)
```bash
cd REPO_ROOT/examples
python3 test_simple_navigation.py
```
To see the graphics, you can open another terminal and run
```bash
GAZEBO_MODEL_PATH=`pwd`/../python/social_bot/models gzclient
```
#### [Grocery Ground Test](python/social_bot/envs/grocery_ground.py)
```bash
python3 python/social_bot/envs/grocery_ground.py
```

## Environments and Tasks

### [Simple Navigation](python/social_bot/envs/simple_navigation.py)

* [Goal task](python/social_bot/teacher_tasks.py). 
    
    Gif to be updated    

### [Grocery Ground](python/social_bot/envs/grocery_ground.py)

* [Goal task](python/social_bot/envs/grocery_ground.py). 

    <img src="media/grocery_ground_pioneer.gif" width="320" height="240" alt="pioneer"/>

    A task to chase a goal. 
    
    The agent will receive reward 1 when it is close enough to the goal, and get -1 if moving away from the goal too much or timeout.

### [PR2 Gripping](python/social_bot/envs/pr2.py)

* Gripping task

    Gif To be updated

    A task that the agent need to use its grippers or fingers to grip a beer.

    A simple reward shaping is used to guide the agent's gripper to get close to the target, open the gripper and lift the target up.


### [iCub Walking](python/social_bot/envs/icub_walk.py)

* Humanoid walking task

    <img src="media/icub_walk.gif" width="320" height="240" alt="pioneer"/>

    A task that the agent learns how to walk. 
    
    reward = not_fall_bonus + truncked_walk_velocity - ctrl_cost
 

## Training Examples
### [Simple Navigation Training Example](https://github.com/HorizonRobotics/alf/blob/master/alf/examples/ac_simple_navigation.gin)
Training with [Agent Learning Framework (Alf)](https://github.com/HorizonRobotics/alf)
```bash
cd ALF_REPO_ROOT/examples/
python -m alf.bin.main --root_dir=~/tmp/simple_navigation --gin_file=ac_simple_navigation.gin --alsologtostderr
```
### [PR2 Training Example](https://github.com/HorizonRobotics/alf/blob/master/alf/examples/ppo_pr2.gin)
Training with Alf:
```bash
python -m alf.bin.main --root_dir=~/tmp/pr2_gripping --gin_file=ALF_REPO_ROOT/examples/ppo_pr2.gin --alsologtostderr
```

### [Grocery Ground Training Example](examples/grocery_alf_ppo.gin)
Training with Alf:
```bash
python -m alf.bin.main --root_dir=~/tmp/grocery_ppo --gin_file=grocery_alf_ppo.gin --alsologtostderr
```

### [iCub Walking Task Training Example](examples/icub_alf_ppo.gin)
Training with Alf:
```bash
python -m alf.bin.main --root_dir=~/tmp/icub_ppo --gin_file=icub_alf_ppo.gin --alsologtostderr
```
Or you can train it with tf-agent SAC, which might be better at sample efficiency:
```bash
python examples/train_icub_walk_sac.py \
    --root_dir=~/tmp/ICubWalkSAC
```

## Trouble shooting

### python
You need to make sure the python you use matches the python found by cmake. You can check this by looking at REPO_ROOT/build/CMakeCache.txt

### display

You need to make sure your `DISPLAY` environment variable points to a valid display, otherwise camera sensor cannot be created and there will be the following error:
```
[Err] [CameraSensor.cc:112] Unable to create CameraSensor. Rendering is disabled.
...
gazebo::rendering::Camera*]: Assertion `px != 0' failed.
```
You can find out the correct value for `DISPLAY` envorinment variable by running `echo $DISPLAY` in a terminal opened from the desktop. You can verify whether the `DISPLAY` is set correctly by running `gazebo`
