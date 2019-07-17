# SocialBot

A python environment for developing interactive learning agent with language communication ability.

## Install dependency
SocialBot is built on top of [Gazebo simulator](http://gazebosim.org). You need to install Gazebo first using the following command:
```bash
curl -sSL http://get.gazebosim.org | sh
```
If you already have a gazebo in your system, please make sure its version is greater than 9.6. You can check gazebo version by running `gazebo --version`. SocialRobot had been tested with Gazebo 9.6 and Gazebo 10.0.

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
```bash
cd REPO_ROOT/examples
python3 test_simple_navigation.py
```
To see the graphics, you can open another terminal and run
```bash
GAZEBO_MODEL_PATH=`pwd`/../python/social_bot/models gzclient
```

## Environments and Tasks

### [Simple Navigation](python/social_bot/envs/simple_navigation.py)

* [Goal task](python/social_bot/teacher_tasks.py). 
    
    To be updated

### [Grocery Ground](python/social_bot/envs/grocery_ground.py)

* [Goal task](python/social_bot/teacher_tasks.py). 

    <img src="examples/grocery_ground_pioneer.gif" width="320" height="240" alt="pioneer"/>

### [PR2 Gripping](python/social_bot/envs/pr2.py)

* Gripping task

    To be updated

### [iCub Walking](python/social_bot/envs/icub_walk.py)

* Humanoid walking task

    <img src="examples/icub_walk.gif" width="320" height="240" alt="pioneer"/>


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
