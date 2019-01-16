# SocialBot

A python environment for developing interactive learning agent with language communication ability.

## Install dependency
SocialBot is based on top of [Gazebo simulator](http://gazebosim.org). You need to install Gazebo first using the following command:
```bach
curl -sSL http://get.gazebosim.org | sh
```

## To compile
```bash
cd REPO_ROOT
mkdir build
cd build
cmake ..
make -j
```

## To run test
```bash
cd REPO_ROOT/socail_bot
PYTHONPATH=`pwd`/../build/social_bot GAZEBO_MODEL_PATH=`pwd`/../models python test.py
```

## Trouble shooting
You need make sure the python you use matches the python found by cmake. You can check this by looking at REPO_ROOT/build/CMakeCache.txt
