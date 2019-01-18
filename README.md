# SocialBot

A python environment for developing interactive learning agent with language communication ability.

## Install dependency
SocialBot is built on top of [Gazebo simulator](http://gazebosim.org). You need to install Gazebo first using the following command:
```bash
curl -sSL http://get.gazebosim.org | sh
```

You also need to install the following packages:
```bash
pip3 install numpy matplotlib gym
apt install python3-tk
```

## To compile
```bash
cd REPO_ROOT
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
To see the graphics, you can open another terminal and run `gzclient`

## Trouble shooting
You need to make sure the python you use matches the python found by cmake. You can check this by looking at REPO_ROOT/build/CMakeCache.txt
