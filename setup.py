# Copyright (c) 2019 Horizon Robotics. All Rights Reserved.

from setuptools import setup, find_packages
import os
import shutil
import glob

for f in glob.glob("build/pygazebo/pygazebo*.so"):
    shutil.copy2(f, "python/social_bot")

setup(
    name='social_bot',
    version='0.0.1',
    install_requires=[
        'gin-config == 0.1.3', 'gym == 0.12.5', 'numpy >= 1.13.3',
        'matplotlib', 'psutil', 'lxml >= 3.5.0', 'Pillow', 'absl-py', 'pytest',
        'PyUserInput', 'python-xlib'
    ],
    package_dir={'': 'python'},
    packages=find_packages('python'),
    package_data={'social_bot': ['models', 'worlds', 'pygazebo*.so']},
)
