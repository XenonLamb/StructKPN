#!/usr/bin/env python

import os

from setuptools import setup, find_packages

this_file = os.path.dirname(__file__)

setup(
    name="tile_matching_package",
    version="0.1",
    description="The tile-based L2 search layer in HDRPlus",
    url="https://github.sz.sensetime.com/zhangyu1/mfsr-pytorch",
    author="Yu Zhang",
    author_email="zhangyu1@sensetime.com",
    # Require cffi
    install_requires=["cffi>=1.0.0"],
    setup_requires=["cffi>=1.0.0"],
    # Exclude the build files.
    packages=find_packages(exclude=["build"]),
    # Package where to put the extensions. Has to be a prefix of build.py
    ext_package="",
    # Extensions to compile
    cffi_modules=[
        os.path.join(this_file, "build.py:ffi")
    ],
)
