import copy
import os
import setuptools
from setuptools import setup
from setuptools.command.build_ext import build_ext
import setup_pycute
import setup_library
setup_library.perform_setup()
setup_pycute.perform_setup()
setup(name='cutlass', version='3.4.0', description='CUTLASS Pythonic Interface', package_dir={'': '.'}, packages=['cutlass', 'cutlass.emit', 'cutlass.op', 'cutlass.utils', 'cutlass.backend', 'cutlass.backend.utils'], setup_requires=['pybind11'], install_requires=['bfloat16', 'cuda-python>=11.8.0', 'pybind11', 'scikit-build', 'treelib', 'pydot'])