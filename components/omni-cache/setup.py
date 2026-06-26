# setup.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
from setuptools import setup, find_packages, Extension
import pybind11
import torch
import subprocess


class PathManagerBase:
    def __init__(self):
        # torch
        torch_root = os.path.dirname(torch.__file__)
        self.torch_inc = os.path.join(torch_root, "include")
        self.torch_csrc_inc = os.path.join(torch_root, "include/torch/csrc/api/include")
        self.torch_lib = os.path.join(torch_root, "lib")

        # ascend
        ascend_root = os.getenv('ASCEND_TOOLKIT_HOME', None)
        if ascend_root is None:
            raise EnvironmentError("Environment variable 'ASCEND_TOOLKIT_HOME' is not set. Please set this environment variable before running the program.")
        self.ascend_inc = os.path.join(ascend_root, "include")
        self.ascend_lib = os.path.join(ascend_root, "lib64")

    def check(self):
        if not os.path.exists(self.torch_inc):
            raise FileNotFoundError(f"PyTorch include path not found: {self.torch_inc}")
        if not os.path.exists(self.torch_lib):
            raise FileNotFoundError(f"PyTorch lib path not found: {self.torch_lib}")

    def get_include_dirs(self):
        include_dirs = [self.header, self.ascend_inc, self.torch_inc]
        if os.path.exists(self.torch_csrc_inc):
            include_dirs.append(self.torch_csrc_inc)
        return include_dirs

    def get_library_dirs(self):
        return [self.torch_lib, self.ascend_lib]

    def get_extra_link_args(self):
        lib_dirs = self.get_library_dirs()
        link_args = [f"-L{x}" for x in lib_dirs]
        link_args.extend([f"-Wl,-rpath={x}" for x in lib_dirs])
        return link_args


from setuptools.command.build_py import build_py as _build_py


class build_py(_build_py):
    def run(self):
        root_dir = os.path.dirname(os.path.abspath(__file__))

        # 1. Build tensor_register_lib
        tensor_register_dir = os.path.join(
            root_dir,
            "omni_cache",
            "cache",
            "device_backend",
            "ascend",
            "tensor_register_lib",
        )
        if os.path.isdir(tensor_register_dir):
            subprocess.check_call(["bash", "build.sh"], cwd=tensor_register_dir)
        else:
            print(f"Warning: tensor_register_lib directory not found at {tensor_register_dir}")

        # 2. Build ox backend
        ox_dir = os.path.join(
            root_dir,
            "omni_cache",
            "connector",
            "backends",
            "ox",
        )
        if os.path.isdir(ox_dir):
            subprocess.check_call(["bash", "build_deps.sh"], cwd=ox_dir)
            subprocess.check_call(["make"], cwd=ox_dir)
        else:
            print(f"Warning: ox directory not found at {ox_dir}")

        super().run()


# Define the extension module of omni-cache
setup(
    name='omni-cache',
    version='0.1.0',
    description='PD-separated KV cache connectors for vLLM',
    packages=find_packages(
        exclude=(
            "build",
        )
    ),
    install_requires=[
        'torch',
        'torch_npu',
        'pybind11',
        'msgpack',
    ],
    include_package_data=True,
    cmdclass={"build_py": build_py},
)
