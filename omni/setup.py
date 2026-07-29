# setup.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import sys
from setuptools import setup, find_packages, Extension
import torch


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
            raise EnvironmentError(
                "Environment variable 'ASCEND_TOOLKIT_HOME' is not set. Please set this environment variable " \
                "before running the program."
            )
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


class AllocatorPathManager(PathManagerBase):
    def __init__(self):
        super().__init__()
        self.header = "omni_npu/allocator"
        self.sources = [
            "src/omni_npu/allocator/npu_mem_allocator.cpp"
        ]
        self.check()

    def check(self):
        super().check()

    def get_include_dirs(self):
        return super().get_include_dirs()

    def get_library_dirs(self):
        return super().get_library_dirs()

    def get_extra_link_args(self):
        return super().get_extra_link_args()

alloc_paths = AllocatorPathManager()

# 定义扩展模块
ext_modules = [
    Extension(
        "omni_npu.allocator.npu_mem_allocator",
        sources=alloc_paths.sources,
        include_dirs=alloc_paths.get_include_dirs(),
        language='c++',
        extra_compile_args=[
            '-std=c++17',
            '-pthread',
        ],
        extra_link_args=[
            '-pthread',
            '-lascendcl',
            '-ltorch',
            '-ltorch_python',
        ] + alloc_paths.get_extra_link_args(),
        library_dirs=alloc_paths.get_library_dirs(),
        libraries=['torch', 'torch_python', 'ascendcl']
    ),
]

# ────────────────────────────────────────────────────────────
# LoPT C++ extensions (pybind11)
# These are optional — LoPT falls back to standard tokenization
# if the extensions are not available.
# ────────────────────────────────────────────────────────────
_lopt_extensions = []
try:
    import pybind11

    _lopt_src_dir = "src/omni_npu/lopt/csrc"
    _lopt_extensions = [
        Extension(
            "Cpp_match_merge",
            sources=[os.path.join(_lopt_src_dir, "match_merge.cpp")],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17"],
        ),
    ]
except ImportError:
    print(
        "pybind11 not available — skipping LoPT C++ extension build. "
        "LoPT will fall back to standard tokenization.",
        file=sys.stderr,
    )

ext_modules.extend(_lopt_extensions)


setup(
    name='omni-npu',
    version='0.2.0',
    description='Omni Infer v1',
    packages=find_packages(
        exclude=(
            "build"
        )
    ),
    install_requires=[
        'torch',
        'torch_npu',
    ],
    ext_modules=ext_modules,
)
