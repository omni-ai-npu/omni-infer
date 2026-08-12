# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os
import sys

import pybind11
import torch
from setuptools import Extension, find_namespace_packages, setup


OMNI_NPU_ROOT = "omni"
OMNI_NPU_SUBPACKAGES = find_namespace_packages(
    where=OMNI_NPU_ROOT,
    exclude=(
        "build",
        "build.*",
        "docs",
        "docs.*",
        "src",
        "src.*",
        "tests",
        "tests.*",
    ),
)
OMNI_NPU_PACKAGES = [
    "omni_npu",
    *(f"omni_npu.{package}" for package in OMNI_NPU_SUBPACKAGES),
]
OMNI_NPU_PACKAGE_DIR = {
    "omni_npu": OMNI_NPU_ROOT,
    **{
        f"omni_npu.{package}": os.path.join(OMNI_NPU_ROOT, *package.split("."))
        for package in OMNI_NPU_SUBPACKAGES
    },
}


class PathManagerBase:
    def __init__(self):
        torch_root = os.path.dirname(torch.__file__)
        self.torch_inc = os.path.join(torch_root, "include")
        self.torch_csrc_inc = os.path.join(torch_root, "include/torch/csrc/api/include")
        self.torch_lib = os.path.join(torch_root, "lib")

        ascend_root = os.getenv("ASCEND_TOOLKIT_HOME") or os.getenv("ASCEND_HOME_PATH")
        if ascend_root is None:
            raise EnvironmentError(
                "Environment variable 'ASCEND_TOOLKIT_HOME' or 'ASCEND_HOME_PATH' "
                "is not set. Please configure the Ascend toolkit before building "
                "omni_infer."
            )
        self.ascend_inc = os.path.join(ascend_root, "include")
        self.ascend_lib = os.path.join(ascend_root, "lib64")
        self.torch_npu_inc_dirs = self._get_torch_npu_include_dirs()
        self.torch_npu_lib_dirs = self._get_torch_npu_library_dirs()

    def check(self):
        if not os.path.isdir(self.torch_inc):
            raise FileNotFoundError(f"PyTorch include path not found: {self.torch_inc}")
        if not os.path.isdir(self.torch_lib):
            raise FileNotFoundError(f"PyTorch lib path not found: {self.torch_lib}")
        if not os.path.isdir(self.ascend_inc):
            raise FileNotFoundError(f"Ascend include path not found: {self.ascend_inc}")
        if not os.path.isdir(self.ascend_lib):
            raise FileNotFoundError(f"Ascend library path not found: {self.ascend_lib}")

    def get_include_dirs(self):
        include_dirs = [self.header, self.ascend_inc, self.torch_inc]
        if os.path.exists(self.torch_csrc_inc):
            include_dirs.append(self.torch_csrc_inc)
        include_dirs.extend(self.torch_npu_inc_dirs)
        return include_dirs

    def get_library_dirs(self):
        return [self.torch_lib, self.ascend_lib] + self.torch_npu_lib_dirs

    def get_extra_link_args(self):
        lib_dirs = self.get_library_dirs()
        link_args = [f"-L{x}" for x in lib_dirs]
        link_args.extend([f"-Wl,-rpath={x}" for x in lib_dirs])
        return link_args

    def _get_torch_npu_include_dirs(self):
        try:
            import torch_npu
        except ImportError:
            return []

        torch_npu_root = os.path.dirname(torch_npu.__file__)
        candidates = [
            torch_npu_root,
            os.path.join(torch_npu_root, "include"),
            os.path.dirname(torch_npu_root),
        ]
        return [path for path in candidates if os.path.exists(path)]

    def _get_torch_npu_library_dirs(self):
        try:
            import torch_npu
        except ImportError:
            return []

        torch_npu_root = os.path.dirname(torch_npu.__file__)
        candidates = [
            os.path.join(torch_npu_root, "lib"),
            torch_npu_root,
        ]
        return [path for path in candidates if os.path.exists(path)]


class AllocatorPathManager(PathManagerBase):
    def __init__(self):
        super().__init__()
        self.header = os.path.join(OMNI_NPU_ROOT, "allocator")
        self.sources = [
            os.path.join(self.header, "npu_mem_allocator.cpp"),
        ]
        self.uva_sources = [
            os.path.join(self.header, "npu_view.cpp"),
        ]
        self.check()

    def check(self):
        super().check()
        if not os.path.isdir(self.header):
            raise FileNotFoundError(
                f"omni_npu source directory not found: {self.header}. "
                "Move the contents of omni/src/omni_npu directly into omni "
                "before building."
            )
        for source in self.sources + self.uva_sources:
            if not os.path.isfile(source):
                raise FileNotFoundError(f"Extension source not found: {source}")


allocator_paths = AllocatorPathManager()


ext_modules = [
    Extension(
        "omni_npu.allocator.npu_mem_allocator",
        sources=allocator_paths.sources,
        include_dirs=allocator_paths.get_include_dirs(),
        language="c++",
        extra_compile_args=[
            "-std=c++17",
            "-pthread",
        ],
        extra_link_args=[
            "-pthread",
        ] + allocator_paths.get_extra_link_args(),
        library_dirs=allocator_paths.get_library_dirs(),
        libraries=["torch", "torch_python", "ascendcl"],
    ),
    Extension(
        "omni_npu.allocator.npu_uva",
        sources=allocator_paths.uva_sources,
        include_dirs=allocator_paths.get_include_dirs(),
        language="c++",
        extra_compile_args=[
            "-std=c++17",
            "-pthread",
        ],
        extra_link_args=[
            "-pthread",
        ] + allocator_paths.get_extra_link_args(),
        library_dirs=allocator_paths.get_library_dirs(),
        libraries=["torch", "torch_python", "ascendcl", "torch_npu"],
    ),
]

# LoPT can fall back to standard tokenization when this optional extension is
# unavailable, but pybind11 is normally present through pyproject.toml.
try:
    lopt_source = os.path.join(OMNI_NPU_ROOT, "lopt", "csrc", "match_merge.cpp")
    if not os.path.isfile(lopt_source):
        raise FileNotFoundError(f"LoPT extension source not found: {lopt_source}")
    ext_modules.append(
        Extension(
            "Cpp_match_merge",
            sources=[lopt_source],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=["-std=c++17"],
        )
    )
except (ImportError, FileNotFoundError) as error:
    print(
        f"Skipping optional LoPT C++ extension: {error}. "
        "LoPT will fall back to standard tokenization.",
        file=sys.stderr,
    )


# Distribution metadata is defined in pyproject.toml. The package mapping keeps
# ``import omni_npu`` stable while the physical source directory is ``omni``.
setup(
    packages=OMNI_NPU_PACKAGES,
    package_dir=OMNI_NPU_PACKAGE_DIR,
    ext_modules=ext_modules,
)
