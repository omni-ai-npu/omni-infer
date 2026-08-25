# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import os

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

    def get_include_dirs(self, header):
        include_dirs = [header, self.ascend_inc, self.torch_inc]
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


class PathManager(PathManagerBase):
    def __init__(self):
        super().__init__()
        self.allocator_header = os.path.join(OMNI_NPU_ROOT, "allocator")
        self.allocator_sources = [
            os.path.join(self.allocator_header, "npu_mem_allocator.cpp"),
        ]
        self.uva_sources = [
            os.path.join(self.allocator_header, "npu_view.cpp"),
        ]
        self.kv_offload_header = os.path.join(
            OMNI_NPU_ROOT, "v1", "kv_offload", "cpu", "csrc"
        )
        self.host_register_sources = [
            os.path.join(self.kv_offload_header, "tensor_register.cpp"),
        ]
        self.zero_copy_sources = [
            os.path.join(self.kv_offload_header, "zero_copy_npu.cpp"),
        ]
        self.swap_blocks_sources = [
            os.path.join(self.kv_offload_header, "swap_blocks_batch.cpp"),
        ]
        self.lopt_header = os.path.join(OMNI_NPU_ROOT, "lopt", "csrc")
        self.lopt_sources = [
            os.path.join(self.lopt_header, "match_merge.cpp"),
        ]
        self.check()

    def check(self):
        super().check()
        if not os.path.isdir(self.allocator_header):
            raise FileNotFoundError(
                f"omni_npu source directory not found: {self.allocator_header}. "
                "Move the contents of omni/src/omni_npu directly into omni "
                "before building."
            )
        if not os.path.isdir(self.kv_offload_header):
            raise FileNotFoundError(
                f"KV offload source directory not found: {self.kv_offload_header}"
            )
        if not os.path.isdir(self.lopt_header):
            raise FileNotFoundError(
                f"LoPT source directory not found: {self.lopt_header}"
            )
        for source in (
            self.allocator_sources
            + self.uva_sources
            + self.host_register_sources
            + self.zero_copy_sources
            + self.swap_blocks_sources
            + self.lopt_sources
        ):
            if not os.path.isfile(source):
                raise FileNotFoundError(f"Extension source not found: {source}")


def get_cann_memcpy_batch_flags(ascend_inc):
    cann_batch_flags = []
    acl_rt_header = os.path.join(ascend_inc, "acl", "acl_rt.h")
    try:
        with open(acl_rt_header, encoding="utf-8", errors="ignore") as acl_rt:
            acl_rt_text = acl_rt.read()
        if (
            "aclrtMemcpyBatchAsync" in acl_rt_text
            and "aclrtMemcpyBatchAttr" in acl_rt_text
        ):
            cann_batch_flags.append("-DCANN_MEMCPY_BATCH_ASYNC")
    except OSError:
        pass
    return cann_batch_flags


paths = PathManager()


def _allocator_extension_kwargs():
    return dict(
        include_dirs=paths.get_include_dirs(paths.allocator_header),
        language="c++",
        extra_compile_args=[
            "-std=c++17",
            "-pthread",
        ],
        extra_link_args=[
            "-pthread",
        ] + paths.get_extra_link_args(),
        library_dirs=paths.get_library_dirs(),
    )


ext_modules = [
    Extension(
        "omni_npu.allocator.npu_mem_allocator",
        sources=paths.allocator_sources,
        libraries=["torch", "torch_python", "ascendcl"],
        **_allocator_extension_kwargs(),
    ),
    Extension(
        "omni_npu.allocator.npu_uva",
        sources=paths.uva_sources,
        libraries=["torch", "torch_python", "ascendcl", "torch_npu"],
        **_allocator_extension_kwargs(),
    ),
    Extension(
        "omni_npu.v1.kv_offload.cpu._host_register",
        sources=paths.host_register_sources,
        include_dirs=[
            pybind11.get_include(),
            paths.ascend_inc,
        ],
        language="c++",
        extra_compile_args=["-std=c++17", "-pthread"],
        extra_link_args=["-pthread"] + paths.get_extra_link_args(),
        library_dirs=paths.get_library_dirs(),
        libraries=["ascendcl"],
    ),
    Extension(
        "omni_npu.v1.kv_offload.cpu._zero_copy_npu",
        sources=paths.zero_copy_sources,
        include_dirs=paths.get_include_dirs(paths.kv_offload_header) + [
            pybind11.get_include()
        ],
        language="c++",
        extra_compile_args=["-std=c++17", "-pthread", "-O3"],
        extra_link_args=["-pthread"] + paths.get_extra_link_args(),
        library_dirs=paths.get_library_dirs(),
        libraries=["torch", "torch_python", "ascendcl", "torch_npu"],
    ),
    Extension(
        "omni_npu.v1.kv_offload.cpu._swap_blocks_batch",
        sources=paths.swap_blocks_sources,
        include_dirs=paths.get_include_dirs(paths.kv_offload_header) + [
            pybind11.get_include()
        ],
        language="c++",
        extra_compile_args=(
            ["-std=c++17", "-pthread", "-O3"] + get_cann_memcpy_batch_flags(
                paths.ascend_inc
            )
        ),
        extra_link_args=["-pthread"] + paths.get_extra_link_args(),
        library_dirs=paths.get_library_dirs(),
        libraries=["torch", "torch_python", "ascendcl", "torch_npu"],
    ),
    Extension(
        "Cpp_match_merge",
        sources=paths.lopt_sources,
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=["-std=c++17"],
    ),
]


# Distribution metadata is defined in pyproject.toml. The package mapping keeps
# ``import omni_npu`` stable while the physical source directory is ``omni``.
setup(
    packages=OMNI_NPU_PACKAGES,
    package_dir=OMNI_NPU_PACKAGE_DIR,
    ext_modules=ext_modules,
)
