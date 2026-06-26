# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
from torch.utils.cpp_extension import BuildExtension, CppExtension

cann_root = os.environ.get(
    "ASCEND_TOOLKIT_HOME",
    os.environ.get("ASCEND_HOME", "/usr/local/Ascend/ascend-toolkit/latest")
)

include_dirs = [os.path.join(cann_root, "include")]
library_dirs = [os.path.join(cann_root, "lib64")]
libraries = ["ascendcl"]

extension = CppExtension(
    name="zero_copy_npu",
    sources=["zero_copy_npu.cpp"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    extra_compile_args=["-O3", "-std=c++17", "-fPIC"],
    language="c++",
)

from setuptools import setup
setup(
    name="zero_copy_npu",
    version="0.1",
    description="Hugepage zero-copy NPU tensor",
    ext_modules=[extension],
    cmdclass={"build_ext": BuildExtension},
    zip_safe=False,
    install_requires=[
        'torch',
        'torch_npu',
        'pybind11',
    ],
    include_package_data=True
)
