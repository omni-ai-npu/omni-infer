# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from omni_npu.compilation.decorators import patch_compile_decorators
from omni_npu.compilation.acl_graph import (
    ACLGraphWrapper,
    GraphTaskEntry,
    OpDescriptor,
)


__all__ = [
    "patch_compile_decorators",
    "ACLGraphWrapper",
    "GraphTaskEntry",
    "OpDescriptor",
]
