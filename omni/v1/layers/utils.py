# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
from contextlib import nullcontext

import torch
import torchair


def get_npu_execution_type(stream_label):
    if stream_label is None:
        return nullcontext()
    # Using strings to determine whether to include an item in the image,
    # and later we will use logical differentiation based on parameters.
    elif isinstance(stream_label, str):
        return torchair.scope.npu_stream_switch(stream_label)  # Graph GE/ACL
    elif isinstance(stream_label, torch.npu.Stream):
        return torch.npu.stream(stream_label)  # eager
    return nullcontext()


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    import math

    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


def named_stream(name): # not for graph
    if name == "current":
        return torch.npu.current_stream()
    if not hasattr(named_stream, name):
        setattr(named_stream, name, torch.npu.Stream())
    return getattr(named_stream, name)
