# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass
from typing import Callable, Optional

import torch


def named_stream(name):
    if name == "current":
        return torch.npu.current_stream()
    if not hasattr(named_stream, name):
        setattr(named_stream, name, torch.npu.Stream())
    return getattr(named_stream, name)


CUBE_SIDE_TASKS_KEY = "cube_side_tasks"
CUBE_SIDE_STREAM_NAME = "sub_stream"


@dataclass
class CubeSideTask:
    # A Vector-bound callback that a Hifloat8 cube matmul will run on a side
    # stream concurrent with itself. Registered by the caller on
    # forward_context.additional_kwargs["cube_side_tasks"][layer.prefix].
    fn: Callable[[], None]
    done_event: Optional["torch.npu.Event"] = None
