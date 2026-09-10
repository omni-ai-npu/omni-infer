# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Device aliases used by the upstream V2 runner."""

from __future__ import annotations

import torch
import torch_npu  # noqa: F401

from vllm.logger import init_logger

logger = init_logger(__name__)

_CUDA_TO_NPU_ATTRS: dict[str, str] = {
    "Event": "Event",
    "Stream": "Stream",
    "stream": "stream",
    "default_stream": "default_stream",
    "current_stream": "current_stream",
    "set_stream": "set_stream",
    "synchronize": "synchronize",
    "current_device": "current_device",
    "mem_get_info": "mem_get_info",
    "graph_pool_handle": "graph_pool_handle",
    "CUDAGraph": "NPUGraph",
    "graph": "graph",
}


def install_torch_cuda_aliases() -> None:
    """Point CUDA stream and graph APIs at their NPU equivalents."""
    for cuda_attr, npu_attr in _CUDA_TO_NPU_ATTRS.items():
        setattr(torch.cuda, cuda_attr, getattr(torch.npu, npu_attr))
    logger.info_once("[omni-npu/mrv2] installed torch.cuda NPU aliases")


def snapshot_cuda_attrs() -> dict[str, object]:
    """Capture aliased attributes for test isolation."""
    return {name: getattr(torch.cuda, name) for name in _CUDA_TO_NPU_ATTRS}


def restore_cuda_attrs(snapshot: dict[str, object]) -> None:
    """Restore attributes captured by snapshot_cuda_attrs."""
    for name, value in snapshot.items():
        setattr(torch.cuda, name, value)
