# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

"""Layer-parallel communication ops.

This module provides thin wrappers around vLLM's collectives (via
`vllm.distributed.parallel_state.GroupCoordinator`), scoped to a specific
`layer_name_inside_block` and an optional `tensor_tag` ("x" / "y").

All wrappers are **no-ops** if the resolved group is missing or has
`world_size <= 1` (common in single-rank / non-distributed runs).
"""

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from .parallel_state_ext import get_layer_parallel_group, get_local_world_group

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

__all__ = [
    "all_gather_local",
    "reduce_scatter_local",
    "all_to_all_local",
    "layer_parallel_all_reduce",
    "layer_parallel_all_gather",
    "layer_parallel_reduce_scatter",
    "layer_parallel_all2all_single",
]


def all_gather_local(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    group = get_local_world_group()
    if group.world_size <= 1:
        return input_
    return group.all_gather(input_, dim)


def reduce_scatter_local(input_: torch.Tensor) -> torch.Tensor:
    group = get_local_world_group()
    if group.world_size <= 1:
        return input_
    return group.reduce_scatter(input_)


def all_to_all_local(
    input_: torch.Tensor,
    scatter_dim: int = 0,
    gather_dim: int = -1,
) -> torch.Tensor:
    group = get_local_world_group()
    if group.world_size <= 1:
        return input_

    communicator = getattr(group, "device_communicator", None)
    if communicator is None or not hasattr(communicator, "all_to_all"):
        raise RuntimeError("local world group has no device communicator")

    ndim = input_.dim()
    scatter_dim = scatter_dim + ndim if scatter_dim < 0 else scatter_dim
    gather_dim = gather_dim + ndim if gather_dim < 0 else gather_dim
    if scatter_dim < 0 or scatter_dim >= ndim:
        raise ValueError(f"Invalid scatter_dim={scatter_dim} for input with dim={ndim}.")
    if gather_dim < 0 or gather_dim >= ndim:
        raise ValueError(f"Invalid gather_dim={gather_dim} for input with dim={ndim}.")

    split_size = input_.size(scatter_dim)
    if split_size % group.world_size != 0:
        raise ValueError(
            f"Input size along scatter_dim={scatter_dim} must be divisible by "
            f"world_size={group.world_size}, got {split_size}."
        )

    return communicator.all_to_all(
        input_, scatter_dim=scatter_dim, gather_dim=gather_dim
    )


def _get_group(
    layer_name_inside_block: str,
    tensor_tag: str | None = None,
) -> GroupCoordinator | None:
    """Resolve the `GroupCoordinator` for a layer (fast-path for `world_size <= 1`)."""
    group = get_layer_parallel_group(layer_name_inside_block, tensor_tag)
    if group is None or group.world_size <= 1:
        return None
    return group


def layer_parallel_all_reduce(
    input_: torch.Tensor,
    layer_name_inside_block: str,
    tensor_tag: str | None = None,
) -> torch.Tensor:
    """All-reduce `input_` over the layer-parallel group."""
    group = _get_group(layer_name_inside_block, tensor_tag)
    if group is None:
        return input_
    return group.all_reduce(input_)


def layer_parallel_all_gather(
    input_: torch.Tensor,
    layer_name_inside_block: str,
    tensor_tag: str | None = None,
    dim: int = -1,
) -> torch.Tensor:
    """All-gather `input_` over the layer-parallel group and concat on `dim`."""
    group = _get_group(layer_name_inside_block, tensor_tag)
    if group is None:
        return input_
    return group.all_gather(input_, dim=dim)


def layer_parallel_reduce_scatter(
    input_: torch.Tensor,
    layer_name_inside_block: str,
    tensor_tag: str | None = None,
    dim: int = -1,
) -> torch.Tensor:
    """Reduce-scatter `input_` over the layer-parallel group along `dim`."""
    group = _get_group(layer_name_inside_block, tensor_tag)
    if group is None:
        return input_
    return group.reduce_scatter(input_, dim=dim)


def layer_parallel_all2all_single(
    input_: torch.Tensor,
    layer_name_inside_block: str,
    tensor_tag: str | None = None,
    dim: int = 0,
) -> torch.Tensor:
    """ProcessGroup-based all_to_all_single along `dim` (PyTorch standard collective)."""
    group = _get_group(layer_name_inside_block, tensor_tag)
    if group is None:
        return input_

    if not isinstance(dim, int):
        raise TypeError(f"dim must be int, got {type(dim)!r}.")

    process_group = getattr(group, "device_group", None)
    if process_group is None:
        raise RuntimeError(
            "layer_parallel_all2all_single requires GroupCoordinator.device_group "
            "(a torch.distributed ProcessGroup)."
        )

    ndim = input_.dim()
    dim = dim + ndim if dim < 0 else dim
    if dim < 0 or dim >= ndim:
        raise ValueError(
            f"Invalid dim={dim} for input_ with dim={ndim}."
        )

    world_size = getattr(group, "world_size", None) or dist.get_world_size(
        group=process_group
    )
    split_size = input_.size(dim)
    if split_size % world_size != 0:
        raise ValueError(
            f"Input size along dim={dim} (resolved dim={dim}) must be "
            f"divisible by world_size={world_size}, got {split_size}."
        )

    if dim != 0:
        in_buf = input_.transpose(0, dim).contiguous()
    else:
        in_buf = input_.contiguous() if not input_.is_contiguous() else input_

    out_buf = torch.empty_like(in_buf)
    dist.all_to_all_single(out_buf, in_buf, group=process_group)

    if dim != 0:
        return out_buf.transpose(0, dim).contiguous()
    return out_buf
