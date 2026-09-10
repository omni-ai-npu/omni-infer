# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

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
import torch_npu
import torchair as tng

import torch.distributed as dist

from .parallel_state_ext import (
    get_layer_parallel_group,
    get_local_world_group,
    get_local_group_from_list,
    get_round_cross_group_from_list,
    get_cross_group_from_list
)
from .utils import get_round_swap_perm

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
    "layer_parallel_dp2tp_all2all",
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


def layer_parallel_dp2tp_single(
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
            f"Invalid dim={dim} for input with dim={ndim}."
        )

    world_size = getattr(group, "world_size", None) or dist.get_world_size(
        group=process_group
    )
    split_size = input_.shape[dim]
    if split_size % world_size != 0:
        raise ValueError(
            f"Input size along dim={dim} (resolved dim={dim}) must be "
            f"divisible by world_size={world_size}, got {split_size}."
        )

    splitted_size = split_size // world_size
    splitted_shape = [
        *input_.shape[:dim],
        world_size,
        splitted_size,
        *input_.shape[dim + 1:]
    ]
    if dim != 0:
        in_buf = input_.view(*splitted_shape).transpose(0, dim).contiguous()
    else:
        in_buf = input_.contiguous() if not input_.is_contiguous() else input_
        in_buf = in_buf.view(*splitted_shape)

    out_buf = torch.empty_like(in_buf)
    dist.all_to_all_single(out_buf, in_buf, group=process_group)

    out_shape = [
        out_buf.shape[0] * out_buf.shape[1],
        *out_buf.shape[2:]
    ]

    return out_buf.view(*out_shape)


def layer_parallel_dp2tp_all2all(
    input_: torch.Tensor,
    layer_name_inside_block: str,
    tensor_tag: str | None = None,
    dim: int = -1,
) -> torch.Tensor:
    """Exchange feature shards to convert DP-local activations into TP input.

    A 2-D ``[T, H]`` input is converted into ``[T * world_size,
    H / world_size]``. Each rank sends a different feature shard to every
    peer and receives its local feature shard for all peers' tokens.
    """
    group = _get_group(layer_name_inside_block, tensor_tag)
    if group is None:
        return input_

    if not isinstance(dim, int):
        raise TypeError(f"dim must be int, got {type(dim)!r}.")

    process_group = getattr(group, "device_group", None)
    if process_group is None:
        raise RuntimeError(
            "layer_parallel_dp2tp_all2all requires "
            "GroupCoordinator.device_group (a torch.distributed ProcessGroup)."
        )

    ndim = input_.dim()
    dim = dim + ndim if dim < 0 else dim
    if dim <= 0 or dim >= ndim:
        raise ValueError(
            f"DP2TPAll2All requires a non-leading feature dim, got dim={dim} "
            f"for input with dim={ndim}."
        )

    world_size = getattr(group, "world_size", None) or dist.get_world_size(
        group=process_group
    )
    feature_size = input_.shape[dim]
    if feature_size % world_size != 0:
        raise ValueError(
            f"Input feature size along dim={dim} must be divisible by "
            f"world_size={world_size}, got {feature_size}."
        )

    feature_size_per_rank = feature_size // world_size
    split_shape = [
        *input_.shape[:dim],
        world_size,
        feature_size_per_rank,
        *input_.shape[dim + 1:],
    ]
    in_buf = input_.reshape(*split_shape).movedim(dim, 0).contiguous()
    out_buf = torch.empty_like(in_buf)
    dist.all_to_all_single(out_buf, in_buf, group=process_group)

    out_shape = [
        out_buf.shape[0] * out_buf.shape[1],
        *input_.shape[1:dim],
        feature_size_per_rank,
        *input_.shape[dim + 1:],
    ]
    return out_buf.reshape(*out_shape)


def all_gather_cross(input_: torch.Tensor, idx: int, dim=-1) -> torch.Tensor:
    return get_cross_group_from_list(idx).all_gather(input_, dim)


def all_gather_local_parallel(input_: torch.Tensor, idx: int, dim=-1) -> torch.Tensor:
    return get_local_group_from_list(idx).all_gather(input_, dim)


def all_gather_into_tensor_local_parallel(output_: torch.Tensor, input_: torch.Tensor, idx: int) -> torch.Tensor:
    return get_local_group_from_list(idx).all_gather_into_tensor(output_, input_)


def all_gather_round_pipeline_two_tensor(
    inputs: tuple[torch.Tensor, torch.Tensor],
    node_rank: int,
    num_nodes: int,
    streams: list[torch.npu.Stream],
    dim=-1,
) -> tuple[torch.Tensor, torch.Tensor]:
    perm = get_round_swap_perm(node_rank, num_nodes)
    local_size = get_local_group_from_list(idx=0).world_size
    world_size = num_nodes * local_size
    input_large, input_small = inputs
    cur_stream = torch.npu.current_stream()

    with torch.npu.stream(streams[num_nodes - 1]):
        small_local_ag = all_gather_local_parallel(input_small, idx=1, dim=dim)

    local_ag_dim = input_large.shape[0] * local_size
    large_ag = input_large.new_empty((input_large.shape[0] * world_size,) + input_large.shape[1:])
    with torch.npu.stream(streams[0]):
        start = perm[0] * local_ag_dim
        all_gather_into_tensor_local_parallel(
            large_ag[start:start + local_ag_dim],
            input_large,
            idx=0
        )

    for idx in range(num_nodes - 1):
        cur_internode_stream = streams[idx]
        start = perm[idx + 1] * local_ag_dim
        with torch.npu.stream(cur_internode_stream):
            round_swp = get_round_cross_group_from_list(idx=idx).swap(input_large, method="all2allv")
            all_gather_into_tensor_local_parallel(
                large_ag[start:start + local_ag_dim],
                round_swp,
                idx=idx + 2
            )

    with torch.npu.stream(streams[num_nodes - 1]):
        small_ag = all_gather_cross(small_local_ag, idx=1, dim=dim)
    for idx in range(num_nodes):
        cur_stream.wait_stream(streams[idx])
    return large_ag, small_ag