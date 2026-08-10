# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from __future__ import annotations

"""Layer-parallel group helpers for vLLM (omni-npu extension).

This module reads `layer_parallel_config` from `vllm_config.additional_config` and
creates per-layer/per-tensor-tag process groups (via vLLM's `GroupCoordinator`).

It also provides utility helpers to split inputs evenly across the layer-parallel
world size, with optional padding + all-gather unpadding.
"""

import os
from typing import Any

import torch
import torch.distributed as dist
import torch_npu

from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform

from vllm.distributed import parallel_state
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_tp_group,
    get_world_group as _get_world_group,
    init_model_parallel_group,
)
from vllm.logger import init_logger

from omni_npu import envs

logger = init_logger(__name__)

_DIE_PER_NODE_910C = 16
_DIE_PER_NODE_910B = 8
_DIE_PER_NODE_950 = 16

device_name = torch_npu.npu.get_device_name(0)


def get_npu_device_count():
    is_a2_device = device_name.startswith("Ascend910B")
    is_a5_device = device_name.startswith("Ascend950")
    if is_a2_device:
        return _DIE_PER_NODE_910B
    elif is_a5_device:
        return _DIE_PER_NODE_950
    else:
        return _DIE_PER_NODE_910C

_NUM_COMM_GROUP = 2

__all__ = [
    "ensure_layer_parallel_initialized",
    "initialize_local_world_group",
    "get_local_world_group",
    "get_world_group",
    "get_layer_parallel_group",
    "get_layer_transform_type",
    "get_layer_dim",
    "get_layer_parallel_world_size",
    "get_layer_parallel_rank",
    "maybe_pad_and_slice",
    "maybe_unpad_and_all_gather",
    "get_local_group_from_list",
    "get_round_cross_group_from_list",
    "get_moe_dispatch_ep_group",
    "destroy_parallel_state_ext_groups",
]

_LOCAL_WORLD = None
_LOCAL_COMM_LIST = None
_CROSS_COMM_LIST = None
_CROSS_ROUND_COMM_LIST = None

# Layer-parallel communication registry (fixed name: _LAYER_COMM_DICT).
# key: layer name inside a block in layer_parallel_config (e.g., "self_attn.q_proj")
# value:
#   {
#     "parallel_group": GroupCoordinator | None,
#     "x_transform": dict[str, Any] | None,  # {"type": str, "dim": int, "parallel_group": GroupCoordinator | None}
#     "y_transform": dict[str, Any] | None,  # {"type": str, "dim": int, "parallel_group": GroupCoordinator | None}
#   }
_LAYER_COMM_DICT: dict[str, dict[str, Any]] | None = None

# Cache process groups by normalized group_ranks so layers sharing the same
# tp_size_or_ranks reuse a single communication domain.
_TP_SIZE_OR_RANKS_GROUP_CACHE: dict[tuple[tuple[int, ...], ...], GroupCoordinator] = {}

# Supported tensor transform keys.
_TENSOR_TRANSFORM_KEYS = ("x_transform", "y_transform")


def destroy_parallel_state_ext_groups() -> None:
    """Destroy the Ascend A2 (910B) extension groups owned by this module.

    These three lists are only populated on Ascend910B, by
    `initialize_local_comm_group_list` / `initialize_round_swap_comm_group_list` /
    `initialize_cross_comm_group_list` (called from `ParallelStatePatch.
    initialize_model_parallel`). On non-A2 devices they stay `None`, so each
    block below is a no-op. Does not touch `_LOCAL_WORLD` or `_LAYER_COMM_DICT`,
    which belong to the separate layer-parallel feature.
    """
    # Created by initialize_local_comm_group_list (single-node A2 groups).
    global _LOCAL_COMM_LIST
    if _LOCAL_COMM_LIST:
        for group in _LOCAL_COMM_LIST:
            group.destroy()
    _LOCAL_COMM_LIST = None

    # Created by initialize_cross_comm_group_list (cross-node A2 groups).
    global _CROSS_COMM_LIST
    if _CROSS_COMM_LIST:
        for group in _CROSS_COMM_LIST:
            group.destroy()
    _CROSS_COMM_LIST = None

    # Created by initialize_round_swap_comm_group_list (cross-node round-swap
    # schedule, only when num_nodes >= 2 and even).
    global _CROSS_ROUND_COMM_LIST
    if _CROSS_ROUND_COMM_LIST:
        for group in _CROSS_ROUND_COMM_LIST:
            group.destroy()
    _CROSS_ROUND_COMM_LIST = None


def get_world_group():
    return _get_world_group()


def get_moe_dispatch_ep_group():
    group = getattr(parallel_state, "_MOE_DISPATCH_EP", None)
    if group is None:
        raise RuntimeError("_MOE_DISPATCH_EP is None")
    return group


def calculate_effective_local_size(local_size: int, world_size: int) -> int:
    effective_local_size = min(local_size, world_size)
    if effective_local_size < local_size:
        logger.info(
            "Note: Using only %s of %s available NPU devices",
            effective_local_size,
            local_size,
        )
    if world_size % effective_local_size != 0:
        raise AssertionError(
            f"world_size ({world_size}) must be divisible by "
            f"effective_local_size ({effective_local_size})"
        )
    return effective_local_size


def initialize_local_world_group(backend: str | None = None) -> None:
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")

    global _LOCAL_WORLD
    if _LOCAL_WORLD is not None:
        return

    world_size: int = dist.get_world_size()

    # External-DP launch scripts pin each worker to one NPU via
    # ASCEND_RT_VISIBLE_DEVICES, so device_count() is often 1. Prefer cluster
    # env vars set by ansible when present.
    local_size: int | None = None

    # Multi-node Prefill: ranks per node = world_size // NNODES.
    nnodes = int(os.getenv("NNODES", "0"))
    if nnodes > 1 and world_size % nnodes == 0:
        local_size = world_size // nnodes

    if local_size is None:
        # Decode per-node NPU count (NUM_SERVERS). Prefill docker does not set this.
        num_servers = int(os.getenv("NUM_SERVERS", "0"))
        if num_servers > 1 and world_size % num_servers == 0:
            local_size = num_servers

    if local_size is None:
        # Dev/mock: infer local size from visible devices when cluster env is absent.
        if envs.OMNI_NO_NPU_MOCK:
            visible = os.getenv("ASCEND_RT_VISIBLE_DEVICES", "")
            local_size = len(visible.split(",")) if visible else 1
        else:
            # Last resort: local NPU count on this process (often 1 under external DP).
            local_size = torch.npu.device_count()

    local_size = calculate_effective_local_size(local_size, world_size)

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // local_size
    group_ranks = []
    for i in range(num_local_groups):
        ranks = list(range(i * local_size, (i + 1) * local_size))
        group_ranks.append(ranks)

    _LOCAL_WORLD = init_model_parallel_group(
        group_ranks,
        get_world_group().local_rank,
        backend,
        use_message_queue_broadcaster=True,
        group_name="world_local",
    )


def get_local_world_group():
    if _LOCAL_WORLD is None:
        raise RuntimeError("local world group is not initialized")
    return _LOCAL_WORLD


def initialize_local_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    logical_ranks = list(get_world_group().ranks)
    world_size = len(logical_ranks)
    local_size = get_npu_device_count()

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_local_groups: int = world_size // local_size
    global _LOCAL_COMM_LIST
    if _LOCAL_COMM_LIST is not None:
        raise RuntimeError("_LOCAL_COMM_LIST must be None")
    _LOCAL_COMM_LIST = list()
    group_ranks = []
    for i in range(num_local_groups):
        ranks = logical_ranks[i * local_size : (i + 1) * local_size]
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    comm_group_per_server = 1
    # one group for topk and the other is redundant
    total_comm_groups = comm_group_per_server * num_local_groups + _NUM_COMM_GROUP
    for _ in range(total_comm_groups):
        _LOCAL_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                use_message_queue_broadcaster=True,
                group_name="world_local",
            )
        )


def initialize_round_swap_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    from omni_npu.v1.distributed.utils import generate_round_swap_schedule
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    logical_ranks = list(get_world_group().ranks)
    world_size = len(logical_ranks)

    local_size = get_npu_device_count()
    if not world_size % local_size == 0:
        raise RuntimeError(
            f"world_size ({world_size}) must be divisible by local_size ({local_size})"
        )

    num_nodes = world_size // local_size

    backend = backend or torch.distributed.get_backend(
        get_world_group().device_group)

    global _CROSS_ROUND_COMM_LIST
    if _CROSS_ROUND_COMM_LIST is not None:
        raise RuntimeError(
            "pipeline model parallel group is already initialized")
    _CROSS_ROUND_COMM_LIST = []

    round_swap_schedule = generate_round_swap_schedule(num_nodes)
    group_ranks_rounds = []
    for round_pairs in round_swap_schedule:
        group_ranks = []
        for i in range(local_size):
            for a, b in round_pairs:
                group_ranks.append(
                    [
                        logical_ranks[a * local_size + i],
                        logical_ranks[b * local_size + i],
                    ]
                )
        group_ranks_rounds.append(group_ranks)

    _CROSS_ROUND_COMM_LIST = [
        init_model_parallel_group(group_ranks_rounds[i],
            get_world_group().local_rank,
            backend,
            group_name=f"world_round{i}_cross")
        for i in range(num_nodes - 1)
    ]


def initialize_cross_comm_group_list(backend) -> None:
    # Get world size and rank. Ensure some consistencies.
    if not torch.distributed.is_initialized():
        raise RuntimeError("torch.distributed must be initialized")
    logical_ranks = list(get_world_group().ranks)
    world_size = len(logical_ranks)
    local_size = get_npu_device_count()

    server_size = world_size // local_size

    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    # Build the pipeline model-parallel groups.
    num_cross_groups: int = world_size // server_size
    global _CROSS_COMM_LIST
    if _CROSS_COMM_LIST is not None:
        raise RuntimeError("pipeline model parallel group is already initialized")
    _CROSS_COMM_LIST = list()
    group_ranks = []
    for i in range(num_cross_groups):
        ranks = [logical_ranks[index] for index in range(i, world_size, num_cross_groups)]
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce

    for _ in range(_NUM_COMM_GROUP):
        _CROSS_COMM_LIST.append(
            init_model_parallel_group(
                group_ranks,
                get_world_group().local_rank,
                backend,
                group_name="world_cross",
            )
        )


def ensure_layer_parallel_initialized(
    backend: str | None = None,
) -> None:
    """Initialize layer-parallel groups from vLLM config (idempotent).

    Safe to call multiple times. If the vLLM config or `layer_parallel_config`
    is missing, initialization becomes a no-op and defaults are used.
    """
    global _LAYER_COMM_DICT
    if _LAYER_COMM_DICT is not None:
        return

    _clear_tp_size_or_ranks_group_cache()

    if not dist.is_initialized():
        logger.warning(
            "Distributed is not initialized, skipping layer parallel initialization"
        )
        # Mark as initialized to avoid repeated warnings and repeated attempts.
        _LAYER_COMM_DICT = {}
        return

    initialize_local_world_group(backend=backend)

    model_parallel_config = _load_layer_parallel_config_from_model_extra_config()
    if model_parallel_config is None:
        logger.debug("model_parallel_config is None, layer parallel skip initialization")
        _LAYER_COMM_DICT = {}
        return

    layer_parallel_config = model_parallel_config.get("layer_parallel_config") or {}
    if not layer_parallel_config:
        _LAYER_COMM_DICT = {}
        return

    _LAYER_COMM_DICT = {}

    vllm_config = None
    try:
        vllm_config = get_current_vllm_config()
    except Exception as e:
        logger.debug(f"Failed to get vllm_config from framework: {e}")
    parallel_config = getattr(vllm_config, "parallel_config", None) if vllm_config else None
    local_rank = getattr(parallel_config, "local_rank", 0)
    if local_rank is None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    backend = backend or getattr(current_platform, "dist_backend", "hccl")

    for layer_name_inside_block, config in layer_parallel_config.items():
        if not isinstance(config, dict):
            continue

        layer_cfg: dict[str, Any] = {
            "parallel_group": _create_group_from_tp_size_or_ranks(
                config.get("tp_size_or_ranks"),
                local_rank,
                backend,
                f"layer_{layer_name_inside_block}",
            )
        }

        for tag in _TENSOR_TRANSFORM_KEYS:
            transform_cfg = _parse_tensor_transform_cfg(
                config.get(tag),
                local_rank,
                backend,
                f"layer_{layer_name_inside_block}_{tag}",
            )
            if transform_cfg is not None:
                layer_cfg[tag] = transform_cfg

        _LAYER_COMM_DICT[layer_name_inside_block] = layer_cfg


def get_layer_parallel_group(
    layer_name_inside_block: str,
    tensor_tag: str | None = None,
) -> GroupCoordinator | None:
    """Get the effective `GroupCoordinator` for a layer and optional tensor tag.

    Fallback order (most specific to least specific):
    - x/y transform group (when `tensor_tag` is "x" or "y")
    - per-layer group
    - global tensor-parallel group (`get_tp_group()`)

    Notes:
    - If the global TP group is not initialized, this function returns `None`
      (instead of raising an assertion from vLLM's `get_tp_group()`).
    """
    try:
        default_tp_group: GroupCoordinator | None = get_tp_group()
    except AssertionError:
        default_tp_group = None

    # If layer-parallel is not initialized, fall back to TP group (or None).
    if _LAYER_COMM_DICT is None:
        return default_tp_group

    layer_cfg: dict[str, Any] | None = _LAYER_COMM_DICT.get(layer_name_inside_block)

    # Fallback order: x/y transform group -> layer group -> global TP group.
    if layer_cfg and tensor_tag in ("x", "y"):
        transform_key = f"{tensor_tag}_transform"
        transform_cfg: dict[str, Any] | None = layer_cfg.get(transform_key)
        if transform_cfg is not None:
            transform_group = transform_cfg.get("parallel_group")
            if transform_group is not None:
                return transform_group

    if layer_cfg is not None:
        layer_group = layer_cfg.get("parallel_group")
        if layer_group is not None:
            return layer_group

    return default_tp_group


def get_layer_transform_type(layer_name_inside_block: str, tensor_tag: str | None = None) -> str:
    """Return the configured transform op type for a layer's tensor tag (default: 'NoOp')."""
    if _LAYER_COMM_DICT is None:
        return "NoOp"
    layer_cfg = _LAYER_COMM_DICT.get(layer_name_inside_block)
    if not layer_cfg:
        return "NoOp"
    if tensor_tag in ("x", "y"):
        transform_key = f"{tensor_tag}_transform"
        transform_cfg = layer_cfg.get(transform_key)
        if transform_cfg:
            return transform_cfg.get("type", "NoOp")
    return "NoOp"


def get_layer_dim(layer_name_inside_block: str, tensor_tag: str | None = None) -> int:
    """Return the configured dim for a layer's tensor tag (default: 0)."""
    if _LAYER_COMM_DICT is None:
        return 0
    layer_cfg = _LAYER_COMM_DICT.get(layer_name_inside_block)
    if not layer_cfg:
        return 0
    if tensor_tag in ("x", "y"):
        transform_key = f"{tensor_tag}_transform"
        transform_cfg = layer_cfg.get(transform_key)
        if transform_cfg:
            return transform_cfg.get("dim", 0)
    return 0


def get_layer_parallel_world_size(layer_name_inside_block: str, tensor_tag: str | None = None) -> int:
    """Return world size for the effective group (default: 1)."""
    group = get_layer_parallel_group(layer_name_inside_block, tensor_tag)
    return group.world_size if group else 1


def get_layer_parallel_rank(layer_name_inside_block: str, tensor_tag: str | None = None) -> int:
    """Return rank within the effective group (default: 0)."""
    group = get_layer_parallel_group(layer_name_inside_block, tensor_tag)
    return group.rank_in_group if group else 0


def maybe_pad_and_slice(
    input_: torch.Tensor,
    dim: int = 0,
    layer_name_inside_block: str | None = None,
) -> tuple[torch.Tensor, int]:
    """Optionally pad `input_` on `dim` and return the rank-local slice.

    Returns `(slice_tensor, original_length_on_dim)`.
    """
    if dim < 0:
        dim += input_.dim()
    if dim < 0 or dim >= input_.dim():
        raise ValueError(f"Invalid dim={dim} for tensor with dim={input_.dim()}")
    if layer_name_inside_block is None or not is_layer_parallel_input_split_enabled():
        return input_, input_.shape[dim]

    world_size = get_layer_parallel_world_size(layer_name_inside_block)
    if world_size <= 1:
        return input_, input_.shape[dim]

    rank = get_layer_parallel_rank(layer_name_inside_block)
    orig_size = input_.shape[dim]
    pad_size = (world_size - (orig_size % world_size)) % world_size

    if pad_size > 0:
        padding = [0] * (2 * input_.dim())
        # torch.nn.functional.pad uses reversed dimension order.
        padding_idx = (input_.dim() - 1 - dim) * 2 + 1
        padding[padding_idx] = pad_size
        input_ = torch.nn.functional.pad(input_, padding)

    chunk_size = input_.shape[dim] // world_size
    start = rank * chunk_size
    end = (rank + 1) * chunk_size

    slices = [slice(None)] * input_.dim()
    slices[dim] = slice(start, end)
    output = input_[tuple(slices)]

    return output, orig_size


def maybe_unpad_and_all_gather(
    input_: torch.Tensor,
    actual_length: int,
    dim: int = 0,
    layer_name_inside_block: str | None = None,
) -> torch.Tensor:
    """All-gather `input_` along `dim` and remove padding based on `actual_length`."""
    if dim < 0:
        dim += input_.dim()
    if dim < 0 or dim >= input_.dim():
        raise ValueError(f"Invalid dim={dim} for tensor with dim={input_.dim()}")
    if layer_name_inside_block is None or not is_layer_parallel_input_split_enabled():
        return input_

    group = get_layer_parallel_group(layer_name_inside_block)
    if group is None or group.world_size <= 1:
        return input_

    output = group.all_gather(input_, dim=dim)

    if output.shape[dim] > actual_length:
        slices = [slice(None)] * output.dim()
        slices[dim] = slice(0, actual_length)
        output = output[tuple(slices)]

    return output


def _load_layer_parallel_config_from_model_extra_config() -> dict[str, Any] | None:
    """Load model_parallel_config from model_extra_config (config_loader)."""
    parallel_cfg = _get_model_parallel_config()
    if parallel_cfg is None:
        return None

    layer_parallel_config = getattr(parallel_cfg, "layer_parallel_config", {}) or {}
    return {
        "layer_parallel_config": layer_parallel_config,
    }


def is_layer_parallel_input_split_enabled() -> bool:
    """Read input_split from model_extra_config (config_loader)."""
    return bool(getattr(_get_model_parallel_config(), "input_split", False))


def _get_model_parallel_config() -> Any | None:
    """Return model parallel config from model_extra_config, if available."""
    try:
        from omni_npu.model_config.config_loader.loader import model_extra_config
    except Exception as e:
        logger.debug(f"Failed to import model_extra_config: {e}")
        return None

    parallel_cfg = getattr(model_extra_config, "parall_config", None)
    return parallel_cfg


_CANONICAL_COMM_OP_TYPE_ALIASES: dict[str, str] = {
    "noop": "NoOp",
    "none": "NoOp",
    "no": "NoOp",
    "all2all": "ALL2ALL",
    "all2allv": "ALL2ALL",
    "alltoall": "ALL2ALL",
    "alltoallv": "ALL2ALL",
    "allreduce": "AllReduce",
    "allgather": "AllGather",
    "reducescatter": "ReduceScatter",
    "dp2tp2": "DP2TP",
    "dp2tpall2all": "DP2TPAll2All",
}
_CANONICAL_COMM_OP_TYPES: set[str] = {
    "NoOp",
    "ALL2ALL",
    "AllReduce",
    "AllGather",
    "ReduceScatter",
    "DP2TP",
    "DP2TPAll2All",
}


def _normalize_comm_op_type(op_type: Any) -> str:
    """Normalize op type strings to canonical values."""
    if not isinstance(op_type, str):
        return "NoOp"
    normalized = op_type.strip()
    if not normalized:
        return "NoOp"
    key = normalized.replace("_", "").replace("-", "").replace(" ", "").lower()
    # allow canonical names without case sensitivity
    for canonical in _CANONICAL_COMM_OP_TYPES:
        if normalized.lower() == canonical.lower():
            return canonical
    return _CANONICAL_COMM_OP_TYPE_ALIASES.get(key, "NoOp")


def _parse_tensor_transform_cfg(
    transform_cfg: Any,
    local_rank: int,
    backend: str,
    group_name: str,
) -> dict[str, Any] | None:
    """Parse x_transform / y_transform into a uniform dict."""
    if not isinstance(transform_cfg, dict):
        return None
    t_type = _normalize_comm_op_type(transform_cfg.get("type", "NoOp"))
    dim = int(transform_cfg.get("dim", 0) or 0)
    return {
        "type": t_type,
        "dim": dim,
        "parallel_group": _create_group_from_tp_size_or_ranks(
            transform_cfg.get("tp_size_or_ranks"),
            local_rank,
            backend,
            group_name,
        ),
    }


def _group_ranks_cache_key(group_ranks: list[list[int]]) -> tuple[tuple[int, ...], ...]:
    """Build a hashable cache key from group_ranks."""
    return tuple(tuple(ranks) for ranks in group_ranks)


def _clear_tp_size_or_ranks_group_cache() -> None:
    """Clear the tp_size_or_ranks group cache."""
    _TP_SIZE_OR_RANKS_GROUP_CACHE.clear()


def _create_group_from_tp_size_or_ranks(
    tp_size_or_ranks: Any,
    local_rank: int,
    backend: str,
    group_name: str,
) -> GroupCoordinator | None:
    """Parse tp_size_or_ranks and create or reuse a GroupCoordinator."""
    group_ranks = _tp_size_or_ranks_to_group_ranks(tp_size_or_ranks, group_name)
    if group_ranks is None:
        return None

    cache_key = _group_ranks_cache_key(group_ranks)
    cached_group = _TP_SIZE_OR_RANKS_GROUP_CACHE.get(cache_key)
    if cached_group is not None:
        logger.debug(
            "Reusing communication group for %s (same tp_size_or_ranks as an existing group)",
            group_name,
        )
        return cached_group

    group = init_model_parallel_group(
        group_ranks=group_ranks,
        local_rank=local_rank,
        backend=backend,
        group_name=group_name,
    )
    _TP_SIZE_OR_RANKS_GROUP_CACHE[cache_key] = group
    return group


def _tp_size_or_ranks_to_group_ranks(
    spec: Any,
    group_name: str,
) -> list[list[int]] | None:
    """Convert tp_size_or_ranks into group_ranks for init_model_parallel_group."""
    if not spec:
        return None

    if isinstance(spec, list):
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before parsing tp_size_or_ranks."
            )
        world = dist.get_world_size()

        # Normalize `spec` into `list[list[int]]`.
        # - list[int] -> [list[int]]
        # - list[list[int]] -> list[list[int]]
        if all(isinstance(x, list) for x in spec):
            group_ranks = spec
        elif all(isinstance(x, int) for x in spec):
            group_ranks = [spec]
        else:
            raise RuntimeError(
                f"Invalid tp_size_or_ranks={spec!r} for {group_name}: "
                "expected list[int] or list[list[int]]."
            )

        flat: list[int] = []
        for grp in group_ranks:
            if not isinstance(grp, list) or not grp:
                raise RuntimeError(
                    f"Invalid tp_size_or_ranks={spec!r} for {group_name}: "
                    "each group must be a non-empty list of global ranks."
                )
            seen_in_grp: set[int] = set()
            for r in grp:
                if not isinstance(r, int):
                    raise RuntimeError(
                        f"Invalid tp_size_or_ranks={spec!r} for {group_name}: "
                        "rank ids must be integers."
                    )
                if r < 0 or r >= world:
                    raise RuntimeError(
                        f"Invalid rank id {r} in tp_size_or_ranks for {group_name}: "
                        f"expected 0 <= rank < world_size({world})."
                    )
                if r in seen_in_grp:
                    raise RuntimeError(
                        f"Duplicate rank {r} within a single group in tp_size_or_ranks "
                        f"for {group_name}."
                    )
                seen_in_grp.add(r)
                flat.append(r)

        if len(set(flat)) != len(flat):
            raise RuntimeError(
                f"tp_size_or_ranks for {group_name} contains duplicate ranks across groups."
            )
        if set(flat) != set(range(world)):
            raise RuntimeError(
                f"tp_size_or_ranks for {group_name} must cover all ranks in the "
                f"default process group. Got {len(set(flat))}/{world} ranks. "
                "If you are using DP/PP/ExternalDP, provide a list-of-lists that "
                "partitions the full world."
            )
        return group_ranks

    if isinstance(spec, int):
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before parsing tp_size_or_ranks."
            )
        tp_size = spec
        if tp_size <= 0:
            raise RuntimeError(
                f"Invalid tp_size_or_ranks={spec!r} for {group_name}: "
                "tp_size must be a positive integer."
            )
        world = dist.get_world_size()
        if world % tp_size != 0:
            raise RuntimeError(
                f"Invalid tp_size_or_ranks={tp_size} for {group_name}: "
                f"world_size({world}) must be divisible by tp_size({tp_size})."
            )
        return torch.arange(world).reshape(-1, tp_size).tolist()

    logger.warning(f"Unsupported tp_size_or_ranks type: {type(spec)} for {group_name}")
    return None


def get_cross_group_from_list(idx: int) -> GroupCoordinator:
    return _CROSS_COMM_LIST[idx]


def get_local_group_from_list(idx: int) -> GroupCoordinator:
    return _LOCAL_COMM_LIST[idx]


def get_round_cross_group_from_list(idx: int) -> GroupCoordinator:
    return _CROSS_ROUND_COMM_LIST[idx]
