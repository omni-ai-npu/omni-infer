# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Replace Triton _compute_slot_mapping_kernel on NPU.

When Triton is unavailable, vLLM's @triton.jit decorator leaves a plain Python
function that cannot be invoked as kernel[(grid,)]. This patch provides a
torch/NPU fallback with the same semantics as block_table.py.
"""

from collections.abc import Callable

import torch

import vllm.v1.worker.block_table as block_table_module

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


class _FuncWrapper:
    def __init__(self, func: Callable) -> None:
        self.func = func

    def __getitem__(self, *_args, **_kwargs) -> Callable:
        return self.func


def _compute_slot_mapping_kernel_impl(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_table_stride: int,
    block_size: int,
    slot_mapping: torch.Tensor,
    TOTAL_CP_WORLD_SIZE: int,
    TOTAL_CP_RANK: int,
    CP_KV_CACHE_INTERLEAVE_SIZE: int,
    PAD_ID: int,
    BLOCK_SIZE: int,
) -> None:
    del block_table_stride, BLOCK_SIZE  # kept for signature parity with Triton kernel
    num_reqs = query_start_loc.shape[0] - 1

    if TOTAL_CP_WORLD_SIZE == 1:
        for req_idx in range(num_reqs):
            start_idx = int(query_start_loc[req_idx].item())
            end_idx = int(query_start_loc[req_idx + 1].item())
            if start_idx >= end_idx:
                continue
            req_positions = positions[start_idx:end_idx]
            block_indices = req_positions // block_size
            block_numbers = block_table[req_idx, block_indices].to(torch.int64)
            block_offsets = req_positions % block_size
            slot_mapping[start_idx:end_idx] = (
                block_numbers * block_size + block_offsets
            )
    else:
        virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
        for req_idx in range(num_reqs):
            start_idx = int(query_start_loc[req_idx].item())
            end_idx = int(query_start_loc[req_idx + 1].item())
            if start_idx >= end_idx:
                continue
            req_positions = positions[start_idx:end_idx]
            block_indices = req_positions // virtual_block_size
            block_numbers = block_table[req_idx, block_indices].to(torch.int64)
            virtual_block_offsets = req_positions - block_indices * virtual_block_size
            is_local = (
                (virtual_block_offsets // CP_KV_CACHE_INTERLEAVE_SIZE)
                % TOTAL_CP_WORLD_SIZE
            ) == TOTAL_CP_RANK
            local_block_offsets = (
                virtual_block_offsets
                // (TOTAL_CP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
            ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
                virtual_block_offsets % CP_KV_CACHE_INTERLEAVE_SIZE
            )
            slot_ids = block_numbers * block_size + local_block_offsets
            slot_ids = torch.where(
                is_local,
                slot_ids,
                torch.full_like(slot_ids, PAD_ID),
            )
            slot_mapping[start_idx:end_idx] = slot_ids

    if num_tokens < max_num_tokens:
        slot_mapping[num_tokens:max_num_tokens] = PAD_ID


_compute_slot_mapping_kernel = _FuncWrapper(_compute_slot_mapping_kernel_impl)


@register_patch("NPUBlockTablePatch", block_table_module)
class NPUBlockTablePatch(VLLMPatch):
    """Use torch slot-mapping kernel when Triton is unavailable on NPU."""

    _attr_names_to_apply = ["_compute_slot_mapping_kernel"]
    _compute_slot_mapping_kernel = _compute_slot_mapping_kernel
