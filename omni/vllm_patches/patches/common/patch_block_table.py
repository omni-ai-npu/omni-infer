# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Replace Triton _compute_slot_mapping_kernel on NPU.

When Triton is unavailable, vLLM's @triton.jit decorator leaves a plain Python
function that cannot be invoked as kernel[(grid,)]. This patch provides a
torch/NPU fallback with the same semantics as block_table.py.

The fallback needs a per-token request id. Deriving it here is expensive on
NPU: searchsorted has no device kernel and runs on AI_CPU, while
repeat_interleave has to materialise one element per token and costs ~3.5 ms
per prefill step on P (vs ~0.15 ms for searchsorted). The model runner already
builds exactly this array on the host -- ``np.repeat(arange(num_reqs),
num_scheduled_tokens)`` in ``_prepare_inputs`` -- and uploads it to
``runner.req_indices``.

The upstream call site does not forward it, and it runs inside
``_prepare_inputs`` where a subclass cannot reach, so the block table borrows
the buffer from the owning runner instead. The source is attached to the
runner's own ``MultiGroupBlockTable`` rather than kept process-wide: with two
runners in one process a global would let the second one's buffer be read
through the first one's block table, and an equal token count would slip past
the freshness check and silently produce a wrong slot_mapping. When no source
is attached (unit tests, dummy runs) the kernel falls back to searchsorted.
"""

from collections.abc import Callable

import torch

import vllm.v1.worker.block_table as block_table_module
from vllm.logger import logger
from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

_REQ_INDICES_SOURCE_ATTR = "_omni_req_indices_source"


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
    req_indices: torch.Tensor | None = None,
    shared: dict | None = None,
) -> None:
    del block_table_stride, BLOCK_SIZE  # kept for signature parity with Triton kernel
    num_reqs = query_start_loc.shape[0] - 1

    if num_tokens > 0 and num_reqs > 0:
        if req_indices is not None:
            # Prebuilt on the host by the model runner; no device work.
            # Callers hand over an exact-length view, so only re-slice on the
            # (unexpected) mismatch rather than on every group.
            if req_indices.shape[0] != num_tokens:
                req_indices = req_indices[:num_tokens]
        else:
            # In serving this means the req_indices binding is broken.
            logger.info_once(
                "slot-mapping fallback: deriving per-token request ids "
                "via searchsorted (AI_CPU)."
            )
            token_indices = torch.arange(
                num_tokens,
                device=query_start_loc.device,
                dtype=query_start_loc.dtype,
            )
            req_indices = torch.searchsorted(
                query_start_loc[1:],
                token_indices,
                right=True,
            )
        req_positions = positions[:num_tokens]
        flat_block_table = block_table.reshape(-1)
        blocks_per_req = block_table.shape[1]

        # Group-invariant intermediates, reused across KV groups via `shared`.
        cache_key = (block_size, blocks_per_req)
        cached = shared.get(cache_key) if shared is not None else None

        if TOTAL_CP_WORLD_SIZE == 1:
            if cached is None:
                block_indices = req_positions // block_size
                flat_block_indices = req_indices * blocks_per_req + block_indices
                block_offsets = req_positions % block_size
                if shared is not None:
                    shared[cache_key] = (flat_block_indices, block_offsets)
            else:
                flat_block_indices, block_offsets = cached
            block_numbers = flat_block_table[flat_block_indices].to(torch.int64)
            slot_mapping[:num_tokens] = (
                block_numbers * block_size + block_offsets
            )
        else:
            if cached is None:
                virtual_block_size = block_size * TOTAL_CP_WORLD_SIZE
                block_indices = req_positions // virtual_block_size
                flat_block_indices = req_indices * blocks_per_req + block_indices
                virtual_block_offsets = (
                    req_positions - block_indices * virtual_block_size
                )
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
                if shared is not None:
                    shared[cache_key] = (
                        flat_block_indices, is_local, local_block_offsets
                    )
            else:
                flat_block_indices, is_local, local_block_offsets = cached
            block_numbers = flat_block_table[flat_block_indices].to(torch.int64)
            slot_ids = block_numbers * block_size + local_block_offsets
            slot_mapping[:num_tokens] = torch.where(is_local, slot_ids, PAD_ID)

    if num_tokens < max_num_tokens:
        slot_mapping[num_tokens:max_num_tokens] = PAD_ID


_compute_slot_mapping_kernel = _FuncWrapper(_compute_slot_mapping_kernel_impl)


def compute_slot_mapping(
    self,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    req_indices: torch.Tensor | None = None,
    shared: dict | None = None,
) -> None:
    num_tokens = positions.shape[0]
    total_cp_world_size = self.pcp_world_size * self.dcp_world_size
    total_cp_rank = self.pcp_rank * self.dcp_world_size + self.dcp_rank
    block_table_module._compute_slot_mapping_kernel[(num_reqs + 1,)](
        num_tokens,
        self.max_num_batched_tokens,
        query_start_loc,
        positions,
        self.block_table.gpu,
        self.block_table.gpu.stride(0),
        self.block_size,
        self.slot_mapping.gpu,
        TOTAL_CP_WORLD_SIZE=total_cp_world_size,
        TOTAL_CP_RANK=total_cp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=self.cp_kv_cache_interleave_size,
        PAD_ID=block_table_module.PAD_SLOT_ID,
        BLOCK_SIZE=1024,
        req_indices=req_indices,
        shared=shared,
    )


def compute_slot_mapping_multi(
    self,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    req_indices: torch.Tensor | None = None,
) -> None:
    if req_indices is None:
        # Resolve once for all groups: every KV cache group maps the same
        # tokens, so looking up and slicing per group only repeats work.
        source = getattr(self, _REQ_INDICES_SOURCE_ATTR, None)
        if source is not None:
            req_indices = source(positions.shape[0])
    # Per-call cache: only valid for this step's tokens.
    shared: dict = {}
    for block_table in self.block_tables:
        block_table.compute_slot_mapping(
            num_reqs, query_start_loc, positions, req_indices, shared
        )


@register_patch("NPUBlockTablePatch", block_table_module)
class NPUBlockTablePatch(VLLMPatch):
    """Use torch slot-mapping kernel when Triton is unavailable on NPU."""

    _attr_names_to_apply = ["_compute_slot_mapping_kernel"]
    _compute_slot_mapping_kernel = _compute_slot_mapping_kernel


@register_patch("NPUBlockTableComputeSlotMappingPatch", BlockTable)
class NPUBlockTableComputeSlotMappingPatch(VLLMPatch):
    """Accept a prebuilt per-token request id and forward it to the kernel."""

    _attr_names_to_apply = ["compute_slot_mapping"]
    compute_slot_mapping = compute_slot_mapping


@register_patch("NPUMultiGroupBlockTableSlotMappingPatch", MultiGroupBlockTable)
class NPUMultiGroupBlockTableSlotMappingPatch(VLLMPatch):
    """Pass the prebuilt request ids through to every KV cache group."""

    _attr_names_to_apply = [
        "compute_slot_mapping",
        "_omni_bind_req_indices_source",
    ]
    compute_slot_mapping = compute_slot_mapping_multi

    def _omni_bind_req_indices_source(
        self,
        source: Callable[[int], torch.Tensor | None] | None,
    ) -> None:
        """Attach a callable returning request ids for the current step."""
        setattr(self, _REQ_INDICES_SOURCE_ATTR, source)
