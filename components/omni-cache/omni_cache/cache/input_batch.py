# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""OmniCache-aware decode InputBatch skeleton.

This module intentionally preserves vLLM's ``InputBatch`` behavior today.  It
adds only a typed policy surface where future OmniCache block-table and
slot-mapping behavior can move back into vLLM's normal metadata path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import numpy as np
import torch

from vllm.v1.kv_cache_interface import (
    DSAAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MomeSpec,
    ShareKVSlidingWindowSpec,
)
from vllm.v1.worker.block_table import BlockTable, MultiGroupBlockTable
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.utils import CpuGpuBuffer
from vllm.logger import init_logger

if TYPE_CHECKING:
    from omni_cache.cache.decode.decode_omni_cache import DecodeOmniCache

logger = init_logger("vllm.v1.omni")


@dataclass(slots=True)
class OmniCacheSpecPolicy:
    """No-op policy for one KV cache group."""

    group_idx: int
    spec: KVCacheSpec

    def initialize_group_block_table(self, input_batch: "OmniCacheInputBatch") -> None:
        return None

    def uses_omni_block_table(self) -> bool:
        return False

    def wrap_block_table(
        self,
        input_batch: "OmniCacheInputBatch",
        block_table: BlockTable,
    ) -> BlockTable:
        return block_table

    def fill_omni_block_row(
        self,
        input_batch: "OmniCacheInputBatch",
        block_table: "OmniCacheBlockTable",
        row_idx: int,
    ) -> None:
        return None


class DSAPolicy(OmniCacheSpecPolicy):
    """DSA keeps normal BlockTable semantics and owns GatherSelection state."""

    def wrap_block_table(
        self,
        input_batch: "OmniCacheInputBatch",
        block_table: BlockTable,
    ) -> BlockTable:
        omni_cache = input_batch.omni_cache
        if omni_cache is None or not omni_cache.enable_gs:
            return block_table
        return DSAGatherSelectionBlockTable.from_block_table(
            block_table, input_batch, self
        )


class MomePolicy(OmniCacheSpecPolicy):
    """Skeleton for one-HBM-lane-per-request MoME state metadata."""

    def uses_omni_block_table(self) -> bool:
        return True

    def wrap_block_table(
        self,
        input_batch: "OmniCacheInputBatch",
        block_table: BlockTable,
    ) -> BlockTable:
        return OmniCacheBlockTable.from_block_table(block_table, input_batch, self)

    def fill_omni_block_row(
        self,
        input_batch: "OmniCacheInputBatch",
        block_table: "OmniCacheBlockTable",
        row_idx: int,
    ) -> None:
        row = block_table.get_numpy_array()[row_idx]
        row.fill(0)
        row[0] = input_batch.get_hbm_lane_for_row(row_idx) + 1


@dataclass(slots=True)
class ShareKVSlidingWindowPolicy(OmniCacheSpecPolicy):
    """Static lane-local ring-buffer metadata for MLA sliding-window groups."""

    req_offset: int = field(init=False)

    def __post_init__(self) -> None:
        self.req_offset = 0

    def initialize_group_block_table(self, input_batch: "OmniCacheInputBatch") -> None:
        from omni_cache.cache.decode.hbm_buffer_utils import _req_offset_for_spec

        self.req_offset = int(
            _req_offset_for_spec(self.spec, max_model_len=input_batch.max_model_len)
        )
        if self.req_offset <= 0:
            raise RuntimeError(
                f"Invalid SWA req_offset {self.req_offset} "
                f"for {type(self.spec).__name__}"
            )

    def uses_omni_block_table(self) -> bool:
        return True

    def wrap_block_table(
        self,
        input_batch: "OmniCacheInputBatch",
        block_table: BlockTable,
    ) -> BlockTable:
        return OmniCacheBlockTable.from_block_table(block_table, input_batch, self)

    def fill_omni_block_row(
        self,
        input_batch: "OmniCacheInputBatch",
        block_table: "OmniCacheBlockTable",
        row_idx: int,
    ) -> None:
        row = block_table.get_numpy_array()[row_idx]
        lane = input_batch.get_hbm_lane_for_row(row_idx)
        base = lane * self.req_offset
        sliding_window = int(self.spec.sliding_window)
        block_size = int(self.spec.block_size)
        num_prompt_tokens = int(input_batch.num_prompt_tokens[row_idx])
        num_skipped_tokens = max(0, num_prompt_tokens - sliding_window + 1)
        num_skipped_blocks = num_skipped_tokens // block_size
        if num_skipped_blocks >= row.shape[0]:
            raise RuntimeError(
                "SWA Omni block table row is too narrow for skipped blocks: "
                f"row_width={row.shape[0]}, skipped_blocks={num_skipped_blocks}, "
                f"prompt_tokens={num_prompt_tokens}, "
                f"sliding_window={sliding_window}, block_size={block_size}"
            )
        start_col = num_skipped_blocks

        row.fill(0)
        offsets = np.arange(row.shape[0] - start_col, dtype=row.dtype)
        row[start_col:] = base + (offsets % self.req_offset) + 1


def _make_policy(group_idx: int, spec: KVCacheSpec) -> OmniCacheSpecPolicy:
    if isinstance(spec, DSAAttentionSpec):
        return DSAPolicy(group_idx, spec)
    if isinstance(spec, MomeSpec):
        return MomePolicy(group_idx, spec)
    if isinstance(spec, ShareKVSlidingWindowSpec):
        return ShareKVSlidingWindowPolicy(group_idx, spec)
    return OmniCacheSpecPolicy(group_idx, spec)


class OmniCacheBlockTable(BlockTable):
    """BlockTable whose public table stores OmniCache HBM block ids.

    The inherited ``block_table`` and ``slot_mapping`` stay attention-facing.
    Scheduler-provided vLLM block ids are stored in ``vllm_block_table``.
    """

    @classmethod
    def from_block_table(
        cls,
        block_table: BlockTable,
        input_batch: "OmniCacheInputBatch",
        policy: OmniCacheSpecPolicy,
    ) -> "OmniCacheBlockTable":
        table = cls.__new__(cls)
        table.__dict__.update(block_table.__dict__)
        table.input_batch = input_batch
        table.policy = policy
        table.vllm_block_table = table._make_buffer(
            table.max_num_reqs,
            table.max_num_blocks_per_req,
            dtype=torch.int32,
        )
        return table

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        self.num_blocks_per_row[row_idx] = 0
        self._append_vllm_row(block_ids, row_idx)
        self.policy.fill_omni_block_row(self.input_batch, self, row_idx)

    def append_row(self, block_ids: list[int], row_idx: int) -> None:
        self._append_vllm_row(block_ids, row_idx)

    def move_row(self, src: int, tgt: int) -> None:
        num_blocks = self.num_blocks_per_row[src]
        self.block_table.np[tgt] = self.block_table.np[src]
        self.vllm_block_table.np[tgt, :num_blocks] = (
            self.vllm_block_table.np[src, :num_blocks]
        )
        self.num_blocks_per_row[tgt] = num_blocks

    def swap_row(self, src: int, tgt: int) -> None:
        src_tgt, tgt_src = [src, tgt], [tgt, src]
        self.num_blocks_per_row[src_tgt] = self.num_blocks_per_row[tgt_src]
        self.block_table.np[src_tgt] = self.block_table.np[tgt_src]
        self.vllm_block_table.np[src_tgt] = self.vllm_block_table.np[tgt_src]

    def clear(self) -> None:
        super().clear()
        self.vllm_block_table.gpu.fill_(0)
        self.vllm_block_table.cpu.fill_(0)

    def _append_vllm_row(self, block_ids: list[int], row_idx: int) -> None:
        if not block_ids:
            return

        normalized_block_ids: list[int] | np.ndarray = block_ids
        if self.use_hybrid_blocks:
            normalized_block_ids = self.map_to_kernel_blocks(
                np.array(block_ids),
                self.blocks_per_kv_block,
                self._kernel_block_arange,
            )

        num_blocks = len(normalized_block_ids)
        start = self.num_blocks_per_row[row_idx]
        self.num_blocks_per_row[row_idx] += num_blocks
        self.vllm_block_table.np[row_idx, start : start + num_blocks] = (
            normalized_block_ids
        )


class DSAGatherSelectionBlockTable(BlockTable):
    """Normal DSA BlockTable plus row-owned GatherSelection buffers."""

    @classmethod
    def from_block_table(
        cls,
        block_table: BlockTable,
        input_batch: "OmniCacheInputBatch",
        policy: DSAPolicy,
    ) -> "DSAGatherSelectionBlockTable":
        table = cls.__new__(cls)
        table.__dict__.update(block_table.__dict__)
        table.input_batch = input_batch
        table.policy = policy
        table.gs_status = None
        table.gs_table = None
        table._initialize_gs_buffers()
        input_batch._set_gs_block_table(table)
        return table

    def _initialize_gs_buffers(self) -> None:
        omni_cache: "DecodeOmniCache" = self.input_batch.omni_cache
        if omni_cache is None:
            raise RuntimeError(
                "Cannot initialize GatherSelection buffers without omni_cache"
            )

        required_attrs = (
            "num_layers",
            "selection_state_size",
            "s_max_block_num",
        )
        missing_attrs = [
            attr for attr in required_attrs if not hasattr(omni_cache, attr)
        ]
        if missing_attrs:
            raise RuntimeError(
                "Cannot initialize GatherSelection buffers: missing "
                f"omni_cache attributes {missing_attrs}"
            )

        num_layers = omni_cache.num_layers
        state_size = omni_cache.selection_state_size
        table_width = omni_cache.s_max_block_num

        if num_layers <= 0 or state_size <= 0 or table_width <= 0:
            raise RuntimeError(
                "Invalid GatherSelection buffer shape: "
                f"num_layers={num_layers}, state_size={state_size}, "
                f"table_width={table_width}"
            )

        self.gs_status = torch.full(
            (
                num_layers,
                self.max_num_reqs,
                state_size,
            ),
            -1,
            dtype=torch.int32,
            device=self.input_batch.device,
        ).contiguous()
        self.gs_table = CpuGpuBuffer(
            self.max_num_reqs,
            table_width,
            dtype=torch.int32,
            device=self.input_batch.device,
            pin_memory=self.input_batch.pin_memory,
        )

        self.gs_table.np[:] = np.arange(
            self.max_num_reqs * table_width,
            dtype=np.int32,
        ).reshape(self.max_num_reqs, table_width)

    def has_gs_buffers(self) -> bool:
        return self.gs_status is not None and self.gs_table is not None

    def commit_gs_buffers(self, num_reqs: int) -> None:
        if not self.has_gs_buffers():
            raise RuntimeError(f"Gather selection buffers not initialized!")
        self.gs_table.copy_to_gpu(num_reqs)

    def get_gs_table_tensor(self, num_reqs: int) -> torch.Tensor:
        if not self.has_gs_buffers():
            raise RuntimeError("GatherSelection buffers are not initialized")
        return self.gs_table.gpu[:num_reqs]

    def get_gs_status_tensor(self, layer_idx: int, num_reqs: int) -> torch.Tensor:
        if not self.has_gs_buffers():
            raise RuntimeError("GatherSelection buffers are not initialized")
        return self.gs_status[layer_idx, :num_reqs]

    def add_row(self, block_ids: list[int], row_idx: int) -> None:
        super().add_row(block_ids, row_idx)
        self.gs_status[:, row_idx, :].fill_(-1)

    def move_row(self, src: int, tgt: int) -> None:
        super().move_row(src, tgt)
        self.gs_status[:, tgt, :].copy_(self.gs_status[:, src, :])
        self.gs_status[:, src, :].fill_(-1)
        tgt_table = self.gs_table.np[tgt].copy()
        self.gs_table.np[tgt] = self.gs_table.np[src]
        self.gs_table.np[src] = tgt_table

    def swap_row(self, src: int, tgt: int) -> None:
        super().swap_row(src, tgt)
        src_status = self.gs_status[:, src, :].clone()
        self.gs_status[:, src, :].copy_(self.gs_status[:, tgt, :])
        self.gs_status[:, tgt, :].copy_(src_status)
        src_table = self.gs_table.np[src].copy()
        self.gs_table.np[src] = self.gs_table.np[tgt]
        self.gs_table.np[tgt] = src_table

    def clear(self) -> None:
        super().clear()
        self.gs_status.fill_(-1)
        self.gs_table.np[:] = np.arange(
            self.max_num_reqs * self.gs_table.np.shape[1],
            dtype=np.int32,
        ).reshape(self.max_num_reqs, self.gs_table.np.shape[1])


class OmniCacheMultiGroupBlockTable(MultiGroupBlockTable):
    """Multi-group table with OmniCache-aware tables for selected groups."""

    def __init__(
        self,
        input_batch: "OmniCacheInputBatch",
        delegate: MultiGroupBlockTable,
        policies: list[OmniCacheSpecPolicy],
    ) -> None:
        self.__dict__.update(delegate.__dict__)
        self.input_batch = input_batch
        self.policies = policies
        self.block_tables = [
            policy.wrap_block_table(input_batch, block_table)
            for block_table, policy in zip(delegate.block_tables, policies)
        ]

    def commit_block_table(self, num_reqs):
        for block_table in self.block_tables:
            if isinstance(block_table, DSAGatherSelectionBlockTable):
                block_table.commit_gs_buffers(num_reqs)
        return super().commit_block_table(num_reqs)


class OmniCacheInputBatch(InputBatch):
    """InputBatch extension point for OmniCache decode metadata.

    The constructor mirrors vLLM's ``InputBatch`` and adds KV spec context.
    Runtime behavior is intentionally identical to vLLM in this first step.
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        pin_memory: bool,
        vocab_size: int,
        block_sizes: list[int],
        kernel_block_sizes: list[int],
        *,
        kv_cache_specs: list[KVCacheSpec],
        kv_cache_config: KVCacheConfig,
        omni_cache: "DecodeOmniCache | None" = None,
        logitsprocs: Any | None = None,
        logitsprocs_need_output_token_ids: bool = False,
        is_spec_decode: bool = False,
        is_pooling_model: bool = False,
        num_speculative_tokens: int = 0,
        cp_kv_cache_interleave_size: int = 1,
    ) -> None:
        self.kv_cache_specs = kv_cache_specs
        self.kv_cache_config = kv_cache_config
        self.kernel_block_sizes = kernel_block_sizes
        self.omni_cache = omni_cache
        self._gs_block_table: DSAGatherSelectionBlockTable | None = None
        self.spec_policies = [
            _make_policy(group_idx, spec)
            for group_idx, spec in enumerate(kv_cache_specs)
        ]
        self.has_hbm_lane_policy = any(
            isinstance(policy, (MomePolicy, ShareKVSlidingWindowPolicy))
            for policy in self.spec_policies
        )

        super().__init__(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            device=device,
            pin_memory=pin_memory,
            vocab_size=vocab_size,
            block_sizes=block_sizes,
            kernel_block_sizes=kernel_block_sizes,
            logitsprocs=logitsprocs,
            logitsprocs_need_output_token_ids=logitsprocs_need_output_token_ids,
            is_spec_decode=is_spec_decode,
            is_pooling_model=is_pooling_model,
            num_speculative_tokens=num_speculative_tokens,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        )

        self.block_table = OmniCacheMultiGroupBlockTable(
            self, self.block_table, self.spec_policies
        )

        for policy in self.spec_policies:
            policy.initialize_group_block_table(self)

        if (
            self.omni_cache is not None
            and self.omni_cache.enable_gs
            and self._gs_block_table is None
        ):
            raise RuntimeError(
                "GatherSelection is enabled, but no DSA block table was "
                "installed on OmniCacheInputBatch"
            )

        logger.warning(
            "[OMNI-INPUT-BATCH] constructed OmniCacheInputBatch "
            "max_num_reqs=%d max_model_len=%d groups=%s "
            "has_hbm_lane_policy=%s",
            max_num_reqs,
            max_model_len,
            [
                f"{policy.group_idx}:{type(policy).__name__}"
                for policy in self.spec_policies
            ],
            self.has_hbm_lane_policy,
        )

    def _set_gs_block_table(self, block_table: DSAGatherSelectionBlockTable) -> None:
        if self._gs_block_table is not None:
            raise RuntimeError(
                "Multiple DSA GatherSelection block tables are not supported"
            )
        self._gs_block_table = block_table

    def _require_gs_block_table(self) -> DSAGatherSelectionBlockTable:
        if self._gs_block_table is None:
            raise RuntimeError("GatherSelection buffers are not initialized")
        return self._gs_block_table

    def has_gs_buffers(self) -> bool:
        return (
            self._gs_block_table is not None
            and self._gs_block_table.has_gs_buffers()
        )

    def commit_gs_buffers(self, num_reqs: int) -> None:
        if self._gs_block_table is not None:
            self._gs_block_table.commit_gs_buffers(num_reqs)

    def get_gs_table_tensor(self, num_reqs: int) -> torch.Tensor:
        return self._require_gs_block_table().get_gs_table_tensor(num_reqs)

    def get_gs_status_tensor(self, layer_idx: int, num_reqs: int) -> torch.Tensor:
        return self._require_gs_block_table().get_gs_status_tensor(
            layer_idx, num_reqs
        )

    def _register_add_request(self, request: CachedRequestState) -> int:
        req_index = super()._register_add_request(request)
        if self.has_hbm_lane_policy:
            self.get_hbm_lane(request.req_id)
        return req_index

    def add_request(self, request: CachedRequestState) -> int:
        try:
            req_index = super().add_request(request)
        except Exception:
            if self.has_hbm_lane_policy:
                self.release_hbm_lane(request.req_id)
            raise
        return req_index

    def remove_request(self, req_id: str) -> int | None:
        req_index = super().remove_request(req_id)
        if self.has_hbm_lane_policy and req_index is not None:
            self.release_hbm_lane(req_id)
        return req_index

    def get_hbm_lane(self, req_id: str) -> int:
        if self.omni_cache is None or not hasattr(
            self.omni_cache, "get_hbm_lane_for_request"
        ):
            raise RuntimeError(
                f"No global HBM lane map available for request {req_id}"
            )
        lane = self.omni_cache.get_hbm_lane_for_request(req_id)
        if lane is None:
            raise RuntimeError(f"No HBM lane reserved for request {req_id}")
        return lane

    def get_hbm_lane_for_row(self, row_idx: int) -> int:
        if row_idx >= len(self._req_ids):
            raise RuntimeError(f"No request at row {row_idx}")
        req_id = self._req_ids[row_idx]
        if req_id is None:
            raise RuntimeError(f"No request at row {row_idx}")
        return self.get_hbm_lane(req_id)

    def release_hbm_lane(self, req_id: str) -> int | None:
        if self.omni_cache is None or not hasattr(
            self.omni_cache, "release_hbm_lane_for_request"
        ):
            return None
        return self.omni_cache.release_hbm_lane_for_request(req_id)

__all__ = [
    "DSAGatherSelectionBlockTable",
    "DSAPolicy",
    "MomePolicy",
    "OmniCacheBlockTable",
    "OmniCacheMultiGroupBlockTable",
    "OmniCacheInputBatch",
    "OmniCacheSpecPolicy",
    "ShareKVSlidingWindowPolicy",
]
