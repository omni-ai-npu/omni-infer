# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Prefill-side OmniCache implementation.

Split across three modules:
- prefill_omni_cache.py      core class + init + APC helpers + D2H sync + MoME snapshot
- device_cache_init.py       device cache initialization
- mome_state_utils.py        MoME state save / restore helpers
"""

import copy
import os
import threading
from collections import defaultdict
from typing import List, Optional, Tuple, Union

import torch

from vllm.logger import init_logger
from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import KVCacheConfig, UniformTypeKVCacheSpecs

from omni_cache.cache.core.base import BaseOmniCache
from omni_cache.cache.utils.debug import should_log_rank, summarize_array
from omni_npu.worker.npu_model_runner import NPUModelRunner
from omni_cache.cache.transfer_engine import TransferManager, TransferBuffers

from .device_cache_init import _compute_group_max_blocks, PrefillDeviceCacheMixin
from .prefix_copy_ops import (
    PrefixCopyMeta,
    compute_prefix_segments,
    get_current_rank_host_data as _get_current_rank_host_data,
)
from .mome_state_utils import (
    _copy_mome_slots_to_host,
    _copy_mome_single_state_to_host,
    _copy_mome_states_to_host_unified,
    _restore_mome_slots_from_host,
    _restore_mome_single_state_from_host,
    _restore_mome_states_from_host_unified,
    request_slots_empty,
    request_slots_to_int64,
)

logger = init_logger("vllm.v1.omni")


def _detect_enable_dsa_from_config(vllm_config: VllmConfig) -> bool:
    try:
        additional_config = vllm_config.additional_config or {}
        return additional_config.get("enable_dsa", False)
    except Exception:
        return False


class PrefillOmniCache(PrefillDeviceCacheMixin, BaseOmniCache):
    """OmniCache for the prefill / producer side."""

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner,
        max_num_batched_tokens: int,
        max_num_seqs: int,
        max_model_len: int,
        vllm_config: VllmConfig = None,
    ):
        super().__init__(kv_cache_config, runner, vllm_config)
        self.max_num_seqs = max_num_seqs  # Save max_num_seqs

        # Initialize TransferManager for H2D/D2H operations
        self.transfer_manager = TransferManager(self, max_num_batched_tokens, max_num_seqs)
        self.transfer_manager.initialize_prefill(kv_cache_config, max_num_batched_tokens, max_num_seqs, max_model_len)

        self._init_dp_sharding()
        self._init_prefix_buffer(max_num_seqs, max_model_len)

        from omni_cache.cache.core.constants import NZ_DIM

        self._nz_size = NZ_DIM

        self.stage_record = 0
        self._host_pool_write_lock = threading.Lock()

    def _init_dp_sharding(self) -> None:
        """Initialize DP sharding configuration."""
        if self.dp_world_size_local > 1:
            blocks_per_rank = self.host_block_offset[1] - self.host_block_offset[0]
            total_blocks = self.dp_world_size_local * blocks_per_rank
        else:
            blocks_per_rank = self.host_cache.kvi_tensors[0].shape[1]
            total_blocks = blocks_per_rank

        distributed_kv_caches = []
        for raw_cache in self.host_cache.kvi_tensors:
            num_layers = raw_cache.shape[0]
            inner_dims = raw_cache.shape[2:]
            target_shape = (self.dp_world_size_local, blocks_per_rank, *inner_dims)

            if self.dp_world_size_local > 1:
                layers = [raw_cache[i, :total_blocks].view(target_shape) for i in range(num_layers)]
            else:
                layers = [raw_cache[i, :total_blocks].unsqueeze(0) for i in range(num_layers)]
            distributed_kv_caches.append(tuple(layers))

        self.host_cache.kvi_tensors = distributed_kv_caches

    def _shard_kv_cache_by_dp_rank(self) -> None:
        """Legacy method name for backward compatibility. Calls _init_dp_sharding."""
        self._init_dp_sharding()

    # ── APC / prefix helpers ──────────────────────────────────────────────

    def get_prefill_prefix_copy_meta(
        self,
        vllm_config,
        kv_lens: "np.ndarray",
        query_lens_list: List[int],
        block_tables: "np.ndarray",
    ) -> Optional["PrefixCopyMeta"]:
        result = compute_prefix_segments(
            kv_lens, query_lens_list, block_tables, self.block_size, self.arange_cpu, self.device
        )
        if result is None or result[0] == [[]]:
            return None
        all_segs, q_lens, q_slots = result
        return PrefixCopyMeta(consecutive_blocks=all_segs, query_lens=q_lens, query_slots=q_slots)

    def compute_prefix_meta_from_metadata(
        self,
        vllm_config: "VllmConfig",
        metadata,
        common_attn_metadata,
    ) -> Optional["PrefixCopyMeta"]:
        """Compute prefix_meta from attention metadata for APC H2D.

        This is a convenience wrapper that extracts the necessary fields
        from metadata objects and calls get_prefill_prefix_copy_meta.

        Args:
            vllm_config: vLLM configuration
            metadata: Attention metadata (NPUMLAMetadata, NPUMomeMetadata, etc.)
            common_attn_metadata: Common attention metadata with block tables

        Returns:
            PrefixCopyMeta or None if APC is not enabled
        """
        if (
            not vllm_config.cache_config.enable_prefix_caching
            and not vllm_config.scheduler_config.enable_chunked_prefill
        ):
            logger.debug("compute_prefix_meta: prefix_caching disabled")
            return None

        num_reqs = getattr(metadata, "num_reqs", 0)
        if num_reqs <= 0:
            logger.debug("compute_prefix_meta: num_reqs=%d", num_reqs)
            return None

        # Extract query lens and computed tokens from common metadata
        query_start_loc = getattr(common_attn_metadata, "query_start_loc_cpu", None)
        if query_start_loc is None:
            logger.debug("compute_prefix_meta: query_start_loc_cpu is None")
            return None

        query_seq_lens_cpu = query_start_loc[1:] - query_start_loc[:-1]

        # Get query_lens from prefill metadata
        prefill_meta = getattr(metadata, "prefill", None)
        if prefill_meta is not None:
            prefill_query_start_loc = getattr(prefill_meta, "query_start_loc", None)
            if prefill_query_start_loc is not None:
                query_lens = prefill_query_start_loc[1:] - prefill_query_start_loc[:-1]
                query_lens_list = query_lens.tolist()
            else:
                logger.debug("compute_prefix_meta: prefill.query_start_loc is None")
                return None
        else:
            logger.debug("compute_prefix_meta: metadata.prefill is None")
            return None

        num_computed_tokens_cpu = common_attn_metadata.seq_lens_cpu - query_seq_lens_cpu

        # Get block tables
        block_table_tensor = getattr(common_attn_metadata, "block_table_tensor", None)
        if block_table_tensor is None:
            logger.debug("compute_prefix_meta: block_table_tensor is None")
            return None
        block_tables = block_table_tensor.cpu().numpy()[:num_reqs]
        if should_log_rank(self):
            logger.warning(
                "[APCDBG/PREFIX_META] tp_rank=%s dp_rank=%s stage=%s "
                "group_idx=%s layer_name=%s num_reqs=%d %s %s %s %s",
                getattr(self, "tp_rank", None),
                getattr(self, "dp_local_rank", None),
                getattr(self, "stage_record", None),
                None,
                None,
                num_reqs,
                summarize_array("seq_lens_cpu", common_attn_metadata.seq_lens_cpu),
                summarize_array("query_lens_list", query_lens_list),
                summarize_array("kv_lens", num_computed_tokens_cpu[:num_reqs]),
                summarize_array("block_tables", block_tables),
            )

        return self.get_prefill_prefix_copy_meta(
            vllm_config,
            kv_lens=num_computed_tokens_cpu[:num_reqs],
            query_lens_list=query_lens_list,
            block_tables=block_tables,
        )

    def _init_prefix_buffer(self, max_num_seqs: int, max_model_len: int) -> None:
        """Initialize prefix buffer for chunked prefill and MoME APC buffers."""
        self.prefix_buffer_npu = torch.empty(
            max_num_seqs * max_model_len,
            1,
            self.head_size,
            dtype=self.dtype,
            device=self.device,
        )
        self.h2d_stream = torch.npu.Stream(device=self.device)
        self.h2d_event = torch.npu.Event(blocking=False, enable_timing=False)

    # ── Token indices ─────────────────────────────────────────────────────

    def init_batch_token_indices_hybrid(self, orig_slot_mapping: torch.Tensor):
        group_idx = self.count_called
        if group_idx >= self.num_attn_group:
            current_block_size = self.kv_cache_config.kv_cache_groups[1].kv_cache_spec.block_size
        else:
            current_block_size = self.kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec.block_size
        current_node_block_size = divide_or_raise(current_block_size, self.tp_nnodes)
        current_rank_block_size = divide_or_raise(current_block_size, self.tp_world_size)
        blocks, slots = (orig_slot_mapping // current_block_size, orig_slot_mapping % current_block_size)
        ranks, rank_slots = (slots // current_rank_block_size, slots % current_node_block_size)
        self.batch_token_indices[group_idx] = torch.nonzero(
            (ranks == self.tp_rank) & (orig_slot_mapping > 0), as_tuple=True
        )[0]

        token_idx = torch.nonzero((ranks == self.tp_rank) & (orig_slot_mapping > 0), as_tuple=True)[0]
        slot_mapping = blocks[token_idx] * current_node_block_size + rank_slots[token_idx]
        num_tokens = token_idx.shape[0]
        self.batch_slots_cpu[group_idx][:num_tokens].copy_(slot_mapping, non_blocking=True)
        self.count_called = (self.count_called + 1) % self.num_called_builder

    def init_batch_token_indices(self, orig_slot_mapping: torch.Tensor):
        blocks, slots = (orig_slot_mapping // self.block_size, orig_slot_mapping % self.block_size)
        ranks, rank_slots = (slots // self.rank_block_size, slots % self.node_block_size)
        self.batch_token_indices = torch.nonzero((ranks == self.tp_rank) & (orig_slot_mapping > 0), as_tuple=True)[0]

        token_idx = torch.nonzero((ranks == self.tp_rank) & (orig_slot_mapping > 0), as_tuple=True)[0]
        slot_mapping = blocks[token_idx] * self.node_block_size + rank_slots[token_idx]
        num_tokens = token_idx.shape[0]
        self.batch_slots_cpu[:num_tokens].copy_(slot_mapping, non_blocking=True)

    # ── Volatile metadata ─────────────────────────────────────────────────

    def get_volatile_metadata(
        self,
        query_lens_list,
        seq_lens_list,
        graph_pad_size,
        pad_slot_id,
        orig_slot_mapping,
    ):
        """Wrapper for get_volatile_metadata function."""
        return get_volatile_metadata(
            self,
            query_lens_list,
            seq_lens_list,
            graph_pad_size,
            pad_slot_id,
            orig_slot_mapping,
        )

    # ── D2H / H2D synchronization ─────────────────────────────────────────

    def get_current_rank_host_data(self, layer_idx, prefix_meta):
        return _get_current_rank_host_data(self, layer_idx, prefix_meta)

    def synchronize_h2d(self, prefix_meta: "PrefixCopyMeta", attn_names: list[str]) -> None:
        """Synchronize H2D transfer for prefill."""
        return self.transfer_manager.sync_h2d_prefill(prefix_meta, attn_names)

    def synchronize_d2h(
        self,
        attn_names: list[str],
        attn_metadatas: list[str],
        kv_event: torch.npu.Event,
    ) -> None:
        """Unified D2H synchronization supporting single and multiple layers."""
        if self.is_hybrid_attn and len(attn_names) > 1:
            return self.transfer_manager.sync_d2h_hybrid(attn_names, attn_metadatas, kv_event)
        return self.transfer_manager.sync_d2h_prefill(attn_names, attn_metadatas, kv_event)

    def synchronize_d2h_hybrid(
        self, layer_name_list: List[str], attn_metadata_list: List[str], kv_event: torch.npu.Event
    ) -> None:
        """Synchronize D2H transfer for hybrid attention mode."""
        return self.transfer_manager.sync_d2h_hybrid(layer_name_list, attn_metadata_list, kv_event)

    def snapshot_mome_states(self) -> None:
        """Snapshot per-layer MoMe conv_states into the host pool.

        Called once per prefill forward pass (fired from the
        post_model_forward_decorator on PanguV2MoEForCausalLM.forward).
        """
        try:
            from vllm.forward_context import get_forward_context
            from vllm.v1.kv_cache_interface import MambaSpec
        except ImportError:
            return

        model = getattr(self.runner, "model", None)
        if model is None or not hasattr(model, "model"):
            return

        try:
            ctx = get_forward_context()
            attn_metadata = getattr(ctx, "attn_metadata", None)
            virtual_engine = getattr(ctx, "virtual_engine", 0)
        except Exception:
            return
        if not isinstance(attn_metadata, dict):
            return

        mome_group_indices: List[int] = [
            i for i, g in enumerate(self.kv_cache_config.kv_cache_groups) if isinstance(g.kv_cache_spec, MambaSpec)
        ]
        if not mome_group_indices:
            return

        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None:
            layers = getattr(model, "layers", None)
        if layers is None:
            return

        host_kvi = getattr(self.host_cache, "kvi_tensors", None)
        if host_kvi is None:
            return

        for layer_idx, decoder_layer in enumerate(layers):
            self_attn = getattr(decoder_layer, "self_attn", None)
            if self_attn is None:
                continue
            mome_attn = getattr(self_attn, "conv", None) or getattr(self_attn, "mome_attn", None)
            if mome_attn is None:
                continue
            mome_kv = getattr(mome_attn, "kv_cache", None)
            if not mome_kv or virtual_engine >= len(mome_kv):
                continue
            state_tuple = mome_kv[virtual_engine]
            if not isinstance(state_tuple, tuple):
                continue

            attn_prefix = getattr(self_attn, "prefix", None)
            if attn_prefix is None:
                continue
            mome_meta = attn_metadata.get(f"{attn_prefix}.conv") or attn_metadata.get(f"{attn_prefix}.mome")
            if mome_meta is None:
                continue
            num_prefills = getattr(mome_meta, "num_prefills", 0)
            if num_prefills == 0:
                continue
            cache_indices = getattr(mome_meta, "cache_indices", None)
            if cache_indices is None:
                continue

            try:
                if cache_indices.dim() > 1:
                    cache_indices = cache_indices[:, 0]
                num_decodes = getattr(mome_meta, "num_decodes", 0)
                prefill_slots = cache_indices[num_decodes:]
            except Exception:
                continue
            if prefill_slots.numel() == 0:
                continue

            try:
                host_layer = host_kvi[0][layer_idx]
            except (IndexError, TypeError):
                continue
            try:
                _copy_mome_states_to_host_unified(
                    state_tuple,
                    host_layer,
                    prefill_slots,
                    dp_local_rank=self.dp_local_rank,
                )
            except Exception as e:
                logger.debug("snapshot_mome_states: layer=%d err=%s", layer_idx, e)


# ── Module-level helpers retained for external callers ────────────────────

__all__ = [
    "PrefillOmniCache",
    "_compute_group_max_blocks",
    "_copy_mome_slots_to_host",
    "_copy_mome_single_state_to_host",
    "_copy_mome_states_to_host_unified",
    "_restore_mome_slots_from_host",
    "_restore_mome_single_state_from_host",
    "_restore_mome_states_from_host_unified",
    "request_slots_empty",
    "request_slots_to_int64",
]
