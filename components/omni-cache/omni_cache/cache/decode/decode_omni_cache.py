# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Decode cache implementation for OmniCache.

This module contains the DecodeOmniCache class implementation.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from vllm.logger import init_logger

from omni_cache.cache.core.base import BaseOmniCache

logger = init_logger("vllm.v1.omni")
from omni_cache.cache.core.constants import (
    SIZE_BYTES_PER_LAYER_DECODE,
    BLOCK_SIZE,
    HEAD_DIM,
    LOCAL_DP_SIZE,
    NZ_DIM,
    PRE_CALC_PREFILL_BLOCK_NUM,
    ENABLE_C8_INDEXER,
    ENABLE_HOST_MAPPING,
)
from omni_cache.cache.utils.ops import divide_or_raise
from omni_cache.cache.utils.support import _is_pangu_v2_model

try:
    from omni_npu.attention.backends.attention import NPUMetadata as load_npu_attention_metadata
    from omni_npu.attention.backends.stride_compress import StridedCompressAttentionMetadata
    from omni_npu.attention.backends.build_utils import (
        _layout_and_computed_lens_kernel,
        _scatter_copy_and_locate_kernel,
    )
except ImportError:
    pass

from omni_cache.cache.transfer_engine import (
    calc_cache_shape_for_decode,
    calculate_kv_xsfer_params,
    build_npu_block_layers,
    build_npu_block_layers_hybrid,
    build_npu_block_layers_hbm,
    prepare_h2d_copy_args as build_h2d_ops,
    prepare_h2d_copy_args_hybrid as build_h2d_ops_hybrid,
    prepare_h2d_copy_args_hbm_buffer as build_h2d_ops_hbm_buffer,
    TransferManager,
)
from omni_cache.attention.metadata import (
    fake_build_compress,
    post_process_fake_metadata,
    fake_build_attention,
    construct_fake_attn_metadata,
)
from omni_cache.cache.utils.ops import (
    _is_hybrid_attention_enabled,
    generate_full_block_slot,
    pad_inputs,
    pad_tensor,
)

# Import HBM buffer utilities
from .hbm_buffer_utils import construct_hbm_buffer

# Import host KV cache utilities
from .host_kv_cache_utils import parse_pa_kv_cache

# Import gather selection components
from omni_cache.gather_selection import (
    GatherSelectionUpdater,
    gather_selection,
    initialize_selection_buffers,
    get_block_table_np,
)

# Import types needed for type hints
try:
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.config import VllmConfig
    from omni_npu.worker.npu_model_runner import NPUModelRunner
    import torch
    from vllm.model_executor.models.utils import extract_layer_index
except ImportError:
    pass


class DecodeOmniCache(BaseOmniCache):
    """Decode cache implementation for OmniCache."""

    def __init__(self, kv_cache_config: KVCacheConfig, runner: NPUModelRunner, vllm_config: VllmConfig = None):
        super().__init__(kv_cache_config, runner, vllm_config)

        # Initialize TransferManager for H2D/D2H operations
        self.transfer_manager = TransferManager(self)
        self.transfer_manager.initialize_decode()

        if self.is_hybrid_attn:
            self.sorted_layer_names = [str(i) for i in range(self.num_layers)]
        else:
            layer_name_list = []
            for kv_cache_group in kv_cache_config.kv_cache_groups:
                layer_name_list.extend(kv_cache_group.layer_names)
            self.layer_indices = {layer_name: extract_layer_index(layer_name) for layer_name in layer_name_list}
            self.sorted_layer_names = list(
                sorted(
                    layer_name_list,
                    key=lambda k: self.layer_indices[k],
                )
            )

        # Store model config and DSA flag for later use
        self.vllm_config = vllm_config

        dp_rank = self.dp_local_rank

        blocks_per_rank = self.host_cache.num_blocks

        self.blk_table_buffers = {}
        self.slot_mapping_buffers = {}
        # only work for mtp-1
        self.record_attn_metadata_mtp = None

        # Defensive defaults — these attributes are re-assigned to
        # Manager().dict() in BaseOmniCache.update_kv_cache_spec when
        # ENABLE_HOST_MAPPING=1. Pre-initialize so forked subprocesses
        # (process_manager) don't AttributeError if they happen to run a
        # check before the manager swaps these in.
        self.record_batch_idx_to_req = None
        self.record_req_ids_swa_block = None
        self.req_id_to_idx = None
        self.hbm_buffer_pool = None
        self.hbm_buffer_block_table_pool = None

        self.block_table = torch.arange(blocks_per_rank, dtype=torch.long, device=self.device)
        if hasattr(torch, "npu") and torch.npu.is_available():
            self._copy_stream = getattr(self, "_copy_stream", torch.npu.Stream())
        else:
            self._copy_stream = getattr(self, "_copy_stream", None)

        self.block_status = {}
        # Initialize block status: use environment variable for test flexibility
        # Default to False for production, True only when testing
        initial_block_status = os.getenv("OMNI_CACHE_TEST_BLOCK_STATUS", "0") == "1"
        self.block_status[tuple(list(range(blocks_per_rank)))] = initial_block_status

        if self.enable_dsa:
            self.gather_selec_updater = GatherSelectionUpdater(self)

        if self.is_hybrid_attn:
            self.last_num_non_zeros = {}
            self.record_req_sched_times = {}

            self.MAX_LOGIC_BLOCKS = (self.vllm_config.model_config.max_model_len // 128) + 10
            self._slot_templates = {}
            self._max_template_len = 128

            # Pre-allocate buffers for vectorized fake metadata processing
            # These are reused on every call to avoid allocation overhead
            max_num_reqs = self.vllm_config.scheduler_config.max_num_seqs * 2  # for two swa groups

            # Map from (req_id, group_id) -> buffer index
            self._sk_to_state_idx = {}
            self._free_state_indices = set(range(max_num_reqs))  # Available buffer slots

            import numpy as np

            # Scalar buffers for each state (flat, contiguous)
            self._base_addrs_buffer = np.zeros(max_num_reqs, dtype=np.int32)
            self._win_sizes_buffer = np.zeros(max_num_reqs, dtype=np.int32)
            self._max_lbs_buffer = np.zeros(max_num_reqs, dtype=np.int64)
            self._num_assigneds_buffer = np.zeros(max_num_reqs, dtype=np.int32)
            self._is_switching_buffer = np.zeros(max_num_reqs, dtype=np.bool_)

            # Flat buffers for logic_to_phys and logic_valid for all states
            self._logic_to_phys_buffer = np.zeros(max_num_reqs * self.MAX_LOGIC_BLOCKS, dtype=np.int32)
            self._logic_valid_buffer = np.zeros(max_num_reqs * self.MAX_LOGIC_BLOCKS, dtype=np.bool_)

            # Temporary buffers for batch processing
            self._start_lb_idxs_buffer = np.zeros(max_num_reqs, dtype=np.int32)
            self._L_currs_buffer = np.zeros(max_num_reqs, dtype=np.int64)
            self._batch_state_indices_buffer = np.zeros(max_num_reqs, dtype=np.int32)

    def _post_process_fake_metadata(
        self,
        num_reqs,
        slot_mapping,
        block_table,
        common_attn_metadata,
        cur_metadata_grp_id,
        query_start_loc,
        seq_lens_cpu,
        draft_index,
    ):
        """Post-process fake metadata - delegates to attn_metadata module.

        Args:
            num_reqs: Number of requests in the batch
            slot_mapping: Slot mapping array to be filled
            block_table: Block table array to be filled
            common_attn_metadata: Common attention metadata
            cur_metadata_grp_id: Current metadata group ID
            query_start_loc: Query start locations per request
            seq_lens_cpu: Sequence lengths on CPU
            draft_index: Optional draft index for MTP

        Returns:
            Tuple of (block_table, slot_mapping) after processing
        """
        return post_process_fake_metadata(
            self,
            num_reqs,
            slot_mapping,
            block_table,
            common_attn_metadata,
            cur_metadata_grp_id,
            query_start_loc,
            seq_lens_cpu,
            draft_index,
        )

    def _compute_slot_mapping_vectorized(self, slot_mapping, slot_infos):
        """Compute slot mapping - delegates to attn_metadata module.

        VECTORIZED: Eliminates Python loop by collecting all data into arrays
        and using a single Numba JIT call. Uses pre-allocated buffers.

        Args:
            slot_mapping: Slot mapping array to fill
            slot_infos: List of _SlotInfo objects
        """
        from omni_cache.attention.metadata.sliding_window_attention import _compute_slot_mapping_vectorized

        return _compute_slot_mapping_vectorized(self, slot_mapping, slot_infos)

    def _fake_build_compress(self, builder, common_attn_metadata):
        """Build compressed attention metadata - delegates to metadata module.

        Args:
            builder: Attention metadata builder with buffers
            common_attn_metadata: Common attention metadata

        Returns:
            Attention metadata for compressed attention
        """
        return fake_build_compress(self, builder, common_attn_metadata)

    def get_fake_kv_cache(self, prefix):
        for grp_idx, _ in enumerate(self.kv_cache_config.kv_cache_groups):
            layer_name_list = self.kv_cache_config.kv_cache_groups[grp_idx].layer_names
            if prefix in layer_name_list:
                kv_cache = self.hbm_buffer_pool[grp_idx][prefix]
                return kv_cache

    def calc_cache_shape(self) -> Tuple[Tuple[int, ...], int]:
        """Calculate cache shape using utility function."""
        return calc_cache_shape_for_decode(
            num_layers=self.num_layers,
            block_size=self.block_size,
            head_size=self.head_size,
            dtype=self.dtype,
        )

    def calculate_kv_xsfer_params(self) -> Tuple[int, int]:
        """Calculate KV transfer parameters using utility function."""
        return calculate_kv_xsfer_params(
            shape=self.shape,
            num_blocks=self.num_blocks,
            dp_local_rank=self.dp_local_rank,
        )

    def initialize_device_cache(
        self, kv_cache_config: KVCacheConfig, runner: NPUModelRunner
    ) -> Optional[Dict[str, Tuple[torch.Tensor]]]:
        """Initialize device cache and selection buffers.

        Args:
            kv_cache_config: KV cache configuration.
            runner: Model runner instance.

        Returns:
            Dictionary of KV caches or None.
        """
        kv_caches = {}

        # Set bsz_seq for selection buffer initialization
        self.bsz_seq = runner.graph_block_tables.shape[0]

        # Initialize selection buffers for DSA GatherSelection.
        if self.enable_gs:
            initialize_selection_buffers(self)

        if not ENABLE_HOST_MAPPING:
            kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(kv_cache_config)
            kv_caches = runner._reshape_kv_cache_tensors(
                kv_cache_config, kv_cache_raw_tensors, runner._prepare_kernel_block_sizes(kv_cache_config)
            )

            # Per-row raw tensor pointers for opaque device-side H2D.
            # Each entry in kv_cache_config.kv_cache_tensors is shared across all
            # attention groups that alias that row (Pangu V2 hybrid: 16 rows
            # shared across 6 kv_cache_groups).
            self.raw_tensors_by_row = []
            for kv_tensor in kv_cache_config.kv_cache_tensors:
                layer_name = kv_tensor.shared_by[0]
                layer_entry = kv_caches[layer_name]
                if isinstance(layer_entry, tuple):
                    first_view = layer_entry[0]
                else:
                    first_view = layer_entry
                self.raw_tensors_by_row.append(first_view)

            # page_size_padded: per-block byte count in the opaque device layout.
            # Read from any non-Mamba group's spec (all attention groups share
            # the same padded page size).
            from vllm.v1.kv_cache_interface import MambaSpec

            self.page_size_padded = None
            for grp in kv_cache_config.kv_cache_groups:
                if not isinstance(grp.kv_cache_spec, MambaSpec):
                    self.page_size_padded = int(grp.kv_cache_spec.page_size_bytes)
                    break
            if self.page_size_padded is None:
                self.page_size_padded = int(kv_cache_config.kv_cache_groups[0].kv_cache_spec.page_size_bytes)
        else:
            try:
                if _is_pangu_v2_model(self.vllm_config):
                    # Identify the DSA group dynamically instead of hardcoding
                    # group index 3.
                    dsa_layer_names = set()
                    try:
                        from vllm.v1.kv_cache_interface import DSAAttentionSpec

                        for group_config in self.kv_cache_config.kv_cache_groups:
                            if isinstance(group_config.kv_cache_spec, DSAAttentionSpec):
                                dsa_layer_names.update(group_config.layer_names)
                                break
                    except ImportError:
                        pass

                    kv_caches = {}
                    for group_idx, group_config in enumerate(self.kv_cache_config.kv_cache_groups):
                        for _layer_idx, layer_name in enumerate(group_config.layer_names):
                            kv_caches[layer_name] = self.hbm_buffer_pool[group_idx][layer_name]
                    for layer_name, kv_cache_tmp in kv_caches.items():
                        if layer_name in dsa_layer_names:
                            host_cache_pa_tmp = self.host_cache_pa[layer_name]
                            kv_caches[layer_name] = (host_cache_pa_tmp[0], kv_cache_tmp[0])
                else:
                    kv_caches = self.host_cache_pa
            except Exception:
                pass
        return kv_caches

    def gather_selection(self, attn_kwargs, layer_idx, compress_ratio=1, attn_type="SAS"):
        """Perform gather selection operation.

        Delegates to the gather_selection module for the core implementation.

        Args:
            attn_kwargs: Dictionary containing attention-related tensors.
            layer_idx: Current layer index.
            compress_ratio: Compression ratio.
        """
        return gather_selection(self, attn_kwargs, layer_idx, compress_ratio, attn_type)

    def build_h2d_ops_hybrid(self, local_block_ids: List[List[int]], tp_nnodes: int = 1):
        """Build H2D operations for hybrid attention.

        Delegates to transfer_engine module for unified H2D operation building.
        """
        return build_h2d_ops_hybrid(self, local_block_ids, tp_nnodes)

    def build_h2d_ops_hbm_buffer(self, local_block_ids: List[List[int]], request_id: str):
        """Build H2D operations using HBM buffer.

        Delegates to transfer_engine module for unified H2D operation building.
        """
        return build_h2d_ops_hbm_buffer(self, local_block_ids, request_id)

    def reserve_hbm_lane_for_request(self, req_id: str) -> int:
        """Reserve or return the HBM lane owned by ``req_id``.

        This path is used only by the ``USE_OMNI_INPUT_BATCH`` flow. H2D is
        the writer, and ``OmniCacheInputBatch`` later reads the reservation
        when building attention-facing rows.
        """
        if self.req_id_to_idx is None:
            raise RuntimeError("HBM lane maps are not initialized")

        with self._lane_lock:
            lane = self.req_id_to_idx.get(req_id, None)
            if lane is not None:
                logger.info(
                    "HBM lane already reserved for request %s: lane=%s",
                    req_id,
                    lane,
                )
                return lane

            used_lanes = {int(mapped_lane) for mapped_lane in self.req_id_to_idx.values()}
            for lane in range(self.num_max_batch_pool):
                if lane not in used_lanes:
                    self.req_id_to_idx[req_id] = lane
                    return lane

        raise RuntimeError(f"No free HBM lane for request {req_id}")

    def get_hbm_lane_for_request(self, req_id: str) -> int | None:
        """Return the pre-reserved HBM lane for ``req_id`` if present."""
        if self.req_id_to_idx is None:
            return None
        lane = self.req_id_to_idx.get(req_id, None)
        return lane

    def release_hbm_lane_for_request(self, req_id: str) -> int | None:
        """Release the HBM lane owned by ``req_id`` from both map views."""
        if self.req_id_to_idx is None:
            return None

        with self._lane_lock:
            lane = self.req_id_to_idx.pop(req_id, None)
            if lane is None:
                logger.info(
                    "No HBM lane reservation found when releasing request %s",
                    req_id,
                )
                return None
            return lane

    def build_h2d_ops(self, local_block_ids: List[List[int]], request_id: Optional[str]):
        """Build H2D operations for decode.

        Delegates to transfer_engine module for unified H2D operation building.
        """
        return build_h2d_ops(self, local_block_ids, request_id)

    def synchronize_h2d(self, batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes):
        """Synchronize H2D transfer for decode.

        Delegates to TransferManager for unified H2D handling.
        """
        return self.transfer_manager.sync_h2d_decode(
            batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes
        )

    def synchronize_d2h(self, attn_names: list[str], attn_metadatas: list[str], kv_event: torch.npu.Event) -> None:
        return

    @staticmethod
    def initialize_decode_omni_cache(vllm_config, model_runner):
        # This method is for initializing host tensor and
        #  regestering before load_model in decode side.
        from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        from vllm.v1.worker.gpu.attn_utils import get_kv_cache_spec
        from omni_cache.cache.core.base import _detect_enable_dsa_from_config

        # block_size is the real model block size; do not override by
        # env var, otherwise the host pool slot is sized for the wrong
        # block geometry and later views run off the end of storage.
        block_size = vllm_config.cache_config.block_size

        if _detect_enable_dsa_from_config(vllm_config):
            # DSA without Pangu V2: indexer-only head_size for the fake
            # spec.  Pangu V2 overrides this below with 352 to match the
            # unified pool layout.
            head_size = 128
        else:
            head_size = vllm_config.model_config.hf_config.head_dim

        # For hybrid-attention models (Pangu V2 hybrid), the unified
        # host-pool slot must fit the concatenated per-token attention
        # head dim (nope + rope [+ indexer]).  FullAttentionSpec doubles
        # head_size internally (K + V), so the fake-spec head_size here
        # is half the concatenated head dim.  Derived from hf_config, so
        # this stays correct when the model's head dims change.
        if _is_pangu_v2_model(vllm_config):
            hf = vllm_config.model_config.hf_config
            # What the KV cache actually stores per token in MLA/DSA:
            # compressed KV (kv_lora_rank) + rotary (qk_rope_head_dim)
            # [+ indexer logits (index_head_dim) for DSA]. NOT
            # qk_nope_head_dim, which is on the Q path only.
            kv_lora = getattr(hf, "kv_lora_rank", 0) or 0
            rope = getattr(hf, "qk_rope_head_dim", 0) or 0
            indexer = getattr(hf, "index_head_dim", 0) or getattr(hf, "indexer_head_dim", 0) or 0
            concat = kv_lora + rope + indexer
            if concat > 0:
                head_size = (concat + 1) // 2  # FullAttentionSpec 2x K+V factor

        attn_group_size = int(os.getenv("HYBRID_ATTN_GROUP_SIZE", "0"))
        if attn_group_size:
            num_hidden_layer = attn_group_size
        else:
            num_hidden_layer = vllm_config.model_config.hf_config.num_hidden_layers
        layer_names = [f"model.layers.{i}.self_attn.attn" for i in range(num_hidden_layer)]

        def create_fake_kv_cache_config(
            layer_names: list[str] = None,
            block_size: int = 128,
            head_size: int = 128,
            dtype: torch.dtype = torch.bfloat16,
            vllm_config: VllmConfig = None,
        ) -> dict[str, FullAttentionSpec]:

            fake_spec = {}
            for layer_name in layer_names:
                fake_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=head_size,
                    dtype=dtype,
                )
            logger.debug(f"<<< in initialize_decode_omni_cache: {fake_spec=}")
            _, total_gpu_memory = torch.npu.mem_get_info()
            available_mem = vllm_config.cache_config.gpu_memory_utilization * total_gpu_memory
            available_mem_list = [available_mem] * len(fake_spec)
            fake_kv_cache_configs = get_kv_cache_configs(vllm_config, [fake_spec], available_mem_list)[0]

            return fake_kv_cache_configs

        fake_kv_cache_config = create_fake_kv_cache_config(
            layer_names=layer_names,
            block_size=block_size,
            head_size=head_size,
            dtype=vllm_config.model_config.dtype,
            vllm_config=vllm_config,
        )

        from omni_cache.cache.core.base import create_omni_cache

        omni_cache = create_omni_cache(
            kv_cache_config=fake_kv_cache_config,
            vllm_config=vllm_config,
            runner=model_runner,
        )
        return omni_cache

    @staticmethod
    def _update_status_buffered(
        omni_cache,
        reshaped_status: torch.Tensor,
        old_req_ids: List[Optional[str]],
        new_req_ids: List[Optional[str]],
        *,
        fill_value: Any = -1,
    ) -> None:
        """Update status buffer - delegates to GatherSelectionUpdater."""
        GatherSelectionUpdater._update_status_buffered(
            omni_cache,
            reshaped_status,
            old_req_ids,
            new_req_ids,
            fill_value=fill_value,
        )

    @staticmethod
    def _reorder_block_table_only(
        omni_cache, reshaped_table: torch.Tensor, old_req_ids: List[str], new_req_ids: List[str]
    ) -> None:
        """Reorder block table - delegates to GatherSelectionUpdater."""
        GatherSelectionUpdater._reorder_block_table_only(
            omni_cache,
            reshaped_table,
            old_req_ids,
            new_req_ids,
        )

    @staticmethod
    def maybe_update_selection_kv_block_status(input_batch, omni_cache, num_scheduled_tokens):
        """Maybe update selection KV block status - delegates to GatherSelectionUpdater."""
        GatherSelectionUpdater.maybe_update_selection_kv_block_status(
            input_batch,
            omni_cache,
            num_scheduled_tokens,
        )

    @property
    def enable_gs(self):
        return self.enable_dsa and int(os.getenv("DISABLE_GATHER_SELECTION", "0")) == 0

    @property
    def use_input_batch_lane_mapping(self) -> bool:
        """Whether H2D owns the global lane mapping for OmniCacheInputBatch."""
        return int(os.getenv("USE_OMNI_INPUT_BATCH", "0")) == 1
