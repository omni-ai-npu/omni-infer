# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import copy
import math
import os
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch_npu

from vllm.config import VllmConfig
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.distributed.parallel_state import get_dp_group, get_tp_group, get_world_group
from vllm.logger import init_logger
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.kv_cache_interface import KVCacheConfig, MLAAttentionSpec, AttentionSpec

from omni_cache.cache.core.constants import (
    SIZE_BYTES_PER_LAYER_PREFILL,
    SIZE_BYTES_PER_LAYER_DECODE,
    BLOCK_SIZE,
    HEAD_DIM,
    LOCAL_DP_SIZE,
    NZ_DIM,
    PRE_CALC_PREFILL_BLOCK_NUM,
    ENABLE_C8_INDEXER,
    ENABLE_HOST_MAPPING,
)
from omni_cache.cache.memory.memory_pool import KVCacheMemoryPool
from omni_npu.worker.npu_model_runner import NPUModelRunner
from omni_cache.cache.utils.ops import (
    _batch_fill_block_table_with_reorder,
    _batch_fill_slot_mapping,
    _is_hybrid_attention_enabled,
    calculate_copy_block_stats,
    divide_or_raise,
    generate_full_block_slot,
    pad_inputs,
    pad_tensor,
    torch_to_numpy_zero_copy,
)
from omni_cache.cache.utils.support import (
    _is_pangu_v2_model,
    reset_static_forward_context_kv_cache,
    resolve_kv_spec_state,
    resolve_parallel_cache_state,
    resolve_requested_device,
)

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from omni_cache.cache.decode.hbm_buffer_utils import construct_hbm_buffer
    from omni_cache.cache.decode.host_kv_cache_utils import parse_pa_kv_cache

logger = init_logger("vllm.v1.omni")


def _detect_enable_dsa_from_config(vllm_config: VllmConfig) -> bool:
    """Detect whether DSA is enabled from vllm_config."""
    if vllm_config is None:
        return False
    if _is_pangu_v2_model(vllm_config):
        return True
    model_config = getattr(vllm_config, "model_config", None)
    if model_config is None:
        return False
    if not getattr(model_config, "use_mla", False):
        return False
    hf_config = getattr(model_config, "hf_config", None)
    return hf_config is not None and hasattr(hf_config, "index_topk")


if ENABLE_HOST_MAPPING:
    try:
        import triton  # noqa: F401
    except ImportError:
        triton = None
    from ..device_backend.ascend.ops.triton_ops import build_fake_block_table_kernel_compress


@dataclass
class RequestWindowState:
    """State for a single request's block table mapping.

    Uses __slots__ and stores only the base address and state index.
    All other data (win_size, max_lb, num_assigned, logic_to_phys, logic_valid)
    is kept in pre-allocated contiguous buffers for zero-copy access.
    Reading max_lb or num_assigned directly accesses the buffers.
    """

    __slots__ = ["base", "win_size", "state_idx", "_omni_cache_ref"]

    base: int
    win_size: int
    state_idx: int
    _omni_cache_ref: object

    @property
    def max_lb(self) -> int:
        """Read max_lb directly from buffer."""
        return self._omni_cache_ref._max_lbs_buffer[self.state_idx]

    @property
    def num_assigned(self) -> int:
        """Read num_assigned directly from buffer."""
        return self._omni_cache_ref._num_assigneds_buffer[self.state_idx]

    @property
    def logic_to_phys(self) -> np.ndarray:
        """Get view of logic_to_phys buffer for this state."""
        offset = self.state_idx * self._omni_cache_ref.MAX_LOGIC_BLOCKS
        return self._omni_cache_ref._logic_to_phys_buffer[offset:offset + self._omni_cache_ref.MAX_LOGIC_BLOCKS]

    @property
    def logic_valid(self) -> np.ndarray:
        """Get view of logic_valid buffer for this state."""
        offset = self.state_idx * self._omni_cache_ref.MAX_LOGIC_BLOCKS
        return self._omni_cache_ref._logic_valid_buffer[offset:offset + self._omni_cache_ref.MAX_LOGIC_BLOCKS]


@dataclass
class _SlotInfo:
    """Minimal dataclass for slot mapping information."""

    __slots__ = ["q_start", "q_len", "last_seq_len", "state"]

    q_start: int
    q_len: int
    last_seq_len: int
    state: RequestWindowState


class BaseOmniCache(ABC):
    MEMMAP_PATH = os.environ.get("OMNI_CACHE_MMAP_PATH", "/dev/hugepages/omni_cache")
    GATHER_SELECTION_POOL_SIZE = 8192

    def __init__(self, kv_cache_config: KVCacheConfig, runner: NPUModelRunner, vllm_config: VllmConfig = None):
        parallel_state = resolve_parallel_cache_state(
            vllm_config=vllm_config,
            get_tp_group=get_tp_group,
            get_dp_group=get_dp_group,
            local_dp_size=LOCAL_DP_SIZE,
        )
        self.tp_rank = parallel_state.tp_rank
        self.tp_local_rank = parallel_state.tp_local_rank
        self.tp_world_size = parallel_state.tp_world_size
        self.dp_world_size = parallel_state.dp_world_size
        self.dp_local_rank = parallel_state.dp_local_rank
        self.dp_world_size_local = parallel_state.dp_world_size_local
        self.local_dp_size = LOCAL_DP_SIZE

        self.device = resolve_requested_device(runner.device, torch)
        self.vllm_config = vllm_config
        self.enable_dsa = _detect_enable_dsa_from_config(vllm_config)
        self.is_pangu_v2 = _is_pangu_v2_model(vllm_config)

        hf_config = getattr(getattr(vllm_config, "model_config", None), "hf_config", None)
        if hf_config is not None:
            kv_lora = getattr(hf_config, "kv_lora_rank", 0) or 512
            qk_rope = getattr(hf_config, "qk_rope_head_dim", 0) or 64
            self.k_rope_size = qk_rope
            self.kvcache_size = kv_lora + qk_rope if self.is_pangu_v2 else kv_lora
            self.index_topk = getattr(hf_config, "index_topk", 0) or 2048
        else:
            self.k_rope_size = 64
            self.kvcache_size = 576
            self.index_topk = 2048

        self.enable_omni_cache = int(os.getenv("ENABLE_OMNI_CACHE", "0"))
        self.kv_cache_config = kv_cache_config
        self._apply_kv_spec_state(
            resolve_kv_spec_state(
                kv_cache_config=kv_cache_config,
                vllm_config=vllm_config,
                enable_dsa=self.enable_dsa,
                enable_c8_indexer=ENABLE_C8_INDEXER,
                mla_attention_spec_type=MLAAttentionSpec,
                force_hybrid_attn=False,  # only for quick debugging, need to remove for stable version
            )
        )

        # Pangu V2 PD: force block_size=128 so prefill and decode agree
        # on the OX transfer layout. The attn_spec may report 64 (TP=8
        # prefill), but the attention kernel and OX must share the same
        # tokens-per-block. cache_config.block_size is patched to 128
        # by PanguV2HybridForCausalLMConfig; if that hasn't landed yet,
        # fall back to 128 unconditionally.
        if self.is_pangu_v2:
            _cc_bs = getattr(
                getattr(vllm_config, "cache_config", None),
                "block_size",
                None,
            )
            logger.warning(
                "[PANGU-V2-BLOCK-SIZE] self.block_size=%d, cache_config.block_size=%s, forcing to 128",
                self.block_size,
                _cc_bs,
            )
            self.block_size = 128

        self.shape, self.num_blocks = self.calc_cache_shape()

        logger.warning(f"**BaseOmniCache**: {self.shape=}, {self.tp_world_size=}")

        self.host_cache = self.initialize_shared_memory()
        self.ascend_cl_stream = self.host_cache.ascend_cl_stream
        self.host_swap_tensor = self.host_cache.shared_tensor_npu
        # DSA-split: secondary host pool wiring
        # When ENABLE_OMNI_CACHE_DSA_SPLIT=1, allocate a second
        # narrow-head hugetlbfs pool (default head_size=576 bf16).
        # The primary stays host-only and serves OX pulls; the
        # connector copies DSA slots over to this secondary
        # post-pull. Attention read-path rewire lives in
        # hbm_buffer_utils/host_kv_cache_utils (done separately).
        self.dsa_host_pool = None
        try:
            from omni_cache.cache.memory.dsa_host_pool import (
                maybe_build_from_primary as _dsa_maybe_build,
            )

            _dsa_pool = _dsa_maybe_build(self.host_cache)
            if _dsa_pool is not None:
                _dsa_pool.register_mmu(self.host_cache)
                self.dsa_host_pool = _dsa_pool
                logger.warning(
                    "[DSA-SPLIT] secondary host pool active: layers=%d blocks=%d block_size=%d head=%d",
                    _dsa_pool.num_layers,
                    _dsa_pool.num_blocks,
                    _dsa_pool.block_size,
                    _dsa_pool.head_size,
                )
        except ImportError:
            pass
        except Exception as _e:
            logger.warning("[DSA-SPLIT] secondary pool init failed: %s", _e)
        self.block_len_dtype, self.dp_offset = self.calculate_kv_xsfer_params()

        self.device_cache = None
        self.runner = runner

    @property
    def enable_gs(self) -> bool:
        return False

    @property
    def selection_state_size(self) -> int:
        return self.GATHER_SELECTION_POOL_SIZE * 2

    @property
    def s_max_block_num(self) -> int:
        # NOTE(runze): for DSA, all heads share the same topK indices
        # so we hardcode head_num to be 1 here
        head_num = 1
        bs = self.block_size
        return (self.GATHER_SELECTION_POOL_SIZE + head_num * bs - 1) // (head_num * bs)

    @property
    def use_input_batch_lane_mapping(self) -> bool:
        return False

    def _layer_name_to_group_and_layer_idx(self, layer_name):
        for grp_idx in range(len(self.kv_cache_config.kv_cache_groups)):
            layer_name_list = self.kv_cache_config.kv_cache_groups[grp_idx].layer_names
            if layer_name in layer_name_list:
                layer_idx = layer_name_list.index(layer_name)
                return grp_idx, layer_idx
        raise ValueError(f"Invalid layer name '{layer_name}' that cannot be found in any kv group.")

    def _apply_kv_spec_state(self, spec_state) -> None:
        self.num_spec_token = spec_state.num_spec_token
        self.is_hybrid_attn = spec_state.is_hybrid_attn
        self.num_layers = spec_state.num_layers
        self.block_size = spec_state.block_size
        self.head_sizes = spec_state.head_sizes
        self.head_size = spec_state.head_size
        self.dtype = spec_state.dtype
        self.block_num = spec_state.block_num
        self.block_bytes = spec_state.block_bytes

    def ensure_device_cache_initialized(self):
        if self.device_cache:
            return
        self.device_cache = self.initialize_device_cache(self.kv_cache_config, self.runner)

    def update_kv_cache_spec(self, kv_cache_config: KVCacheConfig, vllm_config: VllmConfig):
        self.kv_cache_config = kv_cache_config
        self._apply_kv_spec_state(
            resolve_kv_spec_state(
                kv_cache_config=kv_cache_config,
                vllm_config=vllm_config,
                enable_dsa=_detect_enable_dsa_from_config(vllm_config),
                enable_c8_indexer=ENABLE_C8_INDEXER,
                mla_attention_spec_type=MLAAttentionSpec,
            )
        )

        self.shape, self.num_blocks = self.calc_cache_shape()

        logger.warning(f"**BaseOmniCache**: {self.shape=}, {self.tp_world_size=}")
        self.block_len_dtype, self.dp_offset = self.calculate_kv_xsfer_params()

        decode_omni_cache_cls = None
        try:
            # NOTE: DecodeOmniCache lives at omni_cache.cache.decode.decode_omni_cache,
            # not omni_cache.cache.core.decode_omni_cache — use two-level relative
            # import so the check actually matches decode instances.
            from ..decode.decode_omni_cache import DecodeOmniCache as decode_omni_cache_cls
        except Exception:
            decode_omni_cache_cls = None

        if ENABLE_HOST_MAPPING and decode_omni_cache_cls is not None and isinstance(self, decode_omni_cache_cls):
            # import at runtime to avoid circular import
            from omni_cache.cache.decode.host_kv_cache_utils import parse_pa_kv_cache
            from omni_cache.cache.decode.hbm_buffer_utils import construct_hbm_buffer

            self.host_cache_pa = parse_pa_kv_cache(self, self.host_cache.kvi_tensors_swap)
            self.metadata_grp_id = 0
            self.num_max_batch_pool = self.vllm_config.scheduler_config.max_num_seqs + 5
            self.hbm_buffer_pool, self.hbm_buffer_block_table_pool = construct_hbm_buffer(self)
            self.num_called_builder = len(kv_cache_config.kv_cache_groups) + self.num_spec_token
            self.num_attn_group = len(kv_cache_config.kv_cache_groups)
            self.req_swa_record = {}

            import multiprocessing
            from multiprocessing import current_process

            curr_proc = current_process()
            was_daemon = curr_proc.daemon
            curr_proc.daemon = False
            try:
                manager = multiprocessing.Manager()
                self.record_batch_idx_to_req = manager.dict()
                self.record_req_ids_swa_block = manager.dict()
                self.req_id_to_idx = manager.dict()
                self._lane_lock = manager.Lock()
            finally:
                curr_proc.daemon = was_daemon
            for idx_batch in range(self.num_max_batch_pool):
                self.record_batch_idx_to_req[idx_batch] = None

            self.id_to_idx_table = torch.full(
                (self.vllm_config.scheduler_config.max_num_seqs,), -1, device=self.device, dtype=torch.int32
            )
            self.indices_cpu_buffer = torch.zeros(
                self.vllm_config.scheduler_config.max_num_seqs, dtype=torch.int32
            ).pin_memory()

    def update_model_runner(self, model_runner: NPUModelRunner):
        self.runner = model_runner

    @abstractmethod
    def calc_cache_shape(self) -> Tuple[Tuple[int, ...], int]:
        pass

    @abstractmethod
    def calculate_kv_xsfer_params(self) -> Tuple[int, int]:
        pass

    @abstractmethod
    def initialize_device_cache(self, kv_cache_config: KVCacheConfig, runner: NPUModelRunner):
        pass

    def initialize_shared_memory(self) -> KVCacheMemoryPool:
        total_numel = math.prod(self.shape)
        itemsize = self.dtype.itemsize

        return KVCacheMemoryPool(
            BaseOmniCache.MEMMAP_PATH,
            total_numel * itemsize,
            self.shape,
            self.dp_local_rank,
            self.device,
            self.is_hybrid_attn,
            self.enable_dsa,
            vllm_config=self.vllm_config,
        )

    def __getitem__(self, index: int):
        return self.host_cache

    @abstractmethod
    def synchronize_h2d(self) -> None:
        pass

    @abstractmethod
    def synchronize_d2h(self, attn_names: list[str], attn_metadatas: list[str], kv_event: torch.npu.Event) -> None:
        pass


def create_omni_cache(kv_cache_config: KVCacheConfig, vllm_config: VllmConfig, runner: NPUModelRunner) -> BaseOmniCache:
    """
    Factory function to create the appropriate BaseOmniCache instance based on the is_prefill flag.

    Args:
        kv_cache_config: Configuration for the KV cache
        vllm_config: The VllmConfig object
        runner: NPU model runner instance

    Returns:
        PrefillOmniCache or DecodeOmniCache instance based on the is_prefill flag
    """
    from .. import set_omni_cache

    is_prefill = vllm_config.kv_transfer_config.kv_role != "kv_consumer"
    if is_prefill:
        from ..prefill.prefill_omni_cache import PrefillOmniCache

        max_num_batched_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        max_model_len = runner.max_model_len
        set_omni_cache(
            PrefillOmniCache(
                kv_cache_config,
                runner,
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs,
                max_model_len=max_model_len,
                vllm_config=vllm_config,
            )
        )
        from .. import omni_cache

        runner.kv_caches = None

        # Initialize device_cache for omni_cache
        omni_cache.ensure_device_cache_initialized()

        # Create OmniKvCacheAccessor class with access to omni_cache instance
        # This avoids importing omni_cache in the forward method of npu_compressed_mqa.py
        class OmniKvCacheAccessor:
            """Wrapper that provides dynamic access to omni_cache device_cache.

            This wrapper is indexed by engine_id (like original kv_cache) but internally
            retrieves the appropriate cache from omni_cache.device_cache based on the
            current stage_record and layer_name.
            """

            def __init__(self, layer_name, omni_cache):
                self.layer_name = layer_name
                self._omni_cache = omni_cache

            def __getitem__(self, engine_id):
                # Use the current stage_record to get the right cache stage
                stage = self._omni_cache.stage_record
                return self._omni_cache.device_cache[stage][self.layer_name]

            def __len__(self):
                # Return the number of stages for compatibility
                return len(self._omni_cache.device_cache)

        # Replace kv_cache with OmniKvCacheAccessor for all layers that use omni_cache
        static_ctx = vllm_config.compilation_config.static_forward_context
        for layer_name, layer_obj in static_ctx.items():
            # Check if this layer should use omni_cache (has an entry in device_cache)
            if layer_name in omni_cache.device_cache[0]:
                # Replace with dynamic accessor that has access to omni_cache
                layer_obj.kv_cache = OmniKvCacheAccessor(layer_name, omni_cache)
    else:
        from ..decode.decode_omni_cache import DecodeOmniCache

        set_omni_cache(DecodeOmniCache(kv_cache_config, runner, vllm_config=vllm_config))
        from .. import omni_cache
    return omni_cache


def bind_kv_cache(*args, **kwargs):
    """Compatibility hook for legacy call sites/tests.

    Real binding is handled by cache creation flow in current implementation.
    """
    return None


@dataclass
class PrefixCopyMeta:
    consecutive_blocks: list[list[Tuple[int, int]]]
    """The starts and ends of consecutive full blocks."""

    query_lens: list[int]
    """Number of tokens per sample in current batch."""

    query_slots: torch.Tensor
    """The positions to store the KV of current tokens."""

    num_actual_tokens: int = None
    """Total number of tokens in current batch."""

    num_copy_blocks: int = None
    """Total number of blocks in KV cache to copy."""

    last_block_id: Optional[int] = None
    """The last block which might be partially filled. In APC, it must be None."""

    last_block_len: Optional[int] = None
    """Number of tokens filled in the last block."""

    def __post_init__(self):
        if len(self.consecutive_blocks) != len(self.query_lens):
            raise RuntimeError(f"Lengths mismatch! {len(self.consecutive_blocks)=}, while {len(self.query_lens)=}.")
        self.num_actual_tokens = sum(self.query_lens)
        total_copy_ops, total_copy_blocks = calculate_copy_block_stats(self.consecutive_blocks)

        if get_world_group() == 0:
            logger.debug(
                f"!!! Totally {total_copy_ops} copy operations with {total_copy_blocks} blocks will be executed. ***"
            )

        self.num_copy_blocks = total_copy_blocks
