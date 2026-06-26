# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Buffer and stream management for KV swap operations.

This module provides classes for managing CPU buffers, CUDA/NPU streams,
and thread pools used in H2D (Host-to-Device) and D2H (Device-to-Host)
transfer operations.
"""

from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import torch
from concurrent.futures import Future, ThreadPoolExecutor

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


class TransferBuffers:
    """Manages CPU buffers and related resources for H2D/D2H transfers.

    Args:
        cache: The parent cache instance (PrefillOmniCache or DecodeOmniCache)
    """

    def __init__(self, cache):
        self.cache = cache
        self.batch_buffer_cpu: List[Tuple] = []  # CPU buffers for D2H
        self.batch_slots_cpu: Optional[torch.Tensor] = None  # Slot mappings on CPU
        self.arange_cpu: Optional[torch.Tensor] = None
        self.arange: Optional[torch.Tensor] = None

    def _init_basic_attrs(self, max_num_batched_tokens: int, max_model_len: int) -> None:
        """Initialize basic cache attributes."""
        cache = self.cache
        cache.max_num_batched_tokens = max_num_batched_tokens
        cache.max_model_len = max_model_len

    def _init_arange_tensors(self, max_model_len: int) -> None:
        """Initialize arange tensors for indexing."""
        self.arange_cpu = torch.arange(
            max_model_len, device='cpu', dtype=torch.int64
        )
        self.arange = self.arange_cpu.to(device=self.cache.device)

    def _get_mome_head_size(self, spec):
        assert hasattr(spec, "shapes"), f"{type(spec)} has no attribute shapes"
        total_size = sum(shape[0] for shape in spec.shapes)
        return total_size

    def _get_head_size_config(self, kv_cache_config: "KVCacheConfig") -> Tuple[List, List]:
        """Get head sizes and group specs based on attention mode.

        Returns:
            Tuple of (group_specs, head_sizes_buf)
        """
        cache = self.cache
        if cache.is_hybrid_attn:
            group_specs = [g.kv_cache_spec for g in kv_cache_config.kv_cache_groups]
            head_sizes_buf = [ self._get_mome_head_size(s) if type(s).__name__ == "MomeSpec" else \
                s.head_size * s.num_kv_heads for s in group_specs]
        else:
            group_specs = [None] * len(cache.head_sizes)
            head_sizes_buf = cache.head_sizes
        return group_specs, head_sizes_buf

    def _create_cpu_tensor(self, size_d: int, dtype, pin: bool = True) -> torch.Tensor:
        """Create a CPU tensor with optional pinned memory.

        Args:
            size_d: Size of the second dimension
            dtype: Data type of the tensor
            pin: Whether to use pinned memory

        Returns:
            Created CPU tensor
        """
        cache = self.cache
        pin_enabled = torch.cuda.is_available() or (
            hasattr(torch, "npu") and torch.npu.is_available()
        )
        return torch.empty(
            cache.max_num_batched_tokens * 2 // cache.tp_world_size,
            size_d,
            dtype=dtype,
            device='cpu',
            pin_memory=(pin and pin_enabled),
        )

    def _create_single_buffer(self, D: int, spec) -> Tuple:
        """Create a single buffer tuple based on spec.

        Args:
            D: Head size dimension
            spec: KV cache spec or None

        Returns:
            Buffer tuple (tensor or tuple of tensors)
        """
        cache = self.cache
        has_separate_kvscale = spec and getattr(spec, "separate_kvscale", False)

        if has_separate_kvscale:
            return (
                self._create_cpu_tensor(D, spec.dtype),
                self._create_cpu_tensor(1, spec.separate_kvscale)
            )

        if cache.is_hybrid_attn:
            return (self._create_cpu_tensor(D, cache.dtype),)

        return self._create_cpu_tensor(D, cache.dtype)

    def _clone_speculation_buffers(self) -> None:
        """Clone buffers for speculative decoding tokens."""
        cache = self.cache
        if cache.num_spec_token <= 0 or not cache.is_hybrid_attn:
            return

        source_idx = 1 if len(self.batch_buffer_cpu) > 1 else 0
        source_tuple = self.batch_buffer_cpu[source_idx]
        for _ in range(cache.num_spec_token):
            cloned_tuple = tuple(t.clone() for t in source_tuple)
            self.batch_buffer_cpu.append(cloned_tuple)

    def _log_buffer_shapes(self) -> None:
        """Log buffer shapes for debugging."""
        from vllm.logger import init_logger
        logger = init_logger("vllm.v1.omni")
        cache = self.cache

        try:
            if cache.is_hybrid_attn:
                shape_info = []
                for i, buf in enumerate(self.batch_buffer_cpu):
                    shapes = "/".join([str(list(t.shape)) for t in buf])
                    shape_info.append(f"Grp{i}[{shapes}]")
                logger.info(f"**PrefillOmniCache**: CPU buffer shapes: {', '.join(shape_info)}")
            else:
                shapes = "/".join([str(list(t.shape)) for t in self.batch_buffer_cpu])
                logger.info(f"**PrefillOmniCache**: CPU buffer shapes: {shapes}")
        except Exception as e:
            logger.debug(f"Failed to print shapes: {e}")

    def _init_raw_byte_buffer(self, kv_cache_config: "KVCacheConfig") -> None:
        """Initialize a flat byte buffer for byte-dimension TP splitting in D2H.

        Used by synchronize_d2h_prefill + _update_host_cache_thread path.
        Each TP rank stores block_bytes / tp_size bytes per block.
        Initialized for both non-hybrid and hybrid (single-layer) paths.
        """
        cache = self.cache

        page_size_bytes = getattr(cache, "page_size_bytes", 0)
        if page_size_bytes == 0:
            # Fallback: compute from spec
            groups = kv_cache_config.kv_cache_groups
            if groups:
                page_size_bytes = groups[0].kv_cache_spec.page_size_bytes

        if page_size_bytes == 0:
            self.batch_buffer_raw = None
            return

        elem_size = 2  # bfloat16
        page_elems = page_size_bytes // elem_size
        elems_per_rank = page_elems // cache.tp_world_size
        max_blocks = cache.max_model_len * cache.max_num_seqs // cache.block_size

        pin_enabled = torch.cuda.is_available() or (
            hasattr(torch, "npu") and torch.npu.is_available()
        )
        n_stages = max(1, int(getattr(cache, 'num_stages_layer_copy', 1) or 1))
        n_groups = len(kv_cache_config.kv_cache_groups)
        # Per-(stage, group) host pinned scratch, pre-allocated at init.
        # Multiple sub-attention calls inside one decoder block all hit
        # the same stage_idx but different group_idx, so each (stage,
        # group) pair gets its own pinned tensor and they cannot collide.
        # Each group has its own spec.page_size_bytes / tp_world_size
        # elems_per_rank, so size the buffer to fit that group's pages.
        self.batch_buffer_raw_stages = []
        for _ in range(n_stages):
            stage_dict = {}
            for gi, grp in enumerate(kv_cache_config.kv_cache_groups):
                spec = grp.kv_cache_spec
                grp_page_bytes = getattr(spec, 'page_size_bytes', 0) or page_size_bytes
                grp_elems_per_rank = (grp_page_bytes // elem_size) // cache.tp_world_size
                stage_dict[gi] = torch.empty(
                    max_blocks,
                    grp_elems_per_rank,
                    dtype=torch.bfloat16,
                    device='cpu',
                    pin_memory=pin_enabled,
                )
            self.batch_buffer_raw_stages.append(stage_dict)
        # Legacy single-buffer attribute name (None - the dict layout is
        # the single source of truth).
        self.batch_buffer_raw = None

        # Precompute component byte boundaries (in bfloat16 elements) for host write.
        # Layout in device raw tensor: [comp0 | comp1 | ...] per block.
        comp_elem_sizes = [cache.block_size * hs for hs in cache.head_sizes]
        comp_offsets = [0]
        for s in comp_elem_sizes:
            comp_offsets.append(comp_offsets[-1] + s)
        cache.comp_elem_sizes = comp_elem_sizes
        cache.comp_offsets = comp_offsets

        # Verify: total component elements should match page_elems
        total_comp = sum(comp_elem_sizes)
        if total_comp != page_elems:
            from vllm.logger import init_logger
            _logger = init_logger("vllm.v1.omni")
            _logger.warning(
                f"byte-split: comp_elem_sizes sum={total_comp} != page_elems={page_elems}; "
                f"comp_elem_sizes={comp_elem_sizes}, head_sizes={cache.head_sizes}"
            )

    def _attach_to_cache(self) -> None:
        """Attach buffers to cache.

        Multi-stage hosts:
          - cache.batch_buffer_cpu_stages : list of per-stage [per-group] buffers.
            Index as [stage_idx][group_idx]. Length == num_stages_layer_copy.
          - cache.batch_buffer_raw_stages : list of per-stage flat byte buffers.
            Index as [stage_idx]. Length == num_stages_layer_copy (may be None
            on the byte-split path that doesn't allocate batch_buffer_raw).

        Legacy single-handle attributes (cache.batch_buffer_cpu,
        cache.batch_buffer_raw) point at stage 0 so call sites that haven't
        been threaded with stage_record continue to work when
        num_stages_layer_copy == 1.
        """
        cache = self.cache
        cache.batch_buffer_cpu_stages = self.batch_buffer_cpu_stages
        cache.batch_buffer_cpu = self.batch_buffer_cpu_stages[0]
        if hasattr(self, 'batch_buffer_raw_stages'):
            cache.batch_buffer_raw_stages = self.batch_buffer_raw_stages
            cache._raw_alloc_kwargs = getattr(self, '_raw_alloc_kwargs', None)
            cache.batch_buffer_raw = None
        else:
            cache.batch_buffer_raw_stages = None
            cache.batch_buffer_raw = None
        cache.arange_cpu = self.arange_cpu
        cache.arange = self.arange

    def initialize_cpu_buffers(
        self,
        kv_cache_config: "KVCacheConfig",
        max_num_batched_tokens: int,
        max_model_len: int,
    ) -> None:
        """Initialize CPU buffers for D2H transfers.

        This method replaced prefill_omni_cache.py:_init_cpu_buffers() (removed)

        Args:
            kv_cache_config: KV cache configuration
            max_num_batched_tokens: Maximum number of batched tokens
            max_model_len: Maximum model length
        """
        self._init_basic_attrs(max_num_batched_tokens, max_model_len)
        self._init_arange_tensors(max_model_len)

        group_specs, head_sizes_buf = self._get_head_size_config(kv_cache_config)
        self.cache.head_sizes_buf = head_sizes_buf

        n_stages = max(1, int(getattr(self.cache, 'num_stages_layer_copy', 1) or 1))
        # batch_buffer_cpu is shaped [stage][group]. Existing call sites
        # that don't yet pass a stage index keep working against
        # cache.batch_buffer_cpu (= stage 0) when num_stages == 1.
        self.batch_buffer_cpu_stages = [
            [
                self._create_single_buffer(D, spec)
                for D, spec in zip(head_sizes_buf, group_specs)
            ]
            for _ in range(n_stages)
        ]
        self.batch_buffer_cpu = self.batch_buffer_cpu_stages[0]

        self._init_raw_byte_buffer(kv_cache_config)
        # FIXME(runze): this is only for DSV4, should be refactored later
        # self._clone_speculation_buffers()
        self._log_buffer_shapes()
        self._attach_to_cache()

    def initialize_token_indices(self, max_num_batched_tokens: int) -> None:
        """Initialize token indices buffers.

        This method replaced prefill_omni_cache.py:_init_token_indices() (removed)

        Args:
            max_num_batched_tokens: Maximum number of batched tokens
        """
        cache = self.cache
        pin_enabled = torch.cuda.is_available() or (
            hasattr(torch, "npu") and torch.npu.is_available()
        )

        n_stages = max(1, int(getattr(cache, 'num_stages_layer_copy', 1) or 1))
        if cache.is_hybrid_attn:
            cache.num_called_builder = (
                len(cache.kv_cache_config.kv_cache_groups) + cache.num_spec_token
            )
            cache.num_attn_group = len(cache.kv_cache_config.kv_cache_groups)
            cache.count_called = 0
            cache.batch_token_indices = [None] * cache.num_called_builder
            self.batch_slots_cpu_stages = [
                torch.zeros(
                    cache.num_called_builder,
                    max_num_batched_tokens * 2 // cache.tp_world_size,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=pin_enabled,
                )
                for _ in range(n_stages)
            ]
        else:
            self.batch_slots_cpu_stages = [
                torch.zeros(
                    max_num_batched_tokens * 2 // cache.tp_world_size,
                    dtype=torch.int64,
                    device="cpu",
                    pin_memory=True,
                )
                for _ in range(n_stages)
            ]
            cache.batch_token_indices = None
        # Stage-0 alias for legacy callers.
        self.batch_slots_cpu = self.batch_slots_cpu_stages[0]

        # Attach to cache for backward compatibility
        cache.batch_slots_cpu_stages = self.batch_slots_cpu_stages
        cache.batch_slots_cpu = self.batch_slots_cpu


class StreamManager:
    """Manages CUDA/NPU streams for H2D/D2H operations.

    Args:
        device: The torch device (NPU/GPU)
    """

    def __init__(self, device):
        self.device = device
        self.h2d_stream: Optional[torch.Stream] = None  # For prefill H2D
        self.d2h_stream: Optional[torch.Stream] = None  # For prefill D2H
        self.decode_h2d_stream: Optional[torch.Stream] = None  # For decode H2D
        self.copy_stream: Optional[torch.Stream] = None  # General copy stream
        self.h2d_event: Optional[torch.Event] = None  # Event for H2D completion
        self.d2h_event: Optional[torch.Event] = None  # Event for D2H completion

    def initialize_prefill_streams(self, cache) -> None:
        """Initialize streams for prefill operations.

        This method replaced parts of prefill_omni_cache.py:_init_streams_and_pools()
        and _init_prefix_buffer()

        Args:
            cache: The PrefillOmniCache instance
        """
        self.d2h_stream = torch.npu.Stream(device=self.device)
        self.h2d_stream = torch.npu.Stream(device=self.device)
        self.h2d_event = torch.npu.Event(blocking=False, enable_timing=False)
        self.d2h_event = torch.npu.Event(blocking=False, enable_timing=False)

        # Attach to cache for backward compatibility
        cache.d2h_stream = self.d2h_stream
        cache.h2d_stream = self.h2d_stream
        cache.h2d_event = self.h2d_event

    def initialize_decode_streams(self, cache) -> None:
        """Initialize streams for decode operations.

        This method replaces parts of decode_omni_cache.py:__init__

        Args:
            cache: The DecodeOmniCache instance
        """
        self.decode_h2d_stream = torch.npu.Stream(device=self.device)

        if hasattr(torch, "npu") and torch.npu.is_available():
            self.copy_stream = torch.npu.Stream()
        else:
            self.copy_stream = None

        # Attach to cache for backward compatibility
        cache.decode_h2d_stream = self.decode_h2d_stream
        cache._copy_stream = self.copy_stream


class ThreadPoolManager:
    """Manages thread pools for async operations."""

    def __init__(self):
        self.d2h_thrp: Optional[ThreadPoolExecutor] = None
        self.copy_future: Optional[Future] = None
        self.copy_futures: List[Optional[Future]] = []

    def initialize(self, cache, max_workers: int = 1) -> None:
        """Initialize thread pool.

        This method replaced parts of prefill_omni_cache.py:_init_streams_and_pools() (removed)

        Args:
            cache: The cache instance
            max_workers: Maximum number of worker threads
        """
        self.d2h_thrp = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="D2H_Worker"
        )
        self.copy_future = None

        import os as _os_pp
        # OMNI_CACHE_PREFILL_NUM_STAGES (>=1) sets the prefill ping-pong
        # depth. Default keeps the prior behaviour: 2 for non-hybrid
        # (already shipping), 1 for hybrid (Pangu V2). Opt in to >=2 on
        # hybrid to overlap D2H with the next layer's compute. Pairs
        # with the qa_conv-gated wait in MOMEAttnPlugin.pre_attn.
        try:
            _override = int(_os_pp.environ.get('OMNI_CACHE_PREFILL_NUM_STAGES', '0'))
        except ValueError:
            _override = 0
        if _override >= 1:
            cache.num_stages_layer_copy = _override
        elif not cache.is_hybrid_attn:
            cache.num_stages_layer_copy = 2
        else:
            cache.num_stages_layer_copy = 1
        cache.stage_record = 0
        # Per-stage tracking lists. A decoder block performs MULTIPLE
        # synchronize_d2h_prefill calls (DSA pre_attn, MoMe post_attn,
        # SWA / MLA, ...) all sharing one stage_idx because stage_record
        # only advances at the block boundary. To keep all of those
        # waits visible at the boundary, store a LIST of events/futures
        # per stage_idx; sync_d2h_prefill APPENDS, _moe_post_sync DRAINS.
        # copy_futures legacy behaviour (single Future per stage) is kept
        # for the older _wait_for_pending_d2h reader; the new
        # copy_futures_stages is the authoritative list-of-futures view.
        self.copy_futures: List[Optional[Future]] = [None] * cache.num_stages_layer_copy
        cache.copy_futures_stages: List[dict] = [
            {} for _ in range(cache.num_stages_layer_copy)
        ]
        cache.d2h_event_stages: List[dict] = [
            {} for _ in range(cache.num_stages_layer_copy)
        ]
        cache.h2d_event_stages: List[dict] = [
            {} for _ in range(cache.num_stages_layer_copy)
        ]
        # post-copy event onto cache.d2h_stream; consumed at the layer
        # boundary by MoEAttnPlugin._moe_post_sync to guarantee the HBM
        # source for stage S is no longer being read by the d2h_stream
        # before the next decoder block writes that stage.
        if cache.num_stages_layer_copy > 1 and cache.is_hybrid_attn:
            from vllm.logger import init_logger as _il
            _il('vllm.v1.omni').warning(
                '[PINGPONG] hybrid prefill ping-pong enabled: num_stages_layer_copy=%d',
                cache.num_stages_layer_copy,
            )

        # Attach to cache for backward compatibility
        cache.d2h_thrp = self.d2h_thrp
        cache.copy_future = self.copy_future
        cache.copy_futures = self.copy_futures

        # Attach to cache for backward compatibility
        cache.d2h_thrp = self.d2h_thrp
        cache.copy_future = self.copy_future
        cache.copy_futures = self.copy_futures
