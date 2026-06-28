# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Device cache initialization mixin for PrefillOmniCache."""
import os
from collections import defaultdict
from copy import deepcopy
from typing import List, Optional, Tuple, Union

import torch
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheTensor, UniformTypeKVCacheSpecs
from vllm.logger import init_logger

logger = init_logger("vllm.v1.omni")
from omni_cache.cache.core.constants import ENABLE_C8_INDEXER, LOCAL_DP_SIZE
from omni_cache.cache.utils.ops import divide_or_raise
from omni_npu.worker.npu_model_runner import NPUModelRunner
from omni_cache.cache.transfer_engine import (
    calc_cache_shape_for_prefill,
    calculate_kv_xsfer_params as _calculate_kv_xsfer_params,
)


def _compute_group_max_blocks(kv_cache_config, max_model_len: int,
                              default_block_size: int,
                              kernel_block_sizes=None,
                              max_num_batched_tokens: int = 0,
                              max_num_seqs: int = 1) -> list[int]:
    """Per-group upper bound on blocks per request, counted in *kernel
    blocks* — the unit vLLM's BlockTable actually stores.

    When an attention spec uses hybrid virtual blocks
    (`kernel_block_size < spec.block_size`), each KV-manager block
    becomes `spec.block_size // kernel_block_size` kernel blocks, which
    widens `BlockTable.block_table.np`. Counting in kernel blocks
    matches the true `max_num_blocks_per_req` and avoids leaving the
    expanded tail of block_table un-remapped.

    `kernel_block_sizes` comes from
    `NPUModelRunner._prepare_kernel_block_sizes(kv_cache_config)`; when
    absent, we fall back to `spec.block_size`.

    `max_num_batched_tokens` is a batch-level budget; it is divided by
    `max_num_seqs` to obtain a per-request cap before computing blocks.
    """
    try:
        from vllm.v1.kv_cache_interface import MambaSpec
    except Exception:
        MambaSpec = None  # type: ignore[assignment]

    # Per-request token budget: ceil(batch budget / num_seqs)
    _per_req_tokens = (max_num_batched_tokens + max_num_seqs - 1) // max_num_seqs if max_num_seqs > 0 else 0

    out: list[int] = []
    for gid, group in enumerate(kv_cache_config.kv_cache_groups):
        spec = group.kv_cache_spec
        spec_bs = getattr(spec, "block_size", default_block_size) or default_block_size
        # Prefer kernel_block_size (what BlockTable.block_size is set to
        # in hybrid mode). Fall back to spec.block_size if unavailable.
        if kernel_block_sizes is not None and gid < len(kernel_block_sizes):
            eff_bs = int(kernel_block_sizes[gid]) or int(spec_bs)
        else:
            eff_bs = int(spec_bs)
        if MambaSpec is not None and isinstance(spec, MambaSpec):
            out.append((_per_req_tokens + eff_bs - 1) // eff_bs + 4)
            continue
        sliding_window = getattr(spec, "sliding_window", None)
        if sliding_window:
            max_tokens = _per_req_tokens + int(sliding_window)
            out.append((max_tokens + eff_bs - 1) // eff_bs + 4)
            continue
        out.append((max_model_len + eff_bs - 1) // eff_bs + 4)
    return out




class PrefillDeviceCacheMixin:
    """Device cache initialization for PrefillOmniCache."""

    def _construct_device_cache(self, kv_caches, num_stages=1):
        if not self.is_hybrid_attn:
            device_cache_buf = []
            for i in range(num_stages):
                if isinstance(kv_caches, tuple):
                    curr_kv = tuple(
                            t.clone() if i > 1 else t for t in kv_caches
                        )
                else:
                    curr_kv = kv_caches.clone() if i > 1 else kv_caches
                current_kv_raw = curr_kv
                current_group_dict = {}

                for kv_group in self.kv_cache_config.kv_cache_groups:
                    spec = kv_group.kv_cache_spec
                    if isinstance(current_kv_raw, tuple):
                        # Derive DSA head splits (nope, rope, indexer)
                        # from the real model config — no hardcoded
                        # [512, 64, 128] list.
                        if self.enable_dsa:
                            from omni_cache.cache.utils.support import (
                                _pangu_kv_head_dims_from_hf,
                            )
                            _n, _r, _i = _pangu_kv_head_dims_from_hf(
                                self.vllm_config.model_config.hf_config
                            )
                            head_sizes = [_n, _r, _i]
                        else:
                            head_sizes = self.head_sizes

                        shared_view = []
                        for kvi_idx in range(len(current_kv_raw)):
                            view_shape = (-1, spec.block_size, 1, head_sizes[kvi_idx])
                            shared_view.append(current_kv_raw[kvi_idx].view(*view_shape))
                        shared_view = tuple(shared_view)
                    else:
                        view_shape = (-1, spec.block_size, spec.head_size)
                        shared_view = (current_kv_raw.view(*view_shape),)

                    for layer_name in kv_group.layer_names:
                        current_group_dict[layer_name] = shared_view

                device_cache_buf.append(current_group_dict)
            return device_cache_buf
        if ENABLE_C8_INDEXER:
            device_cache_buf = []
            for i in range(num_stages):
                if isinstance(kv_caches, tuple):
                    curr_kv = tuple(
                        t.clone() if i > 1 else t for t in kv_caches
                    )
                else:
                    curr_kv = kv_caches.clone() if i > 1 else kv_caches
                (current_kv_indexer, current_kv_scale) = curr_kv
                current_group_dict = {}
                for kv_group in self.kv_cache_config.kv_cache_groups:
                    spec = kv_group.kv_cache_spec
                    if hasattr(spec, "separate_kvscale") and spec.separate_kvscale:
                        shared_view = (current_kv_indexer, current_kv_scale)
                    else:
                        view_shape = (-1, spec.block_size, spec.head_size)
                        shared_view = current_kv_indexer.view(
                            dtype=spec.dtype
                        ).view(*view_shape)
                    for layer_name in kv_group.layer_names:
                        current_group_dict[layer_name] = shared_view
                device_cache_buf.append(current_group_dict)
        else:
            from vllm.v1.kv_cache_interface import MambaSpec
            device_cache_buf = []
            for i in range(num_stages):
                current_group_dict = {}
                for kv_group in self.kv_cache_config.kv_cache_groups:
                    # Mamba/MoME groups hold per-layer state across tokens and
                    # must NOT share a single buffer across layers. Attention
                    # groups can share one buffer (each layer's KV is fresh
                    # per forward pass and offloaded via D2H between layers).
                    per_layer = isinstance(kv_group.kv_cache_spec, MambaSpec)

                    def _pick(layer_name):
                        if isinstance(kv_caches, dict):
                            kv = kv_caches[layer_name]
                        else:
                            kv = kv_caches  # backward compat
                        if i > 0:
                            if isinstance(kv, tuple):
                                kv = tuple(t.clone() for t in kv)
                            else:
                                kv = kv.clone()
                        return kv

                    if per_layer:
                        for layer_name in kv_group.layer_names:
                            current_group_dict[layer_name] = _pick(layer_name)
                    else:
                        group_kv = _pick(kv_group.layer_names[0])
                        for layer_name in kv_group.layer_names:
                            current_group_dict[layer_name] = group_kv
                device_cache_buf.append(current_group_dict)
        return device_cache_buf

    def calc_cache_shape(self) -> Tuple[Tuple[int, ...], int]:
        self.tp_node_id = self.tp_rank // LOCAL_DP_SIZE
        if self.tp_world_size > LOCAL_DP_SIZE:
            self.tp_nnodes = divide_or_raise(
                self.tp_world_size, LOCAL_DP_SIZE
            )
        else:
            self.tp_nnodes = 1
        self.node_block_size = divide_or_raise(
            self.block_size, self.tp_nnodes
        )
        # Pangu V2 PD: node_block_size must equal block_size (128)
        # so the OX server layout matches decode. TP splitting on
        # block_size is unnecessary — each TP rank's host pool spans
        # all tokens in the block.
        if self.is_pangu_v2:
            logger.warning(
                "[PANGU-V2-BLOCK-SIZE] calc_cache_shape: block_size=%d, "
                "node_block_size=%d, tp_nnodes=%d, forcing node_block_size=128",
                self.block_size, self.node_block_size, self.tp_nnodes,
            )
            self.node_block_size = 128

        self.rank_block_size = divide_or_raise(
            self.block_size, self.tp_world_size
        )

        shape, num_blocks = calc_cache_shape_for_prefill(
            num_layers=self.num_layers,
            block_size=self.node_block_size,
            head_size=self.head_size,
            dtype=self.dtype,
        )
        if self.dp_world_size_local > 1:
            dp_offset_size = num_blocks // self.dp_world_size_local
            self.host_block_offset = [
                i * dp_offset_size for i in range(self.dp_world_size_local)
            ]
        else:
            self.host_block_offset = [0]
        logger.debug(
            f"++++++++ **PrefillOmniCache**: {self.num_layers=}; "
            f"{shape=}; {num_blocks=}; {self.host_block_offset=}"
        )

        return shape, num_blocks

    def calculate_kv_xsfer_params(self) -> Tuple[int, int]:
        """Calculate KV transfer parameters.

        Delegates to the unified calculate_kv_xsfer_params from h2d module.
        """
        return _calculate_kv_xsfer_params(
            self.shape, self.num_blocks, self.dp_local_rank
        )

    def initialize_device_cache(
        self,
        kv_cache_config: KVCacheConfig,
        runner: NPUModelRunner
    ) -> Optional[Tuple[torch.Tensor]]:
        # Per-group packed layout: each kv_cache_group gets its own slice of
        # the shared aliased raw_tensor. The volatile swap in
        # `PrepareInputsPlugin.post_prepare_inputs` assigns flat fake IDs
        # 1..N across all groups so groups can't collide on shared pages.
        # For Pangu V2 hybrid this makes one buffer fit `max_num_reqs *
        # total_blocks_per_req + 1` blocks instead of the full
        # kv_cache_config.num_blocks.
        # Kernel block sizes may differ from `spec.block_size` via the
        # hybrid-virtual-block path (`_prepare_kernel_block_sizes`).
        # BlockTable.np is sized in *kernel* blocks, and the volatile
        # remap writes into it, so we must size `total_blocks_per_req`
        # against kernel_block_size too.
        try:
            _kernel_block_sizes = runner._prepare_kernel_block_sizes(kv_cache_config)
        except Exception:
            _kernel_block_sizes = None
        _max_bt = runner.vllm_config.scheduler_config.max_num_batched_tokens
        _max_seqs = runner.vllm_config.scheduler_config.max_num_seqs
        group_max_blocks = _compute_group_max_blocks(
            kv_cache_config, runner.max_model_len, self.block_size,
            kernel_block_sizes=_kernel_block_sizes,
            max_num_batched_tokens=_max_bt,
            max_num_seqs=_max_seqs,
        )
        group_offsets = [0]
        for gmb in group_max_blocks:
            group_offsets.append(group_offsets[-1] + gmb)
        total_blocks_per_req = group_offsets[-1]
        # Keep legacy name for downstream call sites that expect it.
        max_num_blocks_per_req = total_blocks_per_req
        num_blocks = runner.max_num_reqs * total_blocks_per_req
        volatile_table = torch.arange(
            1, 1 + num_blocks,
            dtype=torch.int32,
            device=runner.device
        ).view(runner.max_num_reqs, total_blocks_per_req)

        # Fake ID 0 is reserved as the "unallocated" sentinel, so the buffer
        # holds `num_blocks + 1` slots.
        target_num_blocks = num_blocks + 1

        # Stash so volatile swap and D2H scatter can derive each group's
        # slice of the flat fake-ID range.
        self.group_max_blocks = list(group_max_blocks)
        self.group_offsets = list(group_offsets)
        self.total_blocks_per_req = int(total_blocks_per_req)
        logger.warning(
            "[PACKED-HBM] group_max_blocks=%s total_blocks_per_req=%d "
            "max_num_reqs=%d target_num_blocks=%d group_offsets=%s PACKED=%s",
            group_max_blocks, total_blocks_per_req, runner.max_num_reqs,
            target_num_blocks, self.group_offsets, os.getenv("OMNI_CACHE_PACKED_HBM", "0"),
        )

        if self.is_hybrid_attn:
            scale_rate = (
                len(kv_cache_config.kv_cache_groups[0].layer_names)
                // (self.num_stages_layer_copy * 4)
            )
        else:
            scale_rate = self.num_layers // (self.num_stages_layer_copy * 4)

        # record the original raw tensors
        self.device_raw_tensors = []
        self.page_size_bytes = 0  # set below for non-hybrid non-C8 path

        if ENABLE_C8_INDEXER:
            # TODO(runze) adapt C8 in the future
            kv_cache_layer_indexer, kv_cache_layer_scale = (
                self._build_device_kv_cache_layer(kv_cache_config, runner)
            )

            shape_indexer = list(kv_cache_layer_indexer.shape)
            shape_indexer[0] = min(
                kv_cache_layer_indexer.shape[0] * scale_rate,
                self.num_blocks // self.dp_world_size_local
            )
            kv_cache_fake_indexer = kv_cache_layer_indexer.new_zeros(
                shape_indexer
            )

            shape_scale = list(kv_cache_layer_scale.shape)
            shape_scale[0] = min(
                kv_cache_layer_scale.shape[0] * scale_rate,
                self.num_blocks // self.dp_world_size_local
            )
            kv_cache_fake_scale = kv_cache_layer_scale.new_zeros(shape_scale)
            kv_caches = self._construct_device_cache(
                (kv_cache_fake_indexer, kv_cache_fake_scale),
                self.num_stages_layer_copy
            )
        else:
            kv_cache_groups = kv_cache_config.kv_cache_groups
            # HBM sizing modes (opt-in independently from the swap so we
            # can narrow accuracy issues):
            #   OMNI_CACHE_PACKED_HBM=1 → shrink raw_tensor to the
            #     `num_blocks + 1` volatile-working-set size. Only
            #     safe when the swap keeps kernels inside that range.
            #   default → keep the legacy
            #     `min(orig_num_blocks * scale_rate, num_blocks /
            #     dp_world_size_local)` sizing (works with or without
            #     the swap because real scheduler block_ids always
            #     fit).
            if not int(os.getenv("OMNI_CACHE_PACKED_HBM", "0")):
                orig_num_blocks = kv_cache_config.num_blocks
                target_num_blocks = min(
                    orig_num_blocks * scale_rate,
                    self.num_blocks // self.dp_world_size_local,
                )
                logger.warning(
                    "[PACKED-HBM] LEGACY mode: orig_num_blocks=%d "
                    "scale_rate=%d target_num_blocks=%d",
                    orig_num_blocks, scale_rate, target_num_blocks,
                )
            else:
                logger.warning(
                    "[PACKED-HBM] PACKED mode: target_num_blocks=%d "
                    "= %d * %d + 1",
                    target_num_blocks, runner.max_num_reqs,
                    total_blocks_per_req,
                )

            # Determine how model runners should initialize the KV cache tensors.
            if len(kv_cache_groups) == 1 and isinstance(
                kv_cache_groups[0].kv_cache_spec, UniformTypeKVCacheSpecs
            ):
                # Special case: DeepSeek V3.2 (vLLM)
                # NOTE(runze): currently impossible to come into this branch
                per_layer_specs = kv_cache_groups[0].kv_cache_spec.kv_cache_specs
                layer_idx2names: dict[int, list[str]] = defaultdict(list)
                for layer_name in per_layer_specs.keys():
                    # group layer names by layer index
                    layer_idx2names[extract_layer_index(layer_name)].append(layer_name)
                first_layer_names = next(iter(layer_idx2names.values()))
                assert all([len(names) == len(first_layer_names) for names in layer_idx2names.values()]), \
                    f"For UniformTypeKVCacheSpecs, not all layers have same number of attention sublayers."
                device_cache_size = sum(  # sum or tuple?
                    [per_layer_specs[ln].page_size_bytes * target_num_blocks for ln in first_layer_names]
                )
            else:
                # General case
                uniform_page_size = kv_cache_groups[0].kv_cache_spec.page_size_bytes
                self.page_size_bytes = uniform_page_size
                device_cache_size = uniform_page_size * target_num_blocks

            kv_caches = []
            for _i in range(self.num_stages_layer_copy):
                raw_tensor = torch.zeros(
                    device_cache_size, dtype=torch.int8, device=self.device
                )
                kv_cache_raw_tensors = {}
                for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
                    # map all layers to the same underlying raw tensor
                    for layer_name in kv_cache_tensor.shared_by:
                        kv_cache_raw_tensors[layer_name] = raw_tensor
                kernel_block_sizes = runner._prepare_kernel_block_sizes(kv_cache_config)
                reshaped_kv_caches = runner._reshape_kv_cache_tensors(
                    kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes
                )
                self.device_raw_tensors.append(
                    raw_tensor.view(target_num_blocks, -1).view(torch.bfloat16)
                )
                kv_caches.append(reshaped_kv_caches)
        self.volatile_table = volatile_table
        return kv_caches

    def _build_device_kv_cache_layer(self, kv_cache_config_fake, runner):
        kv_cache_raw_tensors = runner._allocate_kv_cache_tensors(
            kv_cache_config_fake
        )
        kernel_block_sizes = runner._prepare_kernel_block_sizes(
            kv_cache_config_fake
        )
        kv_caches = runner._reshape_kv_cache_tensors(
            kv_cache_config_fake, kv_cache_raw_tensors, kernel_block_sizes
        )

        if ENABLE_C8_INDEXER:
            for cache_item in kv_caches.values():
                if isinstance(cache_item, tuple):
                    return cache_item
            raise RuntimeError(
                f"Failed to find any quantized KV Cache (tuple format) in layers. "
                f"Available layer count: {len(kv_caches)}. "
                f"Check if the model's quantization spec is correctly configured."
            )
        else:
            return kv_caches  # dict[str, Union[Tensor, Tuple[Tensor, ...]]]

