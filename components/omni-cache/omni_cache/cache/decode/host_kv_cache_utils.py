# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Host-side KV cache utilities for decode cache.

This module contains functions for parsing and managing KV cache data
in host (CPU) memory, including PA (PagedAttention) format parsing.

Dispatch logic:
  - MambaSpec / MomeSpec: byte-packed state tensors, shapes from spec.shapes
  - DSAAttentionSpec: (nope_and_rope, indexer) split
  - ShareKVSlidingWindowSpec / SlidingWindowSpec: (nope, rope) split
  - Other AttentionSpec: single component
"""

import os
from math import prod
from typing import Dict, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    import torch


# ---------------------------------------------------------------------------
# Spec-type detection (lazy imports to avoid circular deps at module load)
# ---------------------------------------------------------------------------

def _import_spec_types():
    """Lazily import spec types; returns a namespace-like dict."""
    try:
        from vllm.v1.kv_cache_interface import (
            AttentionSpec, MambaSpec, FullAttentionSpec,
            MLAAttentionSpec, SlidingWindowSpec,
        )
    except ImportError:
        return None
    try:
        from vllm.v1.kv_cache_interface import DSAAttentionSpec
    except ImportError:
        DSAAttentionSpec = None
    try:
        from omni_npu.vllm_patches.patches.models.pangu_v2_hybrid.patch_kv_cache_interface import (
            MomeSpec, ShareKVSlidingWindowSpec,
        )
    except ImportError:
        MomeSpec = None
        ShareKVSlidingWindowSpec = None
    return {
        "AttentionSpec": AttentionSpec,
        "MambaSpec": MambaSpec,
        "MomeSpec": MomeSpec,
        "DSAAttentionSpec": DSAAttentionSpec,
        "FullAttentionSpec": FullAttentionSpec,
        "MLAAttentionSpec": MLAAttentionSpec,
        "SlidingWindowSpec": SlidingWindowSpec,
        "ShareKVSlidingWindowSpec": ShareKVSlidingWindowSpec,
    }


# ---------------------------------------------------------------------------
# Head-size resolution for attention specs
# ---------------------------------------------------------------------------

def _resolve_attention_head_sizes(spec, host_cache, hf_config=None) -> list[int]:
    """Derive per-component head sizes from an AttentionSpec.

    The unified host pool has shape (num_layers, num_blocks, block_size,
    head_dim) where head_dim is the effective slot width from
    page_size_padded.  Each attention sub-type splits this head_dim
    differently.

    Args:
        spec: An AttentionSpec (or subclass) instance.
        host_cache: kvi_tensors_swap list — used to read the actual
            head_dim of the pool.

    Returns:
        List of per-component head sizes whose elements sum to ≤
        host-cache head_dim.
    """
    types = _import_spec_types()
    if types is None:
        return [host_cache[0].shape[-1]]

    DSAAttentionSpec = types["DSAAttentionSpec"]
    ShareKVSlidingWindowSpec = types["ShareKVSlidingWindowSpec"]
    SlidingWindowSpec = types["SlidingWindowSpec"]
    MLAAttentionSpec = types["MLAAttentionSpec"]

    actual_head_dim = host_cache[0].shape[-1]
    spec_head_size = getattr(spec, "head_size", actual_head_dim)

    if DSAAttentionSpec is not None and isinstance(spec, DSAAttentionSpec):
        # For Pangu V2 DSA: indexer head_size is 128.
        # Try to read from spec.head_sizes (future-proof), else derive.
        head_sizes_attr = getattr(spec, "head_sizes", None)
        if head_sizes_attr is not None:
            return list(head_sizes_attr)
        indexer_dim = None
        if hf_config is not None:
            indexer_dim = (
                getattr(hf_config, "index_head_dim", None)
                or getattr(hf_config, "indexer_head_dim", None)
            )
        if not indexer_dim:
            from omni_cache.cache.memory.constants import INDEXER_HEAD_DIM
            indexer_dim = INDEXER_HEAD_DIM
        nope_rope_dim = spec_head_size - indexer_dim
        if nope_rope_dim <= 0:
            nope_rope_dim = spec_head_size
            indexer_dim = 0
        return [nope_rope_dim, indexer_dim] if indexer_dim > 0 else [nope_rope_dim]

    # ShareKVSlidingWindowSpec / SlidingWindowSpec: [nope, rope]
    is_swa = (ShareKVSlidingWindowSpec is not None
              and isinstance(spec, ShareKVSlidingWindowSpec))
    is_sliding = isinstance(spec, SlidingWindowSpec)
    if is_swa or is_sliding:
        # MLA-style SWA: the kept slot is [c_kv_latent, rope] per token
        # where c_kv = kv_lora_rank and rope = qk_rope_head_dim.
        # Read both from hf_config when the caller has supplied it;
        # otherwise fall back to the historical defaults (512, 64).
        _kv_lora = None
        _rope = None
        if hf_config is not None:
            _kv_lora = getattr(hf_config, "kv_lora_rank", None)
            _rope = getattr(hf_config, "qk_rope_head_dim", None)
        _kv_lora = _kv_lora or 512
        _rope = _rope or 64
        if spec_head_size >= _kv_lora + _rope:
            return [_kv_lora, _rope]
        return [spec_head_size]

    # MLAAttentionSpec
    if MLAAttentionSpec is not None and isinstance(spec, MLAAttentionSpec):
        head_size_v = getattr(spec, "head_size_v", None)
        if head_size_v is not None and head_size_v > 0:
            return [spec_head_size, head_size_v]
        return [spec_head_size]

    # Default (FullAttentionSpec / generic AttentionSpec)
    return [spec_head_size]


# ---------------------------------------------------------------------------
# View helpers
# ---------------------------------------------------------------------------

def _host_cache_view(
    host_cache,
    head_sizes: list[int],
    block_size: int,
    layer_idx: int,
    dtype,
) -> list:
    """Create strided views of the unified host pool for an attention layer.

    Uses the *actual* host-cache layout for strides (block_size × head_dim)
    so that layers whose component head_sizes don't fill the full slot still
    address the right bytes.

    Args:
        host_cache: kvi_tensors_swap list; host_cache[0] has shape
            (num_layers, num_blocks, actual_block_size, actual_head_dim).
        head_sizes: Per-component head sizes [h0, h1, ...].
        block_size: Token-block size from the spec (must equal
            actual_block_size for attention types).
        layer_idx: Index into the num_layers dimension.
        dtype: Target dtype for the view.

    Returns:
        List of tensor views, one per component, each shaped
        (num_blocks, block_size, h_i).
    """
    num_layers, num_blocks, actual_block_size, actual_head_dim = host_cache[0].shape
    layer_buffer = host_cache[0][layer_idx]

    head_offsets = []
    offset = 0
    for h in head_sizes:
        head_offsets.append(offset)
        offset += h

    # Block stride must match the host pool's physical layout.
    block_stride = actual_block_size * actual_head_dim

    layer_tensor = []
    for i, h in enumerate(head_sizes):
        head_view = layer_buffer.as_strided(
            size=(num_blocks, block_size, h),
            stride=(block_stride, h, 1),
            storage_offset=layer_buffer.storage_offset() + head_offsets[i],
        )
        if head_view.dtype != dtype and head_view.element_size() == dtype.itemsize:
            head_view = head_view.view(dtype)
        layer_tensor.append(head_view)
    return layer_tensor


def _host_cache_view_mamba(host_cache, spec, layer_idx: int) -> list:
    """Slice packed MoME/Mamba state views from the unified host pool.

    All layout parameters are derived from `spec` and the actual host
    tensor (no hardcoded block_size / num_total_tokens / shape):

      * `spec.num_total_tokens`        — tokens packed per block
                                         (MomeSpec = kernel_size - 1 +
                                         num_speculative_tokens; plain
                                         MambaSpec is effectively 1).
      * `spec.shapes`                  — per-token feature shape of each
                                         state (e.g. ((1024,), (512,),
                                         (6144,)) for Pangu V2 MoMe).
      * `spec.dtypes` / `spec.dtype`   — per-state element dtypes.
      * `slot_bytes_per_block`         — read from `flat_bytes.shape[1]`;
                                         this is the host pool's real
                                         per-block byte width, not a
                                         hardcoded value.

    Packing mirrors `_copy_mome_states_to_host_unified` on the prefill
    side: states are written contiguously starting at offset 0 of each
    block slot; each state occupies
    ``prod(shape_i) * dtype_bytes * num_total_tokens`` bytes.

    If the pool slot is smaller than that total, the allocator and the
    current MomeSpec disagree (e.g. after an omni-npu MoMe version
    bump). We log once and return correctly-shaped zero views so the
    decode pipeline stays well-typed and can boot; callers must not
    treat zeroed views as valid KV transport.
    """
    import torch
    from vllm.utils.torch_utils import get_dtype_size

    layer_data = host_cache[0][layer_idx]
    num_blocks = layer_data.shape[0]

    flat = layer_data.reshape(num_blocks, -1)
    flat_bytes = flat.view(torch.uint8).reshape(num_blocks, -1)
    slot_bytes_per_block = flat_bytes.shape[1]

    types = _import_spec_types()
    mome_spec_type = types["MomeSpec"] if types is not None else None
    is_mome = mome_spec_type is not None and isinstance(spec, mome_spec_type)
    num_total_tokens = spec.num_total_tokens if is_mome else 1

    # Plan per-state byte widths purely from spec.
    state_plans = []
    total_needed = 0
    for i, shape in enumerate(spec.shapes):
        dtype_i = spec.dtypes[i] if i < len(spec.dtypes) else spec.dtype
        dtype_bytes = get_dtype_size(dtype_i)
        inner_numel = prod(shape)
        byte_width = inner_numel * dtype_bytes * num_total_tokens
        state_plans.append((tuple(shape), dtype_i, byte_width))
        total_needed += byte_width

    if total_needed > slot_bytes_per_block:
        import logging
        logging.getLogger("vllm.v1.omni").warning(
            "[mome-host-view] layer=%d slot_bytes=%d needed=%d "
            "num_total_tokens=%d shapes=%s — returning zeros "
            "(host-pool slot cannot hold this MomeSpec packing).",
            layer_idx, slot_bytes_per_block, total_needed,
            num_total_tokens, [tuple(s) for s in spec.shapes],
        )
        return [
            torch.zeros(
                (num_blocks, num_total_tokens, *shape),
                dtype=dtype_i,
                device=layer_data.device,
            )
            for shape, dtype_i, _ in state_plans
        ]

    result = []
    running_off = 0
    for shape, dtype_i, byte_width in state_plans:
        state_bytes = flat_bytes[:, running_off:running_off + byte_width]
        state_view = state_bytes.view(dtype_i).reshape(
            num_blocks, num_total_tokens, *shape,
        )
        result.append(state_view)
        running_off += byte_width
    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def parse_pa_kv_cache(decode_cache, host_cache) -> Dict[str, tuple]:
    """Parse PA-format KV caches from shared memory.

    Converts raw host memory tensors into properly shaped and typed
    KV cache tensors for each layer.  Dispatches by kv_cache_spec type:
      - MambaSpec / MomeSpec → byte-packed states via _host_cache_view_mamba
      - AttentionSpec subtypes → stride-based views via _host_cache_view

    Args:
        decode_cache: DecodeOmniCache instance with configuration.
        host_cache: Host cache tensors (kvi_tensors_swap) from shared memory.

    Returns:
        Dict mapping layer_name → tuple of tensors.  Tuple length depends
        on the spec type:
          - MoME: (q_lora, kv_lora, o_proj) — 3 tensors
          - DSA:  (nope_and_rope, indexer) — 2 tensors
          - SWA:  (nope, rope) — 2 tensors
          - Other attention: (kv,) — 1 tensor
    """
    import torch

    types = _import_spec_types()
    if types is None:
        return {}

    MambaSpec = types["MambaSpec"]
    AttentionSpec = types["AttentionSpec"]

    group_cache: Dict[str, tuple] = {}
    for _group_idx, group_config in enumerate(decode_cache.kv_cache_config.kv_cache_groups):
        spec = group_config.kv_cache_spec

        if isinstance(spec, MambaSpec):
            # MoME / Mamba: byte-packed state tensors.
            for layer_idx, layer_name in enumerate(group_config.layer_names):
                layer_tensor = _host_cache_view_mamba(host_cache, spec, layer_idx)
                group_cache[layer_name] = tuple(layer_tensor)
        elif isinstance(spec, AttentionSpec):
            # Attention types (DSA, SWA, MLA, Full, etc.)
            has_separate_kvscale = getattr(spec, "separate_kvscale", False)

            if has_separate_kvscale:
                # Quantized path: host_cache has multiple kvi components;
                # component 0 = data, component 1 = scale.
                _parse_attention_with_kvscale(
                    host_cache, spec, group_config, decode_cache,
                    group_cache,
                )
            else:
                _mc = getattr(decode_cache.vllm_config, "model_config", None)
                _hf = getattr(_mc, "hf_config", None) if _mc else None
                head_sizes = _resolve_attention_head_sizes(spec, host_cache, hf_config=_hf)
                block_size = spec.block_size
                for layer_idx, layer_name in enumerate(group_config.layer_names):
                    layer_tensor = _host_cache_view(
                        host_cache, head_sizes, block_size, layer_idx, spec.dtype,
                    )
                    # DSA-SPLIT: redirect nope+rope view to secondary host pool
                    # When the DSA split feature is on, component 0 (nope+rope,
                    # 576-byte head) of DSA layers is backed by decode_cache's
                    # secondary dsa_host_pool — a narrower hugetlbfs buffer that
                    # is MMU-aliased to HBM. Component 1 (indexer) still comes
                    # from a separately allocated HBM buffer.
                    _dsa_pool = getattr(decode_cache, 'dsa_host_pool', None)
                    _is_dsa_spec = False
                    try:
                        _spec_types = _import_spec_types() or {}
                        _DSA = _spec_types.get('DSAAttentionSpec')
                        _is_dsa_spec = (_DSA is not None and isinstance(spec, _DSA))
                    except Exception:
                        _is_dsa_spec = False
                    if _dsa_pool is not None and _is_dsa_spec:
                        _sec = getattr(_dsa_pool, 'device_tensor', None)
                        if _sec is None:
                            _sec = getattr(_dsa_pool, 'shared_tensor', None)
                        if _sec is not None and layer_idx < _sec.shape[0]:
                            _sub = _sec[layer_idx][..., :_dsa_pool.head_size]
                            if _sub.dtype != spec.dtype:
                                _sub = _sub.view(spec.dtype)
                            layer_tensor = [_sub] + list(layer_tensor)[1:]
                    group_cache[layer_name] = tuple(layer_tensor)
        else:
            # Unknown spec type — fall back to basic view.
            for layer_idx, layer_name in enumerate(group_config.layer_names):
                head_sizes = [host_cache[0].shape[-1]]
                block_size = getattr(spec, "block_size", host_cache[0].shape[2])
                dtype = getattr(spec, "dtype", torch.bfloat16)
                layer_tensor = _host_cache_view(
                    host_cache, head_sizes, block_size, layer_idx, dtype,
                )
                group_cache[layer_name] = tuple(layer_tensor)

    return group_cache


def _parse_attention_with_kvscale(
    host_cache,
    spec,
    group_config,
    decode_cache,
    group_cache: dict,
) -> None:
    """Handle attention layers that carry a separate kvscale tensor."""
    dtype_bytes = spec.dtype.itemsize if hasattr(spec.dtype, "itemsize") else 2
    page_padded = getattr(spec, "page_size_padded", None)

    for layer_idx, layer_name in enumerate(group_config.layer_names):
        if page_padded is None or page_padded <= 0:
            tensor_shape = (decode_cache.num_blocks, spec.block_size, spec.head_size)
        else:
            slot_elems = page_padded // dtype_bytes
            inner = slot_elems // spec.block_size
            tensor_shape = (decode_cache.num_blocks, spec.block_size, inner)

        layer_tensor = []
        for idx_kvi, _ in enumerate(host_cache):
            if idx_kvi == 0:
                layer_tensor.append(
                    host_cache[idx_kvi][layer_idx]
                    .view(dtype=spec.dtype)
                    .view(tensor_shape)
                )
            else:
                # Scale tensor — typically small (block_size-level granularity).
                # Use the first layer's scale shape as template.
                scale_dtype = getattr(spec, "separate_kvscale", spec.dtype)
                scale_shape = (
                    tensor_shape[0],
                    getattr(spec, "block_size", 128),
                    tensor_shape[2],
                )
                layer_tensor.append(
                    host_cache[idx_kvi]
                    .view(dtype=scale_dtype)
                    .view(scale_shape)
                )
        group_cache[layer_name] = tuple(layer_tensor)
