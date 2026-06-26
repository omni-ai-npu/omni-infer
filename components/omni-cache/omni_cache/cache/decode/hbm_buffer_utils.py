# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""HBM buffer utilities for decode cache.

On the decode side every attention / MoME layer owns a dedicated contiguous
HBM buffer that persists for the whole decode lifetime of a request: we H2D
the request's KV / state once at prefill-to-decode handoff, and the many
decode rounds that follow read and write the same HBM storage.

Sizing policy is dispatched by `isinstance(spec, ...)` because the six Pangu
V2 groups land in an order (MoME, MoME, MoME, DSA, SWA, SWA) that doesn't
match any fixed grp_idx heuristic.

Per-type sizing:
- MambaSpec / MomeSpec: one per-request state slot per layer (no paging).
- SlidingWindowSpec: ceil(sliding_window / block_size) + 1 blocks per req.
- Everything else (DSA / MLA / full attention): ceil(max_model_len /
  block_size) blocks per req — enough to cover the full sequence.

Each group also gets a metadata dict with:
    {"kind", "block_size", "req_offset", "block_table", "block_table_ts"}
"""
import dataclasses

import torch
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    AttentionSpec,
    DSAAttentionSpec,
    ShareKVSlidingWindowSpec,
    MomeSpec,
    SlidingWindowSpec,
    MambaSpec,
)
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from omni_npu.attention.backends.utils import _maybe_padded_raw_tensor_to_strided_caches

logger = init_logger(__name__)


def _is_mome_spec(spec) -> bool:
    return isinstance(spec, MomeSpec)


def _is_attention_spec(spec) -> bool:
    return isinstance(spec, AttentionSpec)


def _is_sliding_window_spec(spec) -> bool:
    return isinstance(spec, SlidingWindowSpec)


def _req_offset_for_spec(spec: KVCacheSpec, max_model_len: int) -> int:
    """Number of HBM block slots reserved per request in this attention group."""
    if _is_sliding_window_spec(spec):
        # The slot_mapping formula in mla_ext._recompute_decode_slot_mapping is
        #   slot = (num_computed - num_leading*block_size) + block_size + lane_base
        # which hovers near sliding_window + block_size (up to 2*block_size-1
        # extra from the block_size addition). We need at least
        # ceil(sliding_window/block_size) + 1 blocks for the window, plus one
        # more for the leading null/rotating block so the addressed slot stays
        # inside this request lane instead of spilling into the next lane.
        return max(1, (spec.sliding_window + spec.block_size - 1) // spec.block_size + 1)
    # DSA / MLA / full attention: cover the full sequence.
    return max(1, (max_model_len + spec.block_size - 1) // spec.block_size + 1)


def _alloc_attention(cache_obj, spec: AttentionSpec, num_blocks: int, device):
    """Block-paged HBM buffer for one attention layer (+ optional scale buddy).

    For DSAAttentionSpec with ENABLE_HOST_MAPPING, the nope+rope data is
    read from host-mapped memory (via NPU MMU).  The HBM buffer only needs
    to hold the indexer component, dramatically reducing HBM usage.
    """
    if not isinstance(spec, AttentionSpec):
        raise ValueError(f"{type(spec)} is not a subclass of AttentionSpec.")
    block_size = spec.block_size
    hf_cfg = get_current_vllm_config().model_config.hf_config

    def _derive_attention_head_sizes(spec, hf_config) -> list[int]:
        """Derive per-component head sizes from an attention spec and actual HBM head_dim.

        Mirrors the logic in host_kv_cache_utils._resolve_attention_head_sizes
        so HBM buffer layout matches the host-side view.
        """
        if isinstance(spec, DSAAttentionSpec):
            indexer_dim = (
                getattr(hf_config, "index_head_dim", None)
                or getattr(hf_config, "indexer_head_dim", None)
            )
            if not indexer_dim:
                from omni_cache.cache.memory.constants import INDEXER_HEAD_DIM
                indexer_dim = INDEXER_HEAD_DIM
            return [indexer_dim]

        # ShareKVSlidingWindowSpec (MLA-style SWA): [c_kv_latent, rope].
        # Both read from hf_config when available; fall back to the
        # DeepSeek-V3 defaults (512, 64).
        if isinstance(spec, ShareKVSlidingWindowSpec):
            kv_lora = getattr(hf_config, "kv_lora_rank", 512)
            rope_dim = getattr(hf_config, "qk_rope_head_dim", 64)
            return [kv_lora, rope_dim]

        return [spec.head_size]

    head_sizes = _derive_attention_head_sizes(spec, hf_cfg)
    # FIXME(runze): using unpadded page_size_bytes here causes accuracy bugs, which should be fixed
    page_size_bytes = block_size * head_sizes[0] * spec.dtype.itemsize \
        if isinstance(spec, DSAAttentionSpec) else spec.page_size_bytes
    tensor_raw = torch.zeros(num_blocks * page_size_bytes, dtype=torch.int8, device=device)
    if spec.num_kv_heads == 1:
        shapes = tuple([(hs,) for hs in head_sizes])
    else:
        shapes = tuple([(spec.num_kv_heads, hs) for hs in head_sizes])
    dtypes = (spec.dtype,) * len(head_sizes)

    kv_caches = _maybe_padded_raw_tensor_to_strided_caches(
        tensor_raw,
        num_blocks=num_blocks,
        block_size=block_size,
        shapes=shapes,
        dtypes=dtypes,
        page_size_bytes=page_size_bytes,
    )

    scale_dtype = getattr(spec, "separate_kvscale", None)
    if scale_dtype:
        scale = torch.zeros((num_blocks, block_size), dtype=scale_dtype, device=device).contiguous()
        kv_caches = (*kv_caches, scale)
    return kv_caches, tensor_raw.view(spec.dtype).view(num_blocks, -1)


def _alloc_mome(cache_obj, spec, num_blocks, device):
    """Per-request state tensors for one MoME layer."""
    if not isinstance(spec, MomeSpec):
        raise ValueError(f"{type(spec)} is not a subclass of MomeSpec.")

    assert len(set(spec.dtypes)) == 1, f"MomeSpec with different dtypes {spec.dtypes} for sub-components is not supported yet."
    # FIXME(runze): using unpadded page_size_bytes here causes accuracy bugs, which should be fixed
    page_size_bytes = spec.page_size_bytes
    tensor_raw = torch.zeros(num_blocks * page_size_bytes, dtype=torch.int8, device=device)

    conv_states = _maybe_padded_raw_tensor_to_strided_caches(
        tensor_raw,
        num_blocks=num_blocks,
        block_size=spec.num_total_tokens,
        shapes=spec.shapes,
        dtypes=spec.dtypes,
        page_size_bytes=page_size_bytes,
    )
    return conv_states, tensor_raw.view(spec.dtypes[0]).view(num_blocks, -1)


def construct_hbm_buffer(decode_cache) -> tuple:
    """Construct decode-side HBM buffers and per-group block-table metadata.

    Returns:
        (hbm_buffer_pool, hbm_buffer_block_table_pool) — two parallel lists,
        one entry per kv_cache_group.
    """
    max_num_reqs = decode_cache.num_max_batch_pool
    max_model_len = decode_cache.vllm_config.model_config.max_model_len
    device = decode_cache.device

    hbm_buffer_pool: list = []
    hbm_buffer_block_table_pool: list = []

    for group_config in decode_cache.kv_cache_config.kv_cache_groups:
        spec = group_config.kv_cache_spec
        group_cache: dict = {}
        group_cache_raw: dict = {}

        if _is_mome_spec(spec):
            req_offset = 1
            num_blocks = max_num_reqs * req_offset + 1  # FIXME(runze): think of a better way to adjust number of device blocks
            for layer_name in group_config.layer_names:
                group_cache[layer_name], group_cache_raw[layer_name] = _alloc_mome(decode_cache, spec, num_blocks, device)
            block_pool = {
                "kind": "mamba",
                "req_offset": 1,
                "hbm_buffer_pool_raw": group_cache_raw,
                "block_size": spec.num_total_tokens,
                "block_table": list(range(max_num_reqs)),  # FIXME: should start from 1
            }
        elif _is_attention_spec(spec):
            req_offset = _req_offset_for_spec(spec, max_model_len=max_model_len)
            num_blocks = max_num_reqs * req_offset + 1  # FIXME(runze): think of a better way to adjust number of device blocks
            # for dsa, indexer cache and kv cache share the same kv group, need make them the same block number
            if isinstance(spec, DSAAttentionSpec):
                num_blocks = decode_cache.num_blocks
            for layer_name in group_config.layer_names:
                group_cache[layer_name], group_cache_raw[layer_name] = _alloc_attention(decode_cache, spec, num_blocks, device)
            block_pool = {
                "kind": "attention",
                "req_offset": req_offset,
                "hbm_buffer_pool_raw": group_cache_raw,
                "block_size": spec.block_size,
                "block_table": list(range(1, num_blocks)),  # FIXME: should be `num_blocks+1`
            }
        else:
            raise NotImplementedError(f"Spec {spec} is not supported yet.")

        logger.debug(
            "[HBM-BUFFER] spec=%s len(group_cache_raw)=%d layer_shapes=%s raw_shape=%s",
            spec, len(group_cache_raw),
            [_t.shape for _t in group_cache[layer_name]],
            group_cache_raw[layer_name].shape,
        )

        # NOTE: block_table_ts is only for DSV4 compress attn metadata build
        block_pool["block_table_ts"] = torch.tensor(
            block_pool["block_table"], dtype=torch.int32, device=device
        )
        hbm_buffer_pool.append(group_cache)
        hbm_buffer_block_table_pool.append(block_pool)

    return hbm_buffer_pool, hbm_buffer_block_table_pool
