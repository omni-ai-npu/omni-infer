# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""KV-cache reshape adaptation for NPU attention backends.

Injected by omni/vllm_patches/usefull_patch/patch_mrv2_attn_utils.py.
"""

from __future__ import annotations

import torch

from vllm.v1.worker.gpu import attn_utils as up_attn_utils

# Captured at import, which the plugin performs before any patch is applied.
_ORIGINAL = up_attn_utils._reshape_kv_cache


def reshape_kv_cache(
    attn_groups,
    kv_cache_raw_tensors: dict[str, torch.Tensor],
    cache_dtype: str,
    kernel_block_sizes: list[int],
    shared_kv_cache_layers: dict[str, str],
    kv_cache_config=None,
) -> dict:
    """Run the upstream reshape, then delegate supported groups to the backend."""
    kv_caches = _ORIGINAL(
        attn_groups=attn_groups,
        kv_cache_raw_tensors=kv_cache_raw_tensors,
        cache_dtype=cache_dtype,
        kernel_block_sizes=kernel_block_sizes,
        shared_kv_cache_layers=shared_kv_cache_layers,
        kv_cache_config=kv_cache_config,
    )

    for group in attn_groups:
        hook = getattr(getattr(group, "backend", None), "reshape_kv_cache", None)
        if hook is None:
            continue
        spec = group.kv_cache_spec
        for layer_name in group.layer_names:
            if layer_name in shared_kv_cache_layers:
                raise RuntimeError(
                    f"[omni-npu/mrv2] backend reshape does not support shared "
                    f"KV-cache layer {layer_name}"
                )
            raw = kv_cache_raw_tensors[layer_name]
            if raw.numel() % spec.page_size_bytes:
                raise ValueError(
                    f"{layer_name}: raw size {raw.numel()} is not divisible by "
                    f"page size {spec.page_size_bytes}"
                )
            num_blocks = raw.numel() // spec.page_size_bytes
            kv_caches[layer_name] = hook(raw, num_blocks, spec)

    return kv_caches
