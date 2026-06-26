# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Layout-aware KV cache extraction for each transfer stage.

The core function ``extract_block`` returns a canonical (block_size, head_dim_i)
numpy array regardless of which physical location the data comes from.
"""

import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from omni_cache.cache.prefill.prefill_omni_cache import PrefillOmniCache
    from omni_cache.cache.decode.decode_omni_cache import DecodeOmniCache

logger = init_logger("vllm.v1.omni")

# Pangu V2 component offsets within the unified head_dim=704 slot.
COMPONENT_OFFSETS = {
    "nope": (0, 512),
    "rope": (512, 576),
    "indexer": (576, 704),
}

COMPONENT_NAMES = ["nope", "rope", "indexer"]

# DSA-SPLIT secondary pool: nope+rope merged into one 576-wide component.
COMPONENT_OFFSETS_DSA_SPLIT = {
    "nope_rope": (0, 576),
    "indexer": (576, 704),
}

MOME_COMPONENT_NAMES = ["mome_state_0", "mome_state_1", "mome_state_2"]


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Convert a CPU tensor to numpy, handling bfloat16 (unsupported by numpy)."""
    if t.dtype == torch.bfloat16:
        t = t.float()
    return t.numpy()


def _get_group_kind(kv_cache_config, group_idx: int) -> str:
    spec = kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec
    name = type(spec).__name__
    if "Mome" in name or "Mamba" in name:
        return "mome"
    if "DSA" in name:
        return "dsa"
    if "Sliding" in name:
        return "swa"
    if "Omni" in name:
        return "omni"
    return "full"


def _get_component_names(group_kind: str, dsa_split: bool = False) -> List[str]:
    if group_kind == "mome":
        return MOME_COMPONENT_NAMES
    if dsa_split and group_kind == "dsa":
        return ["nope_rope", "indexer"]
    return COMPONENT_NAMES


def _slice_component(
    arr: np.ndarray,
    component: str,
    group_kind: str,
    dsa_split: bool = False,
) -> np.ndarray:
    if group_kind == "mome":
        return arr
    if dsa_split and group_kind == "dsa":
        start, end = COMPONENT_OFFSETS_DSA_SPLIT.get(component, (0, arr.shape[-1]))
        return arr[..., start:end]
    start, end = COMPONENT_OFFSETS.get(component, (0, arr.shape[-1]))
    return arr[..., start:end]


def extract_block_prefill_host(
    cache: "PrefillOmniCache",
    group_idx: int,
    virtual_layer_idx: int,
    block_id: int,
    component_idx: int,
) -> np.ndarray:
    """Extract raw block from prefill host pool - dumps entire block, not components."""
    host_cache = cache.host_cache
    kvi = getattr(host_cache, "kvi_tensors", None)

    dp_rank = getattr(cache, "dp_local_rank", 0) or 0

    if kvi and len(kvi) > 0:
        pool = kvi[0]
        # 4D prefill: (num_layers, num_blocks, block_size, head_dim)
        # or after dp sharding: list of per-layer tensors
        if isinstance(pool, (list, tuple)):
            layer_tensor = pool[virtual_layer_idx]
            if layer_tensor.dim() == 4:
                # (dp_world, blocks_per_rank, block_size, head_dim)
                row = layer_tensor[dp_rank, block_id].cpu().float().numpy()
            else:
                row = layer_tensor[block_id].cpu().float().numpy()
        elif pool.dim() == 4:
            row = pool[virtual_layer_idx, block_id].cpu().float().numpy()
        elif pool.dim() == 5:
            row = pool[dp_rank, virtual_layer_idx, block_id].cpu().float().numpy()
        else:
            row = pool[virtual_layer_idx, block_id].cpu().float().numpy()
    else:
        shared = getattr(host_cache, "shared_tensor", None)
        if shared is None:
            return np.zeros((1,), dtype=np.float16)
        if shared.dim() == 4:
            row = shared[virtual_layer_idx, block_id].cpu().float().numpy()
        elif shared.dim() == 5:
            row = shared[dp_rank, virtual_layer_idx, block_id].cpu().float().numpy()
        else:
            row = shared[virtual_layer_idx, block_id].cpu().float().numpy()

    # Return raw flattened block data
    return row.reshape(-1)


def extract_block_decode_host(
    cache: "DecodeOmniCache",
    group_idx: int,
    virtual_layer_idx: int,
    block_id: int,
    component_idx: int,
) -> np.ndarray:
    """Extract raw block from decode host pool - dumps entire block, not components."""
    host_cache = cache.host_cache
    kvi = getattr(host_cache, "kvi_tensors", None)

    dp_rank = getattr(cache, "dp_local_rank", 0) or 0

    if kvi and len(kvi) > 0:
        pool = kvi[0]
        if pool.dim() == 5:
            row = pool[dp_rank, virtual_layer_idx, block_id].cpu().float().numpy()
        elif pool.dim() == 4:
            row = pool[virtual_layer_idx, block_id].cpu().float().numpy()
        else:
            row = pool[virtual_layer_idx, block_id].cpu().float().numpy()
    else:
        shared = getattr(host_cache, "shared_tensor", None)
        if shared is None:
            return np.zeros((1,), dtype=np.float16)
        if shared.dim() == 5:
            row = shared[dp_rank, virtual_layer_idx, block_id].cpu().float().numpy()
        elif shared.dim() == 4:
            row = shared[virtual_layer_idx, block_id].cpu().float().numpy()
        else:
            row = shared[virtual_layer_idx, block_id].cpu().float().numpy()

    # Return the raw flattened block data
    return row.reshape(-1)


def extract_block_prefill_hbm(
    cache: "PrefillOmniCache",
    layer_name: str,
    fake_block_id: int,
    group_idx: int,
    component_idx: int,
) -> np.ndarray:
    """Extract raw block from prefill HBM — the full contiguous block bytes.

    Reads ``cache.device_raw_tensors[stage]`` (``(num_blocks,
    page_size_bytes/2)`` bf16), indexed by ``fake_block_id``. The earlier
    ``cache.device_cache[layer_name]`` path returned this layer's strided
    component views; concatenating those on-the-fly does NOT reproduce the
    physical block layout (the views are disjoint byte ranges into the
    SAME underlying raw tensor, so the per-layer view only covers a slice
    of the block — the rest belongs to other layers sharing the same row).
    """
    stage = getattr(cache, "stage_record", 0)
    raw_by_layer = getattr(cache, "device_raw_tensors_by_layer", None)
    if raw_by_layer is not None and stage < len(raw_by_layer):
        raw_stage = raw_by_layer[stage].get(layer_name)
        if raw_stage is None:
            raw = getattr(cache, "device_raw_tensors", None)
            if not raw or stage >= len(raw):
                return np.zeros((1,), dtype=np.float16)
            raw_stage = raw[stage]
    else:
        raw = getattr(cache, "device_raw_tensors", None)
        if not raw or stage >= len(raw):
            return np.zeros((1,), dtype=np.float16)
        raw_stage = raw[stage]

    if fake_block_id >= raw_stage.shape[0]:
        return np.zeros((1,), dtype=np.float16)

    row = raw_stage[fake_block_id].cpu().float().numpy()
    return row.reshape(-1)


def extract_block_decode_hbm(
    cache: "DecodeOmniCache",
    group_idx: int,
    layer_name: str,
    block_id: int,
    hbm_slot: int,
    component_idx: int,
) -> np.ndarray:
    """Extract raw block from decode HBM - dumps entire block, not components."""
    enable_hm = os.getenv("ENABLE_HOST_MAPPING", "1") == "1"
    hbm_pool = getattr(cache, "hbm_buffer_block_table_pool", None)

    if enable_hm and hbm_pool is not None and group_idx < len(hbm_pool):
        # ``hbm_buffer_pool_raw[layer_name]`` is allocated by
        # hbm_buffer_utils._alloc_{attention,mamba} as a 2D
        # ``(num_blocks, page_size_bytes / dtype.itemsize)`` view of a
        # contiguous per-layer byte buffer — i.e. the raw block layout
        # that H2D writes into. Indexing by ``hbm_slot`` gives the full
        # flat block bytes, without any per-component strided views.
        buf_raw = hbm_pool[group_idx].get("hbm_buffer_pool_raw", {})
        entry = buf_raw.get(layer_name)
        if entry is None:
            return np.zeros((1,), dtype=np.float16)
        # MoMe conv_state lives in the same HBM bytes the decode kernel
        # advances each step. Without a synchronize here, the kernel can
        # shift the slot between this view and the .cpu() copy. Drain
        # the compute stream before snapshotting.
        try:
            import torch as _torch_sync
            _torch_sync.npu.synchronize()
        except Exception:
            pass
        snap = entry[hbm_slot].clone()
        row = snap.cpu().float().numpy()
        return row.reshape(-1)
    else:
        raw_tensors = getattr(cache, "raw_tensors_by_row", None)
        page_size = getattr(cache, "page_size_padded", None)
        if raw_tensors is None or page_size is None:
            return np.zeros((1,), dtype=np.float16)

        for row_idx in range(len(raw_tensors)):
            raw = raw_tensors[row_idx]
            num_blocks = raw.shape[0] // (page_size // 2)
            if hbm_slot < num_blocks:
                slot_bytes = raw[hbm_slot * page_size // 2:(hbm_slot + 1) * page_size // 2]
                row = slot_bytes.cpu().to(torch.uint8).numpy().view(np.bfloat16).astype(np.float32)
                return row
        return np.zeros((1,), dtype=np.float16)