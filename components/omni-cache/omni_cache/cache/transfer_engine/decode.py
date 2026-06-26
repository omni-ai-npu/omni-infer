# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Host-to-Device (H2D) transfer operations for decode cache."""

import copy
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from vllm.logger import init_logger

logger = init_logger("vllm.v1.omni")


def build_npu_block_layers(
    device_cache: Dict[str, Any],
    layer_names: List[str],
    block_id: int,
    enable_dsa: bool = False
) -> List[Tuple]:
    """Build NPU block layers for a single block.

    Args:
        device_cache: Dictionary mapping layer names to cache tensors.
        layer_names: List of layer names to include.
        block_id: Block ID to extract.
        enable_dsa: Whether DSA optimization is enabled.

    Returns:
        List of layer tuples containing cache tensors for the block.
    """
    layers = []
    for layer_name in layer_names:
        if enable_dsa:
            layers.append((
                device_cache[layer_name][2][block_id],  # k_indexer cache
            ))
        else:
            layers.append((
                device_cache[layer_name][0][block_id],
                device_cache[layer_name][1][block_id]
            ))
    return layers


def build_npu_block_layers_hybrid(
    device_cache: Dict[str, Any],
    layer_indices: List[str],
    block_id: int
) -> List[Tuple]:
    """Build NPU block layers for hybrid attention.

    Args:
        device_cache: Dictionary mapping layer names to cache tensors.
        layer_indices: List of layer indices for this group.
        block_id: Block ID to extract.

    Returns:
        List of layer tuples containing cache tensors.
    """
    layers = []
    for idx_layer, layer_name in enumerate(layer_indices):
        cache_entry = device_cache[layer_name]
        if isinstance(cache_entry, tuple):
            layers.append((
                cache_entry[0][block_id],
                cache_entry[1][block_id][idx_layer].unsqueeze(-1),
            ))
        else:
            layers.append((cache_entry[block_id],))
    return layers


def build_npu_block_layers_hbm(
    hbm_buffer_pool: Dict[str, Any],
    layer_indices: List[str],
    blc_idx_req: int
) -> List[Tuple]:
    """Build NPU block layers from HBM buffer pool.

    Args:
        hbm_buffer_pool: HBM buffer pool dictionary.
        layer_indices: List of layer indices.
        blc_idx_req: Block index in the request.

    Returns:
        List of layer tuples containing cache tensors.
    """
    layers = []
    for idx_layer, layer_name in enumerate(layer_indices):
        cache_entry = hbm_buffer_pool[layer_name]
        if isinstance(cache_entry, tuple):
            layers.append((
                cache_entry[0][blc_idx_req],
                cache_entry[1][blc_idx_req][idx_layer].unsqueeze(-1),
            ))
        else:
            layers.append((cache_entry[blc_idx_req],))
    return layers


def prepare_h2d_copy_args(cache, local_block_ids: List[List[int]], request_id: Optional[str]):
    """Build H2D operations for decode.

    Args:
        cache: The DecodeOmniCache instance
        local_block_ids: List of local block IDs per group
        request_id: Current request ID

    Returns:
        Tuple of (batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes)
    """
    # prepare_h2d_copy_args is called from a forked subprocess (process_manager's
    # _process_h2d_address). NPU context can't be re-initialized there, so
    # device_cache must already have been populated by the parent before the
    # fork. If it isn't, we'd silently try to allocate NPU memory in the
    # child and hit a "Cannot re-initialize NPU" error.
    if cache.device_cache is None:
        raise RuntimeError(
            "DecodeOmniCache.device_cache is None in subprocess — parent must "
            "call ensure_device_cache_initialized() before forking."
        )
    from omni_cache.cache.core.constants import ENABLE_HOST_MAPPING

    if cache.is_hybrid_attn and ENABLE_HOST_MAPPING:
        batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes = prepare_h2d_copy_args_hbm_buffer(
            cache, local_block_ids, request_id)
    elif cache.is_hybrid_attn:
        batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes = prepare_h2d_copy_args_hybrid(
            cache, local_block_ids)
    else:
        layer_indices = cache.device_cache.keys()
        npu_blocks = []
        for block_id in local_block_ids[0]:
            layers = []
            for layer_name in layer_indices:
                if cache.enable_dsa:
                    layers.append(
                        (
                            # only keep k_indexer cache on device
                            cache.device_cache[layer_name][2][block_id],
                        )
                    )
                else:
                    layers.append(
                        (
                            cache.device_cache[layer_name][0][block_id],
                            cache.device_cache[layer_name][1][block_id]
                        )
                    )
            npu_blocks.append(layers)
        batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes = cache.host_cache.batch_layer_copy_to_npu(
            local_block_ids[0], npu_blocks)

    return batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes


def prepare_h2d_copy_args_hybrid(cache, local_block_ids: List[List[int]], tp_nnodes: int = 1):
    """Build H2D ops for hybrid attention via byte-level opaque-block H2D.

    For each kv_cache_group that OX populated, emit one memcpy per row per
    (host_src_block, dev_dst_block) pair. Each memcpy copies exactly
    page_size_padded bytes from host_pool[row, src_block] into
    raw_tensors[row].data_ptr() + dst_block * page_size_padded.

    Layout-agnostic: the unified host page already packs every attention
    component (attn kvi slabs + MoMe state) in the exact device layout for
    that row, so one memcpy per (row, block) transfers all components
    atomically. Works for HOST_MAPPING=0 where the decode worker owns its
    own HBM kv_cache (allocated in initialize_device_cache), not the HBM
    buffer pool used by HOST_MAPPING=1.

    Requires decode_omni_cache.initialize_device_cache to have populated
    cache.raw_tensors_by_row and cache.page_size_padded (done in
    the HM=0 branch).
    """
    batch_device_mem: list = []
    batch_device_max: list = []
    batch_host_mem: list = []
    batch_host_sizes: list = []

    raw_tensors = getattr(cache, 'raw_tensors_by_row', None)
    page_size = getattr(cache, 'page_size_padded', None)
    host_kvi = (
        cache.host_cache.kvi_tensors[0]
        if getattr(cache.host_cache, 'kvi_tensors', None)
        else None
    )
    if raw_tensors is None or page_size is None or host_kvi is None:
        logger.warning(
            "[h2d_hybrid] missing raw_tensors_by_row(%s) / page_size_padded(%s) / "
            "host_kvi(%s) — returning no-op",
            raw_tensors is not None, page_size, host_kvi is not None,
        )
        return batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes

    # Collapse the DP leading dim if present. host_kvi shape is either
    # (num_layers, num_blocks, block_size, head_elts) or
    # (dp_world, num_layers, num_blocks, block_size, head_elts).
    if host_kvi.dim() == 5:
        dp_rank = getattr(cache, 'dp_local_rank', 0) or 0
        host_kvi = host_kvi[dp_rank]

    flat_host_ids = local_block_ids[0]
    num_rows = len(raw_tensors)

    # Per-group pairing. The scheduler slices local_block_ids[0] into
    # local_block_ids[1] per kv_cache_group, positionally. Walk groups in
    # order and consume consecutive entries from flat_host_ids.
    flat_idx = 0
    for idx_grp, block_id_grp in enumerate(local_block_ids[1]):
        if not block_id_grp:
            continue
        for dst_block in block_id_grp:
            if flat_idx >= len(flat_host_ids):
                break
            src_block = flat_host_ids[flat_idx]
            flat_idx += 1
            # Sentinel: vLLM reserves block 0 as a zero-padding slot. Writing
            # there would overwrite the shared pad used by every unfilled
            # attention lane.
            if src_block == 0 and dst_block == 0:
                continue
            for row_idx in range(num_rows):
                raw_tensor = raw_tensors[row_idx]
                host_page = host_kvi[row_idx, src_block]
                dst_ptr = raw_tensor.data_ptr() + dst_block * page_size
                src_ptr = host_page.data_ptr()
                batch_device_mem.append(dst_ptr)
                batch_device_max.append(page_size)
                batch_host_mem.append(src_ptr)
                batch_host_sizes.append(page_size)

    return batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes


def prepare_h2d_copy_args_hbm_buffer(cache, local_block_ids: List[List[int]], request_id: str):
    """Build H2D operations using the per-layer HBM buffer pool.

    Dispatch is by kv_cache_group spec `kind` (populated by
    hbm_buffer_utils.construct_hbm_buffer):
      - "attention" groups (DSA / MLA / SWA): page-by-page H2D into the
        request's HBM lane. SlidingWindowSpec groups only pull the last few
        blocks (window), others pull the full sequence.
      - "mamba" groups (MomeSpec): a single per-request state slot.

    Each layer in a group has its own dedicated HBM buffer.

    Args:
        cache: DecodeOmniCache instance.
        local_block_ids: [host_block_ids, [per_group_block_ids]] from the
            scheduler. Element 1 is a list indexed by kv_cache_group index.
        request_id: Request id (used to claim/release a batch slot).

    Returns:
        (batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes)
    """
    from omni_cache.cache.decode.hbm_buffer_utils import _is_sliding_window_spec

    rmap_before = {int(k): v for k, v in cache.record_batch_idx_to_req.items()}
    if int(os.getenv("USE_OMNI_INPUT_BATCH", "0")):
        idx_req = cache.reserve_hbm_lane_for_request(request_id)
    else:
        # Claim an unused HBM lane for this request.
        # Slot 0 is a permanent sentinel (never None). All other lanes
        # in record_batch_idx_to_req are available — the dict is sized
        # to num_max_batch_pool, which is the real limit.
        idx_req = None
        with cache._lane_lock:
            for idx_batch in range(cache.num_max_batch_pool):
                if cache.record_batch_idx_to_req[idx_batch] is None:
                    cache.record_batch_idx_to_req[idx_batch] = request_id
                    idx_req = int(idx_batch)
                    break
        if idx_req is None:
            raise RuntimeError(f"No free HBM lane for request {request_id}")
    logger.debug(
        "[H2D-TRACE] req=%s idx_req=%d pool=%d rmap_before=%s flat_blks=%s",
        request_id, idx_req, cache.num_max_batch_pool, rmap_before,
        local_block_ids[0] if local_block_ids else "none",
    )
    # Invariant: the claimed `idx_req` is the HBM lane for this request. H2D
    # writes below target slots derived from it — mamba -> `idx_req + 1`,
    # sliding-window attention -> `req_offset*idx_req + 1 .. +N`. With
    # OmniCacheInputBatch enabled, attention-facing block tables read the same
    # lane from H2D-owned `omni_cache.req_id_to_idx`. In the legacy path,
    # extended attention builders rebuild that mapping from
    # `record_batch_idx_to_req`.

    batch_device_mem: list = []
    batch_device_max: list = []
    batch_host_mem: list = []
    batch_host_sizes: list = []

    attn_npu_blocks: List[List[Tuple[torch.Tensor, ...]]] = []
    attn_local_block_true: List[int] = []
    local_req_ids_swa_block: List[List[int]] = []

    disable_swa = int(os.getenv("DISABLE_SWA_MAPPING", "0"))

    for idx_grp, block_id_grp in enumerate(local_block_ids[1]):
        layer_names = cache.kv_cache_config.kv_cache_groups[idx_grp].layer_names
        spec = cache.kv_cache_config.kv_cache_groups[idx_grp].kv_cache_spec
        block_table_meta = cache.hbm_buffer_block_table_pool[idx_grp]
        kind = block_table_meta["kind"]
        logger.debug(
            "[H2D-TRACE] grp=%d kind=%s nlayers=%d blocks_in=%s",
            idx_grp, kind, len(layer_names), block_id_grp,
        )

        if kind == "mamba":
            # Per-request state: one host block is enough (the final prefill
            # snapshot). The HBM side is the per-request lane at idx_req.
            if len(block_id_grp) == 0:
                raise RuntimeError(f"No src blocks found! {local_block_ids=}")
            # Use the last block in the group (prefill's final state snapshot)
            src_block_id = block_id_grp[-1]
            # The host pool row addressing matches the prefill side's
            # _layer_name_to_group_and_layer_idx, which uses the layer's
            # POSITION WITHIN THE GROUP (0..len(layer_names)-1) — not the
            # global decoder layer index. A global-index-based read went
            # through an unset `cache.layer_indices` in hybrid_attn mode
            # and crashed the address process.
            for vl_idx, layer_name in enumerate(layer_names):
                buf = cache.hbm_buffer_block_table_pool[idx_grp]['hbm_buffer_pool_raw'][layer_name]
                # Each component is (max_num_reqs, *state_shape); take the
                # idx_req-th slot. Mamba/Mome: each request occupies a single
                # HBM slot at `idx_req + 1` (slot 0 is reserved as a zero sentinel).
                npu_block = buf[idx_req + 1]
                batch_device_mem.append(npu_block.data_ptr())
                batch_device_max.append(npu_block.nbytes)

                # MoMe data is packed at offset 0 (HEAD) of the host pool slot.
                # batch_layer_copy_to_npu copies the TAIL when the host slot is
                # wider than the NPU target, which is correct for DSA indexer
                # but wrong for MoMe. Copy from the head explicitly here.
                host_block = cache.host_cache.get_block(src_block_id)[0]
                cpu_block = host_block[vl_idx]
                copy_size = min(npu_block.nbytes, cpu_block.nbytes)
                batch_host_mem.append(cpu_block.data_ptr())
                batch_host_sizes.append(copy_size)
                logger.debug(
                    "[H2D-PREP] %s: rmap=%s; NPU: addr=0x%x nbytes=%d shape=%s; CPU: addr=0x%x nbytes=%d shape=%s",
                    layer_name, list(cache.record_batch_idx_to_req.items()),
                    npu_block.data_ptr(), npu_block.nbytes, npu_block.shape,
                    cpu_block.data_ptr(), cpu_block.nbytes, cpu_block.shape,
                )
                # Stash tensor refs for the first two layers of group 0 and
                # group 1 — the consistency checker reports mismatches here,
                # so these are the rows where it pays to hash before/after.
            continue

        # Attention path. SlidingWindow only needs the most recent window.
        if _is_sliding_window_spec(spec):
            if int(os.getenv("DISABLE_SWA_MAPPING", "0")):
                continue
            swa_num = (spec.sliding_window + cache.block_size - 1) // cache.block_size + 1
            # Drop null-block entries (rolled-out positions) BEFORE slicing so
            # the first real block lines up with lane slot 1. Otherwise the
            # leading null pulls in whatever stale bytes OX wrote into host
            # block 0 and that gets copied to slot 1 of every SWA layer,
            # producing non-deterministic output across otherwise identical
            # requests even at temperature=0.
            block_id_grp = [b for b in block_id_grp if b != 0]
            block_id_grp = block_id_grp[-swa_num:]
            # Keep the last (req_offset - 1) blocks: reserve 1 for rotation.
            # block_id_grp = block_id_grp[-(req_offset - 1):] if req_offset > 1 else block_id_grp[-1:]
            local_req_ids_swa_block.append(copy.copy(block_id_grp))

        req_offset = cache.hbm_buffer_block_table_pool[idx_grp]["req_offset"]
        block_offset_per_req = req_offset * idx_req
        attn_local_block_true.extend(block_id_grp)
        for blc_idx, _block_id in enumerate(block_id_grp):
            if _is_sliding_window_spec(spec):
                # Each request's window occupies `req_offset` consecutive
                # HBM slots starting at `req_offset * idx_req + 1`. Slot 0
                # of each lane is reserved (zero sentinel / rotation guard).
                hbm_slot = block_offset_per_req + blc_idx + 1
            else:
                hbm_slot = _block_id  # for DSA, we use the same host and device block ids
            layers = []
            for layer_name in layer_names:
                buf = cache.hbm_buffer_block_table_pool[idx_grp]['hbm_buffer_pool_raw'][layer_name]
                if isinstance(buf, tuple):
                    # (data, scale) pair — H2D copies both components.
                    layers.append((buf[0][hbm_slot], buf[1][hbm_slot]))
                else:
                    layers.append((buf[hbm_slot],))
            attn_npu_blocks.append(layers)

    with cache._lane_lock:
        cache.record_req_ids_swa_block[request_id] = local_req_ids_swa_block
    if attn_local_block_true:
        micro_device_mem, micro_device_max, micro_host_mem, micro_host_sizes = (
            cache.host_cache.batch_layer_copy_to_npu(attn_local_block_true, attn_npu_blocks)
        )
        batch_device_mem.extend(micro_device_mem)
        batch_device_max.extend(micro_device_max)
        batch_host_mem.extend(micro_host_mem)
        batch_host_sizes.extend(micro_host_sizes)

    logger.debug(
        "[H2D-TRACE] req=%s done entries=%d host_blocks=%s host_bytes=%d dev_bytes=%d",
        request_id, len(batch_host_mem), attn_local_block_true,
        sum(batch_host_sizes), sum(batch_device_max),
    )
    return batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes
