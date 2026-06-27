# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Synchronization operations for H2D/D2H transfers.

This module provides functions for synchronizing H2D (Host-to-Device)
and D2H (Device-to-Host) transfer operations for both prefill and decode
cache operations.
"""

from typing import TYPE_CHECKING, List, Optional

import re
import torch

from omni_cache.cache.utils.debug import (
    should_log_rank,
    summarize_array,
    summarize_map,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from omni_cache.cache.prefill.prefill_omni_cache import PrefillOmniCache
    from omni_cache.cache.decode.decode_omni_cache import DecodeOmniCache

logger = init_logger("vllm.v1.omni")


def _resolve_next_layer(cache, layer_name: str):
    """Resolve the next global layer name and its group-local index.

    Build the next layer name by incrementing the layer index embedded in
    ``layer_name``. For example, ``model.layers.22.self_attn.attn`` resolves
    to ``model.layers.23.self_attn.attn``. Then find the group that owns that
    next layer and return its group-local layer index.

    Returns:
        (next_layer_name, group_idx, layer_idx) or None if the constructed
        next layer is not found in any KV cache group.
    """
    from vllm.model_executor.models.utils import extract_layer_index
    try:
        global_idx = extract_layer_index(layer_name)
    except (ValueError, IndexError) as e:
        logger.debug("_resolve_next_layer: extract_layer_index failed for %s: %s", layer_name, e)
        return None

    match = re.search(r"(^|\.)(layers\.)(\d+)(\.|$)", layer_name)
    if match is None:
        return None

    next_global_idx = global_idx + 1
    next_layer = (
        layer_name[:match.start(3)]
        + str(next_global_idx)
        + layer_name[match.end(3):]
    )

    try:
        next_group_idx, layer_idx = cache._layer_name_to_group_and_layer_idx(
            next_layer
        )
    except ValueError:
        return None

    return next_layer, next_group_idx, layer_idx


def _wait_for_pending_d2h(cache, context: str = "") -> None:
    """Wait for any pending D2H thread pool operations to complete.

    This is critical before H2D reads from the host pool, to ensure the
    data written by a previous request's D2H is fully materialized.

    Handles both:
    - cache.copy_future (singular, used by single-layer D2H)
    - cache.copy_futures (plural, used by hybrid D2H)

    Args:
        cache: The PrefillOmniCache instance
        context: Context string for logging
    """
    # Wait for single-layer D2H future
    if hasattr(cache, 'copy_future') and cache.copy_future is not None:
        if not cache.copy_future.done():
            try:
                cache.copy_future.result()
            except Exception as e:
                logger.warning("_wait_for_pending_d2h: copy_future wait failed: %s", e)

    # Wait for hybrid D2H futures (array)
    if hasattr(cache, 'copy_futures') and cache.copy_futures:
        for i, fut in enumerate(cache.copy_futures):
            if fut is not None:
                try:
                    fut.result()
                except Exception as e:
                    logger.warning("_wait_for_pending_d2h: copy_futures[%d] wait failed: %s", i, e)


def synchronize_h2d_prefill(
    cache: "PrefillOmniCache",
    prefix_meta,
    layer_names,
    load_next_layer: bool = True,
) -> None:
    """Synchronize H2D for prefill operations.

    This function performs Host-to-Device transfer for chunked prefill,
    loading KV cache from host memory to device memory.

    Args:
        cache: The PrefillOmniCache instance
        prefix_meta: PrefixCopyMeta for chunked prefill
        layer_names: Layer name (str) or list of layer names.
        load_next: If True, load the NEXT layer (used by post_attn prefetch).
                   If False, load the CURRENT layer (used for Layer 0).

    Returns:
        None
    """
    if prefix_meta is None or not layer_names:
        return

    # Normalize to list
    if isinstance(layer_names, str):
        layer_names = [layer_names]

    from vllm.model_executor.models.utils import extract_layer_index

    # CRITICAL: Wait for any pending D2H operations before reading host pool
    _wait_for_pending_d2h(cache, "prefill")

    for layer_name in layer_names:
        if should_log_rank(cache):
            logger.warning(
                "[APCDBG/H2D] tp_rank=%s dp_rank=%s stage=%s group_idx=%s "
                "layer_name=%s load_next_layer=%s source_layer=%s "
                "prefix_segs=%s",
                getattr(cache, "tp_rank", None),
                getattr(cache, "dp_local_rank", None),
                getattr(cache, "stage_record", None),
                None,
                layer_name,
                load_next_layer,
                layer_name,
                getattr(prefix_meta, "consecutive_blocks", None),
            )
        if load_next_layer:
            # Find the next layer within the same group
            result = _resolve_next_layer(cache, layer_name)
            if result is None:
                continue
            target_layer_name, _, layer_idx = result
        else:
            # Use the current layer directly (for Layer 0 H2D during build)
            target_layer_name = layer_name
            try:
                _, layer_idx = cache._layer_name_to_group_and_layer_idx(target_layer_name)
            except (ValueError, IndexError) as e:
                logger.debug("synchronize_h2d_prefill: extract_layer_index failed: %s", e)
                continue

        if layer_idx >= cache.num_layers:
            continue
        if should_log_rank(cache):
            try:
                group_idx, _ = cache._layer_name_to_group_and_layer_idx(target_layer_name)
            except Exception:
                group_idx = None
            logger.warning(
                "[APCDBG/H2D] tp_rank=%s dp_rank=%s stage=%s group_idx=%s "
                "layer_name=%s source_layer=%s target_layer=%s layer_idx=%s "
                "load_next_layer=%s prefix_segs=%s",
                getattr(cache, "tp_rank", None),
                getattr(cache, "dp_local_rank", None),
                getattr(cache, "stage_record", None),
                group_idx,
                target_layer_name,
                layer_name,
                target_layer_name,
                layer_idx,
                load_next_layer,
                getattr(prefix_meta, "consecutive_blocks", None),
            )
        logger.debug(f"<<< before synchronize_h2d_prefill_unified: {target_layer_name=} {layer_idx=} {prefix_meta.consecutive_blocks}")
        if load_next_layer:
            from vllm.forward_context import get_forward_context
            prefix_meta = get_forward_context().attn_metadata[target_layer_name].prefix_meta
        _synchronize_h2d_prefill_unified(cache, prefix_meta, target_layer_name, layer_idx, load_next_layer=load_next_layer)


def _synchronize_h2d_prefill_unified(
    cache: "PrefillOmniCache",
    prefix_meta,
    layer_name: str,
    layer_idx: int,
    load_next_layer: bool = True
) -> None:
    """Unified H2D for both MoME and Attention layers during APC.

    Both MoME and Attention share the same device_raw_tensors layout.
    The only difference is the host data source:
    - MoME: mome_prefix_buffers[layer_name] (uint8, packed bytes)
    - Attention: kvi_tensors (per-component tensors)

    Copy host[real_block_id] → device[fake_block_id] with TP byte split.

    When the volatile swap is active (PACKED_HBM=1), prefix_meta contains
    fake block IDs. The device write must use fake IDs (packed HBM layout),
    but the host read must use real IDs (host pool layout). We resolve the
    mapping the same way D2H does in synchronize_d2h_prefill: look up
    _volatile_real_to_fake_per_group stashed by the swap and invert it.
    """
    if prefix_meta is None:
        return

    # Extract block IDs from prefix_meta (may be fake when volatile swap is active)
    block_ids = []
    for block_ranges in prefix_meta.consecutive_blocks:
        for start_block_id, end_block_id in block_ranges:
            block_ids.extend(range(start_block_id, end_block_id))

    if not block_ids:
        return

    # When volatile swap rewrites block_table.np with fake IDs,
    # prefix_meta.consecutive_blocks contains fake IDs. The device write
    # must use fake IDs (attention kernels use the packed HBM layout), but
    # the host pool read must use REAL IDs (host pool is indexed by the
    # scheduler's real block IDs). Mirror the D2H logic in
    # synchronize_d2h_prefill.
    host_block_ids = block_ids
    group_idx, _ = cache._layer_name_to_group_and_layer_idx(layer_name)
    runner = getattr(cache, "runner", None)
    input_batch = getattr(runner, "input_batch", None) if runner else None
    maps = getattr(input_batch, "_volatile_real_to_fake_per_group", None)
    fake_to_real = None
    if maps is not None and group_idx < len(maps) and maps[group_idx]:
        fake_to_real = {v: k for k, v in maps[group_idx].items()}
        host_block_ids = [fake_to_real.get(int(fid), int(fid)) for fid in block_ids]

    next_layer_device_stage_idx = (cache.stage_record + 1) % cache.num_stages_layer_copy
    raw = cache.device_raw_tensors[next_layer_device_stage_idx]

    # Wait for the D2H stream to finish reading the H2D target stage
    # before H2D overwrites it.  Critical for n_stages=1.
    _evt_stages = getattr(cache, 'd2h_event_stages', None)
    if _evt_stages and 0 <= next_layer_device_stage_idx < len(_evt_stages):
        _evt = _evt_stages[next_layer_device_stage_idx].get(group_idx)
        if _evt is not None:
            _evt.synchronize()

    # Build host shard as 2D [block_num, block_nbytes].
    # view as 2D FIRST (contiguous → O(1) view), THEN index.
    # TP split on block_nbytes: only read the columns this rank owns.
    host_block_ids_t = torch.tensor(host_block_ids, dtype=torch.long)
    page_elems = raw.shape[1]
    elems_per_rank = page_elems // cache.tp_world_size
    rank_start = cache.tp_rank * elems_per_rank
    rank_end = rank_start + elems_per_rank

    # When multi-component support is added, loop over all components
    # and torch.cat their overlapping column ranges.
    if len(cache.host_cache.kvi_tensors) != 1:
        raise RuntimeError("multi-component kvi_tensors not yet supported")
    tensor = cache.host_cache.kvi_tensors[0]
    if isinstance(tensor, tuple):
        host_layer = tensor[layer_idx]
    else:
        host_layer = tensor[layer_idx]
    if host_layer.dim() == 4:
        per_dp = host_layer[cache.dp_local_rank]
    else:
        per_dp = host_layer
    # per_dp: [blocks_per_rank, *inner_dims], contiguous
    # view → 2D [blocks_per_rank, block_nbytes]  (O(1), no copy)
    host_2d = per_dp.view(per_dp.shape[0], -1)
    # TP split: slice the column range this rank owns
    c0 = rank_start
    c1 = rank_end
    host_shard = host_2d[host_block_ids_t, c0:c1]
    if should_log_rank(cache):
        logger.warning(
            "[APCDBG/H2D] tp_rank=%s dp_rank=%s stage=%s group_idx=%s "
            "layer_name=%s layer_idx=%s target_stage=%s block_ids_fake=%s "
            "host_block_ids_real=%s %s raw_shape=%s host_layer_shape=%s "
            "rank_range=(%s,%s)",
            getattr(cache, "tp_rank", None),
            getattr(cache, "dp_local_rank", None),
            getattr(cache, "stage_record", None),
            group_idx,
            layer_name,
            layer_idx,
            next_layer_device_stage_idx,
            block_ids,
            host_block_ids,
            summarize_map("fake_to_real", fake_to_real),
            tuple(raw.shape),
            tuple(host_layer.shape),
            c0,
            c1,
        )

    # H2D + all-gather on h2d_stream
    h2d_event = torch.npu.Event(blocking=False, enable_timing=False)
    with torch.npu.stream(cache.h2d_stream):
        dev_block_ids_t = torch.tensor(block_ids, dtype=torch.long, device=cache.device)
        device_shard = host_shard.to(device=cache.device, non_blocking=True)
        if cache.tp_world_size > 1:
            from vllm.distributed import tensor_model_parallel_all_gather
            device_full = tensor_model_parallel_all_gather(device_shard, dim=1)
        else:
            device_full = device_shard
        raw[dev_block_ids_t] = device_full
        h2d_event.record(cache.h2d_stream)
        cache.h2d_event.record(cache.h2d_stream)

    # Store per-stage per-group event so _moe_post_sync can fence on the
    # correct stage for the correct attention group.
    h2d_stages = getattr(cache, 'h2d_event_stages', None)
    if h2d_stages is not None and 0 <= next_layer_device_stage_idx < len(h2d_stages):
        h2d_stages[next_layer_device_stage_idx][group_idx] = h2d_event

    if not load_next_layer:
        # Layer 0: no MoE boundary before KV is used — sync now.
        cache.h2d_stream.synchronize()
        cache.h2d_event.synchronize()


def _restore_volatile_swap_np_for_d2h(cache: "PrefillOmniCache") -> bool:
    """Restore CPU block_table.np once, just before the first D2H.

    At this point prepare_inputs and attention metadata construction have
    consumed the fake ids, so the persistent vLLM CPU block table can go back
    to real ids before the next scheduler step mutates it. Keep real/fake
    stashes intact because D2H/H2D still use them.
    """
    if cache is None:
        return False
    runner = getattr(cache, "runner", None)
    input_batch = getattr(runner, "input_batch", None) if runner is not None else None
    if input_batch is None or getattr(input_batch, "_volatile_swap_np_restored", False):
        return False

    restore_tables = getattr(input_batch, "_restore_real_block_tables_per_group", None)
    if restore_tables is None:
        restore_tables = getattr(
            cache,
            "_restore_real_block_tables_per_group",
            getattr(cache, "_real_block_tables_per_group", None),
        )
    if restore_tables is None:
        return False

    bt_mgr = getattr(input_batch, "block_table", None)
    staged_bts = getattr(bt_mgr, "block_tables", None)
    if staged_bts is None:
        return False

    restored = False
    for grp_idx, bt_item in enumerate(staged_bts):
        if grp_idx >= len(restore_tables) or restore_tables[grp_idx] is None:
            continue
        if not (
            hasattr(bt_item, "block_table")
            and hasattr(bt_item.block_table, "np")
        ):
            continue
        real_bt = restore_tables[grp_idx]
        bt_np = bt_item.block_table.np
        rows = min(real_bt.shape[0], bt_np.shape[0])
        cols = min(real_bt.shape[1], bt_np.shape[1])
        if rows <= 0 or cols <= 0:
            continue
        bt_np[:rows, :cols] = real_bt[:rows, :cols]
        restored = True

    if restored:
        input_batch._volatile_swap_np_restored = True
    return restored


def synchronize_d2h_prefill(
    cache: "PrefillOmniCache",
    attn_names: list[str],
    attn_metadatas: list[str],
    kv_event: torch.npu.Event
) -> None:
    """Synchronize D2H for prefill operations (unified API).

    This function performs Device-to-Host transfer, offloading KV cache
    from device memory to host memory. Supports single and multiple layers.

    Args:
        cache: The PrefillOmniCache instance
        attn_names: List of layer names
        attn_metadatas: List of attention metadata objects
        kv_event: Event signaling KV computation completion

    Returns:
        None
    """
    _restore_volatile_swap_np_for_d2h(cache)

    if len(attn_names) > 1:
        # Multiple layers: call hybrid version
        return synchronize_d2h_hybrid(cache, attn_names, attn_metadatas, kv_event)

    # Single layer case
    layer_name = attn_names[0]
    metadata = attn_metadatas[0]
    group_idx, layer_idx = cache._layer_name_to_group_and_layer_idx(layer_name)

    # Capture the stage we will read/write THIS call. Do not re-read
    # cache.stage_record after this point - we will advance it ourselves
    # at the end. With num_stages_layer_copy == 1, stage_idx is always 0
    # and behaviour matches the legacy single-stage path.
    n_stages = max(1, int(getattr(cache, 'num_stages_layer_copy', 1) or 1))
    stage_idx = int(getattr(cache, 'stage_record', 0) or 0) % n_stages
    raw_stages_attr = getattr(cache, 'batch_buffer_raw_stages', None)

    # If the same (stage, group) already has an in-flight D2H future, wait
    # for it before submitting a new one.  This guards against the same
    # attention type being invoked more than once per block for the same
    # layer, which would otherwise overwrite the previous future and leak
    # an un-waited D2H scatter.
    fut_stages = getattr(cache, 'copy_futures_stages', None)
    if fut_stages is not None and 0 <= stage_idx < len(fut_stages):
        bucket = fut_stages[stage_idx]
        if isinstance(bucket, dict):
            prev = bucket.get(group_idx)
            if prev is not None and not prev.done():
                try:
                    prev.result()
                except Exception as e:
                    logger.error(
                        "prev D2H scatter failed on stage=%d group=%d: %s",
                        stage_idx, group_idx, e,
                    )
                    raise

    d2h_event = torch.npu.Event(blocking=False, enable_timing=False)

    with torch.npu.stream(cache.d2h_stream):
        from .prefill import _prepare_blocks
        spec = cache.kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec
        block_ids_cpu, block_ids_npu = _prepare_blocks(metadata, cache.device, spec)

        # When volatile swap rewrites block_table.np with fake IDs,
        # _prepare_blocks returns fake IDs. The HBM read must use fake IDs
        # (attention wrote to fake-ID positions), but the host pool scatter
        # must use REAL IDs (decode side indexes host pool by real IDs).
        real_block_ids_cpu = None
        runner = getattr(cache, "runner", None)
        input_batch = getattr(runner, "input_batch", None) if runner else None
        maps = getattr(input_batch, "_volatile_real_to_fake_per_group", None)
        if maps is not None and group_idx < len(maps) and maps[group_idx]:
            fake_to_real = {v: k for k, v in maps[group_idx].items()}
            real_list = [fake_to_real.get(int(fid), int(fid))
                         for fid in block_ids_cpu.tolist()]
            real_block_ids_cpu = torch.tensor(
                real_list, dtype=block_ids_cpu.dtype
            )
        if should_log_rank(cache):
            logger.warning(
                "[APCDBG/D2H_PREP] tp_rank=%s dp_rank=%s stage=%s "
                "group_idx=%s layer_name=%s layer_idx=%s block_ids_fake=%s "
                "block_ids_real=%s %s",
                getattr(cache, "tp_rank", None),
                getattr(cache, "dp_local_rank", None),
                stage_idx,
                group_idx,
                layer_name,
                layer_idx,
                block_ids_cpu.tolist(),
                (real_block_ids_cpu.tolist()
                 if real_block_ids_cpu is not None else None),
                summarize_map(
                    "real_to_fake",
                    maps[group_idx] if maps is not None and group_idx < len(maps) else None,
                ),
            )

        cache.d2h_stream.wait_event(kv_event)
        CHUNK_SIZE = 512

        # KV diagnostics: probe prefill HBM (Stage 1) before D2H overwrites it
        # should be put after cache.d2h_stream.wait_event(kv_event) --> need to make sure the kv is updated 
        from tools.diagnostics.probes import probe_prefill_hbm
        probe_prefill_hbm(cache, attn_names, attn_metadatas, kv_event)

        # Byte-dimension TP split. Each TP rank offloads 1/tp_world_size of
        # every KV block. Read HBM from device_raw_tensors[stage_idx] and
        # copy into the matching host pinned buffer.
        raw_by_layer = getattr(cache, 'device_raw_tensors_by_layer', None)
        if raw_by_layer is not None and stage_idx < len(raw_by_layer):
            raw = raw_by_layer[stage_idx].get(layer_name, cache.device_raw_tensors[stage_idx])
        else:
            raw = cache.device_raw_tensors[stage_idx]
        page_elems = raw.shape[1]
        elems_per_rank = page_elems // cache.tp_world_size
        rank_start = cache.tp_rank * elems_per_rank
        rank_end = rank_start + elems_per_rank
        num_blocks = len(block_ids_npu)

        # Look up the pre-allocated per-(stage, group) host pinned
        # buffer. Allocated once at initialize_cpu_buffers time.
        host_buf = cache.batch_buffer_raw_stages[stage_idx][group_idx]

        for start in range(0, num_blocks, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, num_blocks)
            src = raw[block_ids_npu[start:end], rank_start:rank_end]
            host_buf[start:end].copy_(src, non_blocking=True)

        d2h_event.record(cache.d2h_stream)

    # Use real block IDs for host pool scatter when available (volatile swap
    # case). Fall back to the original IDs (real in non-volatile baseline).
    host_block_ids = real_block_ids_cpu if real_block_ids_cpu is not None else block_ids_cpu
    fut = cache.d2h_thrp.submit(
        _update_host_cache_thread,
        cache, host_block_ids, group_idx, layer_idx, d2h_event, stage_idx,
    )
    # Append to per-stage lists. Multiple sub-attentions in the same
    # decoder block all use this stage_idx; storing a list lets the
    # block-boundary hook see all of them.
    # Store this group's event/future under (stage_idx, group_idx). The
    # block-boundary hook will iterate every entry in the dict for the
    # current stage so all sub-attention sub-calls within a block are
    # waited on, not just the last one.
    fut_stages = getattr(cache, 'copy_futures_stages', None)
    if fut_stages is not None and 0 <= stage_idx < len(fut_stages):
        bucket = fut_stages[stage_idx]
        if isinstance(bucket, dict):
            bucket[group_idx] = fut
        elif isinstance(bucket, list):  # legacy fallback
            bucket.append(fut)
    evt_stages = getattr(cache, 'd2h_event_stages', None)
    if evt_stages is not None and 0 <= stage_idx < len(evt_stages):
        bucket = evt_stages[stage_idx]
        if isinstance(bucket, dict):
            bucket[group_idx] = d2h_event
        elif isinstance(bucket, list):  # legacy fallback
            bucket.append(d2h_event)
    # NOTE: stage_record is NOT advanced here. Each decoder block fires
    # this function several times (DSA pre_attn, MoMe post_attn(o_conv),
    # MLA post_attn, SWA, ...). All sub-calls within one block must
    # read/write the SAME HBM stage where the model's KV/conv state was
    # written; advancing stage_record per-call would make later sub-
    # calls D2H from a different stage than the one they wrote into.
    # The stage rotation happens once per block in MoEAttnPlugin's
    # _moe_post_sync hook (fired by FusedMoE.forward).


def _hash_kv_slots(cache, label: str) -> None:
    """Task #23: hash the actual KV bytes at a handful of (layer, block)
    positions on whichever side is calling. If prefill and decode both
    print the same hashes for the same (layer, block) coordinates, the
    OX transfer is correct. Mismatched hashes pinpoint the break.
    Guarded by `OMNI_CACHE_VERIFY_TRANSFER=1`.
    """
    import os
    if not int(os.getenv("OMNI_CACHE_VERIFY_TRANSFER", "0")):
        return
    tp_rank = getattr(cache, "tp_rank", 0)
    if tp_rank != 0:
        return  # one rank per side is enough
    try:
        host_kvi = getattr(cache.host_cache, "kvi_tensors", None)
        if not host_kvi:
            return
        pool0 = host_kvi[0]
        # pool0 is per-layer list (after dp-sharding) OR a single tensor.
        if isinstance(pool0, (list, tuple)):
            layers = pool0
        else:
            layers = [pool0[i] for i in range(pool0.shape[0])]
        import hashlib
        probes = [0, min(len(layers) // 2, len(layers) - 1), len(layers) - 1]
        for layer_idx in probes:
            lt = layers[layer_idx]
            if lt.dim() == 4 and lt.shape[0] == 1:
                lt = lt[0]
            for block_id in (1, 2, 4):
                if lt.shape[0] <= block_id:
                    continue
                bbytes = lt[block_id].reshape(-1).view(torch.uint8)
                n = int(min(bbytes.numel(), 4096))
                sha = hashlib.sha256(
                    bbytes[:n].cpu().numpy().tobytes()
                ).hexdigest()[:16]
                logger.warning(
                    "[VERIFY-TRANSFER/%s] layer=%d block=%d first%dB sha=%s",
                    label, layer_idx, block_id, n, sha,
                )
    except Exception as e:
        logger.warning("[VERIFY-TRANSFER/%s] hash failed: %s", label, e)


def synchronize_d2h_hybrid(
    cache: "PrefillOmniCache",
    layer_name_list: List[str],
    attn_metadata_list: List,
    kv_event: torch.npu.Event
) -> None:
    """Synchronize D2H for hybrid attention mode.

    This function handles D2H transfer for hybrid attention with
    multiple attention groups.

    Args:
        cache: The PrefillOmniCache instance
        layer_name_list: List of layer names
        attn_metadata_list: List of attention metadata
        kv_event: Event signaling KV computation completion

    Returns:
        None
    """
    # Issue 2: when prefill ping-pong is enabled (num_stages_layer_copy
    # > 1) the wait on copy_futures[stage_record] is now done in
    # MOMEAttnPlugin.pre_attn (qa_conv gate) BEFORE the layer writes
    # HBM[stage_record]. Doing it here is too late (the source bytes
    # were already overwritten by this layer's compute) and is at
    # best redundant. With num_stages_layer_copy == 1 there is no
    # rotation and the prior single-stage behaviour is preserved.
    if getattr(cache, 'num_stages_layer_copy', 1) <= 1:
        if cache.copy_futures[cache.stage_record] is not None:
            try:
                cache.copy_futures[cache.stage_record].result()
            except Exception as e:
                logger.error(f"Previous D2H failed: {e}")
                raise

    d2h_event = torch.npu.Event(blocking=False, enable_timing=False)

    from .prefill import prepare_d2h_metadata, copy_kv_to_buffers, submit_async_updates

    # KV diagnostics: probe prefill HBM (Stage 1) BEFORE D2H starts
    # This must run before copy_kv_to_buffers reads from HBM
    from tools.diagnostics.probes import probe_prefill_hbm
    probe_prefill_hbm(cache, layer_name_list, attn_metadata_list, kv_event)

    ctx = prepare_d2h_metadata(cache, layer_name_list, attn_metadata_list)

    with torch.npu.stream(cache.d2h_stream):
        cache.d2h_stream.wait_event(kv_event)
        copy_kv_to_buffers(
            cache, layer_name_list, attn_metadata_list, ctx, cache.d2h_stream
        )
        d2h_event.record(cache.d2h_stream)

    # Wait for the async D2H to actually land in the host pool before
    # hashing, otherwise we'd hash pre-D2H bytes.
    fut = cache.copy_futures[cache.stage_record - 1] if hasattr(cache, "copy_futures") else None
    if fut is not None:
        try:
            fut.result()
        except Exception:
            pass

    submit_async_updates(cache, ctx, d2h_event)
    cache.stage_record = (cache.stage_record + 1) % cache.num_stages_layer_copy
    _hash_kv_slots(cache, "prefill-post-d2h")


def synchronize_h2d_decode(
    cache: "DecodeOmniCache",
    batch_device_mem,
    batch_device_max,
    batch_host_mem,
    batch_host_sizes
) -> None:
    """Synchronize H2D for decode operations.

    This function performs Host-to-Device transfer for decode operations,
    loading KV cache from host memory to device memory.

    Args:
        cache: The DecodeOmniCache instance
        batch_device_mem: Device memory batch
        batch_device_max: Device memory max sizes
        batch_host_mem: Host memory batch
        batch_host_sizes: Host memory sizes

    Returns:
        None
    """
    # Hash the host pool BEFORE the H2D: these are the bytes OX just
    # pulled from prefill. Matching prefill's post-D2H hashes confirms
    # the byte transfer happened correctly (Task #23).
    _hash_kv_slots(cache, "decode-post-ox")

    # KV diagnostics: probe decode host pool (Stage 3) after OX, before H2D
    # ctxs are passed through from _post_success via the h2d_q batch

    cache.host_cache.memcpy_async(
        batch_device_mem, batch_device_max,
        batch_host_mem, batch_host_sizes
    )
    # The memcpy above is queued on `ascend_cl_stream` asynchronously. The
    # caller `_post_success` marks the request as KV-ready immediately after
    # we return, and the scheduler will then dispatch the first decode step.
    # Without this sync, the first decode step's attention reads HBM slots
    # whose H2D is still in flight, producing garbage logits — the model's
    # top-1 often lands on EOS, so the request returns empty. Fix: block
    # here until the stream's pending copies have actually landed.
    try:
        cache.host_cache.ascend_cl_stream.sync()
    except Exception:
        import logging as _logging
        _logging.getLogger("vllm.v1.omni").exception(
            "synchronize_h2d_decode: stream.sync failed; continuing "
            "without blocking — expect intermittent empty replies."
        )


def synchronize_d2h_decode(
    cache: "DecodeOmniCache",
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    slot_mapping: torch.Tensor,
    layer_idx: int,
    kv_event: torch.npu.Event
) -> None:
    """Synchronize D2H for decode operations (stub).

    D2H transfer is not implemented for decode cache.

    Raises:
        NotImplementedError: D2H not implemented for decode
    """
    raise NotImplementedError("D2H transfer not implemented for decode cache")


def _update_host_cache_thread(
    cache: "PrefillOmniCache",
    block_ids,
    group_idx,
    layer_idx,
    event,
    stage_idx=0,
):
    """Background thread function: scatter per-rank KV from pinned buffer into host pool.

    Moved from prefill_d2h.py — only caller is synchronize_d2h_prefill.
    """
    torch.npu.set_device(cache.device)
    event.synchronize()

    raw_stages = getattr(cache, 'batch_buffer_raw_stages', None)
    if raw_stages is not None and 0 <= stage_idx < len(raw_stages):
        entry = raw_stages[stage_idx]
        if isinstance(entry, dict):
            stage_buf = entry.get(group_idx)
        else:
            stage_buf = entry
    else:
        stage_buf = None
    if stage_buf is None:
        stage_buf = cache.batch_buffer_raw
    buf = stage_buf[:len(block_ids)]
    total_elems = sum(cache.comp_elem_sizes)
    elems_per_rank = total_elems // cache.tp_world_size
    rank_start = cache.tp_rank * elems_per_rank

    with cache._host_pool_write_lock:
        for i, tensor in enumerate(cache.host_cache.kvi_tensors):
            comp_start = cache.comp_offsets[i]
            comp_end = cache.comp_offsets[i + 1]

            overlap_start = max(rank_start, comp_start)
            overlap_end = min(rank_start + elems_per_rank, comp_end)
            if overlap_start >= overlap_end:
                continue

            buf_offset = overlap_start - rank_start
            comp_offset = overlap_start - comp_start
            length = overlap_end - overlap_start

            src_data = buf[:, buf_offset:buf_offset + length]

            target_layer = tensor[layer_idx]
            block_flat_elems = target_layer.shape[-2] * target_layer.shape[-1]
            target_flat = target_layer.view(torch.bfloat16).reshape(
                cache.dp_world_size_local, -1, block_flat_elems
            )
            target_flat[
                cache.dp_local_rank,
                block_ids,
                comp_offset:comp_offset + length
            ] = src_data
