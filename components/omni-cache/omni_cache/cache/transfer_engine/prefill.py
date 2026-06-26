# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Device-to-Host (D2H) transfer operations for prefill cache."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from omni_cache.cache.utils.debug import (
    should_log_rank,
    summarize_array,
)
from vllm.logger import init_logger

if TYPE_CHECKING:
    from omni_cache.cache.prefill.prefill_omni_cache import PrefillOmniCache

logger = init_logger("vllm.v1.omni")


@dataclass
class D2HContext:
    """Context for D2H operations."""
    layer_list: List[int] = field(default_factory=list)
    group_list: List[int] = field(default_factory=list)
    num_token_list: List[int] = field(default_factory=list)
    blocks_cache: Dict[str, Tuple] = field(default_factory=dict)
    async_data: List[dict] = field(default_factory=list)


def prepare_d2h_metadata(
    cache: "PrefillOmniCache",
    layer_name_list: List[str],
    attn_metadata_list: List,
) -> D2HContext:
    """Prepare metadata for D2H transfer.

    Args:
        cache: The PrefillOmniCache instance.
        layer_name_list: List of layer names.
        attn_metadata_list: List of attention metadata.

    Returns:
        D2HContext with prepared metadata.
    """
    ctx = D2HContext()

    for layer_name, metadata in zip(layer_name_list, attn_metadata_list):
        group_idx, layer_idx = cache._layer_name_to_group_and_layer_idx(layer_name)
        ctx.layer_list.append(layer_idx)
        ctx.group_list.append(group_idx)
        # For hybrid, batch_token_indices is a list; find first initialized entry as fallback
        token_indices = cache.batch_token_indices
        if isinstance(token_indices, list):
            entry = token_indices[group_idx]
            if entry is None:
                entry = next((e for e in token_indices if e is not None), None)
            num_tokens = entry.shape[0] if entry is not None else 0
        else:
            num_tokens = token_indices.shape[0] if token_indices is not None else 0
        ctx.num_token_list.append(num_tokens)

        device_cache_tmp = cache.device_cache[cache.stage_record]
        raw_buf = device_cache_tmp[layer_name]
        bufs = raw_buf if isinstance(raw_buf, tuple) else (raw_buf,)
        device = bufs[0].device

        spec = cache.kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec
        blocks_cpu, blocks_npu = _prepare_blocks(metadata, device, spec)
        ctx.blocks_cache[layer_name] = (blocks_cpu, blocks_npu)

    return ctx


def _prepare_blocks(attn_meta, device, spec=None):
    """Prepare block tables from attention metadata.

    Uses per-request ``seq_lens`` to compute the exact number of blocks
    each request needs, then only takes that many entries from the
    block table.  This excludes stale blocks left by a previous longer
    request that occupied the same batch row.

    For MoME/Mamba (cache_indices):
      - non-APC: take all non-zero entries (unchanged).
      - APC:     slice between ``cache_indices_start_sched`` and
                 ``cache_indices_end_sched`` per request.
    """
    def _to_1d(t):
        if t is None:
            return None
        if t.dim() == 2 and t.shape[0] == 1:
            t = t.squeeze(0)
        return t.view(-1)

    def _nonzero_ordered(t):
        """Filter non-zero, preserve first-occurrence order."""
        t = t[t != 0]
        return t

    try:
        from omni_cache.cache import omni_cache as _active_cache
    except Exception:
        _active_cache = None

    # ── MoME / Mamba path (cache_indices) ──────────────────────────────
    if hasattr(attn_meta, "cache_indices") and not hasattr(attn_meta, "block_tables"):
        cache_indices = attn_meta.cache_indices  # 2D: (num_reqs, max_indices)
        if cache_indices is None:
            return torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32, device=device)

        start_sched = getattr(attn_meta, "block_idx_first_scheduled_token", None)
        end_sched = getattr(attn_meta, "block_idx_last_scheduled_token", None)

        if start_sched is not None and end_sched is not None:
            # APC: each request i uses cache_indices[i, start:end]
            num_reqs = cache_indices.shape[0] if cache_indices.dim() >= 2 else len(start_sched)
            result: List[int] = []
            for i in range(min(num_reqs, len(start_sched))):
                s, e = int(start_sched[i]), int(end_sched[i]) + 1
                if e > s:
                    row = cache_indices[i, s:e].detach().cpu()
                    result.extend(_nonzero_ordered(row))
                    if should_log_rank(_active_cache):
                        logger.warning(
                            "[APCDBG/D2H_PREP] tp_rank=%s dp_rank=%s stage=%s "
                            "group_idx=%s layer_name=%s branch=mome req_idx=%d "
                            "sched=(%d,%d) %s row_nonzero=%s",
                            getattr(_active_cache, "tp_rank", None),
                            getattr(_active_cache, "dp_local_rank", None),
                            getattr(_active_cache, "stage_record", None),
                            None,
                            None,
                            i,
                            s,
                            e,
                            summarize_array("cache_indices_row", row),
                            _nonzero_ordered(row).tolist(),
                        )
            blocks_cpu = torch.tensor(result, dtype=torch.int32)
        else:
            # Non-APC: flatten all rows, deduplicate (old logic).
            flat = _to_1d(cache_indices)
            if flat is None:
                return torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32, device=device)
            blocks_cpu = flat.detach().cpu()
            blocks_cpu = _nonzero_ordered(blocks_cpu).to(dtype=torch.int32)

        blocks_npu = blocks_cpu.to(device, non_blocking=False)
        if should_log_rank(_active_cache):
            logger.warning(
                "[APCDBG/D2H_PREP] tp_rank=%s dp_rank=%s stage=%s "
                "group_idx=%s layer_name=%s branch=mome final_blocks=%s %s",
                getattr(_active_cache, "tp_rank", None),
                getattr(_active_cache, "dp_local_rank", None),
                getattr(_active_cache, "stage_record", None),
                None,
                None,
                blocks_cpu.tolist(),
                summarize_array("cache_indices", cache_indices),
            )

        return blocks_cpu, blocks_npu

    # ── Attention path (block_tables / block_table) ────────────────────
    block_size = getattr(spec, "block_size", None) if spec is not None else None
    seq_lens = getattr(attn_meta, "seq_lens", None)

    if seq_lens is None or block_size is None:
        raise RuntimeError(
            "seq_lens or block_size missing from attention metadata "
            "-- cannot compute per-request block count."
        )

    # Read num_computed_tokens recorded by the MoME metadata builder
    # during build().  It tells us how many tokens were already
    # processed before this step, so we only offload blocks for the
    # newly-added tail.
    from omni_cache.cache import omni_cache
    _num_comp = getattr(omni_cache, "num_computed_tokens", None)
    if _num_comp is None:
        raise ValueError(
            "num_computed_tokens not found on omni_cache -- "
            "ensure the MoME metadata builder is active (ENABLE_OMNI_CACHE=1)."
        )

    # Normalize to a 1D CPU integer tensor of shape (num_reqs,)
    if isinstance(_num_comp, torch.Tensor):
        if _num_comp.is_floating_point():
            _num_comp = _num_comp.long()
        _num_comp = _num_comp.detach().cpu()
    elif isinstance(_num_comp, (list, tuple)):
        _num_comp = torch.tensor(_num_comp)
    else:
        _num_comp = torch.tensor([_num_comp])

    # Per-request starting block: token_offset // block_size
    start_blocks = _num_comp // block_size

    # Build the 2D block table: (num_reqs, max_blocks).  Never squeeze
    # to 1D -- batch_size=1 must still go through the per-request loop.
    if hasattr(attn_meta, "block_tables"):
        blocks_tensor = attn_meta.block_tables
    else:
        blocks_tensor = attn_meta.block_table

    if blocks_tensor is None:
        return torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32, device=device)

    if blocks_tensor.dim() == 1:
        blocks_tensor = blocks_tensor.unsqueeze(0)  # (N,) -> (1, N)

    # Merge DSA-specific tables (score_states, kv_states) as extra columns.
    extra = []
    for attr in ('score_states_block_table', 'kv_states_block_table'):
        t = getattr(attn_meta, attr, None)
        if t is not None:
            if t.dim() == 1:
                t = t.unsqueeze(0)
            extra.append(t)
    if extra:
        blocks_tensor = torch.cat([blocks_tensor] + extra, dim=1)

    num_reqs = blocks_tensor.shape[0]
    if hasattr(seq_lens, "detach"):
        seq_lens_cpu = seq_lens.detach().cpu()
    else:
        seq_lens_cpu = seq_lens

    # Expand start_blocks to per-request if it was a scalar
    if start_blocks.dim() == 0:
        start_blocks = start_blocks.expand(num_reqs)

    all_blocks: List[int] = []
    seen = set()
    for i in range(num_reqs):
        sl = int(seq_lens_cpu[i])
        if sl <= 0:
            continue
        total_blocks = (sl + block_size - 1) // block_size
        start_b = int(start_blocks[i])
        # If all blocks were already offloaded, skip this request
        if start_b >= total_blocks:
            continue
        row = blocks_tensor[i, start_b:total_blocks].detach().cpu()
        if should_log_rank(omni_cache):
            logger.warning(
                "[APCDBG/D2H_PREP] tp_rank=%s dp_rank=%s stage=%s "
                "group_idx=%s layer_name=%s branch=attention req_idx=%d "
                "seq_len=%d num_computed=%s start_block=%d total_blocks=%d %s",
                getattr(omni_cache, "tp_rank", None),
                getattr(omni_cache, "dp_local_rank", None),
                getattr(omni_cache, "stage_record", None),
                None,
                None,
                i,
                sl,
                int(_num_comp[i]) if i < _num_comp.numel() else None,
                start_b,
                total_blocks,
                summarize_array("row_slice", row),
            )
        for v in row.tolist():
            iv = int(v)
            if iv != 0 and iv not in seen:
                seen.add(iv)
                all_blocks.append(iv)
    blocks_cpu = _to_1d(torch.tensor(all_blocks, dtype=torch.int32))

    blocks_npu = blocks_cpu.to(device, non_blocking=False)
    if should_log_rank(omni_cache):
        logger.warning(
            "[APCDBG/D2H_PREP] tp_rank=%s dp_rank=%s stage=%s "
            "group_idx=%s layer_name=%s branch=attention %s %s %s "
            "start_blocks=%s final_blocks=%s",
            getattr(omni_cache, "tp_rank", None),
            getattr(omni_cache, "dp_local_rank", None),
            getattr(omni_cache, "stage_record", None),
            None,
            None,
            summarize_array("seq_lens", seq_lens_cpu),
            summarize_array("num_computed", _num_comp),
            summarize_array("blocks_tensor", blocks_tensor),
            start_blocks.tolist(),
            blocks_cpu.tolist(),
        )

    return blocks_cpu, blocks_npu


def _resolve_real_block_tables(metadata, cache=None, group_idx=None) -> Optional[torch.Tensor]:
    """Fetch the real block table stashed by the volatile swap.

    `PrepareInputsPlugin._apply_volatile_block_swap` stores per-group
    copies on `input_batch._real_block_tables_per_group`. Older flows
    kept the table on the per-group attention metadata as
    `_real_block_tables`. Check both so the helper works whether the
    swap happens at prepare-inputs time or inside a metadata builder.

    Returns `None` if neither source carries a table, and the caller
    falls back to the legacy per-block gather D2H path.
    """
    tbl = getattr(metadata, "_real_block_tables", None)
    if tbl is not None:
        return tbl
    prefill = getattr(metadata, "prefill", None)
    if prefill is not None:
        stashed = getattr(prefill, "_real_block_tables", None)
        if stashed is not None:
            return stashed

    if cache is not None and group_idx is not None:
        runner = getattr(cache, "runner", None)
        input_batch = getattr(runner, "input_batch", None) if runner is not None else None
        per_group = getattr(input_batch, "_real_block_tables_per_group", None) if input_batch is not None else None
        if per_group is not None and group_idx < len(per_group) and per_group[group_idx] is not None:
            stashed = per_group[group_idx]
            # The swap stashes numpy arrays (copied from block_table.np).
            # D2H's contiguous-path builder expects a torch.Tensor with
            # `.detach().to('cpu')` — wrap numpy stashes accordingly.
            if isinstance(stashed, torch.Tensor):
                return stashed
            import numpy as _np
            if isinstance(stashed, _np.ndarray):
                return torch.from_numpy(stashed)
    return None


def copy_kv_to_buffers(
    cache: "PrefillOmniCache",
    layer_name_list: List[str],
    attn_metadata_list: List,
    ctx: D2HContext,
    d2h_stream,
) -> None:
    """Copy KV data from device HBM into pinned host buffers (Task #13).

    Contiguous-D2H + CPU-scatter design
    -----------------------------------
    Task #11/#12 made the prefill HBM buffer a dense per-attn-type
    region addressed by a fake `volatile_table` whose entries are
    sequential ids `1..N = max_num_reqs * max_blocks_per_req`. For a
    batch with `num_reqs` active requests, only HBM rows `[1 .. 1 +
    num_reqs * max_blocks_per_req)` can contain fresh KV — a
    contiguous prefix. That regularity lets us replace the old
    per-unique-block advanced-index D2H (one small gather copy per
    block) with ONE contiguous D2H copy per layer buffer and a
    single event record. The NPU driver issues it as one DMA instead
    of N small ones: amortises per-launch overhead and saturates
    link bandwidth.

    The fake->real translation is deferred to the CPU worker
    (`update_host_cache_thread_compress`):
    `host_pool[real_block_table[i,j]] = pinned[volatile_table[i,j]]`
    for every valid `(i, j)` — no NPU work on the scatter side.

    If `_real_block_tables` is missing (e.g. a pass that didn't go
    through Task #12's swap), fall back to the legacy gather path so
    the helper stays safe to enable everywhere.
    """
    volatile_table = getattr(cache, "volatile_table", None)
    for index_tmp, (layer_name, metadata) in enumerate(zip(layer_name_list, attn_metadata_list)):
        layer_idx = ctx.layer_list[index_tmp]
        group_idx = ctx.group_list[index_tmp]
        num_tokens = ctx.num_token_list[index_tmp]

        if not cache.is_hybrid_attn:
            kv_block_size = cache.kv_cache_config.kv_cache_groups[group_idx].kv_cache_spec.block_size

            kv_states = _compute_kv_state_from_kv_cache(
                cache,
                cache.device_cache[cache.stage_record],
                layer_name,
                metadata.slot_mapping,
                kv_block_size,
                layer_idx,
            )

        if cache.enable_dsa and not cache.is_hybrid_attn:
            for i, _ in enumerate(kv_states):
                cache.batch_buffer_cpu[i][:num_tokens].copy_(
                    kv_states[i].squeeze(1)[cache.batch_token_indices],
                    non_blocking=True
                )

        elif cache.is_hybrid_attn:
            device_cache_tmp = cache.device_cache[cache.stage_record]
            raw_buf = device_cache_tmp[layer_name]
            bufs = raw_buf if isinstance(raw_buf, tuple) else (raw_buf,)

            real_block_tables = _resolve_real_block_tables(metadata, cache, group_idx)

            # Contiguous-copy fast path: one D2H per layer buffer, scatter
            # on CPU using real<-fake pairs. Requires the volatile swap
            # to have run (real_block_tables stashed) and the cache to
            # expose `volatile_table`. pg2-compatible: pairs full-width
            # `real_block_tables` with `volatile_table[:num_reqs]`
            # element-wise.
            if real_block_tables is not None and volatile_table is not None:
                num_reqs = int(real_block_tables.shape[0])
                max_blocks_per_req = int(real_block_tables.shape[1])

                # Use the POST-SWAP block_table (fake ids as assigned by
                # PrepareInputsPlugin._apply_volatile_block_swap — flat
                # counter across groups) as the src-HBM id. `volatile_table`
                # is positional and disagrees with the plugin's flat
                # assignment for groups > 0.
                fake_bt = None
                if hasattr(metadata, "prefill") and metadata.prefill is not None:
                    fake_bt = getattr(metadata.prefill, "block_table", None)
                    if fake_bt is None:
                        fake_bt = getattr(metadata.prefill, "block_tables", None)
                if fake_bt is not None:
                    fake_bt_cpu = fake_bt.detach().to('cpu', non_blocking=False)
                    if fake_bt_cpu.shape != real_block_tables.shape:
                        fake_bt_cpu = fake_bt_cpu[: real_block_tables.shape[0], : real_block_tables.shape[1]]
                else:
                    fake_bt_cpu = volatile_table[:num_reqs, :max_blocks_per_req].detach().to(
                        'cpu', non_blocking=False
                    )
                fake_max = int(fake_bt_cpu.max().item()) if fake_bt_cpu.numel() > 0 else 0
                hbm_end = 1 + max(fake_max, num_reqs * max_blocks_per_req)
                if should_log_rank(cache):
                    logger.warning(
                        "[APCDBG/D2H_SCATTER] tp_rank=%s dp_rank=%s stage=%s "
                        "group_idx=%s layer_name=%s layer_idx=%s mode=contiguous "
                        "fake_max=%d hbm_end=%d %s %s",
                        getattr(cache, "tp_rank", None),
                        getattr(cache, "dp_local_rank", None),
                        getattr(cache, "stage_record", None),
                        group_idx,
                        layer_name,
                        layer_idx,
                        fake_max,
                        hbm_end,
                        summarize_array("real_block_tables", real_block_tables),
                        summarize_array("fake_bt_cpu", fake_bt_cpu),
                    )

                layer_async = {
                    'layer_idx': layer_idx,
                    'layer_name': layer_name,
                    'group_idx': group_idx,
                    'blocks_cpu': None,
                    'mode': 'contiguous',
                    'real_block_tables_cpu': torch.as_tensor(
                        real_block_tables
                    ).detach().to('cpu', non_blocking=False),
                    'volatile_table_cpu': fake_bt_cpu,
                    'num_reqs': num_reqs,
                    'max_blocks_per_req': max_blocks_per_req,
                    'bufs': [],
                }

                for i, buf_tmp in enumerate(bufs):
                    if num_reqs == 0 or hbm_end <= 1:
                        layer_async['bufs'].append(None)
                        continue

                    # Single contiguous slice — rows 0..hbm_end (exclusive).
                    # We include row 0 so fake id==0 (the null/unused slot)
                    # indexes into a zero row; the CPU scatter skips those
                    # anyway but keeping the slice aligned at 0 lets us use
                    # the fake id directly without offset arithmetic.
                    slab = buf_tmp[:hbm_end]
                    pinned = torch.empty_like(slab, device='cpu', pin_memory=True)
                    pinned.copy_(slab, non_blocking=True)

                    buf_event = torch.npu.Event(blocking=False)
                    buf_event.record(d2h_stream)

                    layer_async['bufs'].append({
                        'buf_idx': i,
                        'pinned': pinned,
                        'event': buf_event,
                    })

                ctx.async_data.append(layer_async)
                continue

            # Fallback: legacy per-unique-block gather path.
            blocks_cpu, blocks_npu = ctx.blocks_cache[layer_name]
            if should_log_rank(cache):
                logger.warning(
                    "[APCDBG/D2H_SCATTER] tp_rank=%s dp_rank=%s stage=%s "
                    "group_idx=%s layer_name=%s layer_idx=%s mode=gather "
                    "blocks_cpu=%s blocks_npu_numel=%d",
                    getattr(cache, "tp_rank", None),
                    getattr(cache, "dp_local_rank", None),
                    getattr(cache, "stage_record", None),
                    group_idx,
                    layer_name,
                    layer_idx,
                    blocks_cpu.tolist(),
                    int(blocks_npu.numel()),
                )

            layer_async = {
                'layer_idx': layer_idx,
                'layer_name': layer_name,
                'group_idx': group_idx,
                'blocks_cpu': blocks_cpu,
                'mode': 'gather',
                'bufs': []
            }

            for i, buf_tmp in enumerate(bufs):
                if blocks_npu.numel() == 0:
                    layer_async['bufs'].append(None)
                    continue

                selected = buf_tmp[blocks_npu]
                pinned = torch.empty_like(selected, device='cpu', pin_memory=True)
                pinned.copy_(selected, non_blocking=True)

                buf_event = torch.npu.Event(blocking=False)
                buf_event.record(d2h_stream)

                layer_async['bufs'].append({
                    'buf_idx': i,
                    'pinned': pinned,
                    'event': buf_event
                })

            ctx.async_data.append(layer_async)

        else:
            for i, _ in enumerate(kv_states):
                cache.batch_buffer_cpu[i][:num_tokens].copy_(
                    _nd_to_nz(cache, kv_states[i].squeeze(1)),
                    non_blocking=True
                )


def submit_async_updates(
    cache: "PrefillOmniCache",
    ctx: D2HContext,
    d2h_event,
) -> None:
    """Submit async updates to thread pool.

    Args:
        cache: The PrefillOmniCache instance.
        ctx: D2H context with async data.
        d2h_event: D2H completion event.
    """
    if cache.is_hybrid_attn and ctx.async_data:
        cache.copy_futures[cache.stage_record] = cache.d2h_thrp.submit(
            update_host_cache_thread_compress,
            cache,
            ctx.async_data,
            d2h_event,
            cache.host_cache.kvi_tensors,
            cache.dp_local_rank,
            cache.dp_world_size_local
        )
    elif not cache.is_hybrid_attn:
        cache.copy_futures[cache.stage_record] = cache.d2h_thrp.submit(
            update_host_cache_thread,
            cache,
            ctx.num_token_list,
            ctx.layer_list,
            d2h_event,
            ctx.group_list,
            None
        )


def update_host_cache_thread(
    cache: "PrefillOmniCache",
    num_tokens,
    layer_idx,
    event,
    group_idx: Optional = None,
    layer_name_list: Optional = None
):
    """Background thread function for updating host cache.

    Args:
        cache: The PrefillOmniCache instance
        num_tokens: Number of tokens to process
        layer_idx: Layer index or list of layer indices
        event: Synchronization event
        group_idx: Group index or list of group indices
        layer_name_list: List of layer names

    Returns:
        None
    """
    import torch
    torch.npu.set_device(cache.device)
    event.synchronize()

    layers = [layer_idx] if isinstance(layer_idx, int) else layer_idx
    groups = [group_idx] if isinstance(group_idx, int) else (
        group_idx or [None] * len(layers)
    )
    token_counts = [num_tokens] if isinstance(num_tokens, int) else num_tokens
    names = [None] * len(layers) if layer_name_list is None else (
        [layer_name_list] if isinstance(layer_name_list, str) else layer_name_list
    )

    dp_rank = cache.dp_local_rank
    dp_size = cache.dp_world_size_local
    kvi_tensors = cache.host_cache.kvi_tensors

    for layer, grp, n_tokens, name in zip(layers, groups, token_counts, names):
        if cache.is_hybrid_attn:
            if grp == 1 and layer == 21:
                draft_index = 0
                slots = cache.batch_slots_cpu[
                    cache.num_attn_group + draft_index
                ][:n_tokens]
                raw_buf = cache.batch_buffer_cpu[
                    cache.num_attn_group + draft_index
                ]
                bufs = raw_buf if isinstance(raw_buf, tuple) else (raw_buf,)
            else:
                slots = cache.batch_slots_cpu[grp][:n_tokens]
                raw_buf = cache.batch_buffer_cpu[grp]
                bufs = raw_buf if isinstance(raw_buf, tuple) else (raw_buf,)
        else:
            slots = cache.batch_slots_cpu[:n_tokens]
            bufs = cache.batch_buffer_cpu

        for i, buf_tmp in enumerate(bufs):
            target_layer = kvi_tensors[i][layer]
            src_data = buf_tmp[:n_tokens]
            last_dim = buf_tmp.shape[-1]
            target_view = target_layer.view(buf_tmp.dtype).view(
                dp_size, -1, last_dim
            )
            target_view[dp_rank, slots] = src_data

def update_host_cache_thread_compress(
    cache: "PrefillOmniCache",
    async_data: list,
    d2h_event: torch.npu.Event,
    kvi_tensors,
    dp_rank: int,
    dp_size: int
):
    """Background-thread scatter of pinned HBM copies into the host KV pool.

    Contiguous-D2H + CPU-scatter design (Task #13)
    ----------------------------------------------
    `copy_kv_to_buffers` does ONE contiguous D2H per layer buffer,
    covering the HBM prefix addressable by the batch's fake block ids.
    This worker then, on CPU:

    1. Waits for each buffer's D2H event.
    2. For `mode == 'contiguous'`, flattens the (real, fake) block-id
       pairs from the metadata's real table and the cache's volatile
       table, drops padding (real_id == 0), and writes
       `host_pool[real_id] = pinned[fake_id]` in one vectorised
       advanced-index assignment per layer buffer.
    3. For `mode == 'gather'` (legacy fallback when
       `_real_block_tables` is missing), preserves the original
       per-unique-block scatter path.

    No Python loops over individual blocks — one indexed write per
    layer buffer.
    """
    import time
    start_time = time.time()

    try:
        torch.npu.set_device(cache.device)

        for layer_data in async_data:
            for buf_info in layer_data['bufs']:
                if buf_info is None:
                    continue
                buf_info['event'].synchronize()

        for layer_data in async_data:
            layer_idx = layer_data['layer_idx']
            mode = layer_data.get('mode', 'gather')

            if mode == 'contiguous':
                real_bt = layer_data['real_block_tables_cpu']
                fake_bt = layer_data['volatile_table_cpu']
                # Flatten to 1-D pairs and drop padding/null entries. Both
                # real_id == 0 and fake_id == 0 represent unused slots; fake
                # 0 is the packed-HBM null row, not fresh KV.
                real_flat = real_bt.reshape(-1).to(torch.int64)
                fake_flat = fake_bt.reshape(-1).to(torch.int64)
                valid = (real_flat != 0) & (fake_flat != 0)
                real_sel = real_flat[valid]
                fake_sel = fake_flat[valid]
                blocks_cpu = None
                if should_log_rank(cache):
                    pairs = list(zip(real_sel.tolist(), fake_sel.tolist()))[:12]
                    logger.warning(
                        "[APCDBG/D2H_SCATTER] tp_rank=%s dp_rank=%s stage=%s "
                        "group_idx=%s layer_name=%s layer_idx=%s mode=contiguous "
                        "pair_count=%d pairs=%s",
                        getattr(cache, "tp_rank", None),
                        getattr(cache, "dp_local_rank", None),
                        getattr(cache, "stage_record", None),
                        layer_data.get("group_idx"),
                        layer_data.get("layer_name"),
                        layer_idx,
                        int(real_sel.numel()),
                        pairs,
                    )
            else:
                real_sel = None
                fake_sel = None
                blocks_cpu = layer_data['blocks_cpu']

            for buf_info in layer_data['bufs']:
                if buf_info is None:
                    continue

                i = buf_info['buf_idx']
                pinned = buf_info['pinned']

                # For unified-pool hybrid (Pangu), kvi_tensors has 1 component but
                # DSA D2H produces 3 buffers (k_nope, k_rope, indexer). Skip extras.
                if i >= len(kvi_tensors):
                    continue
                target_layer = kvi_tensors[i][layer_idx].view(pinned.dtype)
                block_size = pinned.shape[-2]
                pinned_last_dim = pinned.shape[-1]

                # target_layer shape: either (dp, num_blocks, block_size, head_dim) 4D
                # or (num_blocks, block_size, head_dim) 3D
                if target_layer.dim() == 4:
                    target_view = target_layer
                elif target_layer.dim() == 3:
                    target_view = target_layer.unsqueeze(0)  # add dp_size=1 dim
                else:
                    target_view = target_layer.view(dp_size, -1, block_size, pinned_last_dim)

                tgt_head_dim = target_view.shape[-1]

                if mode == 'contiguous':
                    if real_sel.numel() == 0:
                        continue
                    # Gather fake rows out of the pinned contiguous slab,
                    # then scatter into host-pool rows indexed by the real
                    # block ids. One advanced-indexed write per buffer.
                    src = pinned.index_select(0, fake_sel)
                    if tgt_head_dim == pinned_last_dim:
                        target_view[dp_rank, real_sel] = src
                    else:
                        target_view[dp_rank, real_sel, :, :pinned_last_dim] = src
                else:
                    if tgt_head_dim == pinned_last_dim:
                        target_view[dp_rank, blocks_cpu] = pinned
                    else:
                        # Target has padding; write pinned into prefix of each token
                        target_view[dp_rank, blocks_cpu, :, :pinned_last_dim] = pinned

        logger.debug(f"H2H compress took {time.time() - start_time:.3f}s")

    except Exception as e:
        logger.error(f"Compress thread failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def _compute_kv_state_from_kv_cache(
    cache: "PrefillOmniCache",
    kv_cache,
    layer_name,
    slot_mapping,
    kv_block_size,
    layer_idx,
) -> List[torch.Tensor]:
    """Extract KV state tensors from device cache for a given layer.

    Moved from prefill_d2h.py — only caller is copy_kv_to_buffers.
    """
    kv_layer_data = kv_cache[layer_name] if isinstance(kv_cache, dict) else kv_cache

    if not isinstance(slot_mapping, torch.Tensor):
        slot_mapping = torch.tensor(slot_mapping, device=cache.device)

    invalid_mask = slot_mapping < 0
    if invalid_mask.any():
        first_invalid = torch.nonzero(invalid_mask, as_tuple=False)[0].item()
        slot_mapping = slot_mapping[:first_invalid]
    block_indices = slot_mapping // kv_block_size
    slot_offsets = slot_mapping % kv_block_size

    def get_view_and_index(data, dtype=None):
        assert block_indices.shape == slot_offsets.shape
        linear_indices = block_indices * kv_block_size + slot_offsets
        flat_data = data.view(-1, *data.shape[2:])
        return flat_data[linear_indices]

    if isinstance(kv_layer_data, tuple):
        kv_cache_indexer, kv_cache_scale = kv_layer_data
        index_state = get_view_and_index(kv_cache_indexer)
        scale_state = get_view_and_index(kv_cache_scale[:, layer_idx, ...])
        scale_state = scale_state.unsqueeze(-1)
        return [index_state, scale_state]
    else:
        return [get_view_and_index(kv_layer_data)]


def _nd_to_nz(cache: "PrefillOmniCache", tensor):
    """Convert ND format to NZ format.

    Moved from prefill_d2h.py — only caller is copy_kv_to_buffers.
    """
    from omni_cache.cache.prefill.tensor_utils import nd_to_nz
    return nd_to_nz(
        tensor,
        cache.sum_total_len,
        cache.query_mask,
        cache.sum_query_len,
        cache.block_size,
        cache._nz_size,
        cache.is_hybrid_attn,
        cache.batch_token_indices if not cache.is_hybrid_attn else None,
    )
