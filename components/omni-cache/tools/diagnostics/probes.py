# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Probe functions for KV cache dump and consistency checking.

Each probe is called from an existing code path (at most 2 lines of
instrumentation per hook site). When diagnostics are disabled (the default),
every probe returns immediately.

Four probe points cover the full PD transfer pipeline:
  1. probe_prefill_hbm   — after KV compute on prefill, before D2H
  2. probe_prefill_host  — after D2H scatter into host pool
  3. probe_decode_host   — after OX transfer into decode host pool
  4. probe_decode_hbm    — after H2D copy into decode HBM
"""

import hashlib
import os
import threading
import time
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

from vllm.logger import init_logger

from tools.diagnostics.config import get_config
from tools.diagnostics.snapshot import (
    KVSnap,
    BlockRecord,
    compute_sha256,
    make_block_record,
    save_snap,
)
from tools.diagnostics.normalizer import (
    COMPONENT_NAMES,
    MOME_COMPONENT_NAMES,
    _get_component_names,
    _get_group_kind,
    extract_block_prefill_host,
    extract_block_decode_host,
    extract_block_prefill_hbm,
    extract_block_decode_hbm,
)

if TYPE_CHECKING:
    from omni_cache.cache.prefill.prefill_omni_cache import PrefillOmniCache
    from omni_cache.cache.decode.decode_omni_cache import DecodeOmniCache
    from omni_cache.connector.utils.metadata import PendingReq

logger = init_logger("vllm.v1.omni")

# ---------------------------------------------------------------------------
# Block ID registry: tracks block IDs for linkage across stages
#
# Design principle:
#   - Prefill cannot get request_id, so uses a shared session folder
#   - Decode uses request_id, but saves remote_block_ids for linkage
#   - Comparison uses: decode's remote_block_ids → prefill's block_id
# ---------------------------------------------------------------------------
class BlockIdRegistry:
    _lock = threading.Lock()
    _prefill_blocks: Dict[str, Dict[int, List[int]]] = {}
    _decode_blocks: Dict[str, Dict] = {}
    # Session folder shared by all stages in one PD round
    _session_id: str = "session_0"
    _session_counter: int = 0

    @classmethod
    def new_session(cls) -> str:
        """Generate a new session ID for the next PD round."""
        with cls._lock:
            cls._session_counter += 1
            cls._session_id = f"session_{cls._session_counter}"
            return cls._session_id

    @classmethod
    def current_session_id(cls) -> str:
        """Get the current session ID."""
        with cls._lock:
            return cls._session_id

    @classmethod
    def register_prefill(cls, session_id: str, group_idx: int, block_ids: List[int]):
        with cls._lock:
            if session_id not in cls._prefill_blocks:
                cls._prefill_blocks[session_id] = {}
            # Accumulate unique block IDs, don't overwrite
            existing = cls._prefill_blocks[session_id].get(group_idx, set())
            if isinstance(existing, list):
                existing = set(existing)
            existing.update(block_ids)
            cls._prefill_blocks[session_id][group_idx] = existing

    @classmethod
    def register_decode(cls, request_id: str, local_block_ids, remote_block_ids, per_group=None):
        with cls._lock:
            cls._decode_blocks[request_id] = {
                "local_block_ids": local_block_ids,
                "remote_block_ids": remote_block_ids,
                "per_group": per_group,
            }

    @classmethod
    def get_prefill_blocks(cls, session_id: str) -> Optional[Dict[int, List[int]]]:
        with cls._lock:
            return cls._prefill_blocks.get(session_id)

    @classmethod
    def get_decode_blocks(cls, request_id: str) -> Optional[Dict]:
        with cls._lock:
            return cls._decode_blocks.get(request_id)

    @classmethod
    def clear(cls):
        with cls._lock:
            cls._prefill_blocks.clear()
            cls._decode_blocks.clear()


# ---------------------------------------------------------------------------
# Helper: resolve virtual layer indices for a group
# ---------------------------------------------------------------------------
def _group_virtual_layers(cache, group_idx: int) -> List[int]:
    kv_cfg = getattr(cache, "kv_cache_config", None)
    if kv_cfg is None or group_idx >= len(kv_cfg.kv_cache_groups):
        return [0]
    group = kv_cfg.kv_cache_groups[group_idx]
    num_layers_in_group = len(group.layer_names)
    return list(range(num_layers_in_group))


def _group_layer_names(cache, group_idx: int) -> List[str]:
    kv_cfg = getattr(cache, "kv_cache_config", None)
    if kv_cfg is None or group_idx >= len(kv_cfg.kv_cache_groups):
        return []
    return kv_cfg.kv_cache_groups[group_idx].layer_names


def _filter_check(cfg, group_idx: int, vl_idx: int, comp_name: str) -> bool:
    if cfg.groups is not None and group_idx not in cfg.groups:
        return False
    if cfg.layers is not None and vl_idx not in cfg.layers:
        return False
    if cfg.components is not None and comp_name not in cfg.components:
        return False
    return True


def _capture_snap(
    snap: KVSnap,
    cfg,
) -> None:
    if cfg.dump_enabled and cfg.should_capture():
        save_snap(snap, cfg.dump_dir)


def _save_remote_to_local_mapping(dump_dir: str, request_id: str, mapping: Dict[int, int]) -> None:
    """Save remote_block_id -> local_block_id mapping for comparison tool."""
    import json
    decode_dir = os.path.join(dump_dir, "decode")
    os.makedirs(decode_dir, exist_ok=True)
    mapping_path = os.path.join(decode_dir, f"mapping_{request_id}.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)
    logger.info("[KV-DIAG] saved block mapping %s (%d entries)", mapping_path, len(mapping))


# ---------------------------------------------------------------------------
# Quick verify: log SHA-256 without saving tensors
# ---------------------------------------------------------------------------
def _quick_verify(
    stage: str,
    request_id: str,
    group_idx: int,
    vl_idx: int,
    block_id: int,
    comp_name: str,
    arr: np.ndarray,
) -> None:
    sha = compute_sha256(arr)
    logger.warning(
        "[KV-VERIFY/%s] req=%s grp=%d vl=%d blk=%d comp=%s sha=%s",
        stage, request_id, group_idx, vl_idx, block_id, comp_name, sha,
    )


# ---------------------------------------------------------------------------
# Probe 1: Prefill HBM (after KV compute, before D2H)
# ---------------------------------------------------------------------------
# Accumulator for prefill_hbm across multiple D2H calls
_prefill_hbm_accumulator: Dict[str, KVSnap] = {}


def clear_accumulators():
    """Clear per-request accumulators. Call between diagnostic requests."""
    global _prefill_hbm_accumulator
    _prefill_hbm_accumulator.clear()
    BlockIdRegistry.clear()
    # Reset session so next request gets a fresh session directory
    BlockIdRegistry._session_id = "session_0"
    BlockIdRegistry._session_counter = 0


def probe_prefill_hbm(
    cache: "PrefillOmniCache",
    layer_name_list: List[str],
    attn_metadata_list: List,
    kv_event: torch.npu.Event,
) -> None:
    """Dump raw HBM block data before D2H transfer.

    Dumps the entire block as a flat byte array - no component splitting.
    Accumulates layers across multiple D2H calls and saves at the end.
    """
    global _prefill_hbm_accumulator
    cfg = get_config()
    if not cfg.active:
        return

    kv_cfg = getattr(cache, "kv_cache_config", None)
    if kv_cfg is None:
        return

    try:
        # Use existing session ID (set by first call) or create new one
        session_id = BlockIdRegistry.current_session_id()
        if session_id == "session_0":
            session_id = BlockIdRegistry.new_session()

        # Get or create accumulator for this session
        dp_rank = getattr(cache, "dp_local_rank", 0) or 0
        tp_rank = getattr(cache, "tp_rank", 0) or 0
        acc_key = f"{session_id}_{dp_rank}_{tp_rank}"

        if acc_key not in _prefill_hbm_accumulator:
            _prefill_hbm_accumulator[acc_key] = KVSnap(
                request_id=session_id,
                stage="prefill_hbm",
                dp_rank=dp_rank,
                tp_rank=tp_rank,
                timestamp_ns=time.time_ns(),
            )

        snap = _prefill_hbm_accumulator[acc_key]

        for layer_name, metadata in zip(layer_name_list, attn_metadata_list):
            group_idx, layer_idx = cache._layer_name_to_group_and_layer_idx(layer_name)
            group_kind = _get_group_kind(kv_cfg, group_idx)

            spec = kv_cfg.kv_cache_groups[group_idx].kv_cache_spec
            real_blocks = _resolve_prefill_block_ids(metadata, spec)
            if not real_blocks:
                continue

            BlockIdRegistry.register_prefill(session_id, group_idx, real_blocks)

            fake_blocks = _resolve_volatile_block_ids(metadata)

            for i, (real_id, fake_id) in enumerate(zip(real_blocks, fake_blocks)):
                if real_id == 0:
                    continue
                arr = extract_block_prefill_hbm(
                    cache, layer_name, fake_id, group_idx, 0,
                )
                if arr.size <= 1:
                    continue

                rec = make_block_record(
                    request_id=session_id,
                    stage="prefill_hbm",
                    group_idx=group_idx,
                    group_kind=group_kind,
                    layer_idx=layer_idx,
                    virtual_layer_idx=layer_idx,
                    block_id=real_id,
                    component="raw",
                    component_idx=0,
                    arr=arr,
                )
                snap.block_records.append(rec)
                snap.tensors[(group_idx, layer_idx, real_id, 0)] = arr

                if cfg.verify and not cfg.dump_enabled:
                    _quick_verify(
                        "prefill_hbm", session_id, group_idx, layer_idx,
                        real_id, "raw", arr,
                    )

        # Save accumulated snapshot if we have tensors
        if snap.tensors and cfg.dump_enabled:
            save_snap(snap, cfg.dump_dir)

    except Exception as e:
        logger.warning("[KV-DIAG] probe_prefill_hbm failed: %s", e)


def _resolve_prefill_block_ids(metadata, spec=None) -> List[int]:
    """Resolve block IDs from metadata for all attention types.

    Uses per-request ``seq_lens`` to compute the exact number of blocks
    each request occupies, then only takes that many entries from the
    block table.  This avoids pulling stale blocks from a previous,
    longer request that occupied the same row.

    For DSA/SWA:  ``block_tables`` with ceil(seq_len / block_size) per row.
    For MoME/Mamba (cache_indices):
      - non-APC: take all non-zero entries.
      - APC:     slice between ``cache_indices_start_sched`` and
                 ``cache_indices_end_sched`` per request.
    """
    # ── helpers ──────────────────────────────────────────────────────────
    def _to_1d(t):
        if t is None:
            return None
        if t.dim() == 2 and t.shape[0] == 1:
            t = t.squeeze(0)
        return t.view(-1)

    def _pick_nonzero(t):
        t = t.detach().cpu()
        t = t[t > 0]
        # Preserve order of first occurrence (torch.unique sorts).
        seen = set()
        ordered = []
        for v in t.tolist():
            iv = int(v)
            if iv not in seen:
                seen.add(iv)
                ordered.append(iv)
        return ordered

    # ── MoME / Mamba path (cache_indices) ────────────────────────────────
    if hasattr(metadata, "cache_indices") and not hasattr(metadata, "block_tables"):
        cache_indices = metadata.cache_indices
        if cache_indices is None:
            return []

        # APC: slice per request using sched boundaries
        start_sched = getattr(metadata, "cache_indices_start_sched", None)
        end_sched = getattr(metadata, "cache_indices_end_sched", None)

        if start_sched is not None and end_sched is not None:
            # APC: each request i uses cache_indices[i, start:end]
            num_reqs = cache_indices.shape[0] if cache_indices.dim() >= 2 else len(start_sched)
            result: List[int] = []
            for i in range(min(num_reqs, len(start_sched))):
                s, e = int(start_sched[i]), int(end_sched[i])
                if e > s:
                    row = cache_indices[i, s:e].detach().cpu()
                    row = row[row > 0]
                    seen = set()
                    for v in row.tolist():
                        iv = int(v)
                        if iv not in seen:
                            seen.add(iv)
                            result.append(iv)
            return result
        else:
            # Non-APC: flatten, filter non-zero, unique
            flat = _to_1d(cache_indices)
            if flat is None:
                return []
            return [int(b) for b in _pick_nonzero(flat).tolist()]

    # ── Attention path (block_tables / block_table) ──────────────────────
    block_size = getattr(spec, "block_size", None) if spec is not None else None
    seq_lens = getattr(metadata, "seq_lens", None)

    if seq_lens is None or block_size is None:
        raise RuntimeError(
            "seq_lens or block_size missing from attention metadata "
            "-- cannot compute per-request block count."
        )

    if hasattr(metadata, "block_tables"):
        blocks_tensor = metadata.block_tables
    else:
        blocks_tensor = getattr(metadata, "block_table", None)

    if blocks_tensor is None:
        return []

    if blocks_tensor.dim() == 1:
        blocks_tensor = blocks_tensor.unsqueeze(0)  # (N,) -> (1, N)

    # Merge DSA-specific tables as extra columns.
    extra = []
    for attr in ('score_states_block_table', 'kv_states_block_table'):
        t = getattr(metadata, attr, None)
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
    all_blocks: List[int] = []
    seen = set()
    for i in range(num_reqs):
        sl = int(seq_lens_cpu[i])
        if sl <= 0:
            continue
        n_blocks = (sl + block_size - 1) // block_size
        row = blocks_tensor[i, :n_blocks].detach().cpu()
        for v in row.tolist():
            iv = int(v)
            if iv != 0 and iv not in seen:
                seen.add(iv)
                all_blocks.append(iv)
    return all_blocks


def _resolve_volatile_block_ids(metadata) -> List[int]:
    vt = getattr(metadata, "volatile_table", None)
    if vt is not None:
        if isinstance(vt, torch.Tensor):
            v = vt.cpu().numpy()
        elif isinstance(vt, np.ndarray):
            v = vt
        else:
            return _resolve_prefill_block_ids(metadata)
        return [int(b) for b in v.flat]
    return _resolve_prefill_block_ids(metadata)


# ---------------------------------------------------------------------------
# Probe 2: Prefill Host Pool (after D2H scatter completes)
# ---------------------------------------------------------------------------
def probe_prefill_host(
    cache: "PrefillOmniCache",
    async_data=None,
    kvi_tensors=None,
    dp_rank: int = 0,
) -> None:
    """Dump raw host pool block data after D2H scatter.

    Dumps the entire block as a flat array - no component splitting.
    """
    cfg = get_config()
    if not cfg.active:
        return

    kv_cfg = getattr(cache, "kv_cache_config", None)
    if kv_cfg is None:
        return

    try:
        session_id = BlockIdRegistry.current_session_id()

        snap = KVSnap(
            request_id=session_id,
            stage="prefill_host",
            dp_rank=dp_rank,
            tp_rank=getattr(cache, "tp_rank", 0) or 0,
            timestamp_ns=time.time_ns(),
        )

        prefill_blocks = BlockIdRegistry.get_prefill_blocks(session_id)
        if not prefill_blocks:
            return

        for group_idx, block_ids in prefill_blocks.items():
            group_kind = _get_group_kind(kv_cfg, group_idx)
            layer_names = _group_layer_names(cache, group_idx)

            if isinstance(block_ids, set):
                block_ids = sorted(block_ids)

            for vl_idx, layer_name in enumerate(layer_names):
                for block_id in block_ids:
                    if block_id == 0:
                        continue
                    arr = extract_block_prefill_host(
                        cache, group_idx, vl_idx, block_id, 0,
                    )
                    if arr.size <= 1:
                        continue

                    rec = make_block_record(
                        request_id=session_id,
                        stage="prefill_host",
                        group_idx=group_idx,
                        group_kind=group_kind,
                        layer_idx=vl_idx,
                        virtual_layer_idx=vl_idx,
                        block_id=block_id,
                        component="raw",
                        component_idx=0,
                        arr=arr,
                    )
                    snap.block_records.append(rec)
                    snap.tensors[(group_idx, vl_idx, block_id, 0)] = arr

                    if cfg.verify and not cfg.dump_enabled:
                        _quick_verify(
                            "prefill_host", session_id, group_idx, vl_idx,
                            block_id, "raw", arr,
                        )

        if snap.tensors:
            _capture_snap(snap, cfg)

    except Exception as e:
        logger.warning("[KV-DIAG] probe_prefill_host failed: %s", e)


# ---------------------------------------------------------------------------
# Probe 3: Decode Host Pool (after OX transfer, before H2D)
# ---------------------------------------------------------------------------
def probe_decode_host(
    cache: "DecodeOmniCache",
    ctxs: List["PendingReq"],
) -> None:
    """Dump raw host pool block data after OX transfer.

    Dumps the entire block as a flat array - no component splitting.
    """
    cfg = get_config()
    if not cfg.active:
        return

    kv_cfg = getattr(cache, "kv_cache_config", None)
    if kv_cfg is None:
        return

    # Optional: only dump for a specific request id prefix
    _target_filter = (os.environ.get("OMNI_KV_DUMP_TARGET_REQ") or "").strip()

    try:
        for ctx in ctxs:
            if _target_filter and _target_filter not in ctx.request_id:
                continue
            request_id = ctx.request_id
            local_block_ids = ctx.local_block_ids
            remote_block_ids = getattr(ctx, "remote_block_ids", [])

            per_group = None
            if isinstance(local_block_ids, list) and len(local_block_ids) >= 2:
                if isinstance(local_block_ids[1], (list, tuple)):
                    per_group = local_block_ids[1]

            # Build remote->local mapping for comparison tool
            remote_to_local = {}
            flat_remote = remote_block_ids[0] if isinstance(remote_block_ids[0], list) else remote_block_ids
            flat_local = local_block_ids[0] if isinstance(local_block_ids[0], list) else local_block_ids
            if isinstance(flat_remote, list) and isinstance(flat_local, list):
                for r, l in zip(flat_remote, flat_local):
                    if r != 0 and l != 0:
                        remote_to_local[int(r)] = int(l)

            BlockIdRegistry.register_decode(
                request_id, local_block_ids, remote_block_ids, per_group,
            )

            snap = KVSnap(
                request_id=request_id,
                stage="decode_host",
                dp_rank=getattr(cache, "dp_local_rank", 0) or 0,
                tp_rank=getattr(cache, "tp_rank", 0) or 0,
                timestamp_ns=time.time_ns(),
                local_block_ids=local_block_ids,
                remote_block_ids=remote_block_ids,
                per_group_block_ids=per_group,
            )

            flat_block_ids = local_block_ids
            if isinstance(local_block_ids, list) and len(local_block_ids) >= 1:
                if isinstance(local_block_ids[0], list):
                    flat_block_ids = local_block_ids[0]

            # Walk groups - dump raw blocks
            if kv_cfg and per_group is not None:
                for group_idx, group_block_ids in enumerate(per_group):
                    if group_idx >= len(kv_cfg.kv_cache_groups):
                        break
                    group_kind = _get_group_kind(kv_cfg, group_idx)
                    layer_names = _group_layer_names(cache, group_idx)

                    for vl_idx in range(len(layer_names)):
                        for block_id in group_block_ids:
                            if block_id == 0:
                                continue
                            arr = extract_block_decode_host(
                                cache, group_idx, vl_idx, block_id, 0,
                            )
                            if arr.size <= 1:
                                continue
                            rec = make_block_record(
                                request_id=request_id,
                                stage="decode_host",
                                group_idx=group_idx,
                                group_kind=group_kind,
                                layer_idx=vl_idx,
                                virtual_layer_idx=vl_idx,
                                block_id=block_id,
                                component="raw",
                                component_idx=0,
                                arr=arr,
                            )
                            snap.block_records.append(rec)
                            snap.tensors[(group_idx, vl_idx, block_id, 0)] = arr

                            if cfg.verify and not cfg.dump_enabled:
                                _quick_verify(
                                    "decode_host", request_id, group_idx,
                                    vl_idx, block_id, "raw", arr,
                                )
            else:
                # Flat block list — dump all groups
                for group_idx in range(len(kv_cfg.kv_cache_groups)):
                    group_kind = _get_group_kind(kv_cfg, group_idx)
                    layer_names = _group_layer_names(cache, group_idx)

                    for vl_idx in range(len(layer_names)):
                        for block_id in flat_block_ids:
                            if isinstance(block_id, (list, tuple)):
                                continue
                            if int(block_id) == 0:
                                continue
                            arr = extract_block_decode_host(
                                cache, group_idx, vl_idx, int(block_id), 0,
                            )
                            if arr.size <= 1:
                                continue
                            rec = make_block_record(
                                request_id=request_id,
                                stage="decode_host",
                                group_idx=group_idx,
                                group_kind=group_kind,
                                layer_idx=vl_idx,
                                virtual_layer_idx=vl_idx,
                                block_id=int(block_id),
                                component="raw",
                                component_idx=0,
                                arr=arr,
                            )
                            snap.block_records.append(rec)
                            snap.tensors[(group_idx, vl_idx, int(block_id), 0)] = arr

            if snap.tensors:
                # Save remote_to_local mapping in a separate file for comparison tool
                if cfg.dump_enabled and remote_to_local:
                    _save_remote_to_local_mapping(cfg.dump_dir, request_id, remote_to_local)
                _capture_snap(snap, cfg)

    except Exception as e:
        logger.warning("[KV-DIAG] probe_decode_host failed: %s", e)


# ---------------------------------------------------------------------------
# Probe 4: Decode HBM (after H2D copy completes)
# ---------------------------------------------------------------------------
def probe_decode_hbm(
    cache: "DecodeOmniCache",
    ctxs: List["PendingReq"],
) -> None:
    """Dump raw HBM block data after H2D copy.

    Dumps the entire block as a flat array - no component splitting.
    """
    cfg = get_config()
    if not cfg.active:
        return

    kv_cfg = getattr(cache, "kv_cache_config", None)
    if kv_cfg is None:
        return

    # Optional: only dump for a specific request id prefix
    _target_filter = (os.environ.get("OMNI_KV_DUMP_TARGET_REQ") or "").strip()

    try:
        for ctx in ctxs:
            if _target_filter and _target_filter not in ctx.request_id:
                continue
            request_id = ctx.request_id
            local_block_ids = ctx.local_block_ids

            per_group = None
            if isinstance(local_block_ids, list) and len(local_block_ids) >= 2:
                if isinstance(local_block_ids[1], (list, tuple)):
                    per_group = local_block_ids[1]

            snap = KVSnap(
                request_id=request_id,
                stage="decode_hbm",
                dp_rank=getattr(cache, "dp_local_rank", 0) or 0,
                tp_rank=getattr(cache, "tp_rank", 0) or 0,
                timestamp_ns=time.time_ns(),
                local_block_ids=local_block_ids,
                remote_block_ids=getattr(ctx, "remote_block_ids", []),
                per_group_block_ids=per_group,
            )

            record_map = getattr(cache, "record_batch_idx_to_req", {}) or {}
            idx_req = None
            for idx, rid in record_map.items():
                if rid == request_id:
                    idx_req = int(idx)
                    break

            if kv_cfg and per_group is not None:
                for group_idx, group_block_ids in enumerate(per_group):
                    if group_idx >= len(kv_cfg.kv_cache_groups):
                        break
                    group_kind = _get_group_kind(kv_cfg, group_idx)
                    layer_names = _group_layer_names(cache, group_idx)

                    # Mirror the block filtering that H2D uses so that
                    # blc_idx matches the HBM slot assignment:
                    #   1. Filter null (0) blocks
                    #   2. For SWA, take only the last swa_num blocks
                    effective_block_ids = [b for b in group_block_ids if b != 0]
                    if group_kind == "swa":
                        spec = kv_cfg.kv_cache_groups[group_idx].kv_cache_spec
                        sw = getattr(spec, "sliding_window", None)
                        if sw is not None:
                            block_size = getattr(spec, "block_size", 128)
                            swa_num = (sw + block_size - 1) // block_size + 1
                            effective_block_ids = effective_block_ids[-swa_num:]

                    for vl_idx, layer_name in enumerate(layer_names):
                        for blc_idx, block_id in enumerate(effective_block_ids):
                            hbm_slot = _resolve_hbm_slot(
                                cache, group_idx, block_id, blc_idx,
                                idx_req, group_kind,
                            )
                            arr = extract_block_decode_hbm(
                                cache, group_idx, layer_name,
                                block_id, hbm_slot, 0,
                            )
                            if arr.size <= 1:
                                continue

                            rec = make_block_record(
                                request_id=request_id,
                                stage="decode_hbm",
                                group_idx=group_idx,
                                group_kind=group_kind,
                                layer_idx=vl_idx,
                                virtual_layer_idx=vl_idx,
                                block_id=block_id,
                                component="raw",
                                component_idx=0,
                                arr=arr,
                            )
                            snap.block_records.append(rec)
                            snap.tensors[(group_idx, vl_idx, block_id, 0)] = arr

                            if cfg.verify and not cfg.dump_enabled:
                                _quick_verify(
                                    "decode_hbm", request_id, group_idx,
                                    vl_idx, block_id, "raw", arr,
                                )

            if snap.tensors:
                _capture_snap(snap, cfg)

    except Exception as e:
        logger.warning("[KV-DIAG] probe_decode_hbm failed: %s", e)


def _resolve_hbm_slot(
    cache: "DecodeOmniCache",
    group_idx: int,
    block_id: int,
    blc_idx: int,
    idx_req: Optional[int],
    group_kind: str,
) -> int:
    hbm_pool = getattr(cache, "hbm_buffer_block_table_pool", None)
    if hbm_pool is None or group_idx >= len(hbm_pool):
        return block_id

    meta = hbm_pool[group_idx]
    kind = meta.get("kind", "attention")

    if kind == "mamba":
        if idx_req is not None:
            return idx_req + 1
        return block_id

    if group_kind == "swa":
        req_offset = meta.get("req_offset", 1)
        if idx_req is not None:
            return req_offset * idx_req + blc_idx + 1
        return block_id

    # DSA or full: 1:1 block ID mapping
    return block_id