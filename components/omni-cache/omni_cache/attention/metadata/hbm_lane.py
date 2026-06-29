# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""HBM lane resolution for attention metadata in decode mode.

The decode H2D path (``transfer_engine/decode.py``) claims an HBM lane
``idx_req`` per request via ``record_batch_idx_to_req`` and writes KV data
to slots ``req_offset * idx_req + 1 .. +N`` (or ``idx_req + 1`` for a
Mamba/Mome per-request state). The attention metadata builders in
``omni_cache.attention.backends`` need the inverse mapping — for each
batch row, which lane does this request hold — so the block_table /
slot_mapping it emits address the same HBM slots the H2D just wrote.

In the legacy path, the single source of truth is
``omni_cache.record_batch_idx_to_req`` (lane -> request_id), and
``omni_cache.req_id_to_idx`` is the reverse rebuilt by
``record_current_batch_order``. With ``USE_OMNI_INPUT_BATCH=1``, H2D owns
``omni_cache.req_id_to_idx`` directly and ``record_current_batch_order`` only
syncs current batch order plus device lookup state. In both modes,
``omni_cache.req_ids_update_buffer`` gives the req_ids in batch order.
"""

from __future__ import annotations

from typing import List, Optional


def resolve_batch_idx_reqs(omni_cache, num_reqs: int) -> Optional[List[int]]:
    """Return the HBM lane index ``idx_req`` for each batch row, or None.

    Returns ``None`` when the mapping is not yet available (e.g. the
    GatherSelection updater has not run yet, or the cache is not a
    hybrid/host-mapped decode cache). Callers must fall back to the base
    metadata in that case.
    """
    if omni_cache is None:
        return None

    req_ids_update = getattr(omni_cache, "req_ids_update_buffer", None)
    req_id_to_idx = getattr(omni_cache, "req_id_to_idx", None)
    if not req_ids_update or req_id_to_idx is None:
        return None

    # Graph mode captures at a fixed batch (e.g. 32) and replays at the
    # same batch even when only a subset of rows is live. Returning None
    # for the whole batch let real vllm block ids flow into the captured
    # metadata. Emit per-row lanes with -1 (PAD_SLOT_ID) for padded /
    # unresolved rows so callers can route those slots to pad.
    idx_reqs: list[int] = []
    for i in range(num_reqs):
        if i >= len(req_ids_update):
            idx_reqs.append(-1)
            continue
        rid = req_ids_update[i]
        lane = req_id_to_idx.get(rid, None) if hasattr(req_id_to_idx, "get") else None
        if lane is None:
            idx_reqs.append(-1)
            continue
        idx_reqs.append(int(lane))
    return idx_reqs
