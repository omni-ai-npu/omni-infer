# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Data model for KV cache snapshots: BlockRecord, KVSnap, and .npz serialization."""

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from vllm.logger import init_logger

logger = init_logger("vllm.v1.omni")


@dataclass
class BlockRecord:
    request_id: str
    stage: str
    group_idx: int
    group_kind: str
    layer_idx: int
    virtual_layer_idx: int
    block_id: int
    component: str
    component_idx: int
    shape: Tuple[int, ...]
    sha256: str
    timestamp_ns: int


@dataclass
class KVSnap:
    request_id: str
    stage: str
    dp_rank: int
    tp_rank: int
    timestamp_ns: int
    block_records: List[BlockRecord] = field(default_factory=list)
    tensors: Dict[Tuple[int, int, int, int], np.ndarray] = field(default_factory=dict)
    local_block_ids: Optional[list] = None
    remote_block_ids: Optional[list] = None
    per_group_block_ids: Optional[list] = None


def _array_key(*key) -> str:
    """Serialize a tensor key to an NPZ array name.

    2-tuple: (group_idx, vl_idx) -> 'g0_vl7'
    4-tuple: (group_idx, vl_idx, block_id, comp_idx) -> 'g0_vl7_b1_c0'
    """
    if len(key) == 2:
        return f"g{key[0]}_vl{key[1]}"
    return f"g{key[0]}_vl{key[1]}_b{key[2]}_c{key[3]}"


def compute_sha256(arr: np.ndarray, max_bytes: int = 0) -> str:
    raw = arr.tobytes()
    if max_bytes and len(raw) > max_bytes:
        raw = raw[:max_bytes]
    return hashlib.sha256(raw).hexdigest()[:16]


def save_snap(snap: KVSnap, dump_dir: str) -> str:
    """Save snapshot to disk.

    Folder structure:
      prefill stage: {dump_dir}/prefill/{request_id}/  (request_id is session_N)
      decode stage:  {dump_dir}/decode/{request_id}/   (request_id is actual request_id)

    Comparison tool uses decode's remote_block_ids to match prefill's block_ids.
    """
    # Separate folders for prefill vs decode
    if snap.stage.startswith("prefill"):
        stage_dir = "prefill"
    else:
        stage_dir = "decode"

    req_dir = os.path.join(dump_dir, stage_dir, f"req_{snap.request_id}")
    os.makedirs(req_dir, exist_ok=True)

    arrays = {}
    for key_tuple, arr in snap.tensors.items():
        arrays[_array_key(*key_tuple)] = arr

    meta = {
        "request_id": snap.request_id,
        "stage": snap.stage,
        "dp_rank": snap.dp_rank,
        "tp_rank": snap.tp_rank,
        "timestamp_ns": snap.timestamp_ns,
        "block_records": [asdict(r) for r in snap.block_records],
        "local_block_ids": snap.local_block_ids,
        "remote_block_ids": snap.remote_block_ids,
        "per_group_block_ids": snap.per_group_block_ids,
    }
    arrays["metadata"] = np.array(json.dumps(meta))

    path = os.path.join(
        req_dir,
        f"{snap.stage}_dp{snap.dp_rank}_tp{snap.tp_rank}.npz",
    )
    np.savez(path, **arrays)
    logger.info("[KV-DIAG] saved %s (%d tensors)", path, len(snap.tensors))
    return path


def load_snap(path: str) -> KVSnap:
    data = np.load(path, allow_pickle=False)
    meta = json.loads(str(data["metadata"]))

    tensors = {}
    for key in data.files:
        if key == "metadata":
            continue
        parts = key.split("_")
        grp = int(parts[0][1:])
        vl = int(parts[1][2:])
        if len(parts) == 2:
            # Per-layer key: g{grp}_vl{vl}
            tensors[(grp, vl)] = data[key]
        else:
            # Legacy per-block key: g{grp}_vl{vl}_b{blk}_c{comp}
            blk = int(parts[2][1:])
            comp = int(parts[3][1:])
            tensors[(grp, vl, blk, comp)] = data[key]

    records = [BlockRecord(**r) for r in meta["block_records"]]
    return KVSnap(
        request_id=meta["request_id"],
        stage=meta["stage"],
        dp_rank=meta["dp_rank"],
        tp_rank=meta["tp_rank"],
        timestamp_ns=meta["timestamp_ns"],
        block_records=records,
        tensors=tensors,
        local_block_ids=meta.get("local_block_ids"),
        remote_block_ids=meta.get("remote_block_ids"),
        per_group_block_ids=meta.get("per_group_block_ids"),
    )


def make_block_record(
    request_id: str,
    stage: str,
    group_idx: int,
    group_kind: str,
    layer_idx: int,
    virtual_layer_idx: int,
    block_id: int,
    component: str,
    component_idx: int,
    arr: np.ndarray,
) -> BlockRecord:
    return BlockRecord(
        request_id=request_id,
        stage=stage,
        group_idx=group_idx,
        group_kind=group_kind,
        layer_idx=layer_idx,
        virtual_layer_idx=virtual_layer_idx,
        block_id=block_id,
        component=component,
        component_idx=component_idx,
        shape=tuple(arr.shape),
        sha256=compute_sha256(arr),
        timestamp_ns=time.time_ns(),
    )
