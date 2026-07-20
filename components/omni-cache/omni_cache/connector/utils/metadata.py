# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Metadata and data structures for connector internal use.

This module provides metadata classes, data structures, and utilities
used by the OmniCache connector for managing request metadata,
connector state, and data type conversions.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata


@dataclass
class ReqMeta:
    local_block_ids: List[List[int]]
    remote_block_ids: List[int]
    remote_host: str
    remote_ox_shard_list: str
    spec_token_ids: Optional[List[int]]
    remote_dp_rank: Optional[int]
    remote_request_id: Optional[str]


@dataclass
class ReqMetaPrefill:
    finish_time: float


def _build_req_meta(
    local_block_ids: List[List[int]],
    kv_transfer_params: Dict[str, Any],
) -> ReqMeta:
    if int(os.getenv("ENABLE_MOCK_P", "0")):
        return ReqMeta(
            local_block_ids=local_block_ids,
            remote_block_ids=[1, 2, 3, 4, 5],
            remote_host="0.0.0.0",
            remote_ox_shard_list="127.0.0.1:15077",
            spec_token_ids=[0],
            remote_dp_rank=0,
            remote_request_id="abc-123",
        )

    return ReqMeta(
        local_block_ids=local_block_ids,
        remote_block_ids=kv_transfer_params["remote_block_ids"],
        remote_host=kv_transfer_params["remote_host_ip"],
        remote_ox_shard_list=kv_transfer_params["remote_cluster_id"],
        spec_token_ids=kv_transfer_params["spec_token_ids"],
        remote_dp_rank=kv_transfer_params.get("remote_dp_rank", 0),
        remote_request_id=kv_transfer_params.get("remote_request_id"),
    )


class DatadistConnectorMetadata(KVConnectorMetadata):
    """Metadata for datadist connector (decode path)."""

    def __init__(self):
        self.requests: Dict[str, ReqMeta] = {}

    def add_new_req(
        self,
        request_id: str,
        local_block_ids: List[List[int]],
        kv_transfer_params: Dict[str, Any],
    ):
        self.requests[request_id] = _build_req_meta(local_block_ids, kv_transfer_params)


class DatadistConnectorMetadataPrefill(KVConnectorMetadata):
    """Metadata for datadist connector (prefill path)."""

    def __init__(self):
        self.requests: Dict[str, ReqMetaPrefill] = {}

    def add_new_req(
        self,
        request_id: str,
        finish_time: float,
    ):
        self.requests[request_id] = ReqMetaPrefill(finish_time=finish_time)


class DTypeUtils:
    """Static helper for converting common dtype strings to byte sizes."""

    _MAP: Dict[str, int] = {
        "int8": 1,
        "uint8": 1,
        "byte": 1,
        "int16": 2,
        "uint16": 2,
        "fp16": 2,
        "float16": 2,
        "bf16": 2,
        "bfloat16": 2,
        "int32": 4,
        "uint32": 4,
        "fp32": 4,
        "float32": 4,
        "int64": 8,
        "uint64": 8,
        "fp64": 8,
        "float64": 8,
    }

    @staticmethod
    def size(dtype) -> int:
        if not isinstance(dtype, str):
            dtype = str(dtype)

        key = dtype.lower().replace('torch.', '')

        if key not in DTypeUtils._MAP:
            raise ValueError(
                f"Unsupported data type: {dtype}. "
                f"Supported types: {list(DTypeUtils._MAP.keys())}"
            )
        return DTypeUtils._MAP[key]

    @staticmethod
    def supported() -> list[str]:
        return list(DTypeUtils._MAP.keys())


@dataclass
class _SendItem:
    request_id: str
    remote_ox_shard_list: str
    src_ids: List[int]
    dst_ids: List[int]
    rank_id: int
    src_dp_rank: int = 0


@dataclass
class PendingReq:
    request_id: str
    local_block_ids: List[List[int]]
    remote_block_ids: List[int]
    remote_ox_shard_list: str
    remote_request_id: Optional[str]
    remote_host_ip: str
    dp_rank: int
    t_submit: float = field(default_factory=time.time)
    t_sent: float = 0.0
    t_resp: float = 0.0


__all__ = [
    "ReqMeta",
    "ReqMetaPrefill",
    "DatadistConnectorMetadata",
    "DatadistConnectorMetadataPrefill",
    "DTypeUtils",
    "_SendItem",
    "PendingReq",
]
