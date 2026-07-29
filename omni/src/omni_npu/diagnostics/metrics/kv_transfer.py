# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""worker→前端指标通道，走 vLLM KVConnectorStats（不碰 multiprocess）。

一条通道承载 KV 传输失败 + worker 显存，data 字典分两命名空间：
    {"fail": {kind: count}, "mem": {rank: {"alloc": b, "reserved": b}}}
worker 每步 collect() 塞入 connector 的 stats，经 aggregate 汇到前端单进程 observe 发射。
"""

import os
import threading
from dataclasses import dataclass
from typing import Any, Optional

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
)
from vllm.logger import init_logger

from omni_npu.diagnostics.metrics import worker_mem

logger = init_logger(__name__)

# OMNI_METRICS_KV_TRANSFER_SELFTEST=1 时 DecodeWorker 启动注入一次合成失败，供 smoke test。
_SELFTEST = os.getenv("OMNI_METRICS_KV_TRANSFER_SELFTEST", "0") == "1"

_lock = threading.Lock()
_counts: dict[str, int] = {}  # kind -> 累计失败数（自上次 drain）


def record_failure(kind: str = "pull", n: int = 1) -> None:
    # connector 传输/守护线程里调用，仅内存累加，绝不外抛。kind: pull / send_done。
    try:
        with _lock:
            _counts[kind] = _counts.get(kind, 0) + n
    except Exception:  # noqa: BLE001
        pass


def _drain_failures() -> dict[str, int]:
    with _lock:
        if not _counts:
            return {}
        snap = dict(_counts)
        _counts.clear()
        return snap


def maybe_selftest() -> None:
    if _SELFTEST:
        record_failure("selftest")
        logger.info(
            "kv_transfer selftest failure injected (kind=selftest); "
            'expect omni_npu:kv_transfer_failures{kind="selftest"} at frontend /metrics'
        )


@dataclass
class OmniConnectorStats(KVConnectorStats):
    def reset(self) -> None:
        self.data = {}

    def is_empty(self) -> bool:
        return not (self.data.get("fail") or self.data.get("mem"))

    def aggregate(self, other: "KVConnectorStats") -> "OmniConnectorStats":
        # fail 逐 kind 相加；mem 按 rank 并集（每 rank 仅自己写，故并集即保留 per-rank）。
        fail = dict(self.data.get("fail") or {})
        for k, v in (other.data.get("fail") or {}).items():
            fail[k] = fail.get(k, 0) + v
        mem = dict(self.data.get("mem") or {})
        mem.update(other.data.get("mem") or {})
        return OmniConnectorStats(data={"fail": fail, "mem": mem})

    def reduce(self) -> dict[str, Any]:
        out: dict[str, Any] = {f"fail_{k}": v for k, v in (self.data.get("fail") or {}).items()}
        out["mem_ranks"] = len(self.data.get("mem") or {})
        return out


class OmniConnectorPromMetrics(KVConnectorPromMetrics):
    def __init__(self, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        ln = list(labelnames)
        self._failures = self._counter_cls(
            name="omni_npu:kv_transfer_failures",
            documentation="PD KV transfer failures on decode side (by kind).",
            labelnames=ln + ["kind"],
        )
        self._mem_alloc = self._gauge_cls(
            name="omni_npu:worker_mem_allocated_bytes",
            documentation="Torch allocator allocated bytes per worker rank.",
            labelnames=ln + ["rank"],
        )
        self._mem_reserved = self._gauge_cls(
            name="omni_npu:worker_mem_reserved_bytes",
            documentation="Torch allocator reserved bytes per worker rank.",
            labelnames=ln + ["rank"],
        )
        logger.info(
            "omni connector prom metrics registered (KVConnectorStats channel): "
            "kv_transfer_failures + worker_mem_{allocated,reserved}_bytes"
        )

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        lv = self.per_engine_labelvalues.get(engine_idx)
        if lv is None or not transfer_stats_data:
            return
        for kind, count in (transfer_stats_data.get("fail") or {}).items():
            if count:
                self._failures.labels(*lv, kind).inc(count)
        for rank, m in (transfer_stats_data.get("mem") or {}).items():
            self._mem_alloc.labels(*lv, rank).set(m.get("alloc", 0))
            self._mem_reserved.labels(*lv, rank).set(m.get("reserved", 0))


# ── LLMDataDistConnector 委托的三个入口 ────────────────────────────────────
def collect(rank: Optional[int] = None) -> Optional[OmniConnectorStats]:
    # worker 每步经 connector 调用；仅 worker 角色传 rank 采显存。两者都空返回 None。
    fail = _drain_failures()
    mem = worker_mem.maybe_sample(rank) if rank is not None else None
    if not fail and not mem:
        return None
    return OmniConnectorStats(data={"fail": fail, "mem": mem or {}})


def build_stats(data: Optional[dict[str, Any]] = None) -> OmniConnectorStats:
    return OmniConnectorStats(data=dict(data) if data else {})


def build_prom_metrics(
    vllm_config, metric_types, labelnames, per_engine_labelvalues
) -> OmniConnectorPromMetrics:
    return OmniConnectorPromMetrics(
        vllm_config, metric_types, labelnames, per_engine_labelvalues
    )
