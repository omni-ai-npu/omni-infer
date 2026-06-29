# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Per-layer Prometheus Gauge for omni_cache reuse_rate.

Registered as a ``vllm.stat_logger_plugins`` entry-point so it runs in the
API-server process where ``/metrics`` is served.
"""

import logging

from vllm.config import VllmConfig
from vllm.v1.metrics.loggers import StatLoggerBase

logger = logging.getLogger(__name__)


class ReusRateStatLogger(StatLoggerBase):
    """Stat logger that exposes omni_cache reuse_rate as Prometheus gauges."""

    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self._gauge = None
        self._labels: dict[int, object] = {}

    def _ensure_gauge(self):
        if self._gauge is not None:
            return True
        try:
            from prometheus_client import Gauge
            self._gauge = Gauge(
                name="vllm:omni_cache_reuse_rate",
                documentation="OmniCache KV cache reuse rate per layer.",
                labelnames=["layer"],
                multiprocess_mode="mostrecent",
            )
            return True
        except Exception:
            return False

    def record(self, scheduler_stats=None, iteration_stats=None,
               mm_cache_stats=None, engine_idx=0):
        if scheduler_stats is None:
            return
        reuse_rate = getattr(scheduler_stats, 'reuse_rate', None)
        if reuse_rate is None:
            return
        if not self._ensure_gauge():
            return
        for i, rate in enumerate(reuse_rate):
            if i not in self._labels:
                self._labels[i] = self._gauge.labels(layer=str(i))
            self._labels[i].set(rate)

    def log(self):
        pass

    def log_engine_initialized(self):
        pass
