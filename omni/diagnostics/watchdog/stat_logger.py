# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Omni-npu StatLogger plugin that drives the hang-detection heartbeat.

Registered through vLLM's `vllm.stat_logger_plugins` entry point, so no source
patch or fork is required. See pyproject.toml.

`OmniNpuStatLogger` extends `AggregateStatLoggerBase`. vLLM instantiates it once
per (vllm_config, engine_indexes) and calls:
  - `record()` after each engine scheduling step -> `heartbeat.mark_progress()`
  - `log_engine_initialized()` after engine init -> `heartbeat.mark_initialized()`
  - `record_sleep_state()` on engine sleep/wake -> `heartbeat.mark_sleeping()`
"""

from typing import TYPE_CHECKING, Optional

from vllm.logger import init_logger
from vllm.v1.metrics.loggers import AggregateStatLoggerBase

from omni_npu.diagnostics.watchdog import heartbeat

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.metrics.stats import IterationStats, SchedulerStats

logger = init_logger(__name__)


class OmniNpuStatLogger(AggregateStatLoggerBase):
    """StatLogger that feeds engine progress into the watchdog heartbeat."""

    def __init__(
        self, vllm_config: "VllmConfig", engine_indexes: Optional[list[int]] = None
    ):
        if engine_indexes is None:
            engine_indexes = [0]
        self.engine_indexes = engine_indexes
        self.vllm_config = vllm_config

        model_name = vllm_config.model_config.served_model_name
        logger.info(
            "OmniNpuStatLogger loaded: model=%s engines=%s", model_name, engine_indexes
        )

    def record(
        self,
        scheduler_stats: Optional["SchedulerStats"],
        iteration_stats: Optional["IterationStats"],
        mm_cache_stats=None,
        engine_idx: int = 0,
    ) -> None:
        heartbeat.mark_progress(engine_idx)

    def record_sleep_state(self, sleep: int = 0, level: int = 0) -> None:
        # vLLM signals sleep with arg 1 and wake with arg 0 (see async_llm.py
        # sleep()/wake_up() and PrometheusStatLogger.record_sleep_state).
        # AggregateStatLoggerBase manages multiple engines; the sleep signal
        # applies to every engine owned by this logger instance.
        for engine_idx in self.engine_indexes:
            heartbeat.mark_sleeping(engine_idx, sleeping=bool(sleep))

    def log_engine_initialized(self) -> None:
        for engine_idx in self.engine_indexes:
            heartbeat.mark_initialized(engine_idx)

    def log(self) -> None:  # noqa: D401 - periodic flush hook, no-op for Prometheus
        pass
