# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Plugins for exposing omni_cache reuse_rate through vllm metrics.

- ``patch_scheduler_stats``: a ``vllm.general_plugins`` entry-point that
  monkey-patches ``SchedulerStats`` to carry a ``reuse_rate`` field so the
  value survives msgspec serialisation across the EngineCore → AsyncLLM
  process boundary.

- ``ReusRateStatLogger``: a ``vllm.stat_logger_plugins`` entry-point that
  creates a per-layer Prometheus Gauge in the API-server process and updates
  it on every scheduler step.
"""

import dataclasses
import logging

logger = logging.getLogger(__name__)

_stats_patched = False


def patch_scheduler_stats() -> None:
    """``vllm.general_plugins`` entry-point.

    Adds ``reuse_rate: list[float] | None = None`` to
    ``vllm.v1.metrics.stats.SchedulerStats`` so the field is included in
    msgspec (de)serialisation.  Idempotent — safe to call from multiple
    processes.
    """
    global _stats_patched
    if _stats_patched:
        return
    _stats_patched = True

    try:
        from vllm.v1.metrics.stats import SchedulerStats
    except ImportError:
        return

    _orig_init = SchedulerStats.__init__

    def _patched_init(self, *args, reuse_rate=None, **kwargs):
        _orig_init(self, *args, **kwargs)
        self.reuse_rate = reuse_rate

    SchedulerStats.__init__ = _patched_init
    SchedulerStats.__annotations__['reuse_rate'] = 'list | None'

    new_field = dataclasses.field(default=None)
    new_field.name = 'reuse_rate'
    new_field._field_type = dataclasses._FIELD
    SchedulerStats.__dataclass_fields__['reuse_rate'] = new_field

    logger.info("[omni_cache] SchedulerStats patched with reuse_rate field")
