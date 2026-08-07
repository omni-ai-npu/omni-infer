# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from typing_extensions import override

import torch_npu

from vllm.config import ProfilerConfig
from vllm.logger import init_logger
from vllm.profiler.wrapper import WorkerProfiler

logger = init_logger(__name__)


def _to_bool(value) -> bool:
    """Coerce config values to bool for torch_npu.profiler."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("1", "true")


class NpuProfilerWrapper(WorkerProfiler):
    """NPU worker profiler backed by torch_npu.profiler."""

    def __init__(
        self,
        profiler_config: ProfilerConfig,
        worker_name: str,
        local_rank: int,
    ) -> None:
        super().__init__(profiler_config)

        self.local_rank = local_rank
        self.profiler_config = profiler_config
        torch_profiler_trace_dir = profiler_config.torch_profiler_dir
        if local_rank in (None, 0):
            logger.info_once(
                "NPU profiling enabled. Traces will be saved to: %s; "
                "record_shapes=%s, profile_memory=%s, with_stack=%s, with_flops=%s",
                torch_profiler_trace_dir,
                profiler_config.torch_profiler_record_shapes,
                profiler_config.torch_profiler_with_memory,
                profiler_config.torch_profiler_with_stack,
                profiler_config.torch_profiler_with_flops,
            )

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        )

        profiler_schedule = None
        if (
            profiler_config.warmup_iterations > 0
            or profiler_config.wait_iterations > 0
        ):
            profiler_schedule = torch_npu.profiler.schedule(
                skip_first=0,
                wait=profiler_config.wait_iterations,
                warmup=profiler_config.warmup_iterations,
                active=profiler_config.active_iterations,
                repeat=1,
            )
            if local_rank in (None, 0):
                logger.info_once(
                    "Profiler schedule configured: wait=%d, warmup=%d, active=%d",
                    profiler_config.wait_iterations,
                    profiler_config.warmup_iterations,
                    profiler_config.active_iterations,
                )

        trace_handler = torch_npu.profiler.tensorboard_trace_handler(
            torch_profiler_trace_dir,
            worker_name=worker_name,
        )

        self.profiler = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            schedule=profiler_schedule,
            record_shapes=_to_bool(profiler_config.torch_profiler_record_shapes),
            profile_memory=_to_bool(profiler_config.torch_profiler_with_memory),
            with_stack=_to_bool(profiler_config.torch_profiler_with_stack),
            with_flops=_to_bool(profiler_config.torch_profiler_with_flops),
            experimental_config=experimental_config,
            on_trace_ready=trace_handler,
        )

        self._uses_schedule = profiler_schedule is not None
        self._warmup_steps_remaining = max(
            profiler_config.wait_iterations
            + profiler_config.warmup_iterations
            - 1,
            0,
        )

    @override
    def _start(self) -> None:
        self.profiler.start()

    @override
    def _stop(self) -> None:
        self.profiler.stop()

    @override
    def _profiler_step(self) -> bool:
        if self._uses_schedule:
            self.profiler.step()
            if self._warmup_steps_remaining > 0:
                self._warmup_steps_remaining -= 1
                return False
        return True
