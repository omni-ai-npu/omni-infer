# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# vllm_patches Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
"""Bench-aligned decode patch for PD-disaggregation decode nodes.

In a data-parallel decode deployment, requests finish receiving their remote KV
at different times on different DP ranks. Without coordination, each rank starts
decoding as soon as its own first request is ready, which produces ragged
batches across ranks. This patch holds every rank back until *all* DP ranks have
accumulated at least ``OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD`` finished KV
receptions, then releases everywhere at once so the first decode step runs with a
fuller, balanced batch — giving aligned, comparable batches for benchmarking.

Mechanism: while the gate is closed the engine-core loop moves newly finished
``finished_recving_kv_req_ids`` into a private holding set, so the scheduler's
``_update_waiting_for_remote_kv`` sees them as not-yet-ready and keeps the
requests in ``WAITING_FOR_REMOTE_KVS``. Once an all-reduce (MIN == logical AND)
confirms every rank is ready, the held ids are handed back to the scheduler and
the gate is released. This deliberately does NOT override any Scheduler method,
so it composes with other patches that wrap ``_update_waiting_for_remote_kv``.

Enabling: this patch lives in its own ``pd_bench_aligned_decode`` model-patch
directory, so it is only imported/registered when that directory is selected via
``OMNI_VLLM_PATCHES_DIR`` (e.g. ``OMNI_VLLM_PATCHES_DIR="..., pd_bench_aligned_decode"``).
That makes it independently toggleable from any model's patch set. Note this is a
benchmarking aid: gating decode start until a batch accumulates inflates TTFT and
is not intended for production serving.

Tuning:
    OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD - per-rank ready count required
                                              (default 20)
"""

import torch
from torch.distributed import ProcessGroup, ReduceOp

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.v1.engine.core import DPEngineCoreProc

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


def _bench_aligned_decode_threshold() -> int:
    return envs.OMNI_PD_BENCH_ALIGNED_DECODE_THRESHOLD


@register_patch("BenchAlignedDecodeParallelConfigPatch", ParallelConfig)
class BenchAlignedDecodeParallelConfigPatch(VLLMPatch):
    _attr_names_to_apply = ["all_ready_dp"]

    @staticmethod
    def all_ready_dp(dp_group: ProcessGroup, local_ready: bool) -> bool:
        # MIN over ints == logical AND: True only if EVERY rank is ready.
        tensor = torch.tensor(
            [1 if local_ready else 0], dtype=torch.int32, device="cpu"
        )
        torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=dp_group)
        return bool(tensor.item())


@register_patch("BenchAlignedDecodeEngineCorePatch", DPEngineCoreProc)
class BenchAlignedDecodeEngineCorePatch(VLLMPatch):
    _attr_names_to_apply = ["_has_global_unfinished_reqs"]
    # Captured at import time (before any patch is applied), so this is the
    # genuine upstream implementation. We delegate to it for the periodic
    # finish-sync all-reduce instead of duplicating its body.
    _original_has_global_unfinished_reqs = DPEngineCoreProc._has_global_unfinished_reqs

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Bench-aligned decode gate check. Runs in the same loop position as the
        # original (right before the global-unfinished all-reduce).
        scheduler = self.scheduler
        if not getattr(scheduler, "bench_aligned_decode_released", False):
            held = getattr(scheduler, "_bench_aligned_decode_held_req_ids", None)
            if held is None:
                held = set()
                scheduler._bench_aligned_decode_held_req_ids = held

            # Move newly-finished KV receptions out of the scheduler's view so
            # they accumulate instead of being admitted one at a time. The
            # scheduler's _update_waiting_for_remote_kv then keeps these
            # requests in WAITING_FOR_REMOTE_KVS (its "not in finished" path).
            prev_held = len(held)
            if scheduler.finished_recving_kv_req_ids:
                held.update(scheduler.finished_recving_kv_req_ids)
                scheduler.finished_recving_kv_req_ids.clear()

            # held only grows until release, so log only when it advances.
            if len(held) > prev_held:
                logger.info("Bench-aligned decode waiting, %d held locally", len(held))

            local_ready = len(held) >= _bench_aligned_decode_threshold()
            if ParallelConfig.all_ready_dp(self.dp_group, local_ready):
                # Release: hand the accumulated ids back so the next schedule()
                # admits them together, then open the gate permanently.
                scheduler.finished_recving_kv_req_ids.update(held)
                held.clear()
                scheduler.bench_aligned_decode_released = True
                logger.info("Bench-aligned decode released")

        return BenchAlignedDecodeEngineCorePatch._original_has_global_unfinished_reqs(
            self, local_unfinished
        )
