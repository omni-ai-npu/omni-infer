# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# vllm_patches Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
"""Cross-rank aligned-decode gate for hybrid (proxy-fronted) DP serving.

Purpose
-------
Targets the *hybrid* data-parallel deployment scenario: a model served
data-parallel (e.g. ``vllm serve ... --data-parallel-size 8``) with a global
proxy distributing requests across the DP instances (hybrid / external DP load
balancing). Because requests arrive at different DP ranks at different times,
each rank otherwise starts decoding as soon as its own first request finishes
prefill, producing ragged, differently-sized decode batches across ranks. For
benchmarking TPOT/throughput at a *specified* per-rank batch size you instead
want every rank to begin decoding together, with a full batch.

This patch holds the decode phase on EVERY DP rank until ALL of them have at
least ``OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD`` requests that are ready for decode
(prefill complete). Once all ranks reach the threshold, the gate is released
everywhere on the same engine step, so the first decode batch is full and
aligned across ranks.

This generalizes two existing patches:
  * ``decode_profile_sync`` -- holds decode locally (offline profiling), but only
    correct when every rank has identical work. Here we add a real cross-rank
    barrier so uneven per-rank arrival (the proxy case) is handled correctly.
  * ``pd_bench_aligned_decode`` (!1387) -- same cross-rank barrier idea, but its
    "ready" signal is remote-KV reception (``finished_recving_kv_req_ids``),
    which only exists on PD-disaggregated decode nodes. Here "ready" is local
    prefill completion, which applies to any normal/hybrid DP deployment.

Mechanism
---------
Two cooperating hooks:

1. ``Scheduler.schedule`` (LOCAL, no collective): while the gate is closed,
   prefill-complete (decode-ready) requests are temporarily hidden from
   ``self.running`` so the step schedules ONLY prefills. Requests still
   mid-prefill stay visible so their prefill keeps advancing. The held requests
   are restored (at the front, preserving FCFS) right after, so they remain
   "unfinished" -- keeping the rank inside the DP wave and never decoding until
   release.

2. ``DPEngineCoreProc._has_global_unfinished_reqs`` (CROSS-RANK barrier): every
   engine step it counts this rank's decode-ready requests, forms
   ``local_ready = count >= threshold``, and all-reduces (MIN == logical AND)
   across the DP group. When every rank is ready it flips a shared released flag
   on the scheduler; from then on ``schedule`` runs unmodified.

Why this is deadlock-safe: the barrier all-reduce is placed at the SAME loop
position as vLLM's existing per-step ``has_unfinished_dp`` all-reduce
(``run_busy_loop`` calls ``_has_global_unfinished_reqs`` every iteration of an
active wave). Held requests keep ``local_unfinished`` True, so a gated rank stays
in the wave and keeps calling the barrier; an idle rank that has not yet received
any request still participates in the wave (running DP dummy batches) and reports
``local_ready=False``, holding the whole group until it too fills up. The
released flag is derived from the identical all-reduce result on every rank, so
all ranks stop calling the barrier on the same step.

Enabling
--------
Lives in the shared ``pangu_v2_benchmark`` model-patch directory, so it is only
imported/registered when that directory is selected via ``OMNI_NPU_PATCHES_DIR``
(e.g. ``OMNI_NPU_PATCHES_DIR="..., pangu_v2_benchmark"``). Because it replaces
the global ``Scheduler.schedule``, it additionally engages only when
``OMNI_HYBRID_ALIGNED_DECODE=1`` -- otherwise both hooks are strict no-ops, so the
directory is safe to leave loaded. Mutually exclusive with the other decode-gate
patches (``decode_profile_sync``, ``pd_bench_aligned_decode``): they touch the
same ``Scheduler.schedule`` / ``DPEngineCoreProc`` attrs, so select only one.

Tuning
------
    OMNI_HYBRID_ALIGNED_DECODE            - "1"/"true"/"ALL" to engage the gate.
    OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD  - per-rank decode-ready count required
                                            before release (default 16).

Caveats
-------
This is a benchmarking aid, NOT for production: gating decode start inflates TTFT
for the held requests, and holding prefill-complete requests pins their KV-cache
blocks for the duration of the gate (a large threshold or long prompts can
pressure / exhaust KV cache). The gate is one-shot: once released it stays open
for the lifetime of the engine.
"""

import torch
from torch.distributed import ProcessGroup, ReduceOp

from vllm.config import ParallelConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.core import DPEngineCoreProc

from omni_npu import envs
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)

# Capture the genuine upstream implementations at import time (before any patch
# is applied) so we delegate to them instead of re-implementing their bodies.
_ORIG_SCHEDULE = Scheduler.schedule

_RELEASED_ATTR = "hybrid_aligned_decode_released"

_logged_engage = False


def _hybrid_aligned_decode_enabled() -> bool:
    # Read at call time, not import/apply time: the flag may be set after vLLM
    # (and this patch) is imported but before serving starts.
    return envs.OMNI_HYBRID_ALIGNED_DECODE


def _hybrid_aligned_decode_threshold() -> int:
    return envs.OMNI_HYBRID_ALIGNED_DECODE_THRESHOLD


def _ready_for_decode(request) -> bool:
    """A request is decode-ready once its prompt is fully computed.

    vLLM advances ``num_computed_tokens`` in ``_update_after_schedule`` right
    after a request is scheduled, so this is robust to both chunked prefill and
    async scheduling.
    """
    return request.num_computed_tokens >= request.num_prompt_tokens


@register_patch("HybridAlignedDecodeParallelConfig", ParallelConfig)
class HybridAlignedDecodeParallelConfigPatch(VLLMPatch):
    """Adds a DP all-reduce helper used as the cross-rank release barrier."""

    _attr_names_to_apply = ["all_ready_dp"]

    @staticmethod
    def all_ready_dp(dp_group: ProcessGroup, local_ready: bool) -> bool:
        # MIN over ints == logical AND: True only if EVERY rank is ready.
        tensor = torch.tensor(
            [1 if local_ready else 0], dtype=torch.int32, device="cpu"
        )
        torch.distributed.all_reduce(tensor, op=ReduceOp.MIN, group=dp_group)
        return bool(tensor.item())


@register_patch("HybridAlignedDecodeScheduler", Scheduler)
class HybridAlignedDecodeSchedulerPatch(VLLMPatch):
    """Hold decode-ready requests (schedule prefills only) until released."""

    _attr_names_to_apply = ["schedule"]

    def schedule(self) -> SchedulerOutput:
        # Check the released flag first: once the gate is open the per-step hot
        # path is a single attribute read (no env lookup).
        if getattr(self, _RELEASED_ATTR, False) or not _hybrid_aligned_decode_enabled():
            return _ORIG_SCHEDULE(self)

        # Requests admitted for prefill *this* step are appended to self.running
        # inside _ORIG_SCHEDULE, so right now self.running only holds requests
        # carried over from previous steps. Hide the decode-ready ones so this
        # step schedules ONLY prefills; keep mid-prefill ones so the original
        # scheduler keeps advancing their prefill.
        held = [r for r in self.running if _ready_for_decode(r)]
        if not held:
            # Nothing decode-ready to hold; let the original scheduler run.
            return _ORIG_SCHEDULE(self)

        global _logged_engage
        if not _logged_engage:
            logger.info(
                "[hybrid-aligned-decode] gate closed: holding %d decode-ready "
                "request(s) (waiting=%d, running=%d)",
                len(held), len(self.waiting), len(self.running),
            )
            _logged_engage = True

        self.running = [r for r in self.running if not _ready_for_decode(r)]
        try:
            output = _ORIG_SCHEDULE(self)
        finally:
            # Restore held requests at the FRONT, preserving FCFS order: they
            # were prefilled in earlier steps, so they precede any request still
            # prefilling / newly admitted this step.
            self.running = held + self.running
        return output


@register_patch("HybridAlignedDecodeEngineCore", DPEngineCoreProc)
class HybridAlignedDecodeEngineCorePatch(VLLMPatch):
    """Cross-rank barrier: release the gate once ALL ranks reach the threshold."""

    _attr_names_to_apply = ["_has_global_unfinished_reqs"]
    # Captured at import time (genuine upstream impl); delegated to for the
    # periodic finish-sync all-reduce instead of duplicating its body.
    _original_has_global_unfinished_reqs = (
        DPEngineCoreProc._has_global_unfinished_reqs
    )

    def _has_global_unfinished_reqs(self, local_unfinished: bool) -> bool:
        # Runs in the same loop position as the original (right before the
        # global-unfinished all-reduce), so every rank in an active wave reaches
        # the barrier the same number of times. Check the released flag first:
        # once the gate is open the per-step hot path is a single attribute read
        # (no env lookup, no all-reduce). released flips identically across ranks
        # (it is derived from the all_ready_dp result), so all ranks short-circuit
        # on the same step and stay in lockstep.
        scheduler = self.scheduler
        if not getattr(scheduler, _RELEASED_ATTR, False) and _hybrid_aligned_decode_enabled():
            ready = sum(1 for r in scheduler.running if _ready_for_decode(r))
            local_ready = ready >= _hybrid_aligned_decode_threshold()
            if ParallelConfig.all_ready_dp(self.dp_group, local_ready):
                setattr(scheduler, _RELEASED_ATTR, True)
                logger.info(
                    "[hybrid-aligned-decode] gate released: all DP ranks reached "
                    "the threshold (this rank ready=%d)", ready,
                )

        return HybridAlignedDecodeEngineCorePatch._original_has_global_unfinished_reqs(
            self, local_unfinished
        )
