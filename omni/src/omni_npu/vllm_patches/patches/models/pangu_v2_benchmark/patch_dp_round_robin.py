# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# vllm_patches Reference: https://blog.vllm.ai/2025/11/20/vllm-plugin-system.html
"""Round-robin DP request routing for the internal-LB async client.

Purpose
-------
Forces the internal data-parallel load balancer to dispatch requests strictly
round-robin (request i -> DP rank ``i % dp_size``) instead of the default
greedy least-loaded policy. With a benchmark that fires exactly
``dp_size * max_num_seqs`` requests, round-robin gives EXACTLY ``max_num_seqs``
requests to every rank -- the precondition that makes ``hybrid_aligned_decode``
(threshold = ``max_num_seqs``) align cleanly and deadlock-free:

* every rank reaches exactly the threshold -> the AND-barrier always releases
  (no rank stuck below threshold = no hang);
* no rank exceeds ``max_num_seqs`` -> no ``waiting`` backlog -> the decode gate
  never over-admits past ``max_num_seqs`` (the failure mode that wedged the
  prefill gate under the default uneven LB).

Why server-side
---------------
The benchmark cannot set the per-request ``X-data-parallel-rank`` header (which
vLLM already honors), so the routing decision must be made on the server. The
internal-LB client ``DPLBAsyncMPClient.get_core_engine_for_request`` lives in the
API-server / front-end process; the ``omni_npu_patches`` general plugin is loaded
there too (``EngineArgs.add_cli_args`` -> ``load_general_plugins`` during
``vllm serve`` arg parsing), so this patch takes effect on the routing path.

Default vs round-robin
----------------------
The stock policy (``core_client.py``) scores each engine ``waiting*4 + running``
from coordinator stats refreshed only every ~100ms, with an optimistic local
increment between refreshes. Under a burst it can pile disproportionately onto
one rank before stats catch up -> uneven per-rank counts. This patch replaces the
score loop with a monotonic counter modulo the engine count.

An explicit per-request rank (``request.data_parallel_rank``, set via the
``X-data-parallel-rank`` header) is still honored and bypasses round-robin.

Enabling
--------
Lives in the shared ``pangu_v2_benchmark`` model-patch directory, imported only
when that dir is selected via ``OMNI_NPU_PATCHES_DIR``. It additionally engages
only when ``OMNI_DP_ROUND_ROBIN=1`` -- otherwise the hook delegates to the genuine
upstream greedy LB, so the directory is safe to leave loaded. Orthogonal to (and
bundled with) ``hybrid_aligned_decode`` in the same directory: they patch
different classes (``DPLBAsyncMPClient`` vs ``Scheduler`` / ``DPEngineCoreProc``),
so no conflict.

Caveats
-------
* Internal-LB online DP only (``DPLBAsyncMPClient``, i.e. ``vllm serve
  --data-parallel-size N`` without external LB). External-LB / proxy-fronted
  deployments route upstream and never reach this client.
* Exactness needs a single routing counter: use one API server
  (``--api-server-count 1``, the default). With multiple clients each keeps its
  own counter (offset by ``eng_start_index``); evenness then holds only in
  aggregate, not necessarily per rank.
* Round-robin is by call order, independent of completions, so it stays even
  even if early requests finish before late ones are dispatched.
"""

import os

from vllm.logger import init_logger
from vllm.v1.engine.core_client import DPLBAsyncMPClient

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)

_ENV_FLAG = "OMNI_DP_ROUND_ROBIN"
_COUNTER_ATTR = "_omni_rr_counter"

_logged_engage = False


def _enabled() -> bool:
    # Read at call time, not import/apply time: the flag may be set after vLLM
    # (and this patch) is imported but before serving starts.
    return os.environ.get(_ENV_FLAG, "").strip() in ("1", "true", "True", "ALL")


@register_patch("DPRoundRobinClient", DPLBAsyncMPClient)
class DPRoundRobinClientPatch(VLLMPatch):
    """Route requests round-robin across DP engines instead of least-loaded."""

    _attr_names_to_apply = ["get_core_engine_for_request"]
    # Captured at import time (genuine upstream impl, before this patch is
    # applied); delegated to when disabled or for explicit-rank requests.
    _orig_get_core_engine_for_request = (
        DPLBAsyncMPClient.get_core_engine_for_request
    )

    def get_core_engine_for_request(self, request):
        # Disabled, or an explicit per-request rank was set (X-data-parallel-rank
        # header) -> defer to the genuine upstream greedy/explicit routing.
        if not _enabled() or request.data_parallel_rank is not None:
            return DPRoundRobinClientPatch._orig_get_core_engine_for_request(
                self, request
            )

        num_engines = len(self.core_engines)
        counter = getattr(self, _COUNTER_ATTR, 0)
        # Offset by eng_start_index so multiple clients (api-server-count > 1)
        # start at different ranks; == 0 for the single-client case.
        eng_index = (self.eng_start_index + counter) % num_engines
        setattr(self, _COUNTER_ATTR, counter + 1)

        global _logged_engage
        if not _logged_engage:
            logger.info(
                "[dp-round-robin] routing requests round-robin across %d DP "
                "engines (start_index=%d); greedy LB bypassed",
                num_engines, self.eng_start_index,
            )
            _logged_engage = True

        # Keep the optimistic local load estimate in sync with the choice so any
        # code / metrics reading lb_engines between coordinator refreshes stays
        # consistent (the greedy scorer itself is bypassed).
        self.lb_engines[eng_index][0] += self.client_count

        chosen_engine = self.core_engines[eng_index]
        # Record which engine is chosen, to handle aborts (mirrors upstream).
        self.reqs_in_flight[request.request_id] = chosen_engine
        return chosen_engine
