# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Propagate finished send/recv vote counts from child KV connectors.

LLMDataDistConnector (prefill) reports finished_sending from a single TP
worker (send count = 1). OffloadingConnector loads still need all ranks
(recv count = world_size). Upstream MultiConnector returns None for
get_finished_count, and a single shared expected_finished_count cannot
satisfy both barriers — so we expose separate send/recv aggregations.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


def _aggregate_child_counts(connectors, getter_name: str, reduce_min: bool) -> int | None:
    result = None
    for c in connectors:
        getter = getattr(c, getter_name, None)
        count = getter() if getter is not None else c.get_finished_count()
        if count is None:
            continue
        if result is None:
            result = count
        else:
            result = min(result, count) if reduce_min else max(result, count)
    return result


@register_patch("MultiConnectorGetFinishedCountPatch", MultiConnector)
class MultiConnectorGetFinishedCountPatch(VLLMPatch):
    _attr_names_to_apply = [
        "get_finished_count",
        "get_finished_send_count",
        "get_finished_recv_count",
    ]

    def get_finished_send_count(self) -> int | None:
        # PD pull_done: take the tightest (min) non-None send barrier.
        return _aggregate_child_counts(
            self._connectors, "get_finished_send_count", reduce_min=True
        )

    def get_finished_recv_count(self) -> int | None:
        # Offloading / remote KV loads: take the loosest (max) non-None recv
        # barrier. If every child returns None, Aggregator falls back to
        # world_size — correct for TP-sharded CPU offload loads.
        return _aggregate_child_counts(
            self._connectors, "get_finished_recv_count", reduce_min=False
        )

    def get_finished_count(self) -> int | None:
        # Legacy single-count API: send-oriented min over *child* connectors.
        # Do not call self.get_finished_send_count() — that is this same patch
        # method; aggregating children cannot recurse into self.
        return _aggregate_child_counts(
            self._connectors, "get_finished_send_count", reduce_min=True
        )
