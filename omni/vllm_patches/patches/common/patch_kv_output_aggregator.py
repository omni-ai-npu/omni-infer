# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Split KVOutputAggregator send/recv finished-count barriers.

Upstream uses a single ``expected_finished_count`` for both
``finished_sending`` and ``finished_recving``. Under MultiConnector
(PD LLMDataDist + Offloading) Prefill needs send=1 (pull_done on one
rank) while Offloading loads still need recv=world_size. Patch the
aggregator without modifying vLLM source tree.

Do not patch KVConnectorBase_V1 with get_finished_send/recv: VLLMPatch
shares ``_omni_npu_applied_patches`` via inheritance, so a base patch would
block MultiConnector from applying the same method names.
"""

from __future__ import annotations

from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
from vllm.logger import init_logger
from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


def _finished_count(connector, getter_name: str) -> int | None:
    getter = getattr(connector, getter_name, None)
    if getter is not None:
        return getter()
    return connector.get_finished_count()


@register_patch("KVOutputAggregatorSendRecvSplitPatch", KVOutputAggregator)
class KVOutputAggregatorSendRecvSplitPatch(VLLMPatch):
    """Use separate send/recv expected vote counts in the aggregator."""

    _attr_names_to_apply = [
        "__init__",
        "from_connector",
        "aggregate",
    ]

    def __init__(
        self,
        expected_finished_count: int,
        expected_recv_count: int | None = None,
    ):
        # Complete transfer tracker. Used to track finished requests
        # [req_id -> n_remaining_workers]
        self._recv_remaining_count = dict[str, int]()
        self._send_remaining_count = dict[str, int]()
        self._expected_send_count = expected_finished_count
        self._expected_recv_count = (
            expected_finished_count
            if expected_recv_count is None
            else expected_recv_count
        )
        # Backward-compatible alias; dynamic discovery keeps both in sync.
        self._expected_finished_count = expected_finished_count

    @classmethod
    def from_connector(cls, connector, world_size: int):
        # Optional send/recv hooks (MultiConnector / LLMDataDist); fall back
        # to legacy get_finished_count() when absent.
        send_count = _finished_count(connector, "get_finished_send_count")
        recv_count = _finished_count(connector, "get_finished_recv_count")
        return cls(
            expected_finished_count=send_count or world_size,
            expected_recv_count=recv_count or world_size,
        )

    def aggregate(
        self, outputs: list[ModelRunnerOutput | None], output_rank: int = 0
    ) -> ModelRunnerOutput | None:
        if not outputs[output_rank]:
            return None

        def update_finished_set(
            req_ids: set[str] | None,
            remaining_count_dict: dict[str, int],
            finished_set: set[str],
            expected_count: int,
        ) -> None:
            for req_id in req_ids or ():
                remaining_count = remaining_count_dict.get(req_id, expected_count)
                remaining_count_dict[req_id] = remaining_count - 1
                if remaining_count_dict[req_id] == 0:
                    finished_set.add(req_id)
                    del remaining_count_dict[req_id]

        finished_sending = set[str]()
        finished_recving = set[str]()
        aggregated_kv_connector_stats = None
        aggregated_kv_connector_worker_meta = None
        combined_kv_cache_events = None
        invalid_block_ids = set[int]()
        for model_runner_output in outputs:
            if model_runner_output is None:
                raise RuntimeError("ModelRunnerOutput in outputs must not be None")
            kv_output = model_runner_output.kv_connector_output
            if not kv_output:
                continue
            # Legacy field updates both barriers (NIXL discovery).
            if (
                kv_output.expected_finished_count > 0
                and kv_output.expected_finished_count != self._expected_finished_count
            ):
                logger.debug(
                    "Expected finished requests updated from %d to %d "
                    "(send/recv barriers)",
                    self._expected_finished_count,
                    kv_output.expected_finished_count,
                )
                self._expected_finished_count = kv_output.expected_finished_count
                self._expected_send_count = kv_output.expected_finished_count
                self._expected_recv_count = kv_output.expected_finished_count

            update_finished_set(
                kv_output.finished_sending,
                self._send_remaining_count,
                finished_sending,
                self._expected_send_count,
            )
            update_finished_set(
                kv_output.finished_recving,
                self._recv_remaining_count,
                finished_recving,
                self._expected_recv_count,
            )

            if aggregated_kv_connector_stats is None:
                aggregated_kv_connector_stats = kv_output.kv_connector_stats
            elif kv_connector_stats := kv_output.kv_connector_stats:
                if not isinstance(
                    aggregated_kv_connector_stats, type(kv_connector_stats)
                ):
                    raise TypeError(
                        "kv_connector_stats type mismatch: "
                        f"{type(aggregated_kv_connector_stats)} vs "
                        f"{type(kv_connector_stats)}"
                    )
                aggregated_kv_connector_stats = aggregated_kv_connector_stats.aggregate(
                    kv_connector_stats
                )

            if aggregated_kv_connector_worker_meta is None:
                aggregated_kv_connector_worker_meta = kv_output.kv_connector_worker_meta
            elif kv_connector_worker_meta := kv_output.kv_connector_worker_meta:
                aggregated_kv_connector_worker_meta = (
                    aggregated_kv_connector_worker_meta.aggregate(
                        kv_connector_worker_meta
                    )
                )

            if combined_kv_cache_events is None:
                combined_kv_cache_events = kv_output.kv_cache_events
            elif kv_cache_events := kv_output.kv_cache_events:
                if not isinstance(
                    combined_kv_cache_events,
                    type(kv_cache_events),
                ):
                    raise TypeError(
                        "kv_cache_events type mismatch: "
                        f"{type(combined_kv_cache_events)} vs "
                        f"{type(kv_cache_events)}"
                    )
                worker_kv_cache_events = kv_cache_events.get_all_events()
                combined_kv_cache_events.add_events(worker_kv_cache_events)
                combined_kv_cache_events.increment_workers(1)

            invalid_block_ids |= kv_output.invalid_block_ids

        output = outputs[output_rank]
        if output is None:
            raise RuntimeError(
                f"ModelRunnerOutput at rank {output_rank} must not be None"
            )
        output.kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending or None,
            finished_recving=finished_recving or None,
            kv_connector_stats=aggregated_kv_connector_stats or None,
            kv_cache_events=combined_kv_cache_events or None,
            kv_connector_worker_meta=aggregated_kv_connector_worker_meta or None,
            invalid_block_ids=invalid_block_ids,
            expected_finished_count=self._expected_finished_count,
        )

        return output
