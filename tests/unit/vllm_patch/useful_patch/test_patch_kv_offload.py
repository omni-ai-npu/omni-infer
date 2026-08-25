# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for KV offload MultiConnector / aggregator patches."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from omni_npu.vllm_patches.usefull_patch.patch_kv_output_aggregator import (
    KVOutputAggregatorSendRecvSplitPatch,
    _finished_count,
)
from omni_npu.vllm_patches.usefull_patch.patch_multi_connector import (
    MultiConnectorGetFinishedCountPatch,
    _aggregate_child_counts,
)


class _LegacyConnector:
    def get_finished_count(self):
        return 4


class _SplitConnector:
    def get_finished_send_count(self):
        return 1

    def get_finished_recv_count(self):
        return 8

    def get_finished_count(self):
        return 99


def test_finished_count_prefers_named_getter():
    assert _finished_count(_SplitConnector(), "get_finished_send_count") == 1
    assert _finished_count(_LegacyConnector(), "get_finished_send_count") == 4


def test_from_connector_splits_send_recv():
    agg = KVOutputAggregatorSendRecvSplitPatch.from_connector(_SplitConnector(), 16)
    assert agg._expected_send_count == 1
    assert agg._expected_recv_count == 8

    agg = KVOutputAggregatorSendRecvSplitPatch.from_connector(_LegacyConnector(), 16)
    assert agg._expected_send_count == 4
    assert agg._expected_recv_count == 4


def test_aggregate_none_rank_and_finished_sets():
    agg = KVOutputAggregatorSendRecvSplitPatch(2, 2)
    assert agg.aggregate([None], 0) is None

    kv = SimpleNamespace(
        expected_finished_count=0,
        finished_sending={"r1"},
        finished_recving=None,
        kv_connector_stats=None,
        kv_connector_worker_meta=None,
        kv_cache_events=None,
        invalid_block_ids=set(),
    )
    out = SimpleNamespace(kv_connector_output=kv)
    first = agg.aggregate([out], 0)
    assert first is out
    assert first.kv_connector_output.finished_sending is None

    kv2 = SimpleNamespace(
        expected_finished_count=0,
        finished_sending={"r1"},
        finished_recving={"r2"},
        kv_connector_stats=None,
        kv_connector_worker_meta=None,
        kv_cache_events=None,
        invalid_block_ids={3},
    )
    out2 = SimpleNamespace(kv_connector_output=kv2)
    second = agg.aggregate([out2], 0)
    assert second.kv_connector_output.finished_sending == {"r1"}
    assert second.kv_connector_output.finished_recving is None
    assert 3 in second.kv_connector_output.invalid_block_ids


def test_aggregate_rejects_none_in_loop():
    agg = KVOutputAggregatorSendRecvSplitPatch(1)
    good = SimpleNamespace(kv_connector_output=None)
    with pytest.raises(RuntimeError, match="must not be None"):
        agg.aggregate([good, None], 0)


def test_aggregate_stats_type_mismatch():
    class Stats:
        def aggregate(self, other):
            return self

    class OtherStats:
        pass

    agg = KVOutputAggregatorSendRecvSplitPatch(1)
    kv1 = SimpleNamespace(
        expected_finished_count=0,
        finished_sending=None,
        finished_recving=None,
        kv_connector_stats=Stats(),
        kv_connector_worker_meta=None,
        kv_cache_events=None,
        invalid_block_ids=set(),
    )
    kv2 = SimpleNamespace(
        expected_finished_count=0,
        finished_sending=None,
        finished_recving=None,
        kv_connector_stats=OtherStats(),
        kv_connector_worker_meta=None,
        kv_cache_events=None,
        invalid_block_ids=set(),
    )
    with pytest.raises(TypeError, match="kv_connector_stats type mismatch"):
        agg.aggregate(
            [
                SimpleNamespace(kv_connector_output=kv1),
                SimpleNamespace(kv_connector_output=kv2),
            ],
            0,
        )


def test_aggregate_child_counts_min_max_and_none():
    a = MagicMock()
    a.get_finished_send_count.return_value = 1
    b = MagicMock()
    b.get_finished_send_count.return_value = 3
    c = MagicMock()
    c.get_finished_send_count.return_value = None
    c.get_finished_count.return_value = None
    assert _aggregate_child_counts([a, b], "get_finished_send_count", True) == 1
    assert _aggregate_child_counts([a, b], "get_finished_send_count", False) == 3
    assert _aggregate_child_counts([c], "get_finished_send_count", True) is None


def test_multi_connector_patch_send_recv():
    patch_obj = MultiConnectorGetFinishedCountPatch.__new__(
        MultiConnectorGetFinishedCountPatch
    )

    class Send:
        def get_finished_send_count(self):
            return 1

        def get_finished_recv_count(self):
            return None

        def get_finished_count(self):
            return 1

    class Recv:
        def get_finished_send_count(self):
            return 4

        def get_finished_recv_count(self):
            return 8

        def get_finished_count(self):
            return 4

    patch_obj._connectors = [Send(), Recv()]
    assert patch_obj.get_finished_send_count() == 1
    assert patch_obj.get_finished_recv_count() == 8
    assert patch_obj.get_finished_count() == 1
