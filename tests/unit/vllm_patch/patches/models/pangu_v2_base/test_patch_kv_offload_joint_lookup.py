# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UT for hybrid HBM+DDR joint-lookup patches."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.kv_cache_interface import FullAttentionSpec

from omni_npu.vllm_patches.patches.models.pangu_v2_base.patch_kv_offload_joint_lookup import (
    KVCacheManagerOffloadJointLookupPatch,
    SchedulerOffloadJointLookupPatch,
    full_attention_group_id,
    is_offloading_connector,
    local_hits_from_blocks,
)


def test_full_attention_group_id_uses_the_leading_fa_group():
    groups = [
        SimpleNamespace(spec=MagicMock(spec=FullAttentionSpec), group_ids=[2, 3]),
        SimpleNamespace(spec=SimpleNamespace(), group_ids=[0]),
    ]
    assert full_attention_group_id(groups) == 2


def test_full_attention_group_id_is_none_without_fa():
    groups = [SimpleNamespace(spec=SimpleNamespace(), group_ids=[0])]
    assert full_attention_group_id(groups) is None


def test_is_offloading_connector_walks_multiconnector():
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )

    child = MagicMock(spec=OffloadingConnector)
    multi = MagicMock(spec=MultiConnector)
    multi._connectors = [MagicMock(), child]
    assert is_offloading_connector(child) is True
    assert is_offloading_connector(multi) is True
    assert is_offloading_connector(SimpleNamespace()) is False


def test_local_hits_from_blocks_uses_each_group_block_size():
    groups = [
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=128)),
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=64)),
    ]
    blocks = SimpleNamespace(blocks=([0, 1], [0, 1, 2, 3]))
    assert local_hits_from_blocks(groups, blocks) == (256, 256)


def test_get_computed_blocks_for_connector_falls_back_when_not_hybrid():
    upstream = MagicMock(return_value=("blocks", 128))
    KVCacheManagerOffloadJointLookupPatch._upstream_get_computed_blocks = upstream
    mgr = SimpleNamespace(
        coordinator=SimpleNamespace(),
        kv_cache_config=SimpleNamespace(has_mamba_layers=False),
    )
    bound = KVCacheManagerOffloadJointLookupPatch.get_computed_blocks_for_connector.__get__(
        mgr, type(mgr)
    )
    request = SimpleNamespace()
    assert bound(request) == ("blocks", 128, False)
    upstream.assert_called_once_with(mgr, request)


def test_get_computed_blocks_for_connector_reports_fa_and_divergence():
    coordinator = MagicMock(spec=HybridKVCacheCoordinator)
    coordinator.full_attention_group_id = 0
    coordinator.find_longest_cache_hit_per_group.return_value = (
        "computed",
        (256, 128),
    )
    mgr = SimpleNamespace(
        coordinator=coordinator,
        kv_cache_config=SimpleNamespace(has_mamba_layers=True),
        enable_caching=True,
        create_kv_cache_blocks=MagicMock(return_value="blocks"),
    )
    mgr.prefix_cache_lookup_enabled = (
        KVCacheManagerOffloadJointLookupPatch.prefix_cache_lookup_enabled.__get__(
            mgr, type(mgr)
        )
    )
    bound = KVCacheManagerOffloadJointLookupPatch.get_computed_blocks_for_connector.__get__(
        mgr, type(mgr)
    )
    request = SimpleNamespace(
        skip_reading_prefix_cache=False,
        block_hashes=[1],
        num_tokens=257,
    )
    assert bound(request) == ("blocks", 256, True)
    mgr.create_kv_cache_blocks.assert_called_once_with("computed")


def test_get_computed_blocks_without_connector_uses_upstream():
    upstream = MagicMock(return_value=("blocks", 64))
    KVCacheManagerOffloadJointLookupPatch._upstream_get_computed_blocks = upstream
    mgr = SimpleNamespace(_omni_has_connector=False)
    bound = KVCacheManagerOffloadJointLookupPatch.get_computed_blocks.__get__(
        mgr, type(mgr)
    )
    request = SimpleNamespace()
    assert bound(request) == ("blocks", 64)
    upstream.assert_called_once_with(mgr, request)


def test_get_computed_blocks_wrap_sets_per_group_and_min_for_offload():
    KVCacheManagerOffloadJointLookupPatch._upstream_get_computed_blocks = MagicMock()
    blocks = SimpleNamespace(blocks=([0, 1], [0]))
    mgr = SimpleNamespace(
        _omni_has_connector=True,
        _omni_is_offloading_connector=True,
        log_stats=False,
        enable_caching=True,
        coordinator=SimpleNamespace(scheduler_block_size=128),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=128)),
                SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=128)),
            ]
        ),
    )
    mgr.get_computed_blocks_for_connector = MagicMock(
        return_value=(blocks, 256, False)
    )
    mgr.record_prefix_cache_stats = MagicMock()
    bound = KVCacheManagerOffloadJointLookupPatch.get_computed_blocks.__get__(
        mgr, type(mgr)
    )
    request = SimpleNamespace()
    out_blocks, num_local = bound(request)
    assert out_blocks is blocks
    assert num_local == 128
    assert request.local_computed_tokens_per_group == (256, 128)
    mgr.record_prefix_cache_stats.assert_called_once_with(request, 128)


def test_scheduler_offload_flag_helper():
    sched = SimpleNamespace(connector=None)
    bound = SchedulerOffloadJointLookupPatch._is_offloading_connector.__get__(
        sched, type(sched)
    )
    assert bound() is False
