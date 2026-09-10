# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UT for hybrid HBM+DDR joint-lookup patches."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.kv_cache_interface import FullAttentionSpec

from omni_npu.vllm_patches.patches.models.low_latency.patch_kv_offload_joint_lookup import (
    HybridKVCacheCoordinatorOffloadPatch,
    KVCacheManagerOffloadJointLookupPatch,
    SchedulerOffloadJointLookupPatch,
    _SkipHybridConnectorLookup,
    full_attention_group_id,
    is_offloading_connector,
    local_hits_from_blocks,
)


def _bind(method, instance):
    return method.__get__(instance, type(instance))


def _restore_upstream(cls, name, saved):
    if saved is None:
        if hasattr(cls, name):
            delattr(cls, name)
        return
    setattr(cls, name, saved)


def _attach_prefix_lookup(mgr):
    mgr.prefix_cache_lookup_enabled = _bind(
        KVCacheManagerOffloadJointLookupPatch.prefix_cache_lookup_enabled, mgr
    )
    return mgr


def _hybrid_coordinator(fa_group_id=0, per_group_hits=None):
    coordinator = MagicMock(spec=HybridKVCacheCoordinator)
    coordinator.full_attention_group_id = fa_group_id
    if per_group_hits is not None:
        coordinator.find_longest_cache_hit_per_group.return_value = (
            "computed",
            per_group_hits,
        )
    return coordinator


def _connector_mgr(coordinator, **attrs):
    fields = {
        "coordinator": coordinator,
        "kv_cache_config": SimpleNamespace(has_mamba_layers=True),
        "enable_caching": True,
    }
    fields.update(attrs)
    return _attach_prefix_lookup(SimpleNamespace(**fields))


def _lookup_request(**attrs):
    fields = {
        "skip_reading_prefix_cache": False,
        "block_hashes": [1],
        "num_tokens": 257,
    }
    fields.update(attrs)
    return SimpleNamespace(**fields)


def _bind_connector(mgr):
    return _bind(
        KVCacheManagerOffloadJointLookupPatch.get_computed_blocks_for_connector,
        mgr,
    )


def _bind_computed(mgr):
    return _bind(KVCacheManagerOffloadJointLookupPatch.get_computed_blocks, mgr)


def _bind_record(mgr):
    return _bind(
        KVCacheManagerOffloadJointLookupPatch.record_prefix_cache_stats, mgr
    )


def _stats_mgr(*, log_stats, enable_caching=True, prefix_cache_stats=None):
    return _attach_prefix_lookup(
        SimpleNamespace(
            log_stats=log_stats,
            enable_caching=enable_caching,
            prefix_cache_stats=prefix_cache_stats,
        )
    )


def _two_group_spec(block_size=128):
    return [
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=block_size)),
        SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=block_size)),
    ]


def _offload_wrap_mgr(*, connector_result, **attrs):
    fields = {
        "_omni_is_offloading_connector": True,
        "log_stats": False,
        "coordinator": SimpleNamespace(scheduler_block_size=128),
        "kv_cache_config": SimpleNamespace(kv_cache_groups=_two_group_spec()),
    }
    fields.update(attrs)
    mgr = SimpleNamespace(**fields)
    mgr.get_computed_blocks_for_connector = MagicMock(return_value=connector_result)
    mgr.record_prefix_cache_stats = MagicMock()
    return mgr


def _run_coordinator_offload_init(upstream_init):
    saved = getattr(HybridKVCacheCoordinatorOffloadPatch, "_upstream__init__", None)
    try:
        HybridKVCacheCoordinatorOffloadPatch._upstream__init__ = upstream_init
        coord = SimpleNamespace()
        _bind(HybridKVCacheCoordinatorOffloadPatch.__init__, coord)()
        return coord
    finally:
        _restore_upstream(
            HybridKVCacheCoordinatorOffloadPatch, "_upstream__init__", saved
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
    request = SimpleNamespace()
    assert _bind_connector(mgr)(request) == ("blocks", 128, False)
    upstream.assert_called_once_with(mgr, request)


def test_get_computed_blocks_for_connector_reports_fa_and_divergence():
    coordinator = _hybrid_coordinator(per_group_hits=(256, 128))
    mgr = _connector_mgr(
        coordinator, create_kv_cache_blocks=MagicMock(return_value="blocks")
    )
    assert _bind_connector(mgr)(_lookup_request()) == ("blocks", 256, True)
    mgr.create_kv_cache_blocks.assert_called_once_with("computed")


def test_get_computed_blocks_without_offload_uses_upstream():
    upstream = MagicMock(return_value=("blocks", 64))
    KVCacheManagerOffloadJointLookupPatch._upstream_get_computed_blocks = upstream
    mgr = SimpleNamespace(
        _omni_has_connector=True,
        _omni_is_offloading_connector=False,
    )
    request = SimpleNamespace()
    assert _bind_computed(mgr)(request) == ("blocks", 64)
    upstream.assert_called_once_with(mgr, request)


def test_get_computed_blocks_wrap_sets_per_group_and_min_for_offload():
    KVCacheManagerOffloadJointLookupPatch._upstream_get_computed_blocks = MagicMock()
    blocks = SimpleNamespace(blocks=([0, 1], [0]))
    mgr = _offload_wrap_mgr(
        connector_result=(blocks, 256, False),
        _omni_has_connector=True,
        enable_caching=True,
    )
    request = SimpleNamespace()
    out_blocks, num_local = _bind_computed(mgr)(request)
    assert out_blocks is blocks
    assert num_local == 128
    assert request.local_computed_tokens_per_group == (256, 128)
    mgr.record_prefix_cache_stats.assert_called_once_with(request, 128)


def test_scheduler_offload_flag_helper():
    sched = SimpleNamespace(connector=None)
    assert _bind(SchedulerOffloadJointLookupPatch._is_offloading_connector, sched)() is False


def test_is_offloading_connector_false_for_multiconnector_without_offload():
    from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
        MultiConnector,
    )

    multi = MagicMock(spec=MultiConnector)
    multi._connectors = [MagicMock(), MagicMock()]
    assert is_offloading_connector(multi) is False


def test_coordinator_offload_patch_sets_fa_group_id():
    def init_with_fa(self, *args, **kwargs):
        self.attention_groups = [
            SimpleNamespace(
                spec=MagicMock(spec=FullAttentionSpec), group_ids=[5, 6]
            )
        ]

    assert _run_coordinator_offload_init(init_with_fa).full_attention_group_id == 5


def test_coordinator_offload_patch_sets_none_without_fa():
    def init_without_fa(self, *args, **kwargs):
        self.attention_groups = [
            SimpleNamespace(spec=SimpleNamespace(), group_ids=[0])
        ]

    assert _run_coordinator_offload_init(init_without_fa).full_attention_group_id is None


def test_prefix_cache_lookup_enabled_requires_caching_and_read():
    mgr = SimpleNamespace(enable_caching=True)
    bound = _bind(KVCacheManagerOffloadJointLookupPatch.prefix_cache_lookup_enabled, mgr)
    assert bound(SimpleNamespace(skip_reading_prefix_cache=False)) is True
    assert bound(SimpleNamespace(skip_reading_prefix_cache=True)) is False
    mgr.enable_caching = False
    assert bound(SimpleNamespace(skip_reading_prefix_cache=False)) is False


def test_record_prefix_cache_stats_skips_when_disabled():
    stats = MagicMock()
    mgr = _stats_mgr(log_stats=False, prefix_cache_stats=stats)
    bound = _bind_record(mgr)
    bound(SimpleNamespace(skip_reading_prefix_cache=False), 8)
    stats.record.assert_not_called()

    mgr.log_stats = True
    mgr.enable_caching = False
    bound(SimpleNamespace(skip_reading_prefix_cache=False), 8)
    stats.record.assert_not_called()


def test_record_prefix_cache_stats_raises_when_stats_missing():
    mgr = _stats_mgr(log_stats=True, prefix_cache_stats=None)
    with pytest.raises(RuntimeError, match="prefix_cache_stats is None"):
        _bind_record(mgr)(SimpleNamespace(skip_reading_prefix_cache=False), 1)


def test_record_prefix_cache_stats_records_preempted_hits():
    stats = MagicMock()
    mgr = _stats_mgr(log_stats=True, prefix_cache_stats=stats)
    request = SimpleNamespace(
        skip_reading_prefix_cache=False,
        num_tokens=10,
        num_preemptions=1,
    )
    _bind_record(mgr)(request, 4)
    stats.record.assert_called_once_with(
        num_tokens=10, num_hits=4, preempted=True
    )


def test_get_computed_blocks_for_connector_falls_back_without_fa_group():
    upstream = MagicMock(return_value=("blocks", 64))
    KVCacheManagerOffloadJointLookupPatch._upstream_get_computed_blocks = upstream
    mgr = SimpleNamespace(
        coordinator=_hybrid_coordinator(fa_group_id=None),
        kv_cache_config=SimpleNamespace(has_mamba_layers=True),
    )
    request = SimpleNamespace()
    assert _bind_connector(mgr)(request) == ("blocks", 64, False)
    upstream.assert_called_once_with(mgr, request)


def test_get_computed_blocks_for_connector_skips_disabled_prefix_lookup():
    coordinator = _hybrid_coordinator()
    mgr = _connector_mgr(coordinator, empty_kv_cache_blocks="empty")
    request = SimpleNamespace(skip_reading_prefix_cache=True)
    assert _bind_connector(mgr)(request) == ("empty", 0, False)
    coordinator.find_longest_cache_hit_per_group.assert_not_called()


def test_get_computed_blocks_for_connector_falls_back_when_other_group_outruns_fa():
    upstream = MagicMock(return_value=("fallback", 128))
    KVCacheManagerOffloadJointLookupPatch._upstream_get_computed_blocks = upstream
    coordinator = _hybrid_coordinator(per_group_hits=(128, 256))
    mgr = _connector_mgr(coordinator, create_kv_cache_blocks=MagicMock())
    request = _lookup_request()
    assert _bind_connector(mgr)(request) == ("fallback", 128, False)
    mgr.create_kv_cache_blocks.assert_not_called()
    upstream.assert_called_once_with(mgr, request)


def test_get_computed_blocks_for_connector_reports_no_divergence_when_groups_agree():
    mgr = _connector_mgr(
        _hybrid_coordinator(per_group_hits=(256, 256)),
        create_kv_cache_blocks=MagicMock(return_value="blocks"),
    )
    assert _bind_connector(mgr)(_lookup_request()) == ("blocks", 256, False)


def test_get_computed_blocks_falls_back_when_diverged_hits_misaligned():
    upstream_blocks = SimpleNamespace(blocks=([0], [0]))
    upstream = MagicMock(return_value=(upstream_blocks, 128))
    KVCacheManagerOffloadJointLookupPatch._upstream_get_computed_blocks = upstream
    mgr = _offload_wrap_mgr(
        connector_result=(SimpleNamespace(blocks=([0, 1], [0])), 200, True),
        log_stats=True,
    )
    request = SimpleNamespace()
    out_blocks, num_local = _bind_computed(mgr)(request)
    assert out_blocks is upstream_blocks
    assert num_local == 128
    assert mgr.log_stats is True
    upstream.assert_called_once_with(mgr, request)
    mgr.record_prefix_cache_stats.assert_called_once_with(request, 128)


def test_get_computed_blocks_keeps_connector_hits_when_per_group_empty():
    blocks = SimpleNamespace(blocks=())
    mgr = _offload_wrap_mgr(
        connector_result=(blocks, 256, False),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )
    request = SimpleNamespace()
    out_blocks, num_local = _bind_computed(mgr)(request)
    assert out_blocks is blocks
    assert num_local == 256
    assert not hasattr(request, "local_computed_tokens_per_group")
    mgr.record_prefix_cache_stats.assert_called_once_with(request, 256)


def test_get_computed_blocks_restores_log_stats_after_connector_error():
    mgr = SimpleNamespace(
        _omni_is_offloading_connector=True,
        log_stats=True,
    )
    mgr.get_computed_blocks_for_connector = MagicMock(
        side_effect=RuntimeError("lookup failed")
    )
    with pytest.raises(RuntimeError, match="lookup failed"):
        _bind_computed(mgr)(SimpleNamespace())
    assert mgr.log_stats is True


def test_skip_hybrid_connector_lookup_is_never_the_coordinator():
    assert isinstance(SimpleNamespace(), _SkipHybridConnectorLookup) is False
    assert isinstance(
        MagicMock(spec=HybridKVCacheCoordinator), _SkipHybridConnectorLookup
    ) is False


def test_patch_apply_methods_call_bypass_conflict():
    with patch.object(
        HybridKVCacheCoordinatorOffloadPatch, "apply_bypass_conflict"
    ) as coord_apply, patch.object(
        KVCacheManagerOffloadJointLookupPatch, "apply_bypass_conflict"
    ) as mgr_apply, patch.object(
        SchedulerOffloadJointLookupPatch, "apply_bypass_conflict"
    ) as sched_apply:
        HybridKVCacheCoordinatorOffloadPatch.apply()
        KVCacheManagerOffloadJointLookupPatch.apply()
        SchedulerOffloadJointLookupPatch.apply()
        coord_apply.assert_called_once_with("__init__")
        mgr_apply.assert_called_once_with("get_computed_blocks")
        sched_apply.assert_called_once_with("__init__")


def test_scheduler_init_skips_hybrid_branch_only_for_offload():
    import vllm.v1.core.sched.scheduler as scheduler_mod
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )

    saved = scheduler_mod.HybridKVCacheCoordinator
    saved_upstream = getattr(
        SchedulerOffloadJointLookupPatch, "_upstream__init__", None
    )

    def _bind_init(sched):
        sched._is_offloading_connector = _bind(
            SchedulerOffloadJointLookupPatch._is_offloading_connector, sched
        )
        return _bind(SchedulerOffloadJointLookupPatch.__init__, sched)

    try:
        def init_without_offload(self, *args, **kwargs):
            self.connector = None
            self.kv_cache_manager = SimpleNamespace()

        SchedulerOffloadJointLookupPatch._upstream__init__ = init_without_offload
        sched = SimpleNamespace()
        _bind_init(sched)()
        assert scheduler_mod.HybridKVCacheCoordinator is saved
        assert sched.kv_cache_manager._omni_is_offloading_connector is False

        def init_with_offload(self, *args, **kwargs):
            self.connector = MagicMock(spec=OffloadingConnector)
            self.kv_cache_manager = SimpleNamespace()

        SchedulerOffloadJointLookupPatch._upstream__init__ = init_with_offload
        sched = SimpleNamespace()
        _bind_init(sched)()
        assert scheduler_mod.HybridKVCacheCoordinator is _SkipHybridConnectorLookup
        assert sched.kv_cache_manager._omni_is_offloading_connector is True
    finally:
        scheduler_mod.HybridKVCacheCoordinator = saved
        _restore_upstream(
            SchedulerOffloadJointLookupPatch, "_upstream__init__", saved_upstream
        )
