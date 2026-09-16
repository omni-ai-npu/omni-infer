# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""UT for hybrid lookup, offload admission and waiting-queue scheduling."""

from __future__ import annotations

import unittest
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched.request_queue import (
    FCFSRequestQueue,
    PriorityRequestQueue,
    SchedulingPolicy,
)
from vllm.v1.core.sched.scheduler import PauseState, Scheduler
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.request import RequestStatus

from omni_npu.vllm_patches.patches.models.low_latency.patch_kv_offload_joint_lookup import (
    HybridKVCacheCoordinatorOffloadPatch,
    KVCacheManagerOffloadJointLookupPatch,
    SchedulerOffloadJointLookupPatch,
    _SkipHybridConnectorLookup,
    full_attention_group_id,
    is_offloading_connector,
    local_hits_from_blocks,
    offload_remaining_blocks,
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
        mgr_apply.assert_called_once_with("get_computed_blocks", "allocate_slots")
        sched_apply.assert_called_once_with(
            "__init__", "_request_remaining_blocks",
            "_select_waiting_queue_for_scheduling",
        )
        assert "schedule" not in SchedulerOffloadJointLookupPatch._attr_names_to_apply


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


# Device execution and model outputs are simulated; allocation/scheduling methods
# below are real vLLM code. These are regression UTs, not NPU benchmark tests.
def _block(block_id, ref_cnt=1):
    return SimpleNamespace(block_id=block_id, is_null=block_id == 0, ref_cnt=ref_cnt)


def _request(req_id, tokens, computed=0, status=RequestStatus.WAITING):
    return SimpleNamespace(
        request_id=req_id, num_tokens=tokens, num_prompt_tokens=tokens,
        num_computed_tokens=computed, status=status,
    )


def _manager(cap=None):
    manager = SimpleNamespace(
        block_size=16,
        _max_admission_blocks_per_request=cap,
        req_to_blocks=defaultdict(list),
        num_cached_block={},
        get_num_skipped_tokens=lambda tokens: 0,
        _get_num_evictable_blocks=SingleTypeKVCacheManager._get_num_evictable_blocks,
    )
    manager.get_num_blocks_to_allocate = (
        SingleTypeKVCacheManager.get_num_blocks_to_allocate.__get__(manager)
    )
    return manager


class _PriorityRequest(SimpleNamespace):
    def __lt__(self, other):
        return self.priority < other.priority


class _IdentityRequest(SimpleNamespace):
    __hash__ = object.__hash__
    __eq__ = object.__eq__


def _scheduler(managers, free=32, lookahead=0):
    pool = SimpleNamespace(free=free, next_id=1)
    pool.get_num_free_blocks = lambda: pool.free
    mgr = SimpleNamespace(
        coordinator=SimpleNamespace(single_type_managers=managers),
        empty_kv_cache_blocks=SimpleNamespace(blocks=tuple([] for _ in managers)),
        block_pool=pool, watermark_blocks=0, max_model_len=65536,
        enable_caching=True,
    )
    scheduler = SimpleNamespace(
        kv_cache_manager=mgr, max_model_len=65536,
        num_lookahead_tokens=lookahead, _inflight_prefills=[],
    )
    mgr._omni_offload_scheduler = scheduler
    return scheduler


def _wire_allocator(scheduler):
    """In-memory block backend for the real vLLM allocate_slots method."""
    mgr = scheduler.kv_cache_manager
    coordinator = mgr.coordinator
    pool = mgr.block_pool

    def new_blocks(count):
        if count > pool.free:
            raise AssertionError("physical pool exhausted")
        result = [_block(i) for i in range(pool.next_id, pool.next_id + count)]
        pool.next_id += count
        pool.free -= count
        return result

    def needed(request_id, num_tokens, new_computed_blocks, num_encoder_tokens,
               total_computed_tokens, num_tokens_main_model,
               apply_admission_cap=False):
        return sum(
            manager.get_num_blocks_to_allocate(
                request_id, num_tokens, blocks, total_computed_tokens,
                num_tokens_main_model, apply_admission_cap,
            )
            for manager, blocks in zip(coordinator.single_type_managers,
                                       new_computed_blocks)
        )

    def computed(request_id, new_computed_blocks, num_local_computed_tokens,
                 num_external_computed_tokens):
        for manager, hits in zip(coordinator.single_type_managers,
                                 new_computed_blocks):
            blocks = manager.req_to_blocks[request_id]
            for block in hits:
                if block.ref_cnt == 0:
                    pool.free -= 1
                block.ref_cnt += 1
                blocks.append(block)
            count = (
                num_local_computed_tokens + num_external_computed_tokens + 15
            ) // 16
            skipped = manager.get_num_skipped_tokens(
                num_local_computed_tokens + num_external_computed_tokens
            ) // 16
            blocks.extend([_block(0)] * max(0, skipped - len(blocks)))
            blocks.extend(new_blocks(max(0, count - len(blocks))))
            manager.num_cached_block[request_id] = 0

    def allocate(request_id, num_tokens, num_tokens_main_model, num_encoder_tokens):
        result = []
        for manager in coordinator.single_type_managers:
            blocks = manager.req_to_blocks[request_id]
            added = new_blocks(max(0, (num_tokens + 15) // 16 - len(blocks)))
            blocks.extend(added)
            result.append(added)
        return tuple(result)

    def remove_skipped(request_id, total_computed_tokens, num_prompt_tokens):
        for manager in coordinator.single_type_managers:
            blocks = manager.req_to_blocks.get(request_id, [])
            skipped = manager.get_num_skipped_tokens(total_computed_tokens) // 16
            for i in range(min(skipped, len(blocks))):
                block = blocks[i]
                if not block.is_null:
                    block.ref_cnt -= 1
                    if block.ref_cnt == 0:
                        pool.free += 1
                    blocks[i] = _block(0)

    coordinator.get_num_blocks_to_allocate = needed
    coordinator.remove_skipped_blocks = remove_skipped
    coordinator.allocate_new_computed_blocks = computed
    coordinator.allocate_new_blocks = allocate
    coordinator.cache_blocks = MagicMock()
    mgr.create_kv_cache_blocks = lambda blocks: SimpleNamespace(blocks=blocks)


class TestOffloadAdmission(unittest.TestCase):
    def setUp(self):
        self.upstream = patch.object(
            KVCacheManagerOffloadJointLookupPatch, "_upstream_allocate_slots",
            KVCacheManager.allocate_slots, create=True,
        )
        self.upstream.start()
        self.addCleanup(self.upstream.stop)
        self.selector = patch.object(
            SchedulerOffloadJointLookupPatch,
            "_upstream_select_waiting_queue_for_scheduling",
            Scheduler._select_waiting_queue_for_scheduling, create=True,
        )
        self.selector.start()
        self.addCleanup(self.selector.stop)

    def queues(self, scheduler, skipped=(), waiting=(), priority=False):
        scheduler.policy = (
            SchedulingPolicy.PRIORITY if priority else SchedulingPolicy.FCFS
        )
        queue_type = PriorityRequestQueue if priority else FCFSRequestQueue
        scheduler.skipped_waiting = queue_type()
        scheduler.waiting = queue_type()
        for req in skipped:
            scheduler.skipped_waiting.add_request(req)
        for req in waiting:
            scheduler.waiting.add_request(req)
        scheduler.finished_recving_kv_req_ids = set()
        scheduler.running = []
        return scheduler

    def select(self, scheduler):
        return SchedulerOffloadJointLookupPatch._select_waiting_queue_for_scheduling(
            scheduler
        )

    def schedule_fixture(self, count=32):
        """Real scheduler control flow with fake cache/device metadata backends."""
        manager = _manager()
        scheduler = self.queues(_scheduler([manager], free=8))
        _wire_allocator(scheduler)
        requests = []
        for name in ["loaded"] + ["cold-%d" % i for i in range(count)]:
            req = _IdentityRequest(**vars(_request(name, 128)))
            req.num_preemptions = 0
            req.prefill_stats = None
            req.has_encoder_inputs = False
            req.num_output_placeholders = 0
            req.use_structured_output = False
            requests.append(req)
        loaded = requests[0]
        loaded.status = RequestStatus.WAITING_FOR_REMOTE_KVS
        loaded.num_computed_tokens = 16
        manager.req_to_blocks["loaded"] = [_block(99)]
        manager.num_cached_block["loaded"] = 0
        scheduler.skipped_waiting.add_request(requests[1])
        scheduler.skipped_waiting.add_request(loaded)
        for req in requests[2:]:
            scheduler.waiting.add_request(req)
        scheduler.__dict__.update(
            current_step=0, max_num_scheduled_tokens=128,
            max_num_encoder_input_tokens=0, _pause_state=PauseState.UNPAUSED,
            prefill_capacity_bound=False, lora_config=None,
            num_waiting_for_streaming_input=0, max_num_running_reqs=1,
            has_mamba_layers=False, ec_connector=None,
            num_spec_tokens=0, dynamic_sd_lookup=None,
            scheduler_config=SimpleNamespace(long_prefill_token_threshold=0,
                                             enable_chunked_prefill=True),
            need_mamba_block_aligned_split=False, is_encoder_decoder=False,
            scheduler_reserve_full_isl=True, connector_prefix_cache_stats=None,
            log_stats=False, kv_cache_config=SimpleNamespace(kv_cache_groups=[None]),
            use_v2_model_runner=False, prev_step_scheduled_req_ids=set(),
            needs_kv_cache_zeroing=False, reset_preempted_req_ids=set(),
            finished_req_ids=set(), encoder_cache_manager=MagicMock(),
            defer_block_free=False, enable_return_routed_experts=False,
            requests={req.request_id: req for req in requests},
            _inflight_prefills={loaded}, failed_recving_kv_req_ids=set(),
        )
        scheduler.connector = MagicMock()
        scheduler.connector.get_num_new_matched_tokens.return_value = (0, False)
        scheduler._is_blocked_waiting_status = Scheduler._is_blocked_waiting_status
        for name in ("_update_after_schedule", "_update_waiting_for_remote_kv",
                     "_try_promote_blocked_waiting_request", "_build_kv_connector_meta",
                     "_inflight_prefill_reserved_blocks"):
            setattr(scheduler, name, getattr(Scheduler, name).__get__(scheduler))
        scheduler._make_cached_request_data = MagicMock()
        scheduler._request_remaining_blocks = lambda req: offload_remaining_blocks(
            scheduler, req,
        )
        scheduler._select_waiting_queue_for_scheduling = lambda: self.select(scheduler)
        mgr = scheduler.kv_cache_manager
        mgr.new_step_starts = MagicMock()
        mgr.cache_blocks = MagicMock()
        mgr.get_computed_blocks = lambda req: (mgr.empty_kv_cache_blocks, 0)
        mgr.get_num_common_prefix_blocks = lambda req_id: [0]
        mgr.get_blocks = lambda req_id: SimpleNamespace(
            get_block_ids=lambda: ([b.block_id for b in manager.req_to_blocks[req_id]],),
        )
        mgr.allocate_slots = KVCacheManagerOffloadJointLookupPatch.allocate_slots.__get__(mgr)
        # Output serialization/model-runner data are outside this control-flow test.
        self.enter_schedule_output_stubs()
        return scheduler, manager, requests

    def enter_schedule_output_stubs(self):
        context = patch.dict(
            Scheduler.schedule.__globals__,
            NewRequestData=MagicMock(),
            SchedulerOutput=lambda **kw: SimpleNamespace(
                has_structured_output_requests=False, **kw,
            ),
        )
        context.start()
        self.addCleanup(context.stop)

    def test_complete_schedule_loop_drains_burst_after_ready_first_fix(self):
        scheduler, manager, requests = self.schedule_fixture()
        scheduler.finished_recving_kv_req_ids.add("loaded")
        scheduler._select_waiting_queue_for_scheduling = (
            Scheduler._select_waiting_queue_for_scheduling.__get__(scheduler)
        )
        for _ in range(100):
            output = Scheduler.schedule(scheduler)
            self.assertEqual(output.total_num_scheduled_tokens, 0)
            self.assertEqual(len(scheduler.running), 0)
        self.assertEqual(scheduler.finished_recving_kv_req_ids, {"loaded"})
        scheduler._select_waiting_queue_for_scheduling = lambda: self.select(scheduler)
        completed = []
        for _ in requests:
            output = Scheduler.schedule(scheduler)
            self.assertGreater(output.total_num_scheduled_tokens, 0)
            self.assertEqual(len(scheduler.running), 1)
            req = scheduler.running.pop()
            completed.append(req.request_id)
            self.assertEqual(req.num_computed_tokens, req.num_tokens)
            self.assertNotIn(req, scheduler._inflight_prefills)
            # Model completion/free is simulated; schedule/promotion/accounting are real.
            scheduler.kv_cache_manager.block_pool.free += len(
                manager.req_to_blocks.pop(req.request_id)
            )
            manager.num_cached_block.pop(req.request_id, None)
        self.assertEqual(completed, [req.request_id for req in requests])
        self.assertFalse(scheduler.waiting or scheduler.skipped_waiting)
        self.assertFalse(scheduler.finished_recving_kv_req_ids)
        self.assertEqual(scheduler.kv_cache_manager.block_pool.free, 9)

    def test_complete_schedule_does_not_promote_until_receive_finishes(self):
        scheduler, manager, requests = self.schedule_fixture(count=1)
        for _ in range(10):
            self.assertEqual(Scheduler.schedule(scheduler).total_num_scheduled_tokens, 0)
        self.assertEqual(requests[0].status, RequestStatus.WAITING_FOR_REMOTE_KVS)
        scheduler.kv_cache_manager.cache_blocks.assert_not_called()
        self.assertEqual(len(manager.req_to_blocks["loaded"]), 1)
        scheduler.finished_recving_kv_req_ids.add("loaded")
        output = Scheduler.schedule(scheduler)
        self.assertEqual(output.num_scheduled_tokens, {"loaded": 112})
        scheduler.kv_cache_manager.cache_blocks.assert_called_once_with(requests[0], 16)
        self.assertFalse(scheduler.finished_recving_kv_req_ids)

    def test_ready_load_behind_rejected_head_can_spend_reservation(self):
        manager = _manager()
        scheduler = self.queues(_scheduler([manager], free=8))
        _wire_allocator(scheduler)
        fresh = _request("fresh", 128)
        loaded = _request("loaded", 128, computed=16)
        manager.req_to_blocks[loaded.request_id] = [_block(99)]
        manager.num_cached_block[loaded.request_id] = 0
        scheduler._inflight_prefills = [loaded]
        scheduler.skipped_waiting.add_request(fresh)
        scheduler.skipped_waiting.add_request(loaded)

        # v0.25.1 breaks its waiting loop on allocate_slots(None). The same
        # unadmitted head is retried forever, despite B's reserved capacity.
        for _ in range(100):
            old = Scheduler._select_waiting_queue_for_scheduling(scheduler)
            self.assertIs(old.peek_request(), fresh)
            self.assertIsNone(self.allocate(scheduler, fresh, tokens=128))
        view = self.select(scheduler)
        self.assertIs(view.peek_request(), loaded)
        self.assertEqual(list(scheduler.skipped_waiting), [fresh, loaded])
        result = self.allocate(scheduler, loaded, tokens=112)
        self.assertIsNotNone(result)
        self.assertIs(view.pop_request(), loaded)
        self.assertEqual(list(scheduler.skipped_waiting), [fresh])
        # Complete/free B, then A can enter with the same admission accounting.
        scheduler._inflight_prefills.clear()
        scheduler.kv_cache_manager.block_pool.free += len(
            manager.req_to_blocks.pop(loaded.request_id)
        )
        self.assertIs(self.select(scheduler).peek_request(), fresh)
        self.assertIsNotNone(self.allocate(scheduler, fresh, tokens=128))

    def test_ready_selection_crosses_queues_but_never_starts_unfinished_dma(self):
        fresh = _request("fresh", 128)
        loading = _request("load", 128, 16, RequestStatus.WAITING_FOR_REMOTE_KVS)
        scheduler = self.queues(_scheduler([_manager()]), [fresh], [loading])
        scheduler._inflight_prefills = [loading]
        self.assertIs(self.select(scheduler), scheduler.skipped_waiting)
        scheduler.finished_recving_kv_req_ids.add("load")
        view = self.select(scheduler)
        self.assertIs(view.peek_request(), loading)
        # Even on failed allocation (no pop) or cancellation, no ownership
        # moved to the temporary view and the original queues remain usable.
        self.assertEqual(list(scheduler.waiting), [loading])
        scheduler.waiting.remove_request(loading)
        self.assertIs(self.select(scheduler), scheduler.skipped_waiting)

    def test_ready_selection_preserves_priority_among_admitted_requests(self):
        def req(name, priority):
            value = _PriorityRequest(**vars(_request(name, 128, 16)))
            value.priority = priority
            return value
        fresh, low, high = req("fresh", 0), req("low", 10), req("high", 5)
        scheduler = self.queues(
            _scheduler([_manager()]), [fresh, low], [high], priority=True,
        )
        scheduler._inflight_prefills = [low, high]
        self.assertIs(self.select(scheduler).pop_request(), high)
        self.assertIs(self.select(scheduler).pop_request(), low)
        self.assertIs(self.select(scheduler).peek_request(), fresh)
        self.assertEqual((fresh.priority, low.priority, high.priority), (0, 10, 5))

    def test_non_offload_queue_selection_is_unchanged(self):
        fresh, loaded = _request("fresh", 128), _request("loaded", 128, 16)
        scheduler = self.queues(_scheduler([_manager()]), [fresh, loaded])
        scheduler._inflight_prefills = [loaded]
        del scheduler.kv_cache_manager._omni_offload_scheduler
        self.assertIs(self.select(scheduler).peek_request(), fresh)


    def allocate(self, scheduler, request, tokens=0, **kwargs):
        return KVCacheManagerOffloadJointLookupPatch.allocate_slots(
            scheduler.kv_cache_manager, request, tokens, **kwargs
        )

    def test_null_placeholders_do_not_erase_rolling_reservation(self):
        manager = _manager(cap=4)
        req = _request("long", 3200, computed=1600)
        manager.req_to_blocks[req.request_id] = [_block(0)] * 99 + [_block(1)]
        manager.num_cached_block[req.request_id] = 100
        scheduler = _scheduler([manager])
        # The exact upstream admission predictor returns zero for this table.
        self.assertEqual(manager.get_num_blocks_to_allocate(
            "long", 3200, [], 1600, 3200, apply_admission_cap=True,
        ), 0)
        self.assertEqual(offload_remaining_blocks(scheduler, req), 3)

    def test_shared_pages_are_not_recycling_credit(self):
        manager = _manager(cap=4)
        req = _request("shared", 3200, computed=1600)
        manager.req_to_blocks[req.request_id] = [_block(0)] * 98 + [
            _block(1, ref_cnt=2), _block(2, ref_cnt=1),
        ]
        manager.num_cached_block[req.request_id] = 100
        self.assertEqual(offload_remaining_blocks(_scheduler([manager]), req), 3)

    def test_hybrid_groups_and_lookahead_are_summed(self):
        full, rolling = _manager(), _manager(cap=4)
        req = _request("hybrid", 3200, computed=1600)
        full.req_to_blocks["hybrid"] = [_block(i + 1) for i in range(100)]
        rolling.req_to_blocks["hybrid"] = [_block(0)] * 99 + [_block(101)]
        full.num_cached_block["hybrid"] = rolling.num_cached_block["hybrid"] = 100
        self.assertEqual(offload_remaining_blocks(
            _scheduler([full, rolling], lookahead=3), req,
        ), 101 + 4)

    def test_local_hit_sharing_cannot_steal_another_reservation_credit(self):
        manager = _manager(cap=4)
        manager.get_num_skipped_tokens = lambda tokens: max(0, tokens - 16)
        scheduler = _scheduler([manager], free=4)
        _wire_allocator(scheduler)
        old = _request("old", 3200, computed=1600)
        blocks = [_block(0)] * 99 + [_block(1)]
        manager.req_to_blocks["old"] = blocks
        manager.num_cached_block["old"] = 100
        scheduler._inflight_prefills.append(old)
        self.assertEqual(offload_remaining_blocks(scheduler, old), 3)
        new = _request("new", 1616)
        # New needs one page, but sharing old's retained page also removes one
        # page of old's recycling credit. Four free pages are not sufficient.
        self.assertIsNone(self.allocate(
            scheduler, new, tokens=16, num_new_computed_tokens=1600,
            new_computed_blocks=SimpleNamespace(blocks=(blocks,)),
        ))
        self.assertEqual(blocks[-1].ref_cnt, 1)

    def test_fresh_full_ddr_hit_still_reserves_lookahead(self):
        manager = _manager()
        scheduler = _scheduler([manager], free=4, lookahead=3)
        _wire_allocator(scheduler)
        req = _request("mtp", 64)
        self.assertIsNone(self.allocate(
            scheduler, req, num_external_computed_tokens=64, delay_cache_blocks=True,
        ))
        self.assertEqual(scheduler.kv_cache_manager.block_pool.free, 4)
        self.assertNotIn("mtp", manager.req_to_blocks)

    def test_candidate_suffix_and_other_reservations_must_fit_together(self):
        manager = _manager()
        scheduler = _scheduler([manager], free=9)
        _wire_allocator(scheduler)
        a, b = _request("a", 128), _request("b", 112)
        self.assertIsNotNone(self.allocate(
            scheduler, a, num_external_computed_tokens=32, delay_cache_blocks=True,
        ))
        a.num_computed_tokens = 32
        scheduler._inflight_prefills.append(a)
        self.assertIsNone(self.allocate(
            scheduler, b, num_external_computed_tokens=16, delay_cache_blocks=True,
            full_sequence_must_fit=True, reserved_blocks=6,
        ))
        self.assertNotIn("b", manager.req_to_blocks)
        # Completion of A's DMA retains its reservation until local compute.
        self.assertEqual(offload_remaining_blocks(scheduler, a), 6)
        self.assertIsNotNone(self.allocate(scheduler, a, tokens=96))

    def test_cold_miss_cannot_consume_loaded_request_headroom(self):
        scheduler = _scheduler([_manager()], free=7)
        _wire_allocator(scheduler)
        loaded = _request("loaded", 96)
        self.assertIsNotNone(self.allocate(
            scheduler, loaded, num_external_computed_tokens=32, delay_cache_blocks=True,
        ))
        loaded.num_computed_tokens = 32
        scheduler._inflight_prefills.append(loaded)
        self.assertIsNone(self.allocate(scheduler, _request("cold", 64), tokens=16))
        self.assertIsNotNone(self.allocate(scheduler, loaded, tokens=64))

    def test_running_decode_respects_other_prefill_reservations(self):
        scheduler = _scheduler([_manager()], free=3)
        _wire_allocator(scheduler)
        scheduler._inflight_prefills.append(_request("loading", 48))
        decode = _request("decode", 16, status=RequestStatus.RUNNING)
        self.assertIsNone(self.allocate(scheduler, decode, tokens=16))

    def test_cancel_or_preempt_releases_reservation_without_counter_leak(self):
        scheduler = _scheduler([_manager()], free=4)
        _wire_allocator(scheduler)
        old = _request("old", 64)
        scheduler._inflight_prefills.append(old)
        new = _request("new", 64)
        self.assertIsNone(self.allocate(
            scheduler, new, num_external_computed_tokens=16, delay_cache_blocks=True,
        ))
        scheduler._inflight_prefills.remove(old)
        self.assertIsNotNone(self.allocate(
            scheduler, new, num_external_computed_tokens=16, delay_cache_blocks=True,
        ))

    def test_failed_load_retry_does_not_reserve_itself_twice(self):
        scheduler = _scheduler([_manager()], free=4)
        _wire_allocator(scheduler)
        req = _request("retry", 64)
        scheduler._inflight_prefills.append(req)
        self.assertIsNotNone(self.allocate(scheduler, req, tokens=64))

    def test_failed_load_async_retry_deducts_its_own_upstream_reservation(self):
        scheduler = _scheduler([_manager()], free=4)
        _wire_allocator(scheduler)
        req = _request("retry", 64)
        scheduler._inflight_prefills.append(req)
        self.assertIsNotNone(self.allocate(
            scheduler, req, num_external_computed_tokens=16, delay_cache_blocks=True,
            reserved_blocks=offload_remaining_blocks(scheduler, req),
        ))

    def test_non_offload_scheduler_keeps_upstream_reservation(self):
        scheduler = _scheduler([_manager()])
        del scheduler.kv_cache_manager._omni_offload_scheduler
        with patch.object(SchedulerOffloadJointLookupPatch,
                          "_upstream_request_remaining_blocks",
                          return_value=42, create=True) as upstream:
            req = _request("plain", 64)
            self.assertEqual(SchedulerOffloadJointLookupPatch._request_remaining_blocks(
                scheduler, req,
            ), 42)
            upstream.assert_called_once_with(scheduler, req)

    def test_watermark_and_max_model_length(self):
        scheduler = _scheduler([_manager()], free=4, lookahead=3)
        scheduler.max_model_len = scheduler.kv_cache_manager.max_model_len = 64
        scheduler.kv_cache_manager.watermark_blocks = 1
        _wire_allocator(scheduler)
        req = _request("cap", 64)
        self.assertEqual(offload_remaining_blocks(scheduler, req), 4)
        self.assertIsNone(self.allocate(
            scheduler, req, num_external_computed_tokens=16, delay_cache_blocks=True,
        ))
        self.assertIsNotNone(self.allocate(
            scheduler, req, num_external_computed_tokens=16, delay_cache_blocks=True,
            has_scheduled_reqs=False,
        ))

    def test_without_offload_allocation_is_unchanged(self):
        scheduler = _scheduler([_manager()], free=4)
        _wire_allocator(scheduler)
        del scheduler.kv_cache_manager._omni_offload_scheduler
        self.assertIsNotNone(self.allocate(
            scheduler, _request("plain", 128), num_external_computed_tokens=16,
            delay_cache_blocks=True,
        ))

    def test_high_concurrency_drains_without_abort(self):
        manager = _manager()
        scheduler = _scheduler([manager], free=16)
        _wire_allocator(scheduler)
        requests = [_request(str(i), 128) for i in range(64)]
        pending = list(requests)
        completed = []
        pool = scheduler.kv_cache_manager.block_pool
        for _ in range(128):
            while pending:
                req = pending[0]
                if self.allocate(scheduler, req, num_external_computed_tokens=64,
                                 delay_cache_blocks=True) is None:
                    break
                pending.pop(0)
                req.num_computed_tokens = 64
                scheduler._inflight_prefills.append(req)
            self.assertGreater(len(scheduler._inflight_prefills), 0)
            self.assertLessEqual(sum(
                offload_remaining_blocks(scheduler, req)
                for req in scheduler._inflight_prefills
            ), pool.free)
            for req in list(scheduler._inflight_prefills):
                self.assertIsNotNone(self.allocate(scheduler, req, tokens=64))
                scheduler._inflight_prefills.remove(req)
                pool.free += len(manager.req_to_blocks.pop(req.request_id))
                manager.num_cached_block.pop(req.request_id)
                completed.append(req.request_id)
            if not pending:
                break
        self.assertEqual(len(completed), len(requests))
        self.assertEqual(pool.free, 16)

    def test_rolling_concurrency_reproduces_old_wedge_and_drains_with_guard(self):
        def make():
            manager = _manager(cap=4)
            manager.get_num_skipped_tokens = lambda tokens: max(0, tokens - 16)
            scheduler = _scheduler([manager], free=16)
            _wire_allocator(scheduler)
            return scheduler, manager

        old, old_manager = make()
        old_requests = [_request(str(i), 3200) for i in range(16)]
        for req in old_requests:
            # v0.25.1's capped predictor reports no reservation after each
            # long prefix is loaded: its table has 99 nulls and one real block.
            self.assertIsNotNone(KVCacheManager.allocate_slots(
                old.kv_cache_manager, req, 0, num_external_computed_tokens=1600,
                delay_cache_blocks=True, full_sequence_must_fit=True,
            ))
            req.num_computed_tokens = 1600
            self.assertEqual(old_manager.get_num_blocks_to_allocate(
                req.request_id, 3200, [], 1600, 3200, apply_admission_cap=True,
            ), 0)
        self.assertEqual(old.kv_cache_manager.block_pool.free, 0)
        for req in old_requests:
            self.assertIsNone(KVCacheManager.allocate_slots(
                old.kv_cache_manager, req, 32,
            ))

        scheduler, manager = make()
        pending = [_request(str(i), 3200) for i in range(64)]
        completed = []
        pool = scheduler.kv_cache_manager.block_pool
        for _ in range(2000):
            while pending:
                req = pending[0]
                if self.allocate(scheduler, req, num_external_computed_tokens=1600,
                                 delay_cache_blocks=True,
                                 full_sequence_must_fit=True) is None:
                    break
                pending.pop(0)
                req.num_computed_tokens = 1600
                scheduler._inflight_prefills.append(req)
            self.assertLessEqual(
                sum(offload_remaining_blocks(scheduler, req)
                    for req in scheduler._inflight_prefills),
                pool.free,
            )
            progressed = False
            for req in list(scheduler._inflight_prefills):
                self.assertIsNotNone(self.allocate(scheduler, req, tokens=32))
                req.num_computed_tokens += 32
                progressed = True
                if req.num_computed_tokens == req.num_tokens:
                    scheduler._inflight_prefills.remove(req)
                    pool.free += sum(not block.is_null for block in
                                     manager.req_to_blocks.pop(req.request_id))
                    manager.num_cached_block.pop(req.request_id)
                    completed.append(req.request_id)
            self.assertTrue(progressed)
            if not pending and not scheduler._inflight_prefills:
                break
        self.assertEqual(len(completed), 64)
        self.assertEqual(pool.free, 16)
