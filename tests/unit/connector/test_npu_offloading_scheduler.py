# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU offloading connector scheduler: store cursor and hybrid lookup."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    OffloadingConnectorScheduler,
    RequestOffloadState,
    SchedulerOffloadConfig,
    TransferJobStatus,
)
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadPolicy,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)

from omni_npu.connector.npu_offloading_connector import NPUOffloadingConnector
from omni_npu.connector.npu_offloading_scheduler import (
    NPUOffloadingConnectorScheduler,
    storeable_num_blocks,
)

BLOCK_SIZE = 128
CHUNK_TOKENS = 8192
BLOCKS_PER_CHUNK = CHUNK_TOKENS // BLOCK_SIZE  # 64
PROMPT_TOKENS = CHUNK_TOKENS * 2  # two chunks
TOTAL_BLOCKS = PROMPT_TOKENS // BLOCK_SIZE  # 128


def _group_config(
    group_idx: int,
    is_eagle: bool,
    *,
    sliding_window_size_in_blocks: int | None = None,
) -> GroupOffloadConfig:
    return GroupOffloadConfig(
        group_idx=group_idx,
        gpu_block_size=BLOCK_SIZE,
        offloaded_block_size=BLOCK_SIZE,
        hash_block_size_factor=1,
        kv_event_group_spec=None,
        sliding_window_size_in_blocks=sliding_window_size_in_blocks,
        alignment_block_count=None,
        is_eagle_group=is_eagle,
    )


def _scheduler_config(*group_configs, offload_prompt_only=False):
    return SchedulerOffloadConfig(
        kv_group_configs=tuple(group_configs),
        block_size_factor=1,
        num_workers=1,
        offload_prompt_only=offload_prompt_only,
    )


def _make_req_status(config: SchedulerOffloadConfig) -> RequestOffloadState:
    req = SimpleNamespace(
        request_id="req-0",
        kv_transfer_params=None,
        num_computed_tokens=0,
        num_tokens=PROMPT_TOKENS,
        num_prompt_tokens=PROMPT_TOKENS,
    )
    req_status = RequestOffloadState(
        config=config,
        req=req,
        req_context=ReqContext(req_id="req-0", kv_transfer_params=None),
        offloading_context=RequestOffloadingContext(),
    )
    for group_state in req_status.group_states:
        # key i identifies offloaded block i, so assertions read directly as
        # block indices. GPU block ids start at 1 because 0 is the null block.
        group_state.offload_keys.extend(range(TOTAL_BLOCKS))
        group_state.block_ids.extend(range(1, TOTAL_BLOCKS + 1))
    return req_status


def _empty_prepare_load(keys, ctx):
    return SimpleNamespace()


def _make_scheduler(config, req_status, recorded, store_everything=True):
    """A scheduler instance wired up without touching OffloadingSpec.

    ``__new__`` skips ``__init__`` (which would build a spec and a manager);
    every attribute ``_build_store_jobs`` reads is filled in by hand.
    """
    scheduler = NPUOffloadingConnectorScheduler.__new__(
        NPUOffloadingConnectorScheduler
    )
    job_ids = iter(range(1000))

    def record_store(req, group_config, offloaded_block_idx, offload_key):
        recorded.append(offloaded_block_idx)

    def prepare_store(keys, ctx):
        return PrepareStoreOutput(
            keys_to_store=list(keys) if store_everything else [],
            store_spec=SimpleNamespace(),
            evicted_keys=[],
        )

    def _noop_touch(req_status_):
        return None

    def _next_job_id():
        return next(job_ids)

    scheduler.config = config
    scheduler.manager = SimpleNamespace(prepare_store=prepare_store)
    scheduler._req_status = {"req-0": req_status}
    scheduler._touch = _noop_touch
    scheduler._events_tracker = SimpleNamespace(record_store=record_store)
    scheduler._generate_job_id = _next_job_id
    scheduler._jobs = {}
    scheduler._block_id_to_pending_jobs = {}
    return scheduler


def _run_step(scheduler, num_computed_tokens: int, num_scheduled_tokens: int):
    scheduler._req_status["req-0"].req.num_computed_tokens = num_computed_tokens
    scheduler_output = SimpleNamespace(
        num_scheduled_tokens={"req-0": num_scheduled_tokens}
    )
    return scheduler._build_store_jobs(scheduler_output)


# --------------------------------------------------------------------------
# storeable_num_blocks
# --------------------------------------------------------------------------


def test_storeable_num_blocks_keeps_every_block_for_non_eagle_group():
    assert (
        storeable_num_blocks(_group_config(0, is_eagle=False), CHUNK_TOKENS)
        == BLOCKS_PER_CHUNK
    )


def test_storeable_num_blocks_drops_trailing_block_for_eagle_group():
    assert (
        storeable_num_blocks(_group_config(0, is_eagle=True), CHUNK_TOKENS)
        == BLOCKS_PER_CHUNK - 1
    )


def test_storeable_num_blocks_never_goes_negative():
    config = _group_config(0, is_eagle=True)

    assert storeable_num_blocks(config, 0) == 0
    assert storeable_num_blocks(config, BLOCK_SIZE - 1) == 0


def test_advance_stored_idx_stops_before_the_eagle_trailing_block():
    config = _scheduler_config(
        _group_config(0, is_eagle=False), _group_config(1, is_eagle=True)
    )
    req_status = _make_req_status(config)
    scheduler = _make_scheduler(config, req_status, [])

    scheduler._advance_stored_idx(req_status, CHUNK_TOKENS)

    full_attn_state, eagle_state = req_status.group_states
    assert full_attn_state.next_stored_block_idx == BLOCKS_PER_CHUNK
    # must NOT be BLOCKS_PER_CHUNK: block 63 was never offered for store, and
    # a cursor at 64 would skip it forever
    assert eagle_state.next_stored_block_idx == BLOCKS_PER_CHUNK - 1


# --------------------------------------------------------------------------
# the regression: a cursor left past the last offered block
# --------------------------------------------------------------------------


def test_eagle_group_offloads_a_contiguous_key_sequence_across_chunks():
    config = _scheduler_config(_group_config(0, is_eagle=True))
    req_status = _make_req_status(config)
    recorded: list[int] = []
    scheduler = _make_scheduler(config, req_status, recorded)
    (group_state,) = req_status.group_states

    # chunk 1: tokens 0..8191 -> blocks 0..63 complete, block 63 is the
    # volatile trailing block and is held back.
    assert _run_step(scheduler, 0, CHUNK_TOKENS)
    assert recorded == list(range(BLOCKS_PER_CHUNK - 1))
    assert group_state.next_stored_block_idx == BLOCKS_PER_CHUNK - 1

    # chunk 2: block 63 is no longer trailing, so it must be picked up here.
    assert _run_step(scheduler, CHUNK_TOKENS, CHUNK_TOKENS)
    assert group_state.next_stored_block_idx == TOTAL_BLOCKS - 1

    # The whole point: no gap anywhere in the offloaded key sequence.
    assert recorded == list(range(TOTAL_BLOCKS - 1))


def test_eagle_cursor_holds_back_when_nothing_is_stored():
    """The early-return path, where upstream calls advance_stored_idx."""
    config = _scheduler_config(_group_config(0, is_eagle=True))
    req_status = _make_req_status(config)
    recorded: list[int] = []
    scheduler = _make_scheduler(
        config, req_status, recorded, store_everything=False
    )
    (group_state,) = req_status.group_states

    assert _run_step(scheduler, 0, CHUNK_TOKENS) == {}
    assert recorded == []
    assert group_state.next_stored_block_idx == BLOCKS_PER_CHUNK - 1


def test_non_eagle_group_still_offloads_every_completed_block():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    req_status = _make_req_status(config)
    recorded: list[int] = []
    scheduler = _make_scheduler(config, req_status, recorded)
    (group_state,) = req_status.group_states

    _run_step(scheduler, 0, CHUNK_TOKENS)
    assert group_state.next_stored_block_idx == BLOCKS_PER_CHUNK

    _run_step(scheduler, CHUNK_TOKENS, CHUNK_TOKENS)
    assert group_state.next_stored_block_idx == TOTAL_BLOCKS

    assert recorded == list(range(TOTAL_BLOCKS))


def test_mixed_groups_keep_independent_cursors():
    config = _scheduler_config(
        _group_config(0, is_eagle=False), _group_config(1, is_eagle=True)
    )
    req_status = _make_req_status(config)
    recorded: list[int] = []
    scheduler = _make_scheduler(config, req_status, recorded)
    full_attn_state, eagle_state = req_status.group_states

    _run_step(scheduler, 0, CHUNK_TOKENS)

    assert full_attn_state.next_stored_block_idx == BLOCKS_PER_CHUNK
    assert eagle_state.next_stored_block_idx == BLOCKS_PER_CHUNK - 1


def test_prompt_only_stops_the_cursor_at_the_prompt_boundary():
    config = _scheduler_config(
        _group_config(0, is_eagle=False), offload_prompt_only=True
    )
    req_status = _make_req_status(config)
    req_status.req.num_tokens = PROMPT_TOKENS + CHUNK_TOKENS
    scheduler = _make_scheduler(config, req_status, [])
    (group_state,) = req_status.group_states

    # decode past the prompt: no block beyond the prompt may become eligible
    _run_step(scheduler, PROMPT_TOKENS, CHUNK_TOKENS)

    assert group_state.next_stored_block_idx == TOTAL_BLOCKS


def test_unknown_request_is_skipped():
    config = _scheduler_config(_group_config(0, is_eagle=True))
    scheduler = _make_scheduler(config, _make_req_status(config), [])

    assert (
        scheduler._build_store_jobs(
            SimpleNamespace(num_scheduled_tokens={"other-req": CHUNK_TOKENS})
        )
        == {}
    )


def _src_block_ids(jobs) -> list[int]:
    job = next(iter(jobs.values()))
    return [int(x) for x in job.src_spec.block_ids]


def test_swa_reused_gpu_block_is_stored_once():
    """vLLM #42903: the same physical GPU block must not enter src_spec twice."""
    config = _scheduler_config(
        GroupOffloadConfig(
            group_idx=0,
            gpu_block_size=BLOCK_SIZE,
            offloaded_block_size=BLOCK_SIZE,
            hash_block_size_factor=1,
            kv_event_group_spec=None,
            sliding_window_size_in_blocks=2,
            alignment_block_count=None,
            is_eagle_group=False,
        )
    )
    req_status = _make_req_status(config)
    (group_state,) = req_status.group_states
    group_state.block_ids[:4] = [1, 2, 1, 3]
    recorded: list[int] = []
    scheduler = _make_scheduler(config, req_status, recorded)

    jobs = _run_step(scheduler, 0, 4 * BLOCK_SIZE)
    src_ids = _src_block_ids(jobs)

    assert src_ids == [1, 2, 3]
    assert recorded == [0, 1, 3]


def test_in_flight_gpu_block_is_not_stored_again():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    req_status = _make_req_status(config)
    recorded: list[int] = []
    scheduler = _make_scheduler(config, req_status, recorded)
    scheduler._jobs[99] = TransferJobStatus(
        req_id="other",
        pending_count=1,
        keys=set(),
        is_store=True,
        non_sliding_window_block_ids=[1],
        sliding_window_block_ids=None,
    )

    jobs = _run_step(scheduler, 0, CHUNK_TOKENS)
    src_ids = _src_block_ids(jobs)

    assert 1 not in src_ids
    assert recorded == list(range(1, BLOCKS_PER_CHUNK))


# --------------------------------------------------------------------------
# connector wiring
# --------------------------------------------------------------------------


def _make_connector(role):
    vllm_config = SimpleNamespace(kv_transfer_config=SimpleNamespace())
    with patch(
        "omni_npu.connector.npu_offloading_connector.OffloadingSpecFactory"
        ".create_spec",
        return_value="spec-sentinel",
    ) as create_spec, patch(
        "omni_npu.connector.npu_offloading_connector"
        ".NPUOffloadingConnectorScheduler"
    ) as scheduler_cls, patch(
        "omni_npu.connector.npu_offloading_connector.OffloadingConnectorWorker"
    ) as worker_cls:
        connector = NPUOffloadingConnector(vllm_config, role, SimpleNamespace())
    return connector, create_spec, scheduler_cls, worker_cls


def test_scheduler_role_builds_the_npu_scheduler_from_a_single_spec():
    connector, create_spec, scheduler_cls, worker_cls = _make_connector(
        KVConnectorRole.SCHEDULER
    )

    # one spec only: a second one would build a second offloading manager
    create_spec.assert_called_once()
    scheduler_cls.assert_called_once_with("spec-sentinel")
    worker_cls.assert_not_called()
    assert connector.connector_scheduler is scheduler_cls.return_value
    assert connector.connector_worker is None


def test_worker_role_builds_the_upstream_worker():
    connector, create_spec, scheduler_cls, worker_cls = _make_connector(
        KVConnectorRole.WORKER
    )

    create_spec.assert_called_once()
    worker_cls.assert_called_once_with("spec-sentinel")
    scheduler_cls.assert_not_called()
    assert connector.connector_scheduler is None
    assert connector.connector_worker is worker_cls.return_value


# --------------------------------------------------------------------------
# per-group HBM start for hybrid HBM+DDR lookup
# --------------------------------------------------------------------------


def _make_lookup_scheduler(config, lookup_results: dict):
    scheduler = NPUOffloadingConnectorScheduler.__new__(
        NPUOffloadingConnectorScheduler
    )
    sliding = tuple(
        g.group_idx
        for g in config.kv_group_configs
        if g.sliding_window_size_in_blocks is not None
    )
    full = tuple(
        g.group_idx
        for g in config.kv_group_configs
        if g.sliding_window_size_in_blocks is None
    )
    scheduler.config = config
    scheduler._per_group_local_tokens = {}
    scheduler._sliding_window_groups = sliding
    scheduler._lookup_groups = full + sliding
    scheduler._mamba_align_size = None
    scheduler._blocks_being_loaded = None

    def _lookup(key, ctx):
        return lookup_results.get(key, LookupResult.MISS)

    scheduler.manager = SimpleNamespace(lookup=_lookup)
    return scheduler


def _make_lookup_req_status(
    scheduler,
    *,
    num_tokens: int,
    num_computed_tokens: int,
    offload_keys_per_group: list[list[int]],
):
    req = SimpleNamespace(
        request_id="req-0",
        kv_transfer_params=None,
        num_tokens=num_tokens,
        local_computed_tokens_per_group=None,
    )
    req_status = RequestOffloadState(
        config=scheduler.config,
        req=req,
        req_context=ReqContext(req_id="req-0", kv_transfer_params=None),
        offloading_context=RequestOffloadingContext(policy=OffloadPolicy.BLOCK_LEVEL),
        num_locally_computed_tokens=num_computed_tokens,
    )
    for group_state, keys in zip(req_status.group_states, offload_keys_per_group):
        group_state.offload_keys.extend(keys)
    return req_status


def _gpu_block(block_id: int, hashed: bool):
    return SimpleNamespace(
        block_id=block_id,
        is_null=False,
        block_hash=object() if hashed else None,
    )


def test_group_local_tokens_prefers_per_group_hbm_hits():
    config = _scheduler_config(
        _group_config(0, is_eagle=False), _group_config(1, is_eagle=False)
    )
    req_status = _make_req_status(config)
    scheduler = _make_lookup_scheduler(config, {})
    req_status.num_locally_computed_tokens = BLOCK_SIZE
    scheduler._per_group_local_tokens["req-0"] = (2 * BLOCK_SIZE, BLOCK_SIZE)

    assert scheduler._group_local_tokens(req_status, 0) == 2 * BLOCK_SIZE
    assert scheduler._group_local_tokens(req_status, 1) == BLOCK_SIZE


def test_group_local_tokens_falls_back_to_scalar_local():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    req_status = _make_req_status(config)
    scheduler = _make_lookup_scheduler(config, {})
    req_status.num_locally_computed_tokens = 3 * BLOCK_SIZE

    assert scheduler._group_local_tokens(req_status, 0) == 3 * BLOCK_SIZE


def test_capture_per_group_local_tokens_moves_off_the_request():
    config = _scheduler_config(
        _group_config(0, is_eagle=False), _group_config(1, is_eagle=False)
    )
    scheduler = _make_lookup_scheduler(config, {})
    request = SimpleNamespace(
        request_id="req-0",
        local_computed_tokens_per_group=(2 * BLOCK_SIZE, BLOCK_SIZE),
    )

    scheduler._capture_per_group_local_tokens(request)

    assert scheduler._per_group_local_tokens["req-0"] == (2 * BLOCK_SIZE, BLOCK_SIZE)
    assert request.local_computed_tokens_per_group is None


def test_request_finished_clears_per_group_map():
    config = _scheduler_config(
        _group_config(0, is_eagle=False), _group_config(1, is_eagle=False)
    )
    scheduler = _make_lookup_scheduler(config, {})
    scheduler._per_group_local_tokens["req-0"] = (2 * BLOCK_SIZE, BLOCK_SIZE)
    scheduler._req_status = {}
    request = SimpleNamespace(request_id="req-0")

    with patch.object(
        OffloadingConnectorScheduler,
        "request_finished",
        return_value=(False, None),
    ):
        assert scheduler.request_finished(request) == (False, None)

    assert "req-0" not in scheduler._per_group_local_tokens


def test_lookup_starts_each_group_at_its_own_hbm_hit():
    """FA longer in HBM than DDR must not zero the joint prefix.

    Upstream starts both groups at scalar min. Group 0's first DDR key is
    then a miss (that block lives only in HBM) and the whole lookup returns 0.
    """
    config = _scheduler_config(
        _group_config(0, is_eagle=False), _group_config(1, is_eagle=False)
    )
    # Group 0: HBM covers blocks 0-1; DDR has 2, 3.
    # Group 1: HBM covers block 0; DDR has 1, 2, 3.
    lookup_results = {
        2: LookupResult.HIT,
        3: LookupResult.HIT,
        101: LookupResult.HIT,
        102: LookupResult.HIT,
        103: LookupResult.HIT,
    }
    scheduler = _make_lookup_scheduler(config, lookup_results)
    scheduler._per_group_local_tokens["req-0"] = (2 * BLOCK_SIZE, BLOCK_SIZE)
    req_status = _make_lookup_req_status(
        scheduler,
        num_tokens=4 * BLOCK_SIZE,
        num_computed_tokens=BLOCK_SIZE,
        offload_keys_per_group=[
            [0, 1, 2, 3],
            [100, 101, 102, 103],
        ],
    )

    assert scheduler._lookup(req_status) == 3 * BLOCK_SIZE


def test_lookup_keeps_hbm_prefix_when_one_group_has_no_ddr_suffix():
    config = _scheduler_config(
        _group_config(0, is_eagle=False), _group_config(1, is_eagle=False)
    )
    # Group 0 fully covered by HBM (2 blocks). Group 1 extends via DDR.
    lookup_results = {
        101: LookupResult.HIT,
        102: LookupResult.HIT,
    }
    scheduler = _make_lookup_scheduler(config, lookup_results)
    scheduler._per_group_local_tokens["req-0"] = (2 * BLOCK_SIZE, BLOCK_SIZE)
    req_status = _make_lookup_req_status(
        scheduler,
        num_tokens=3 * BLOCK_SIZE,
        num_computed_tokens=BLOCK_SIZE,
        offload_keys_per_group=[
            [0, 1, 2],
            [100, 101, 102],
        ],
    )

    # assembled min is 2 blocks; scalar local is 1 block → 1 block external
    assert scheduler._lookup(req_status) == BLOCK_SIZE


def test_alloc_does_not_assert_when_a_lagging_group_starts_before_scalar_local():
    config = _scheduler_config(
        _group_config(0, is_eagle=False), _group_config(1, is_eagle=False)
    )
    req_status = _make_req_status(config)
    req_status.num_locally_computed_tokens = 2 * BLOCK_SIZE
    req_status.offloading_context = RequestOffloadingContext(
        policy=OffloadPolicy.BLOCK_LEVEL
    )
    for group_state in req_status.group_states:
        group_state.offload_keys = list(range(4))
    scheduler = _make_scheduler(config, req_status, [])
    scheduler._per_group_local_tokens = {
        "req-0": (2 * BLOCK_SIZE, BLOCK_SIZE)
    }
    scheduler._current_batch_allocated_block_ids = set()
    scheduler._current_batch_load_jobs = {}
    scheduler._blocks_being_loaded = None
    scheduler.manager.prepare_load = _empty_prepare_load

    # 3 GPU blocks = cdiv(local 256 + external 128, 128). Group 0 hashed for
    # 2 blocks; group 1 hashed only for 1 (lagging), so load starts at block 1.
    blocks = SimpleNamespace(
        blocks=(
            [
                _gpu_block(1, hashed=True),
                _gpu_block(2, hashed=True),
                _gpu_block(3, hashed=False),
            ],
            [
                _gpu_block(11, hashed=True),
                _gpu_block(12, hashed=False),
                _gpu_block(13, hashed=False),
            ],
        )
    )

    scheduler.update_state_after_alloc(req_status.req, blocks, BLOCK_SIZE)

    job = next(iter(scheduler._current_batch_load_jobs.values()))
    assert list(job.dst_spec.group_sizes) == [1, 2]


def _alloc_scheduler_for_pending(config, num_cached_blocks: int):
    req_status = _make_req_status(config)
    req_status.num_locally_computed_tokens = 0
    req_status.offloading_context = RequestOffloadingContext(
        policy=OffloadPolicy.BLOCK_LEVEL
    )
    for group_state in req_status.group_states:
        group_state.offload_keys = list(range(num_cached_blocks))
    scheduler = _make_scheduler(config, req_status, [])
    scheduler._per_group_local_tokens = {}
    scheduler._current_batch_allocated_block_ids = set()
    scheduler._current_batch_load_jobs = {}
    scheduler._blocks_being_loaded = None
    scheduler.manager.prepare_load = _empty_prepare_load
    return scheduler, req_status


def test_alloc_allows_mome_pending_up_to_the_window():
    """DDR load only covers the MoME window (1 snapshot), not extra reserved."""
    config = _scheduler_config(
        _group_config(0, is_eagle=False),
        _group_config(1, is_eagle=False, sliding_window_size_in_blocks=1),
    )
    num_blocks = 1
    scheduler, req_status = _alloc_scheduler_for_pending(config, num_blocks)
    blocks = SimpleNamespace(
        blocks=(
            [_gpu_block(i + 1, hashed=False) for i in range(num_blocks)],
            [_gpu_block(100 + i, hashed=False) for i in range(num_blocks)],
        )
    )

    scheduler.update_state_after_alloc(
        req_status.req, blocks, num_blocks * BLOCK_SIZE
    )

    job = next(iter(scheduler._current_batch_load_jobs.values()))
    assert list(job.dst_spec.group_sizes) == [num_blocks, num_blocks]


def test_alloc_rejects_mome_pending_past_the_window():
    config = _scheduler_config(
        _group_config(0, is_eagle=False),
        _group_config(1, is_eagle=False, sliding_window_size_in_blocks=1),
    )
    num_blocks = 2
    scheduler, req_status = _alloc_scheduler_for_pending(config, num_blocks)
    blocks = SimpleNamespace(
        blocks=(
            [_gpu_block(i + 1, hashed=False) for i in range(num_blocks)],
            [_gpu_block(100 + i, hashed=False) for i in range(num_blocks)],
        )
    )

    with pytest.raises(RuntimeError, match="exceeds sliding-window limit 1"):
        scheduler.update_state_after_alloc(
            req_status.req, blocks, num_blocks * BLOCK_SIZE
        )


def test_in_flight_store_ids_skip_load_jobs():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    scheduler = _make_scheduler(config, _make_req_status(config), [])
    scheduler._block_id_to_pending_jobs = {9: {1}}
    scheduler._jobs[1] = TransferJobStatus(
        req_id="req-0",
        pending_count=1,
        keys=set(),
        is_store=False,
        non_sliding_window_block_ids=[20],
        sliding_window_block_ids=[21],
    )

    assert scheduler._in_flight_store_gpu_ids() == {9}


def test_init_creates_per_group_local_tokens():
    spec = MagicMock()
    with patch.object(OffloadingConnectorScheduler, "__init__", return_value=None):
        scheduler = NPUOffloadingConnectorScheduler(spec)

    assert scheduler._per_group_local_tokens == {}


def test_per_group_map_creates_mapping_when_missing():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    scheduler = _make_lookup_scheduler(config, {})
    del scheduler._per_group_local_tokens

    mapping = scheduler._per_group_map()

    assert mapping == {}
    assert scheduler._per_group_local_tokens is mapping


def test_capture_per_group_local_tokens_clears_stale_entry():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    scheduler = _make_lookup_scheduler(config, {})
    scheduler._per_group_local_tokens["req-0"] = (BLOCK_SIZE,)
    request = SimpleNamespace(
        request_id="req-0",
        local_computed_tokens_per_group=None,
    )

    scheduler._capture_per_group_local_tokens(request)

    assert "req-0" not in scheduler._per_group_local_tokens


def test_get_num_new_matched_tokens_captures_then_delegates():
    config = _scheduler_config(
        _group_config(0, is_eagle=False), _group_config(1, is_eagle=False)
    )
    scheduler = _make_lookup_scheduler(config, {})
    request = SimpleNamespace(
        request_id="req-0",
        local_computed_tokens_per_group=(2 * BLOCK_SIZE, BLOCK_SIZE),
    )

    with patch.object(
        OffloadingConnectorScheduler,
        "get_num_new_matched_tokens",
        return_value=(3, True),
    ):
        assert scheduler.get_num_new_matched_tokens(request, 0) == (3, True)

    assert scheduler._per_group_local_tokens["req-0"] == (2 * BLOCK_SIZE, BLOCK_SIZE)
    assert request.local_computed_tokens_per_group is None


def test_lookup_raises_when_offload_keys_are_short():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    scheduler = _make_lookup_scheduler(config, {})
    req_status = _make_lookup_req_status(
        scheduler,
        num_tokens=4 * BLOCK_SIZE,
        num_computed_tokens=0,
        offload_keys_per_group=[[0]],
    )

    with pytest.raises(RuntimeError, match="need at least 4"):
        scheduler._lookup(req_status)


def test_lookup_returns_zero_when_remaining_tokens_are_below_one_block():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    scheduler = _make_lookup_scheduler(config, {0: LookupResult.HIT, 1: LookupResult.HIT})
    req_status = _make_lookup_req_status(
        scheduler,
        num_tokens=BLOCK_SIZE + 10,
        num_computed_tokens=BLOCK_SIZE,
        offload_keys_per_group=[[0, 1]],
    )

    assert scheduler._lookup(req_status) == 0


def test_lookup_sliding_window_reduces_prompt_and_uses_window_hits():
    config = _scheduler_config(
        _group_config(0, is_eagle=False, sliding_window_size_in_blocks=2)
    )
    lookup_results = {
        0: LookupResult.HIT,
        1: LookupResult.HIT,
        2: LookupResult.HIT,
        3: LookupResult.HIT,
    }
    scheduler = _make_lookup_scheduler(config, lookup_results)
    scheduler._mamba_align_size = BLOCK_SIZE
    req_status = _make_lookup_req_status(
        scheduler,
        num_tokens=4 * BLOCK_SIZE,
        num_computed_tokens=0,
        offload_keys_per_group=[[0, 1, 2, 3]],
    )

    assert scheduler._lookup(req_status) == 3 * BLOCK_SIZE


def test_lookup_empty_ddr_slice_keeps_hbm_prefix():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    scheduler = _make_lookup_scheduler(config, {})
    scheduler._per_group_local_tokens["req-0"] = (2 * BLOCK_SIZE,)
    req_status = _make_lookup_req_status(
        scheduler,
        num_tokens=2 * BLOCK_SIZE,
        num_computed_tokens=0,
        offload_keys_per_group=[[0, 1]],
    )

    assert scheduler._lookup(req_status) == 2 * BLOCK_SIZE


def test_lookup_eagle_sliding_window_pops_volatile_block():
    config = _scheduler_config(
        _group_config(0, is_eagle=True, sliding_window_size_in_blocks=2)
    )
    lookup_results = {
        0: LookupResult.HIT,
        1: LookupResult.HIT,
        2: LookupResult.HIT,
        3: LookupResult.HIT,
    }
    scheduler = _make_lookup_scheduler(config, lookup_results)
    req_status = _make_lookup_req_status(
        scheduler,
        num_tokens=4 * BLOCK_SIZE,
        num_computed_tokens=0,
        offload_keys_per_group=[[0, 1, 2, 3]],
    )

    # Window query is 3 blocks (2 + eagle extra); popping the volatile
    # trailing block leaves 3 assembled tokens = 384.
    assert scheduler._lookup(req_status) == 3 * BLOCK_SIZE


def test_lookup_defers_when_backend_returns_pending():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    scheduler = _make_lookup_scheduler(
        config, {0: LookupResult.HIT_PENDING, 1: LookupResult.HIT}
    )
    req_status = _make_lookup_req_status(
        scheduler,
        num_tokens=2 * BLOCK_SIZE,
        num_computed_tokens=0,
        offload_keys_per_group=[[0, 1]],
    )

    assert scheduler._lookup(req_status) is None


def test_lookup_retries_when_a_later_group_tightens_a_deferred_hit():
    config = _scheduler_config(
        _group_config(0, is_eagle=True), _group_config(1, is_eagle=False)
    )
    lookup_results = {
        0: LookupResult.HIT_PENDING,
        1: LookupResult.HIT,
        100: LookupResult.MISS,
        101: LookupResult.MISS,
    }
    scheduler = _make_lookup_scheduler(config, lookup_results)
    req_status = _make_lookup_req_status(
        scheduler,
        num_tokens=2 * BLOCK_SIZE,
        num_computed_tokens=0,
        offload_keys_per_group=[[0, 1], [100, 101]],
    )

    assert scheduler._lookup(req_status) == 0


def test_lookup_rechecks_sliding_window_after_prefix_shrinks():
    config = _scheduler_config(
        _group_config(0, is_eagle=False),
        _group_config(1, is_eagle=False, sliding_window_size_in_blocks=1),
    )
    lookup_results = {
        0: LookupResult.HIT,
        1: LookupResult.HIT,
        2: LookupResult.HIT,
        100: LookupResult.MISS,
        101: LookupResult.MISS,
        102: LookupResult.MISS,
    }
    scheduler = _make_lookup_scheduler(config, lookup_results)
    req_status = _make_lookup_req_status(
        scheduler,
        num_tokens=3 * BLOCK_SIZE,
        num_computed_tokens=0,
        offload_keys_per_group=[[0, 1, 2], [100, 101, 102]],
    )

    assert scheduler._lookup(req_status) == 0


@pytest.mark.parametrize("sliding_window_size_in_blocks", [None, 1])
def test_lookup_delays_when_hit_blocks_are_already_loading(
    sliding_window_size_in_blocks,
):
    config = _scheduler_config(
        _group_config(
            0,
            is_eagle=False,
            sliding_window_size_in_blocks=sliding_window_size_in_blocks,
        )
    )
    lookup_results = {0: LookupResult.HIT, 1: LookupResult.HIT}
    scheduler = _make_lookup_scheduler(config, lookup_results)
    scheduler._blocks_being_loaded = {1}
    req_status = _make_lookup_req_status(
        scheduler,
        num_tokens=2 * BLOCK_SIZE,
        num_computed_tokens=0,
        offload_keys_per_group=[[0, 1]],
    )

    assert scheduler._lookup(req_status) is None


def test_alloc_returns_early_when_no_external_tokens():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    scheduler, req_status = _alloc_scheduler_for_pending(config, 1)
    blocks = SimpleNamespace(blocks=([_gpu_block(1, hashed=False)],))

    scheduler.update_state_after_alloc(req_status.req, blocks, 0)

    assert scheduler._current_batch_load_jobs == {}


def test_alloc_rejects_too_few_gpu_blocks():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    scheduler, req_status = _alloc_scheduler_for_pending(config, 2)
    blocks = SimpleNamespace(blocks=([_gpu_block(1, hashed=False)],))

    with pytest.raises(RuntimeError, match="need at least 2"):
        scheduler.update_state_after_alloc(req_status.req, blocks, 2 * BLOCK_SIZE)


def test_alloc_rejects_hashed_prefix_shorter_than_used_local():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    scheduler, req_status = _alloc_scheduler_for_pending(config, 2)
    scheduler._per_group_local_tokens["req-0"] = (2 * BLOCK_SIZE,)
    blocks = SimpleNamespace(
        blocks=(
            [_gpu_block(1, hashed=False), _gpu_block(2, hashed=False)],
        )
    )

    with pytest.raises(RuntimeError, match="hashed prefix"):
        scheduler.update_state_after_alloc(req_status.req, blocks, 2 * BLOCK_SIZE)


def test_alloc_rejects_too_few_offload_keys():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    scheduler, req_status = _alloc_scheduler_for_pending(config, 1)
    req_status.group_states[0].offload_keys = [0]
    blocks = SimpleNamespace(
        blocks=(
            [_gpu_block(1, hashed=False), _gpu_block(2, hashed=False)],
        )
    )

    with pytest.raises(RuntimeError, match="offload keys"):
        scheduler.update_state_after_alloc(req_status.req, blocks, 2 * BLOCK_SIZE)


def test_alloc_rejects_when_transfer_jobs_are_pending():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    scheduler, req_status = _alloc_scheduler_for_pending(config, 1)
    req_status.transfer_jobs.add(3)
    blocks = SimpleNamespace(blocks=([_gpu_block(1, hashed=False)],))

    with pytest.raises(RuntimeError, match="cannot issue load job"):
        scheduler.update_state_after_alloc(req_status.req, blocks, BLOCK_SIZE)


def test_alloc_records_keys_in_blocks_being_loaded():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    scheduler, req_status = _alloc_scheduler_for_pending(config, 1)
    scheduler._blocks_being_loaded = set()
    blocks = SimpleNamespace(blocks=([_gpu_block(1, hashed=False)],))

    scheduler.update_state_after_alloc(req_status.req, blocks, BLOCK_SIZE)

    assert scheduler._blocks_being_loaded == {0}


def test_store_rejects_offload_key_and_block_id_mismatch():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    req_status = _make_req_status(config)
    req_status.group_states[0].block_ids = [1, 2]
    scheduler = _make_scheduler(config, req_status, [])

    with pytest.raises(RuntimeError, match="offload key count"):
        _run_step(scheduler, 0, CHUNK_TOKENS)


def test_store_rejects_alignment_without_sliding_window():
    config = _scheduler_config(
        GroupOffloadConfig(
            group_idx=0,
            gpu_block_size=BLOCK_SIZE,
            offloaded_block_size=BLOCK_SIZE,
            hash_block_size_factor=1,
            kv_event_group_spec=None,
            sliding_window_size_in_blocks=None,
            alignment_block_count=4,
            is_eagle_group=False,
        )
    )
    req_status = _make_req_status(config)
    scheduler = _make_scheduler(config, req_status, [])

    with pytest.raises(RuntimeError, match="alignment_block_count"):
        _run_step(scheduler, 0, CHUNK_TOKENS)


def test_store_rejects_when_a_load_job_is_pending():
    config = _scheduler_config(_group_config(0, is_eagle=False))
    req_status = _make_req_status(config)
    req_status.transfer_jobs.add(7)
    scheduler = _make_scheduler(config, req_status, [])
    scheduler._jobs[7] = TransferJobStatus(
        req_id="req-0",
        pending_count=1,
        keys=set(),
        is_store=False,
    )

    with pytest.raises(RuntimeError, match="cannot issue store job"):
        _run_step(scheduler, 0, CHUNK_TOKENS)
