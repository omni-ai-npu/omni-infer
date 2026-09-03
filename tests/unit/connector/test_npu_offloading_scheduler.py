# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Store-cursor alignment for the NPU offloading connector scheduler.

Upstream's own ``test_full_attn_store_excludes_trailing_block`` runs a single
prefill step and therefore cannot observe the store cursor carrying over
between steps, which is where the hole in the offloaded key sequence appears.
"""

from types import SimpleNamespace
from unittest.mock import patch

from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    RequestOffloadState,
    SchedulerOffloadConfig,
)
from vllm.v1.kv_offload.base import (
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


def _group_config(group_idx: int, is_eagle: bool) -> GroupOffloadConfig:
    return GroupOffloadConfig(
        group_idx=group_idx,
        gpu_block_size=BLOCK_SIZE,
        offloaded_block_size=BLOCK_SIZE,
        hash_block_size_factor=1,
        kv_event_group_spec=None,
        sliding_window_size_in_blocks=None,
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

    scheduler.config = config
    scheduler.manager = SimpleNamespace(prepare_store=prepare_store)
    scheduler._req_status = {"req-0": req_status}
    scheduler._touch = lambda req_status_: None
    scheduler._events_tracker = SimpleNamespace(record_store=record_store)
    scheduler._generate_job_id = lambda: next(job_ids)
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
