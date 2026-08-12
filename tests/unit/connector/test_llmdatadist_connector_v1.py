# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT


import time
import pytest
import torch
import queue
import copy
import uuid
from types import SimpleNamespace
from collections import defaultdict
from dataclasses import dataclass
from contextlib import contextmanager
from unittest.mock import MagicMock, patch


from vllm.config import VllmConfig, KVTransferConfig, ParallelConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.v1.request import Request, RequestStatus
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.core.kv_cache_manager import KVCacheBlock


from .test_llmdatadist_manager_v1 import (
    _patch_module,
    _mock_parallel,
    SupportLogCompare,
    _LogCompare,
    _stop_daemon,
)

from omni_npu.connector.utils import ParallelDesc
from omni_npu.connector.llmdatadist_connector_v1 import (
    LLMDataDistConnector,
    Metadata,
    SchemePull,
)

DEFAULT_BLOCK_IDS = [1, 2, 3]
DEFAULT_LAYER_IDS = [0, 1, 2, 3]
DEFAULT_TOKEN_NUM = 256

def _make_kv_transfer_params(
    p_blocks: list[list[int]] = [DEFAULT_BLOCK_IDS],
    d_blocks: list[list[int]] = [DEFAULT_BLOCK_IDS],
    pp_layers: list[list[int]] = [DEFAULT_LAYER_IDS],
    token_num: int = DEFAULT_TOKEN_NUM,
):
    desc = ParallelDesc()
    addrs = [str(i) for i in range(desc.size)]
    return dict(
        token_num=token_num,
        workers=addrs.copy(),
        engines=addrs.copy(),
        prefill_req_id="req0",
        pp_layers=pp_layers,
        prefill_blocks=p_blocks,
        decode_blocks=d_blocks,
        parallel=desc.to_list(),
    )

def _make_vllm_config(is_prefill, pp=1, dp=1, pcp=1, tp=1, dcp=1) -> VllmConfig:
    kv_role = "kv_producer" if is_prefill else "kv_consumer"
    return VllmConfig(
        kv_transfer_config=KVTransferConfig(
            kv_role=kv_role,
            kv_connector="LLMDataDistConnector",
        ),
        parallel_config=ParallelConfig(
            pipeline_parallel_size=pp,
            data_parallel_size=dp,
            prefill_context_parallel_size=pcp,
            tensor_parallel_size=tp,
            decode_context_parallel_size=dcp,
        )
    )


class TestSchemePull:

    def test_tp_to_tp(self):
        with _mock_parallel(tp=32): # prefill node
            params = _make_kv_transfer_params()

        for i in range(16):
            with _mock_parallel(tp=16, rank=i):
                pull = SchemePull(params, DEFAULT_LAYER_IDS)
                targets = pull.targets()
                assert len(targets) == 1
                assert str(i * 2) in targets

    def test_tp_to_dp(self):
        with _mock_parallel(tp=16): # prefill node
            params = _make_kv_transfer_params()

        for i in range(16):
            with _mock_parallel(dp=32, rank=i):
                pull = SchemePull(params, DEFAULT_LAYER_IDS)
                targets = pull.targets()
                assert len(targets) == 1
                assert str(i // 2) in targets

    def test_pp_prefill(self):
        with _mock_parallel(pp=2, tp=8): # prefill node
            params = _make_kv_transfer_params(
                pp_layers=[[0, 1, 2, 3], [4, 5, 6]]
            )

        for i in range(16):
            with _mock_parallel(dp=16, rank=i):
                pull = SchemePull(params, [0, 1, 2, 3, 4, 5, 6])
                targets = pull.targets()
                assert len(targets) == 2
                assert targets[str(i // 2)].layer_ids == [0, 1, 2, 3]
                assert targets[str(i // 2 + 8)].layer_ids == [4, 5, 6]

    def test_done_routes_to_last_prefill_pp_stage(self):
        with _mock_parallel(pp=2, tp=16): # prefill node
            params = _make_kv_transfer_params(
                pp_layers=[[0, 1], [2, 3]]
            )

        # Decode has 64 independent DP ranks. Each request is pulled by one
        # rank, and its completion must be visible through the output PP stage.
        for i in range(64):
            with _mock_parallel(dp=64, rank=i):
                worker, req_id, parts = SchemePull(
                    params, DEFAULT_LAYER_IDS).done()
            assert worker == str(16 + i % 16)
            assert req_id == "req0"
            assert parts == 1

    def test_dcp_prefill(self):
        with _mock_parallel(tp=4, dcp=4): # prefill node
            params = _make_kv_transfer_params(
                p_blocks=[[12, 37]],
                d_blocks=[[1, 2, 3, 4, 5, 6, 7]],
            )

        with _mock_parallel(dp=16, rank=0):
            pull = SchemePull(params, [0, 1, 2, 3])
            targets = pull.targets()
            assert len(targets) == 4

            assert targets[str(0)].p_blocks == [12, 37]
            assert targets[str(2)].p_blocks == [12, 37]
            assert targets[str(1)].p_blocks == [12, 37]
            assert targets[str(3)].p_blocks == [12]

            assert targets[str(0)].d_blocks == [1, 5]
            assert targets[str(1)].d_blocks == [2, 6]
            assert targets[str(2)].d_blocks == [3, 7]
            assert targets[str(3)].d_blocks == [4]

    def test_prefix_cache(self):
        pass


@contextmanager
def _mock_cache_manager():

    class CacheManager(SupportLogCompare):
        def __init__(self, port, is_prefill):
            self._init_log_compare()
            self.is_server = False
            if is_prefill:
                self.is_server = True
            self.inited = True
            self.engine_addr = "test_addr"

        def sleep(self):
            self.inited = False
            self.log.runtime("sleep")

        def weakup(self):
            self.inited = True
            self.log.runtime("weakup")

        def register(self, vllm_caches: dict | list | None) -> list[int]:
            self.log.runtime("register")
            return [0, 1, 2, 3]

        def unregister(self):
            assert self.inited
            self.log.runtime("unregister")

        def pull_blocks(
            self,
            addr: str,
            p_blocks: list[int],
            d_blocks: list[int],
            layer_ids: list[int], # None for all
        ) -> bool:
            assert not self.is_server
            self.log.runtime("pull_blocks")
            return True

    import omni_npu.connector.llmdatadist_connector_v1 as module
    with _patch_module(module, CacheManager=CacheManager):
        SupportLogCompare.clear_common()
        def get_instance() -> tuple[_LogCompare, CacheManager]:
            log, core = SupportLogCompare.get_instance()
            return log, core
        yield get_instance


class TestLLMDataDistConnector:

    def test_sleep_wakeup(self):
        with (
            _mock_parallel(tp=16, rank=0),
            _mock_cache_manager() as get_instance,
        ):
            worker = LLMDataDistConnector(
                vllm_config=_make_vllm_config(True),
                role=KVConnectorRole.WORKER,
                kv_cache_config=None,
            )
            log, core = get_instance()
            log.clear()

            worker.register_kv_caches({})
            log.manual("weakup")
            log.manual("register")
            log.compare_and_clear()

            worker.unregister_kv_caches()
            log.manual("sleep") # auto unregister
            log.compare_and_clear()

            worker.register_kv_caches({})
            log.manual("weakup")
            log.manual("register")
            log.compare_and_clear()

            # Stop the worker's background daemon so its thread doesn't leak into
            # later tests (e.g. test_kv_dump's global time.sleep patch).
            _stop_daemon(worker)

    def test_pull_done_and_feedback(self):
        p_vllm_config = _make_vllm_config(is_prefill=True, tp=4)
        d_vllm_config = _make_vllm_config(is_prefill=False, tp=4)

        def gen_p_worker(i):
            with (
                _mock_cache_manager(),
                _mock_parallel(tp=4, rank=i),
            ):
                return LLMDataDistConnector(
                    vllm_config=p_vllm_config,
                    role=KVConnectorRole.WORKER,
                    kv_cache_config=None,
                )
        def gen_d_worker(i):
            with (
                _mock_cache_manager(),
                _mock_parallel(tp=4, rank=i),
            ):
                return LLMDataDistConnector(
                    vllm_config=d_vllm_config,
                    role=KVConnectorRole.WORKER,
                    kv_cache_config=None,
                )

        # ===================== init =========================

        p_workers = [gen_p_worker(i) for i in range(4)]
        d_workers = [gen_d_worker(i) for i in range(4)]

        p_scheduler = LLMDataDistConnector(
            vllm_config=p_vllm_config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        d_scheduler = LLMDataDistConnector(
            vllm_config=d_vllm_config,
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        request = MagicMock(
            request_id="prefill_req_0",
            status=RequestStatus.FINISHED_LENGTH_CAPPED,
            num_prompt_tokens=128,
        )

        # ===================== check finished =========================

        assert p_scheduler.get_finished_count() == 1
        assert d_scheduler.get_finished_count() is None

        prefill_done = []
        decode_done = []
        def finished():
            for i, it in enumerate(p_workers):
                with _mock_parallel(tp=4, rank=i):
                    send_done, recv_done = it.get_finished(None)
                    for req_id in send_done:
                        assert req_id == "prefill_req_0"
                        prefill_done.append(req_id)
            for i, it in enumerate(d_workers):
                with _mock_parallel(tp=4, rank=i):
                    send_done, recv_done = it.get_finished(None)
                    for req_id in recv_done:
                        assert req_id == "decode_req_0"
                        decode_done.append(req_id)
            return len(prefill_done) == 1 and len(decode_done) == 4

        # ===================== runtime =========================

        request.request_id = "prefill_req_0"
        blocks = MagicMock(blocks=[[MagicMock(block_id=2, block_hash=None)] * 9])
        block_ids = tuple([block.block_id for block in group] for group in blocks.blocks)

        # prefill workers register to scheduler
        p_metadata = p_scheduler.build_connector_meta(scheduler_output=None)
        for i, it in enumerate(p_workers):
            with _mock_parallel(tp=4, rank=i):
                it.bind_connector_metadata(p_metadata)
                it.start_load_kv(forward_context=None)

        # prefill scheduler send kv_transfer_params
        p_scheduler.update_state_after_alloc(request, blocks, 0)
        delay_free, kv_transfer_params = p_scheduler.request_finished(request, block_ids)
        assert delay_free

        # prefill workers register to scheduler
        p_metadata = p_scheduler.build_connector_meta(scheduler_output=None)
        for i, it in enumerate(p_workers):
            with _mock_parallel(tp=4, rank=i):
                it.bind_connector_metadata(p_metadata)
                it.start_load_kv(forward_context=None)

        # decode scheduler build metadata
        request.request_id = "decode_req_0"
        request.kv_transfer_params = kv_transfer_params
        d_scheduler.update_state_after_alloc(request, blocks, 0)
        d_metadata = d_scheduler.build_connector_meta(scheduler_output=None)

        for i, it in enumerate(d_workers):
            it.bind_connector_metadata(d_metadata)

        # decode workers pull kv
        # should not finish until all workers pull done
        for i, it in enumerate(d_workers):
            with _mock_parallel(tp=4, rank=i):
                assert not finished()
                it.start_load_kv(forward_context=None)

        time.sleep(1.0)
        assert finished()
        p_scheduler.update_connector_output(MagicMock(finished_sending=["prefill_req_0"]))

        # Stop every scheduler/worker background daemon created above so their
        # threads don't leak into later tests (e.g. test_kv_dump's global
        # time.sleep patch), which is the root of the cross-test pollution here.
        _stop_daemon(*p_workers, *d_workers, p_scheduler, d_scheduler)

    def test_abort_before_build_notifies_last_prefill_pp_stage(self):
        with _mock_parallel(pp=2, tp=16): # prefill node
            kv_transfer_params = _make_kv_transfer_params(
                pp_layers=[[0, 1], [2, 3]]
            )

        d_scheduler = LLMDataDistConnector(
            vllm_config=_make_vllm_config(is_prefill=False, dp=64),
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        request = MagicMock(
            request_id="decode_req_0",
            status=RequestStatus.FINISHED_ABORTED,
            kv_transfer_params=kv_transfer_params,
        )

        with patch(
            "omni_npu.connector.llmdatadist_connector_v1.SimpleClient"
        ) as client_cls:
            delay_free, params = d_scheduler.request_finished(request, [])

        assert not delay_free
        assert params is None
        client_cls.assert_called_once_with("16")
        client_cls.return_value.query.assert_called_once_with(
            "pull_done", req_id="req0", parts=1)
        client_cls.return_value.close.assert_called_once_with(linger=-1)

    def test_timeout_is_reported_by_last_prefill_pp_stage(self):
        p_vllm_config = _make_vllm_config(
            is_prefill=True, pp=2, tp=2)
        metadata = Metadata()
        metadata.req_params = {
            "pending": set(),
            "timeout": {"prefill_req_0"},
        }
        workers = []

        try:
            for rank in (0, 2):
                with (
                    _mock_cache_manager(),
                    _mock_parallel(pp=2, tp=2, rank=rank),
                ):
                    worker = LLMDataDistConnector(
                        vllm_config=p_vllm_config,
                        role=KVConnectorRole.WORKER,
                        kv_cache_config=None,
                    )
                    worker.bind_connector_metadata(metadata)
                    send_done, recv_done = worker.get_finished(None)
                    workers.append(worker)

                if rank == 0:
                    assert send_done == set()
                else:
                    assert send_done == {"prefill_req_0"}
                assert recv_done is None
        finally:
            _stop_daemon(*workers)


class TestConnectorMetrics:
    """C 线指标：连接器的三个 KVConnectorStats 委托方法（转发到 kv_transfer）。"""

    def test_get_kv_connector_stats_worker_passes_rank(self):
        # worker 角色：取 world rank 传给 collect（用于采本 rank 显存）。
        with (
            _mock_parallel(tp=16, rank=3),
            _mock_cache_manager(),
            patch("omni_npu.diagnostics.metrics.kv_transfer.collect") as m,
        ):
            conn = LLMDataDistConnector(
                vllm_config=_make_vllm_config(is_prefill=False, tp=16),
                role=KVConnectorRole.WORKER,
                kv_cache_config=None,
            )
            conn.get_kv_connector_stats()
            m.assert_called_once_with(rank=3)

    def test_get_kv_connector_stats_scheduler_no_rank(self):
        # scheduler 角色：self.worker 为 None → 不采显存，rank=None。
        conn = LLMDataDistConnector(
            vllm_config=_make_vllm_config(is_prefill=False, tp=16),
            role=KVConnectorRole.SCHEDULER,
            kv_cache_config=None,
        )
        with patch("omni_npu.diagnostics.metrics.kv_transfer.collect") as m:
            conn.get_kv_connector_stats()
            m.assert_called_once_with(rank=None)

    def test_build_kv_connector_stats_delegates(self):
        with patch("omni_npu.diagnostics.metrics.kv_transfer.build_stats") as m:
            LLMDataDistConnector.build_kv_connector_stats({"fail": {"pull": 1}})
            m.assert_called_once_with({"fail": {"pull": 1}})

    def test_build_prom_metrics_delegates(self):
        with patch("omni_npu.diagnostics.metrics.kv_transfer.build_prom_metrics") as m:
            LLMDataDistConnector.build_prom_metrics("cfg", "mt", "ln", "pe")
            m.assert_called_once_with("cfg", "mt", "ln", "pe")
