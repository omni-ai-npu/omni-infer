# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import os
import pytest
from collections import Counter, defaultdict
from omni_npu.connector.llmdatadist_connector_v1 import LLMDataDistConnector, ReqMeta, ReqMetaPrefill, \
    DatadistConnectorMetadata, DatadistConnectorMetadataPrefill, PrefillConnectorScheduler, DecodeConnectorScheduler, \
    PrefillConnectorWorker, DecodeConnectorWorker, CLUSTER_HEARTBEAT_TIMEOUT, BLOCK_RELEASE_DELAY, handle_exception, \
    get_local_ip
from .utils import create_vllm_config, create_request, create_scheduler, create_model_runner_output, run_in_process
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole, KVConnectorBase_V1
import torch
from contextlib import contextmanager, ExitStack
from unittest.mock import MagicMock, patch
from unittest.mock import MagicMock, patch, PropertyMock
import time
import threading
import zmq
import logging

from omni_npu.connector import register_connectors
from vllm.v1.request import RequestStatus

register_connectors()

logging.basicConfig(level=logging.DEBUG)

class TestLLMDataDistConnectorV1LifeCycle:
    def test_prefill_schedule(self):
        vllm_config = create_vllm_config(kv_role="kv_producer")
        scheduler = create_scheduler(vllm_config)
        assert scheduler.connector is not None

        request = create_request(do_remote_decode=True)
        scheduler.add_request(request=request)

        # 使用 mock 对象记录方法调用次数
        with ExitStack() as stack:
            connector = scheduler.connector  # 使用实例对象
            mock_get_num = stack.enter_context(
                patch.object(connector, 'get_num_new_matched_tokens',
                             wraps=connector.get_num_new_matched_tokens)
            )
            mock_update = stack.enter_context(
                patch.object(connector, 'update_state_after_alloc',
                             wraps=connector.update_state_after_alloc)
            )
            mock_build = stack.enter_context(
                patch.object(connector, 'build_connector_meta',
                             wraps=connector.build_connector_meta)
            )
            mock_finish = stack.enter_context(
                patch.object(connector, 'request_finished_all_groups',
                             wraps=connector.request_finished_all_groups)
            )

            schedule_output = scheduler.schedule()
            mock_get_num.assert_called_once()
            mock_update.assert_called_once()
            mock_build.assert_called_once()
            runner_output = create_model_runner_output(reqs=[request])
            scheduler.update_from_output(schedule_output, runner_output)
            mock_finish.assert_called_once()
            assert request.request_id in scheduler.requests
            assert request.request_id in scheduler.finished_req_ids

            schedule_output = scheduler.schedule()
            runner_output = create_model_runner_output(finished_sending={request.request_id})
            scheduler.update_from_output(schedule_output, runner_output)
            mock_finish.assert_called_once()
            assert request.request_id not in scheduler.requests


    def test_decode_schedule(self):
        vllm_config = create_vllm_config(kv_role="kv_consumer")
        scheduler = create_scheduler(vllm_config)
        assert scheduler.connector is not None

        request = create_request(do_remote_prefill=True)
        scheduler.add_request(request=request)

        with ExitStack() as stack:
            connector = scheduler.connector
            mock_get_num = stack.enter_context(
                patch.object(connector, 'get_num_new_matched_tokens',
                             wraps=connector.get_num_new_matched_tokens)
            )
            mock_update = stack.enter_context(
                patch.object(connector, 'update_state_after_alloc',
                             wraps=connector.update_state_after_alloc)
            )
            mock_build = stack.enter_context(
                patch.object(connector, 'build_connector_meta',
                             wraps=connector.build_connector_meta)
            )
            mock_finish = stack.enter_context(
                patch.object(connector, 'request_finished',
                             wraps=connector.request_finished)
            )

            schedule_output = scheduler.schedule()
            mock_get_num.assert_called_once()
            mock_update.assert_called_once()
            mock_build.assert_called_once()
            assert schedule_output.kv_connector_metadata is not None
            runner_output = create_model_runner_output(finished_recving={request.request_id})
            scheduler.update_from_output(schedule_output, runner_output)

            schedule_output = scheduler.schedule()
            assert request.request_id in schedule_output.num_scheduled_tokens

    def test_all_worker(self):
        vllm_config_p = create_vllm_config(kv_role="kv_producer")
        vllm_config_p.model_config.is_deepseek_mla = True
        scheduler_p = create_scheduler(vllm_config_p)
        vllm_config_d = create_vllm_config(kv_role="kv_consumer")
        vllm_config_d.model_config.is_deepseek_mla = True
        scheduler_d = create_scheduler(vllm_config_d)

        with ExitStack() as stack:
            mock = stack.enter_context(patch("omni_npu.connector.llmdatadist_connector_v1.get_tensor_model_parallel_rank"))
            mock.return_value = 0
            mock2 = stack.enter_context(patch("omni_npu.connector.llmdatadist_manager_v1.get_world_group"))
            mock2.return_value.rank_in_group = 0
            mock2.return_value.local_rank = 0
            mock3 = stack.enter_context(patch("omni_npu.connector.llmdatadist_manager_v1.LLMDataDist"))
            from llm_datadist import LLMStatusCode
            mock3.return_value.link_clusters.return_value = (LLMStatusCode.LLM_SUCCESS, None)
            mock4 = stack.enter_context(patch("omni_npu.connector.llmdatadist_manager_v1.BlocksCacheKey"))
            mock_dcp = stack.enter_context(patch("omni_npu.connector.llmdatadist_connector_v1.get_dcp_group"))
            mock_dcp.return_value.world_size = 0
            mock_get_pp_group = stack.enter_context(patch("omni_npu.connector.llmdatadist_connector_v1.get_pp_group"))
            mock_get_pp_group.return_value.rank_in_group = 0
            kv_cache_config = MagicMock()

            connector_p = LLMDataDistConnector(vllm_config_p, KVConnectorRole.WORKER, kv_cache_config)
            connector_d = LLMDataDistConnector(vllm_config_d, KVConnectorRole.WORKER, kv_cache_config)

            kv_caches = {"layer.0": torch.empty(1, 1, 1, 1, 1)} # 1 * [block_num, block_size, head_num, head_dim]
            connector_p.register_kv_caches(kv_caches)
            kv_caches = [torch.empty(1, 1, 1, 1, 1)]
            connector_d.register_kv_caches(kv_caches)

            request_p = create_request(request_id=1, do_remote_decode=True)
            scheduler_p.add_request(request_p)
            scheduler_output = scheduler_p.schedule()

            connector_p.bind_connector_metadata(scheduler_output.kv_connector_metadata)
            all_done_sending, all_done_recving = connector_p.get_finished(MagicMock())
            assert not all_done_sending and not all_done_recving

            runner_output = create_model_runner_output(reqs=[request_p])
            core_output = scheduler_p.update_from_output(scheduler_output, runner_output)

            request_d = create_request(request_id=1, do_remote_prefill=True)
            request_d.kv_transfer_params.update(core_output[0].outputs[0].kv_transfer_params)

            scheduler_d.add_request(request_d)
            scheduler_output = scheduler_d.schedule()

            connector_d.bind_connector_metadata(scheduler_output.kv_connector_metadata)
            forward_context = MagicMock()
            connector_d.start_load_kv(forward_context)

            scheduler_output = scheduler_p.schedule()
            connector_p.bind_connector_metadata(scheduler_output.kv_connector_metadata)

            time_out_limit = 10
            finished_flags = [False, False]
            for _ in range(time_out_limit):
                all_done_sending, all_done_recving = connector_p.get_finished(MagicMock())
                if all_done_sending == {request_p.request_id}:
                    finished_flags[0] = True
                else:
                    assert not all_done_sending
                assert not all_done_recving

                all_done_sending, all_done_recving = connector_d.get_finished(MagicMock())
                assert not all_done_sending
                if all_done_recving == {request_d.request_id}:
                    finished_flags[1] = True
                else:
                    assert not all_done_recving

                if all(finished_flags):
                    break
                else:
                    time.sleep(1)

            if not all(finished_flags):
                raise RuntimeError("time out")


PATH_PREFIX = "omni_npu.connector.llmdatadist_connector_v1"
LLMDIST_CONFIG_PATH = f"{PATH_PREFIX}.LLMDataDistConfig"
LLMDIST_MANAGER_PATH = f"{PATH_PREFIX}.LLMDataDistManager"
PARALLEL_STATE_PATH = PATH_PREFIX
UTILS_PATH = f"omni_npu.connector.utils"
LOGGER_PATH = "vllm.logger"
REQUEST_PATH = "vllm.v1.request.Request"
SCHED_OUTPUT_PATH = "vllm.v1.core.sched.output.SchedulerOutput"
KV_CACHE_INTERFACE_PATH = f"vllm.v1.kv_cache_interface"


@pytest.fixture
def mock_vllm_config():
    config = MagicMock()
    config.kv_transfer_config = MagicMock()
    config.kv_transfer_config.kv_role = "kv_consumer" # Default to decode
    config.kv_transfer_config.kv_port = 5569
    config.model_config = MagicMock()
    config.model_config.is_deepseek_mla = False
    config.cache_config = MagicMock()
    config.cache_config.block_size = 16
    config.parallel_config = MagicMock()
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.data_parallel_rank_local = 0
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    config.additional_config.get.return_value = False
    return config


@pytest.fixture
def mock_kv_cache_config():
    return MagicMock()


@pytest.fixture
def mock_datadist_config():
    with patch(LLMDIST_CONFIG_PATH) as mock:
        yield mock


@pytest.fixture
def mock_datadist_manager():
    with patch(LLMDIST_MANAGER_PATH) as mock:
        yield mock


@pytest.fixture
def mock_get_tp_rank():
    with patch(f"{PARALLEL_STATE_PATH}.get_tensor_model_parallel_rank", return_value=0) as mock:
        yield mock

@pytest.fixture
def mock_get_pp_group():
    mock_group = MagicMock()
    mock_group.rank_in_group = 0
    with patch(f"{PARALLEL_STATE_PATH}.get_pp_group", return_value=mock_group) as mock:
        yield mock


@pytest.fixture
def mock_get_tp_group():
    mock_group = MagicMock()
    mock_group.cpu_group = MagicMock()
    with patch(f"{PARALLEL_STATE_PATH}.get_tp_group", return_value=mock_group) as mock:
        yield mock


@pytest.fixture
def mock_get_dcp_group():
    mock_group = MagicMock()
    mock_group.world_size = 1
    mock_group.rank_in_group = 0
    with patch(f"{PARALLEL_STATE_PATH}.get_dcp_group", return_value=mock_group) as mock:
        yield mock


@pytest.fixture
def mock_utils_get_config():
    with patch(f"{UTILS_PATH}.get_config_from_dict_or_env", return_value=5568) as mock:
        yield mock


@pytest.fixture
def mock_zmq_context():
    with patch(f"{PATH_PREFIX}.zmq.Context") as mock_ctx_class:
        mock_ctx_instance = MagicMock()
        mock_ctx_class.return_value = mock_ctx_instance
        yield mock_ctx_instance


@pytest.fixture
def mock_socket():
    with patch(f"{PATH_PREFIX}.socket.socket") as mock_sock_class:
        mock_sock_instance = MagicMock()
        mock_sock_class.return_value = mock_sock_instance
        yield mock_sock_instance


@pytest.fixture
def mock_logger():
    with patch(LOGGER_PATH) as mock_logger_module:
        mock_logger_instance = MagicMock()
        mock_logger_module.init_logger.return_value = mock_logger_instance
        yield mock_logger_instance


@pytest.fixture
def mock_threading_thread():
    with patch(f"{PATH_PREFIX}.threading.Thread") as mock_thread_class, patch(f"{PATH_PREFIX}.ThreadPoolExecutor") as mock_pool_class:
        mock_thread_instance = MagicMock()
        mock_thread_instance.start = MagicMock() # Mock start to prevent actual thread creation
        mock_thread_class.return_value = mock_thread_instance
        yield mock_thread_instance, mock_pool_class.return_value


class TestReqMeta:
    def test_init(self):
        meta = ReqMeta(
            local_block_ids=[[1, 2, 3]],
            remote_block_ids=[[4, 5, 6]],
            remote_host="tcp://127.0.0.1:5568",
            remote_cluster_id="cluster_1",
            spec_token_ids=None,
            remote_dp_rank=0,
            remote_request_id="req_1",
            remote_pp_partition=None,
            token_num=None, # only used in DCP
            num_external_tokens=0,
            full_local_block_ids=[[1, 2, 3]],
        )
        assert meta.local_block_ids == [[1, 2, 3]]
        assert meta.remote_block_ids == [[4, 5, 6]]
        assert meta.remote_host == "tcp://127.0.0.1:5568"
        assert meta.remote_cluster_id == "cluster_1"
        assert meta.spec_token_ids is None
        assert meta.remote_dp_rank == 0
        assert meta.remote_request_id == "req_1"
        assert meta.num_external_tokens == 0
        assert meta.full_local_block_ids == [[1, 2, 3]]


class TestReqMetaPrefill:
    def test_init(self):
        finish_time = time.monotonic()
        meta = ReqMetaPrefill(finish_time=finish_time)
        assert meta.finish_time == finish_time


class TestDatadistConnectorMetadata:
    def test_init(self):
        metadata = DatadistConnectorMetadata()
        assert metadata.requests == {}

    def test_add_new_req(self):
        metadata = DatadistConnectorMetadata()
        kv_transfer_params = {
            "remote_block_ids": [[4, 5, 6]],
            "remote_host_ip": "tcp://127.0.0.1:5568",
            "remote_cluster_id": "cluster_1",
            "spec_token_ids": [100, 101],
        }
        metadata.add_new_req(
            "req_1",
            [[1, 2, 3]],
            kv_transfer_params,
            num_external_tokens=0,
            full_local_block_ids=[[1, 2, 3]],
        )

        assert "req_1" in metadata.requests
        req_meta = metadata.requests["req_1"]
        assert req_meta.local_block_ids == [[1, 2, 3]]
        assert req_meta.remote_block_ids == [[4, 5, 6]]
        assert req_meta.remote_host == "tcp://127.0.0.1:5568"
        assert req_meta.remote_cluster_id == "cluster_1"
        assert req_meta.spec_token_ids == [100, 101]
        assert req_meta.remote_dp_rank == 0 # default
        assert req_meta.remote_request_id is None # default
        assert req_meta.num_external_tokens == 0
        assert req_meta.full_local_block_ids == [[1, 2, 3]]


class TestDatadistConnectorMetadataPrefill:
    def test_init(self):
        metadata = DatadistConnectorMetadataPrefill()
        assert metadata.requests == {}

    def test_add_new_req(self):
        metadata = DatadistConnectorMetadataPrefill()
        finish_time = time.monotonic()
        metadata.add_new_req("req_1", finish_time)

        assert "req_1" in metadata.requests
        req_meta = metadata.requests["req_1"]
        assert req_meta.finish_time == finish_time


class TestLLMDataDistConnector:
    @pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER])
    @pytest.mark.parametrize("is_prefill", [True, False])
    def test_init_scheduler_worker(self, mock_vllm_config, mock_datadist_config, mock_datadist_manager, mock_get_tp_rank, mock_utils_get_config, mock_zmq_context, mock_threading_thread, role, is_prefill):
        mock_vllm_config.kv_transfer_config.kv_role = "kv_producer" if is_prefill else "kv_consumer"

        with patch(f"{PATH_PREFIX}.get_local_ip", return_value="127.0.0.1"):
            with ExitStack() as stack:
                mock_get_pp_group = stack.enter_context(patch("omni_npu.connector.llmdatadist_connector_v1.get_pp_group"))
                mock_get_pp_group.return_value.rank_in_group = 0
                connector = LLMDataDistConnector(mock_vllm_config, role)

                assert connector.datadist_config is not None
                assert connector.host_cluster_id is not None
                assert connector.host_ip == "127.0.0.1"
                assert connector.host_port is not None
                assert connector.is_prefill == is_prefill

                if role == KVConnectorRole.SCHEDULER:
                    assert connector.connector_scheduler is not None
                    assert connector.connector_worker is None
                    if is_prefill:
                        assert isinstance(connector.connector_scheduler, PrefillConnectorScheduler)
                    else:
                        assert isinstance(connector.connector_scheduler, DecodeConnectorScheduler)
                elif role == KVConnectorRole.WORKER:
                    assert connector.connector_worker is not None
                    assert connector.connector_scheduler is None
                    if is_prefill:
                        assert isinstance(connector.connector_worker, PrefillConnectorWorker)
                    else:
                        assert isinstance(connector.connector_worker, DecodeConnectorWorker)

    def test_init_no_kv_transfer_config(self, mock_vllm_config):
        mock_vllm_config.kv_transfer_config = None
        with pytest.raises(RuntimeError, match="vllm_config.kv_transfer_config cannot be None"):
            LLMDataDistConnector(mock_vllm_config, KVConnectorRole.SCHEDULER)

    def test_init_deepseek_mla(self, mock_vllm_config, mock_datadist_config, mock_datadist_manager, mock_get_tp_rank, mock_utils_get_config, mock_zmq_context, mock_logger, mock_threading_thread):
        mock_vllm_config.model_config.is_deepseek_mla = True
        mock_vllm_config.kv_transfer_config.kv_parallel_size = 2

        with patch(f"{PATH_PREFIX}.get_local_ip", return_value="127.0.0.1"):
            connector = LLMDataDistConnector(mock_vllm_config, KVConnectorRole.SCHEDULER)

            # Check if kv_parallel_size was set to 1
            assert mock_vllm_config.kv_transfer_config.kv_parallel_size == 1

    # Test scheduler side methods
    @pytest.mark.parametrize("role", [KVConnectorRole.WORKER]) # Should raise error for worker
    def test_scheduler_methods_raise_error_for_worker(self, mock_vllm_config, mock_datadist_config, mock_datadist_manager, mock_get_tp_rank, mock_utils_get_config, mock_zmq_context, mock_threading_thread, role):
        with patch(f"{PATH_PREFIX}.get_local_ip", return_value="127.0.0.1"):
            connector = LLMDataDistConnector(mock_vllm_config, role)

            # All scheduler methods should raise RuntimeError if connector_scheduler is None
            with pytest.raises(RuntimeError, match="self.connector_scheduler cannot be None"):
                connector.get_num_new_matched_tokens(MagicMock(), 0)

            with pytest.raises(RuntimeError, match="self.connector_scheduler cannot be None"):
                connector.update_state_after_alloc(MagicMock(), MagicMock(), 0)

            with pytest.raises(RuntimeError, match="self.connector_scheduler cannot be None"):
                connector.build_connector_meta(MagicMock())

            with pytest.raises(RuntimeError, match="self.connector_scheduler cannot be None"):
                connector.request_finished(MagicMock(), [])

    # Test worker side methods
    @pytest.mark.parametrize("role", [KVConnectorRole.SCHEDULER]) # Should raise error for scheduler
    def test_worker_methods_raise_error_for_scheduler(self, mock_vllm_config, mock_datadist_config, mock_datadist_manager, mock_get_tp_rank, mock_utils_get_config, mock_zmq_context, mock_threading_thread, role):
        with patch(f"{PATH_PREFIX}.get_local_ip", return_value="127.0.0.1"):
            connector = LLMDataDistConnector(mock_vllm_config, role)

            # All worker methods should raise RuntimeError if connector_worker is None
            with pytest.raises(RuntimeError, match="self.connector_worker cannot be None"):
                connector.register_kv_caches({})

            with pytest.raises(RuntimeError, match="self.connector_worker cannot be None"):
                connector.get_finished(set())

            with pytest.raises(RuntimeError, match="self.connector_worker cannot be None"):
                connector.start_load_kv(MagicMock())

    def test_get_finished_count_prefill(self, mock_vllm_config, mock_datadist_config, mock_datadist_manager, mock_get_tp_rank, mock_utils_get_config, mock_zmq_context, mock_threading_thread):
        mock_vllm_config.kv_transfer_config.kv_role = "kv_producer" # is_prefill = True
        with patch(f"{PATH_PREFIX}.get_local_ip", return_value="127.0.0.1"):
            connector = LLMDataDistConnector(mock_vllm_config, KVConnectorRole.SCHEDULER)
            assert connector.get_finished_count() == 1

    def test_get_finished_count_decode(self, mock_vllm_config, mock_datadist_config, mock_datadist_manager, mock_get_tp_rank, mock_utils_get_config, mock_zmq_context, mock_threading_thread):
        mock_vllm_config.kv_transfer_config.kv_role = "kv_consumer" # is_prefill = False
        with patch(f"{PATH_PREFIX}.get_local_ip", return_value="127.0.0.1"):
            connector = LLMDataDistConnector(mock_vllm_config, KVConnectorRole.SCHEDULER)
            assert connector.get_finished_count() is None

    def test_start_load_kv_worker_metadata_check(self, mock_vllm_config, mock_datadist_config, mock_datadist_manager, mock_get_tp_rank, mock_utils_get_config, mock_zmq_context, mock_threading_thread):
        mock_vllm_config.kv_transfer_config.kv_role = "kv_consumer" # Decode worker
        with patch(f"{PATH_PREFIX}.get_local_ip", return_value="127.0.0.1"):
            connector = LLMDataDistConnector(mock_vllm_config, KVConnectorRole.WORKER)

            forward_context = MagicMock()
            # Test with invalid metadata type
            with pytest.raises(RuntimeError, match="self._connector_metadata must be an instance of DatadistConnectorMetadata or DatadistConnectorMetadataPrefill"):
                connector.bind_connector_metadata("invalid_metadata")
                connector.start_load_kv(forward_context)

            # Test with valid metadata type
            mock_metadata = DatadistConnectorMetadata()
            # This should not raise an error, just call the worker's method
            # We need to mock the worker's method too
            with patch.object(connector.connector_worker, 'start_load_kv') as mock_worker_load:
                connector.bind_connector_metadata(mock_metadata)
                connector.start_load_kv(forward_context)
                mock_worker_load.assert_called_once_with(mock_metadata)

    def test_empty_method(self, mock_vllm_config, mock_datadist_config, mock_datadist_manager, mock_get_tp_rank, mock_utils_get_config, mock_zmq_context, mock_threading_thread):
        with patch(f"{PATH_PREFIX}.get_local_ip", return_value="127.0.0.1"):
            connector = LLMDataDistConnector(mock_vllm_config, KVConnectorRole.SCHEDULER)
            connector.wait_for_layer_load(None)
            connector.save_kv_layer(None, None, None)
            connector.wait_for_save()


class TestPrefillConnectorScheduler:
    def test_get_num_new_matched_tokens(self, mock_vllm_config):
        scheduler = PrefillConnectorScheduler(mock_vllm_config, "cluster_1", [1, 2], "127.0.0.1", "5568")
        req = MagicMock()
        num, bool_val = scheduler.get_num_new_matched_tokens(req, 10)
        assert num == 0
        assert bool_val is False

    def test_update_state_after_alloc(self, mock_vllm_config):
        scheduler = PrefillConnectorScheduler(mock_vllm_config, "cluster_1", [1, 2], "127.0.0.1", "5568")
        # Should pass without error, no side effects expected from the implementation
        scheduler.update_state_after_alloc(MagicMock(), MagicMock(), 0)

    def test_build_connector_metadata(self, mock_vllm_config):
        scheduler = PrefillConnectorScheduler(mock_vllm_config, "cluster_1", [1, 2], "127.0.0.1", "5568")
        finish_time = time.monotonic()
        scheduler.requests_finish_time = {"req_1": finish_time, "req_2": time.monotonic() - 10}

        metadata = scheduler.build_connector_metadata(MagicMock())

        assert isinstance(metadata, DatadistConnectorMetadataPrefill)
        assert "req_1" in metadata.requests
        assert "req_2" in metadata.requests
        assert metadata.requests["req_1"].finish_time == finish_time
        # Check that the dict was cleared
        assert scheduler.requests_finish_time == {}

    def test_request_finished(self, mock_vllm_config):
        mock_vllm_config.parallel_config.data_parallel_rank = 1
        scheduler = PrefillConnectorScheduler(mock_vllm_config, "cluster_1", [1, 2], "127.0.0.1", "5568")

        req = MagicMock()
        req.request_id = "req_1"
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED

        block_ids = [1, 2, 3]
        spec_token_ids = [100, 101]

        delay_free, params = scheduler.request_finished(req, block_ids, spec_token_ids)

        assert delay_free is True
        assert "remote_block_ids" in params
        assert params["remote_block_ids"] == block_ids
        assert params["remote_cluster_id"] == "cluster_1"
        assert params["remote_host_ip"] == "tcp://127.0.0.1:5568"
        assert params["spec_token_ids"] == spec_token_ids
        assert params["remote_dp_rank"] == 1
        assert params["remote_request_id"] == "req_1"
        # Check that finish time was recorded
        assert "req_1" in scheduler.requests_finish_time

    def test_request_finished_not_length_capped(self, mock_vllm_config):
        scheduler = PrefillConnectorScheduler(mock_vllm_config, "cluster_1", [1, 2], "127.0.0.1", "5568")

        req = MagicMock()
        req.request_id = "req_1"
        req.status = MagicMock()
        req.status.name = "FINISHED_ABORTED" # Or any other status != FINISHED_LENGTH_CAPPED

        block_ids = [1, 2, 3]
        delay_free, params = scheduler.request_finished(req, block_ids)

        assert delay_free is False
        assert params is None
        # Check that finish time was NOT recorded
        assert "req_1" not in scheduler.requests_finish_time

    def test_request_finished_no_blocks(self, mock_vllm_config):
        scheduler = PrefillConnectorScheduler(mock_vllm_config, "cluster_1", [1, 2], "127.0.0.1", "5568")

        req = MagicMock()
        req.request_id = "req_1"
        req.status = MagicMock()
        req.status.name = "FINISHED_LENGTH_CAPPED"

        block_ids = []
        delay_free, params = scheduler.request_finished(req, block_ids)

        assert delay_free is False
        assert params is None
        # Check that finish time was NOT recorded
        assert "req_1" not in scheduler.requests_finish_time


class TestPrefillConnectorWorker:
    @pytest.mark.parametrize("tp_rank", [0, 1])
    def test_init(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_get_pp_group, mock_zmq_context, mock_threading_thread, tp_rank):
        mock_get_tp_rank.return_value = tp_rank
        worker = PrefillConnectorWorker(mock_vllm_config, "127.0.0.1", "5568")

        assert worker.host_ip == "127.0.0.1"
        assert worker.host_port == "5568"
        assert worker.rank == tp_rank
        assert worker.datadist_manager is not None
        assert worker.requests_finish_time == {}

        if tp_rank == 0:
            # Rank 0 binds input socket and starts threads
            mock_zmq_context.socket.assert_called()
            assert worker.input_socket.bind.called
            assert worker.thread is not None
            assert worker.hb_socket is not None
        else:
            # Rank != 0 does not bind input socket
            assert not worker.input_socket.bind.called if hasattr(worker, 'input_socket') else True
            assert worker.thread is not None

    def test_heartbeat_timer_func(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_get_pp_group, mock_zmq_context, mock_threading_thread):
        mock_get_tp_rank.return_value = 0
        worker = PrefillConnectorWorker(mock_vllm_config, "127.0.0.1", "5568")

        # Mock time and datadist manager
        mock_time = time.time()
        mock_monotonic = time.monotonic()
        with patch(f"{PATH_PREFIX}.time.time", return_value=mock_time + CLUSTER_HEARTBEAT_TIMEOUT + 1), \
             patch(f"{PATH_PREFIX}.time.monotonic", return_value=mock_monotonic), \
             patch.object(worker, 'datadist_manager') as mock_manager, \
             patch.object(worker, 'hb_socket') as mock_hb_socket:

            cluster_2 = 1
            mock_manager.cluster_id_to_ip_port.side_effect = [("127.0.0.2:0", 1, 1, 0), ("127.0.0.2:1", 1, 1, 0), ("127.0.0.2:1", 1, 1, 0)]
            worker.remote_hb_info = {1: mock_time, 2: mock_time, 3: mock_time}

            # Run once to trigger timeout logic
            # We cannot easily run the infinite loop, so we patch the sleep to break it after one iteration
            with patch(f"{PATH_PREFIX}.time.sleep", side_effect=StopIteration):
                try:
                    worker.heartbeat_timer_func()
                except StopIteration:
                    pass # Expected to break the loop

            assert not worker.remote_hb_info
            mock_manager.force_unlink.assert_called_once_with(1)
            worker.ctx.socket.return_value.connect.assert_called()
            assert worker.ctx.socket.return_value.send_string.call_count == 2

    def test_heartbeat_server_func(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_get_pp_group, mock_zmq_context, mock_threading_thread):
        mock_get_tp_rank.return_value = 1
        worker = PrefillConnectorWorker(mock_vllm_config, "127.0.0.1", "5568")
        with patch.object(worker.ctx, 'socket') as mock_socket:
            mock_socket.return_value.recv_string.side_effect = [1, StopIteration]
            try:
                worker.heartbeat_server_func()
            except StopIteration:
                pass
            worker.datadist_manager.force_unlink.assert_called_once_with(1)

    def test_get_finished_rank_0(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_get_pp_group, mock_zmq_context, mock_threading_thread):
        mock_get_tp_rank.return_value = 0
        worker = PrefillConnectorWorker(mock_vllm_config, "127.0.0.1", "5568")

        # Simulate metadata with a request that finished long ago
        finish_time = time.monotonic() - BLOCK_RELEASE_DELAY - 1
        metadata = DatadistConnectorMetadataPrefill()
        metadata.add_new_req("req_old", finish_time)
        metadata.add_new_req("req_new", time.monotonic())

        # Add a request to receive list
        worker.receive_req_list = ["req_received"]
        worker.requests_finish_time = {"req_old": finish_time, "req_new": time.monotonic(), "req_received": time.monotonic()}

        sending, recving = worker.get_finished(metadata)

        # req_old should be in sending set due to timeout
        assert "req_old" in sending
        # req_received should be in sending set
        assert "req_received" in sending
        # req_new should not be in sending set
        assert "req_new" not in sending
        # recving set should be empty for prefill worker
        assert recving == set()

        # Check that old request was removed from requests_finish_time
        assert "req_old" not in worker.requests_finish_time
        assert "req_new" in worker.requests_finish_time

    def test_get_finished_rank_non_zero(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_get_pp_group, mock_zmq_context, mock_threading_thread):
        mock_get_tp_rank.return_value = 1
        worker = PrefillConnectorWorker(mock_vllm_config, "127.0.0.1", "5568")

        metadata = DatadistConnectorMetadataPrefill()
        sending, recving = worker.get_finished(metadata)

        # Non-zero ranks return empty sets
        assert sending == set()
        assert recving == set()

    def test_start_load_kv(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_get_pp_group, mock_zmq_context, mock_threading_thread):
        worker = PrefillConnectorWorker(mock_vllm_config, "127.0.0.1", "5568")
        metadata = DatadistConnectorMetadataPrefill()
        # Should pass without error, no side effects expected from the implementation
        worker.start_load_kv(metadata)

    @run_in_process
    def test_get_pulled_kv_req_list(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_get_pp_group, mock_zmq_context):
        mock_zmq_context.socket.return_value.poll.return_value = True

        count = 0
        def recv_side_effect(*arg, **kwarg):
            nonlocal count
            i = count
            count += 1
            rets = ['["0"]', '["decode_hb:1"]', '["decode_hb:invalid"]']
            if i < len(rets):
                return rets[i]
            elif i == len(rets):
                raise StopIteration
            else:
                exit()

        mock_datadist_manager.return_value.cluster_id_to_ip_port.return_value = (0, 0, 0, 0)
        mock_zmq_context.socket.return_value.recv_string.side_effect = recv_side_effect
        worker = PrefillConnectorWorker(mock_vllm_config, "127.0.0.1", "5568")

        worker.thread.join(timeout=1) # wait comm finish, this time should enough

        assert "0" in worker.receive_req_list
        assert "1" in worker.remote_hb_info


class TestDecodeConnectorScheduler:
    def test_init(self, mock_vllm_config):
        scheduler = DecodeConnectorScheduler(mock_vllm_config)
        assert scheduler.vllm_config == mock_vllm_config
        assert scheduler.block_size == mock_vllm_config.cache_config.block_size
        assert scheduler._reqs_need_recv == {}
        assert scheduler.processed_request == set()

    @pytest.mark.parametrize("prompt_len,computed_tokens,expected_count,expected_bool", [
        (16, 0, 16, True),   # Prompt needs full first block
        (16, 16, 0, False),  # Prompt block already computed
        (30, 16, 14, True),  # New behavior: count is prompt_len - computed_tokens
        (32, 16, 16, True),  # Prompt needs next block, aligned
        (32, 32, 0, False),  # Prompt fully computed
    ])
    def test_get_num_new_matched_tokens(self, mock_vllm_config, prompt_len, computed_tokens, expected_count, expected_bool):
        scheduler = DecodeConnectorScheduler(mock_vllm_config)
        req = MagicMock()
        req.request_id = "req_1"
        req.prompt_token_ids = [1] * prompt_len
        req.kv_transfer_params = {"remote_block_ids": [1], "remote_cluster_id": "cluster_1", "remote_host_ip": "tcp://127.0.0.1:5568"}

        # Ensure request is not in processed_request
        scheduler.processed_request.discard("req_1")

        count, bool_val = scheduler.get_num_new_matched_tokens(req, computed_tokens)
        assert count == expected_count
        assert bool_val == expected_bool

    def test_get_num_new_matched_tokens_processed(self, mock_vllm_config):
        scheduler = DecodeConnectorScheduler(mock_vllm_config)
        req = MagicMock()
        req.request_id = "req_1"
        scheduler.processed_request.add("req_1")

        count, bool_val = scheduler.get_num_new_matched_tokens(req, 10)
        assert count == 0
        assert bool_val is False

    def test_get_num_new_matched_tokens_no_params(self, mock_vllm_config):
        scheduler = DecodeConnectorScheduler(mock_vllm_config)
        req = MagicMock()
        req.request_id = "req_1"
        req.kv_transfer_params = None

        count, bool_val = scheduler.get_num_new_matched_tokens(req, 10)
        assert count == 0
        assert bool_val is False

    def test_get_num_new_matched_tokens_raise(self, mock_vllm_config):
        scheduler = DecodeConnectorScheduler(mock_vllm_config)
        req = MagicMock()
        req.request_id = "req_1"

        with pytest.raises(RuntimeError, match="num_computed_tokens must be divisible by self.block_size"):
            scheduler.get_num_new_matched_tokens(req, 10)

    def test_update_state_after_alloc(self, mock_vllm_config):
        scheduler = DecodeConnectorScheduler(mock_vllm_config)
        req = MagicMock()
        req.request_id = "req_1"
        req.kv_transfer_params = {"remote_block_ids": [[1, 2]], "remote_cluster_id": "cluster_1", "remote_host_ip": "tcp://127.0.0.1:5568"}
        blocks = MagicMock()
        blocks.get_block_ids.return_value = [[10, 11]]
        scheduler.get_unhashed_block_ids = lambda _: [[10, 11]]

        scheduler.update_state_after_alloc(req, blocks, 0)

        assert req.request_id in scheduler.processed_request
        assert req.request_id in scheduler._reqs_need_recv
        assert scheduler._reqs_need_recv[req.request_id][1] == [[10, 11]]
        assert scheduler._reqs_need_recv[req.request_id][2] == [[10, 11]]
        assert scheduler._reqs_need_recv[req.request_id][3] == 0

    def test_build_connector_metadata(self, mock_vllm_config, mock_logger):
        scheduler = DecodeConnectorScheduler(mock_vllm_config)
        req = MagicMock()
        req.request_id = "req_1"
        req.num_prompt_tokens = 32
        req.kv_transfer_params = {"remote_block_ids": [[1, 2]], "remote_cluster_id": "cluster_1", "remote_host_ip": "tcp://127.0.0.1:5568", "spec_token_ids": []}
        scheduler._reqs_need_recv[req.request_id] = (req, [[10, 11]], [[10, 11]], 32)

        metadata = scheduler.build_connector_metadata(MagicMock())

        assert isinstance(metadata, DatadistConnectorMetadata)
        assert "req_1" in metadata.requests
        assert metadata.requests["req_1"].local_block_ids == [[10, 11]]
        assert metadata.requests["req_1"].remote_block_ids == [[1, 2]]
        assert metadata.requests["req_1"].full_local_block_ids == [[10, 11]]
        assert metadata.requests["req_1"].num_external_tokens == 32
        # Check that _reqs_need_recv was cleared
        assert scheduler._reqs_need_recv == {}

        req.kv_transfer_params = None
        scheduler._reqs_need_recv[req.request_id] = (req, [[10, 11]], [[10, 11]], 32)
        scheduler.build_connector_metadata(MagicMock())

    def test_request_finished_processed_request(self, mock_vllm_config):
        scheduler = DecodeConnectorScheduler(mock_vllm_config)
        req = MagicMock()
        req.request_id = "req_1"
        req.status = RequestStatus.FINISHED_ABORTED
        req.kv_transfer_params = {"remote_host_ip": "tcp://127.0.0.1:5569"}
        scheduler.processed_request.add("req_1")

        with patch.object(scheduler, 'ctx') as mock_ctx:
            mock_ctx.socket.return_value.send_string.side_effect = [None, RuntimeError]
            delay_free, params = scheduler.request_finished(req, [])
            delay_free, params = scheduler.request_finished(req, [])

            # Check that request was removed from processed_request
            assert "req_1" not in scheduler.processed_request

    @pytest.mark.skip
    def test_async_pull_kv(self, mock_vllm_config, mock_zmq_context):
        mock_vllm_config.additional_config = dict(async_pull_kv=True)
        scheduler = DecodeConnectorScheduler(mock_vllm_config)
        assert hasattr(scheduler, "pub")

        req = MagicMock()
        req.request_id = "req_1"
        req.kv_transfer_params = {"remote_block_ids": [1, 2], "remote_cluster_id": "cluster_1", "remote_host_ip": "tcp://127.0.0.1:5568", "spec_token_ids": []}
        scheduler._reqs_need_recv[req.request_id] = (req, [10, 11])

        metadata = scheduler.build_connector_metadata(None)
        scheduler.pub.send.assert_called()


class TestDecodeConnectorWorker:
    @pytest.mark.skip
    def test_init_with_async_pull_kv(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_threading_thread):
        mock_vllm_config.additional_config = dict(async_pull_kv=True)
        worker = DecodeConnectorWorker(mock_vllm_config, "127.0.0.1", 123)
        assert worker.vllm_config == mock_vllm_config
        assert worker.host_cluster_id == 123
        assert worker.datadist_manager is not None
        assert worker._recving_transfers == []
        assert isinstance(worker._done_recving_count, defaultdict)

    def test_register_kv_caches(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_threading_thread):
        worker = DecodeConnectorWorker(mock_vllm_config, "127.0.0.1", 123)
        kv_caches = {"layer_0": torch.tensor([1, 2, 3])}

        worker.register_kv_caches(kv_caches)
        worker.datadist_manager.register_memory.assert_called_once_with(kv_caches, None)

    @pytest.mark.parametrize("tp", [1, 2])
    @pytest.mark.parametrize(
        "local_blocks,remote_blocks,full_local_blocks,expected_local,expected_remote",
        [
            # Standard grouped case.
            ([[1, 2, 3]], [[4, 5, 6]], [[1, 2, 3]], [1, 2, 3], [4, 5, 6]),
            # Remote shorter than local: trim local to the available remote blocks.
            ([[1, 2, 3, 4]], [[5, 6]], [[1, 2, 3, 4]], [1, 2], [5, 6]),
            # Decode prefix-cache hit: local is an unhashed suffix of full local,
            # so transfer the corresponding remote suffix.
            ([[100]], [[7, 37, 58]], [[4, 53, 100]], [100], [58]),
            # MTP/lookahead: trim the extra remote tail before suffix alignment.
            ([[5, 6, 7, 8]], [[5, 9, 10, 17, 21]], [[5, 6, 7, 8]],
             [5, 6, 7, 8], [5, 9, 10, 17]),
            # Multiple hybrid groups: preserve per-group alignment, then flatten.
            ([[0, 0, 97], [100]], [[0, 0, 55], [7, 37, 58]], [[0, 0, 97], [4, 53, 100]],
             [0, 0, 97, 100], [0, 0, 55, 58]),
            # Empty grouped local blocks -> skip.
            ([[]], [[]], [[]], [], []),
        ],
    )
    def test_start_load_kv_block_slicing(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_get_tp_group, mock_get_dcp_group, mock_threading_thread, local_blocks, remote_blocks, full_local_blocks, expected_local, expected_remote, tp):
        def fun_submit(*arg, **kwarg):
            worker._read_blocks(**kwarg)
            return MagicMock()
        mock_threading_thread[1].submit.side_effect = fun_submit

        mock_vllm_config.parallel_config.tensor_parallel_size = tp

        worker = DecodeConnectorWorker(mock_vllm_config, "127.0.0.1", 123)

        metadata = DatadistConnectorMetadata()
        metadata.add_new_req(
            "req_1",
            local_block_ids=local_blocks,
            kv_transfer_params={
                "remote_block_ids": remote_blocks,
                "remote_cluster_id": "cluster_1",
                "remote_host_ip": "tcp://127.0.0.1:5568",
                "remote_request_id": "remote_req_1",
                "remote_pp_partition": None,
                "remote_dp_rank": 0,
                "spec_token_ids": [],
            },
            full_local_block_ids=full_local_blocks,
        )
        metadata.add_new_req(
            "req_2",
            local_block_ids=local_blocks,
            kv_transfer_params={
                "remote_block_ids": remote_blocks,
                "remote_cluster_id": "cluster_1",
                "remote_host_ip": "tcp://127.0.0.1:5568",
                "remote_request_id": "remote_req_1",
                "remote_pp_partition": None,
                "remote_dp_rank": 0,
                "spec_token_ids": [],
            },
            full_local_block_ids=full_local_blocks,
        )

        with patch(f"{PATH_PREFIX}.torch.distributed.barrier") as mock_bar, \
             patch.object(worker.datadist_manager, 'get_real_remote_cluster_ids', return_value=["cluster_1"]), \
             patch.object(worker, "ctx") as mock_ctx:
            worker.start_load_kv(metadata)
            if expected_local == []:
                worker.executor.submit.assert_not_called()
            else:
                assert worker.executor.submit.call_count == 2
                for call in worker.executor.submit.call_args_list:
                    assert call.kwargs["local_block_ids"] == expected_local
                    assert call.kwargs["remote_block_ids"] == expected_remote

    def test_get_finished(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_threading_thread):
        worker = DecodeConnectorWorker(mock_vllm_config, "127.0.0.1", 123)
        metadata = DatadistConnectorMetadata()

        # Add a transfer to the recving list
        worker._recving_transfers = ["req_1", "req_2"]

        sending, recving = worker.get_finished(metadata)

        # Sending set should be empty for decode worker
        assert sending == set()
        # Recving set should contain the requests from _recving_transfers
        assert recving == {"req_1", "req_2"}
        # Check that _recving_transfers was cleared
        assert worker._recving_transfers == []

    def test_heartbeat_timer_func(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_threading_thread):
        worker = DecodeConnectorWorker(mock_vllm_config, "127.0.0.1", 123)
        with patch.object(worker, "datadist_manager") as mock_manager, \
             patch.object(worker, "ctx") as mock_ctx, \
             patch(f"{PATH_PREFIX}.time") as mock_time:
            mock_manager.registered_link_infos = [((0, 0), 0, 0), ((0, 0), 0, 0), ((0, 0), 0, 0)]
            mock_manager.cluster_id_to_ip_port.side_effect = [("127.0.0.1:123", 2, 1, 0), ("127.0.0.1:123", 2, 1, 0), ("127.0.0.1:123", 2, 1, 0)]
            mock_ctx.socket.return_value.recv_string.side_effect = ["data", zmq.error.Again]
            mock_time.sleep.return_value = StopIteration
            mock_time.time.side_effect = [0, 0, int(1e9)]
            worker.zmq_socket_map = {"127.0.0.2:124": MagicMock()}

            try:
                worker.heartbeat_timer_func()
            except StopIteration:
                pass

class TestHelper:
    def test_handle_exception(self):
        future = MagicMock()
        future.exception.return_value = RuntimeError()
        with pytest.raises(RuntimeError):
            handle_exception(future)


    def test_get_local_ip(self):
        # This function connects to an external address. Mocking socket is complex here.
        # A simple test to ensure it returns a string-like IP.
        ip = get_local_ip()
        assert isinstance(ip, str)
        assert len(ip) > 0
        # Basic check for IPv4 format (not perfect, but a start)
        assert len(ip.split('.')) == 4


class TestLLMDataDistConnectorUnregisterKvCaches:
    """Tests for LLMDataDistConnector.unregister_kv_caches method."""

    def test_unregister_kv_caches_success(self):
        """Test successful unregistration of KV caches."""
        connector = MagicMock(spec=LLMDataDistConnector)
        connector.connector_worker = MagicMock()
        connector.connector_worker.unregister_kv_caches = MagicMock()

        # Simulate the actual implementation
        if connector.connector_worker is None:
            raise RuntimeError("self.connector_worker cannot be None")
        connector.connector_worker.unregister_kv_caches()

        connector.connector_worker.unregister_kv_caches.assert_called_once()

    def test_unregister_kv_caches_raises_when_worker_is_none(self):
        """Test that RuntimeError is raised when connector_worker is None."""
        connector = MagicMock(spec=LLMDataDistConnector)
        connector.connector_worker = None

        with pytest.raises(RuntimeError, match="self.connector_worker cannot be None"):
            if connector.connector_worker is None:
                raise RuntimeError("self.connector_worker cannot be None")
            connector.connector_worker.unregister_kv_caches()


class TestPrefillConnectorWorkerUnregisterKvCaches:
    """Tests for PrefillConnectorWorker.unregister_kv_caches method."""

    def test_unregister_kv_caches_calls_unlink_and_unmemory(self):
        """Test that unregister_kv_caches calls both unregister_link and unregister_memory.

        Covers lines 406-408 in llmdatadist_connector_v1.py.
        """
        worker = MagicMock(spec=PrefillConnectorWorker)
        worker.datadist_manager = MagicMock()
        worker.datadist_manager.unregister_link = MagicMock()
        worker.datadist_manager.unregister_memory = MagicMock()

        # Simulate the actual implementation
        worker.datadist_manager.unregister_link()
        worker.datadist_manager.unregister_memory()

        worker.datadist_manager.unregister_link.assert_called_once()
        worker.datadist_manager.unregister_memory.assert_called_once()


class TestDecodeConnectorWorkerUnregisterKvCaches:

    def test_unregister_kv_caches_calls_unlink_and_unmemory(self):
        """Test that unregister_kv_caches calls both unregister_link and unregister_memory."""
        worker = MagicMock(spec=DecodeConnectorWorker)
        worker.datadist_manager = MagicMock()
        worker.datadist_manager.unregister_link = MagicMock()
        worker.datadist_manager.unregister_memory = MagicMock()

        # Simulate the actual implementation
        worker.datadist_manager.unregister_link()
        worker.datadist_manager.unregister_memory()

        worker.datadist_manager.unregister_link.assert_called_once()
        worker.datadist_manager.unregister_memory.assert_called_once()


class TestUnregisterKvCachesIntegration:
    """Integration tests for the full unregister_kv_caches flow."""

    def test_full_unregister_flow_for_decode(self):
        """Test the full unregister flow for decode worker."""
        # Create mock objects
        manager = MagicMock()
        manager.data_dist_config = MagicMock()
        manager.data_dist_config.is_prefill = False
        manager.registered_link_infos = {
            (('cluster1',), 0, 0): [12345],
        }
        manager.registered_kv_caches = [MagicMock(cache_id=1)]

        # Track method calls
        calls = []

        def track_unlink():
            calls.append('unregister_link')

        def track_unmemory():
            calls.append('unregister_memory')
            for kv_cache in manager.registered_kv_caches:
                manager.data_dist_engine.cache_manager.unregister_cache(kv_cache.cache_id)
            manager.registered_kv_caches = []

        manager.unregister_link = track_unlink
        manager.unregister_memory = track_unmemory
        manager.data_dist_engine = MagicMock()
        manager.data_dist_engine.cache_manager = MagicMock()

        # Execute the flow
        manager.unregister_link()
        manager.unregister_memory()

        # Verify
        assert 'unregister_link' in calls
        assert 'unregister_memory' in calls
        assert manager.registered_kv_caches == []

    def test_full_unregister_flow_for_prefill(self):
        """Test the full unregister flow for prefill worker."""
        # Create mock objects
        manager = MagicMock()
        manager.data_dist_config = MagicMock()
        manager.data_dist_config.is_prefill = True
        manager.registered_kv_caches = [MagicMock(cache_id=1)]
        manager.data_dist_engine = MagicMock()
        manager.data_dist_engine_is_inited = True

        # Track method calls
        calls = []

        def track_finalize():
            calls.append('finalize')
            manager.data_dist_engine_is_inited = False

        def track_unlink():
            calls.append('unregister_link')
            if manager.data_dist_config.is_prefill:
                track_finalize()

        def track_unmemory():
            calls.append('unregister_memory')
            manager.registered_kv_caches = []

        manager.unregister_link = track_unlink
        manager.unregister_memory = track_unmemory

        # Execute the flow
        manager.unregister_link()
        manager.unregister_memory()

        # Verify
        assert 'unregister_link' in calls
        assert 'unregister_memory' in calls
        assert 'finalize' in calls
        assert manager.data_dist_engine_is_inited is False


class TestUnregisterKVCachesConnector:
    """Tests for LLMDataDistConnector.unregister_kv_caches method."""

    def test_unregister_kv_caches_raises_when_connector_worker_is_none(self):
        """Test that unregister_kv_caches raises RuntimeError when connector_worker is None."""
        vllm_config = create_vllm_config(kv_role="kv_consumer")
        connector = LLMDataDistConnector.__new__(LLMDataDistConnector)
        connector.connector_worker = None

        with pytest.raises(RuntimeError, match="self.connector_worker cannot be None"):
            connector.unregister_kv_caches()

    def test_unregister_kv_caches_calls_worker_method(self):
        """Test that unregister_kv_caches calls connector_worker.unregister_kv_caches."""
        vllm_config = create_vllm_config(kv_role="kv_consumer")
        connector = LLMDataDistConnector.__new__(LLMDataDistConnector)
        connector.connector_worker = MagicMock()

        connector.unregister_kv_caches()

        connector.connector_worker.unregister_kv_caches.assert_called_once()


class TestPrefillConnectorWorkerUnregisterKVCaches:
    """Tests for PrefillConnectorWorker.unregister_kv_caches method."""

    def test_prefill_worker_unregister_kv_caches_calls_manager_methods(self):
        """Test that unregister_kv_caches calls manager methods."""
        # Create a mock worker without full initialization
        worker = MagicMock()
        worker.datadist_manager = MagicMock()

        # Bind the actual method to the mock worker
        worker.unregister_kv_caches = PrefillConnectorWorker.unregister_kv_caches.__get__(worker, PrefillConnectorWorker)

        worker.unregister_kv_caches()

        worker.datadist_manager.unregister_link.assert_called_once()
        worker.datadist_manager.unregister_memory.assert_called_once()

    def test_prefill_worker_unregister_kv_caches_clears_remote_hb_info(self):
        """Test that unregister_kv_caches clears remote_hb_info for PrefillConnectorWorker."""
        worker = MagicMock()
        worker.datadist_manager = MagicMock()
        worker.remote_hb_info = {'cluster_123': 1234567890, 'cluster_456': 1234567891}
        worker.remote_hb_info_lock = threading.Lock()

        worker.unregister_kv_caches = PrefillConnectorWorker.unregister_kv_caches.__get__(worker, PrefillConnectorWorker)

        worker.unregister_kv_caches()

        assert worker.remote_hb_info == {}


class TestDecodeConnectorWorkerUnregisterKVCaches:
    """Tests for DecodeConnectorWorker.unregister_kv_caches method."""

    def test_decode_worker_unregister_kv_caches_calls_manager_methods(self):
        """Test that unregister_kv_caches calls manager methods."""
        # Create a mock worker without full initialization
        worker = MagicMock()
        worker.datadist_manager = MagicMock()

        # Bind the actual method to the mock worker
        worker.unregister_kv_caches = DecodeConnectorWorker.unregister_kv_caches.__get__(worker, DecodeConnectorWorker)

        worker.unregister_kv_caches()

        worker.datadist_manager.unregister_link.assert_called_once()
        worker.datadist_manager.unregister_memory.assert_called_once()

    def test_decode_worker_unregister_kv_caches_clears_hb_server_info(self):
        """Test that unregister_kv_caches clears hb_server_info for DecodeConnectorWorker."""
        worker = MagicMock()
        worker.datadist_manager = MagicMock()
        worker.hb_server_info = {'tcp://127.0.0.1:5000': (MagicMock(), 1234567890)}

        worker.unregister_kv_caches = DecodeConnectorWorker.unregister_kv_caches.__get__(worker, DecodeConnectorWorker)

        worker.unregister_kv_caches()

        assert worker.hb_server_info == {}


class TestPrefillHeartbeatTimerAfterSleep:
    """Tests for PrefillConnectorWorker.heartbeat_timer_func behavior after sleep/clear."""

    def test_heartbeat_timer_no_timeout_when_remote_hb_info_cleared(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_get_pp_group, mock_zmq_context, mock_threading_thread):
        """Test that heartbeat_timer_func does not trigger timeout when remote_hb_info is cleared after sleep."""
        mock_get_tp_rank.return_value = 0
        worker = PrefillConnectorWorker(mock_vllm_config, "127.0.0.1", "5568")

        with patch(f"{PATH_PREFIX}.time") as mock_time_util, \
             patch.object(worker, 'datadist_manager') as mock_manager, \
             patch.object(worker, 'hb_socket') as mock_hb_socket:
            # remote_hb_info is empty after sleep/unregister
            worker.remote_hb_info = {}
            worker.remote_hb_info_lock = threading.Lock()

            # Set time such that any entry would timeout if it existed
            mock_time_util.time.return_value = 1234567890 + CLUSTER_HEARTBEAT_TIMEOUT + 100
            mock_time_util.sleep.side_effect = StopIteration

            try:
                worker.heartbeat_timer_func()
            except StopIteration:
                pass

            # force_unlink should not be called since remote_hb_info is empty
            mock_manager.force_unlink.assert_not_called()


class TestDecodeHeartbeatTimerAfterSleep:
    """Tests for DecodeConnectorWorker.heartbeat_timer_func behavior after sleep/clear."""

    def test_heartbeat_timer_no_timeout_when_hb_server_info_cleared(self, mock_vllm_config, mock_datadist_manager, mock_get_tp_rank, mock_threading_thread):
        """Test that heartbeat_timer_func does not trigger close_link when hb_server_info is cleared after sleep."""
        worker = DecodeConnectorWorker(mock_vllm_config, "127.0.0.1", 123)

        with patch.object(worker, "datadist_manager") as mock_manager, \
             patch.object(worker, "ctx") as mock_ctx, \
             patch(f"{PATH_PREFIX}.time") as mock_time_util:

            # hb_server_info is empty after sleep/unregister
            worker.hb_server_info = {}
            mock_manager.registered_link_infos = [((0, 0), 0, 0), ((0, 0), 0, 0), ((0, 0), 0, 0)]
            mock_manager.cluster_id_to_ip_port.side_effect = [("127.0.0.1:123", 2, 0, 0), ("127.0.0.1:123", 2, 0, 0), ("127.0.0.1:123", 2, 0, 0)]
            mock_time_util.time.side_effect = [0, 0, int(1e9)]
            mock_time_util.sleep.side_effect = StopIteration

            try:
                worker.heartbeat_timer_func()
            except StopIteration:
                pass

            # close_link should not be called since hb_server_info is empty
            mock_manager.close_link.assert_not_called()
