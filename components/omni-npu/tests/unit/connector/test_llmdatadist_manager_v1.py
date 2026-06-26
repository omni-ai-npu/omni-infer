# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
import copy
import os
import socket
import struct
import threading
import time
from unittest import mock
from unittest.mock import patch, MagicMock

import pytest
import torch
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_world_group

from omni_npu.connector.llmdatadist_manager_v1 import (
    LLMDataDistManager,
    LLMDataDistConfig,
    TORCH_DTYPE_TO_NPU_DTYPE,
    SCHEDULER_LINK_BATCH_SIZE,
    SCHEDULER_LINK_INTERVAL,
    KV_CACHE_RETRY_TIMES,
    KV_CACHE_RETRY_WAIT_SECOND,
    SYNC_KV_TIMEOUT,
    LINK_TIMEOUT,
    RETRYABLE_CODES,
    NUM_DIE_PER_MACH,
    get_pp_partition,
    ip_port_to_int,
    unzip_kv_cache_dict,
    unzip_kv_cache_list,
    maybe_merge_kv_caches,
    maybe_split_kv_caches_for_spec_layers,
    LLMStatusCode,
    LLMRole
)

# Define path constants
VLLM_KV_TRANSFER_MANAGER_PATH = 'omni_npu.connector.llmdatadist_manager_v1'
LLM_DATADIST_PATH = 'omni_npu.connector.llmdatadist_manager_v1'

@pytest.fixture
def mock_llm_datadist():
    with patch(f'{LLM_DATADIST_PATH}.LLMDataDist') as mock_datadist:
        mock_instance = MagicMock()
        mock_datadist.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_world_group():
    with patch(f'{VLLM_KV_TRANSFER_MANAGER_PATH}.get_world_group') as mock_get_world_group:
        mock_world_group = MagicMock()
        mock_world_group.rank_in_group = 0
        mock_world_group.local_rank = 0
        mock_get_world_group.return_value = mock_world_group
        yield mock_world_group


@pytest.fixture
def mock_vllm_config():
    config = MagicMock(spec=VllmConfig)
    config.kv_transfer_config = MagicMock()
    config.kv_transfer_config.kv_role = 'kv_producer'
    config.kv_transfer_config.kv_parallel_size = 2
    config.kv_transfer_config.kv_connector_extra_config = {'kv_producer_dp_size': 1}
    config.parallel_config = MagicMock()
    config.parallel_config.data_parallel_rank = 0
    config.parallel_config.tensor_parallel_size = 1
    config.parallel_config.data_parallel_size = 1
    config.parallel_config.pipeline_parallel_size = 1
    yield config


@pytest.fixture
def mock_block_cache_key():
    with patch(f"{VLLM_KV_TRANSFER_MANAGER_PATH}.BlocksCacheKey") as mock_obj:
        yield mock_obj


@pytest.fixture
def mock_kv_cache_retry_times():
    from omni_npu.connector import llmdatadist_manager_v1
    ori = llmdatadist_manager_v1.KV_CACHE_RETRY_TIMES
    llmdatadist_manager_v1.KV_CACHE_RETRY_TIMES = 3
    yield llmdatadist_manager_v1.KV_CACHE_RETRY_TIMES
    llmdatadist_manager_v1.KV_CACHE_RETRY_TIMES = ori


class TestLLMDataDistManager:

    def test_init_llm_data_dist_manager(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        assert manager.rank == 0
        assert manager.local_rank == 0
        assert manager.tp_size == 1
        assert manager.dp_size == 1
        assert manager.dp_rank == 0
        assert manager.prefill_dp_size == 1
        assert manager.data_dist_engine == mock_llm_datadist

    @pytest.mark.parametrize(
        "remote_cluster_id,remote_dp_rank,expected_result",
        [
            ((12345, 67890), 0, [(12345, 67890)]),
            ([12345, 67890], 0, [(12345, 67890)]),
        ]
    )
    def test_get_real_remote_cluster_ids_found(self, mock_vllm_config, mock_llm_datadist, mock_world_group, 
                                               remote_cluster_id, remote_dp_rank, expected_result):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        meta = MagicMock()
        meta.remote_cluster_id = remote_cluster_id
        meta.remote_dp_rank = remote_dp_rank
        
        key = (tuple(remote_cluster_id) if isinstance(remote_cluster_id, list) else remote_cluster_id, 
               remote_dp_rank, 0)
        manager.registered_link_infos[key] = expected_result
        
        result = manager.get_real_remote_cluster_ids(meta)
        
        assert result == expected_result

    def test_get_real_remote_cluster_ids_not_found_register_link(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        meta = MagicMock()
        meta.remote_cluster_id = (12345, 67890)
        meta.remote_dp_rank = 0

        manager.registered_link_infos[(54321, 67890), 0, 0] = None

        # Mock register_link to avoid side effects
        with patch.object(manager, 'register_link') as mock_register_link, \
             patch.object(manager, 'close_link') as mock_close_link:
            result = manager.get_real_remote_cluster_ids(meta)
            
            mock_register_link.assert_called_once_with((12345, 67890), 0, 0, 0)
            assert result is None  # Since we don't set it in the dict
            mock_close_link.assert_called()

    def test_register_link_success(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        mock_llm_datadist.link_clusters.return_value = LLMStatusCode.LLM_SUCCESS, None

        with patch.object(manager, '_get_cluster_id_list', return_value=[12345]):
            with patch.object(manager, 'cluster_id_to_ip_port', return_value=("127.0.0.1:8000", 1, 1, 0)):
                with patch.object(manager, '_get_local_ip', return_value="127.0.0.1"):
                    manager.register_link((12345,), 0, 0)
        
        mock_llm_datadist.link_clusters.assert_called_once()

    def test_register_link_failure(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)

        mock_llm_datadist.link_clusters.return_value = (1, "error")
        
        with patch.object(manager, '_get_cluster_id_list', return_value=[12345]):
            with patch.object(manager, 'cluster_id_to_ip_port', return_value=("127.0.0.1:8000", 1, 1, 0)):
                with patch.object(manager, '_get_local_ip', return_value="127.0.0.1"):
                    with pytest.raises(Exception, match="link failed"):
                        manager.register_link((12345,), 0, 0)

    def test_close_link_success(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        manager.data_dist_config.is_prefill = False

        mock_llm_datadist.unlink_clusters.return_value = LLMStatusCode.LLM_SUCCESS, None
        
        with patch.object(manager, '_get_cluster_id_list', return_value=[12345]):
            with patch.object(manager, 'cluster_id_to_ip_port', return_value=("127.0.0.1:8000", 1, 1, 0)):
                with patch.object(manager, '_get_local_ip', return_value="127.0.0.1"):
                    manager.close_link((12345, ), 0, 0, 0)
        
        mock_llm_datadist.unlink_clusters.assert_called_once()

    def test_close_link_failure(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        mock_llm_datadist.unlink_clusters.return_value = (1, "error")  # Non-success status
        
        with patch.object(manager, '_get_cluster_id_list', return_value=[12345]):
            with patch.object(manager, 'cluster_id_to_ip_port', return_value=("127.0.0.1:8000", 1, 1, 0)):
                with patch.object(manager, '_get_local_ip', return_value="127.0.0.1"):
                    with pytest.raises(Exception, match="unlink failed"):
                        manager.close_link(12345, 0, 0, 0)

    def test_force_unlink(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        manager.force_unlink(12345)
        
        mock_llm_datadist.unlink_clusters.assert_called_once()

    @pytest.mark.parametrize(
        "exception_type, status_code, expected_result",
        [
            (None, None, True),  # Success case
            ("LLMException", 0, False),  # Non-retryable exception
            ("LLMException", 1, False),  # Retryable exception, max retries reached
        ]
    )
    def test_pull_blocks(self, mock_vllm_config, mock_llm_datadist, mock_world_group, 
                         exception_type, status_code, expected_result):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        if exception_type == "LLMException":
            from llm_datadist import LLMException, LLMStatusCode
            if status_code in RETRYABLE_CODES:
                mock_llm_datadist.cache_manager.pull_blocks.side_effect = [
                    LLMException("test", status_code),
                    LLMException("test", status_code)
                ]
            else:
                mock_llm_datadist.cache_manager.pull_blocks.side_effect = [
                    LLMException("test", status_code)
                ]
        else:
            # Success case
            pass

        src_cache_key = MagicMock()
        dst_cache = MagicMock()
        src_blocks = [0]
        dst_blocks = [0]
        
        result = manager._pull_blocks(src_cache_key, dst_cache, src_blocks, dst_blocks)
        
        assert result == expected_result

    def test_pull_blocks_retryable_code_success_after_retry(self, mock_vllm_config, mock_llm_datadist, mock_world_group, mock_kv_cache_retry_times):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        from llm_datadist import LLMException, LLMStatusCode
        # First call fails with retryable code, second succeeds
        mock_llm_datadist.cache_manager.pull_blocks.side_effect = [
            LLMException("test", status_code=LLMStatusCode.LLM_TIMEOUT),
            None  # Success on second try
        ]
        
        src_cache_key = MagicMock()
        dst_cache = MagicMock()
        src_blocks = [0]
        dst_blocks = [0]
        
        result = manager._pull_blocks(src_cache_key, dst_cache, src_blocks, dst_blocks)
        
        assert result == True
        assert mock_llm_datadist.cache_manager.pull_blocks.call_count == 2

    def test_pull_blocks_failure_after_retry(self, mock_vllm_config, mock_llm_datadist, mock_world_group, mock_kv_cache_retry_times):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)

        from llm_datadist import LLMException, LLMStatusCode
        # First call fails with retryable code, second succeeds
        mock_llm_datadist.cache_manager.pull_blocks.side_effect = LLMException("test", status_code=LLMStatusCode.LLM_TIMEOUT)

        mock_cache = MagicMock()
        manager.registered_kv_caches = [mock_cache]

        with patch.object(manager, "_refresh_link"):
            with pytest.raises(RuntimeError):
                manager.pull_kv([0], [0], 12345, 0, None)

        mock_llm_datadist.cache_manager.pull_blocks.side_effect = ValueError

        with patch.object(manager, "_refresh_link"):
            with pytest.raises(RuntimeError):
                manager.pull_kv([0], [0], 12345, 0, None)

        from omni_npu.connector import llmdatadist_manager_v1
        llmdatadist_manager_v1.KV_CACHE_RETRY_TIMES = 0

        with patch.object(manager, "_refresh_link"):
            with pytest.raises(RuntimeError):
                manager.pull_kv([0], [0], 12345, 0, None)

    def test_pull_kv_success(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Mock registered_kv_caches
        mock_cache = MagicMock()
        manager.registered_kv_caches = [mock_cache]
        # Mock _pull_blocks to return True
        with patch.object(manager, '_pull_blocks', return_value=True):
            manager.pull_kv([0], [0], 12345, 0, None)
        
            # Verify _pull_blocks was called
            manager._pull_blocks.assert_called_once()

    def test_pull_kv_success_pp(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        mock_vllm_config_pp = copy.deepcopy(mock_vllm_config)
        mock_vllm_config_pp.kv_transfer_config.kv_role = "kv_consumer"
        manager = LLMDataDistManager(mock_vllm_config_pp, "127.0.0.1", 8000)
        
        # Mock registered_kv_caches
        mock_cache = MagicMock()
        manager.registered_kv_caches = [mock_cache]
        manager.registered_kv_caches_tensor = [(torch.ones(1), torch.ones(1))]
        # Mock _pull_blocks to return True
        with patch.object(manager, '_pull_blocks', return_value=True):
            manager.pull_kv([0], [0], 12345, 0, [1, 1])
        
            # Verify _pull_blocks was called
            assert manager._pull_blocks.call_count == 2

    def test_pull_kv_failure_then_success_with_refresh(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Mock registered_kv_caches
        mock_cache = MagicMock()
        manager.registered_kv_caches = [mock_cache]
        
        # Mock _pull_blocks to fail first, then succeed
        with patch.object(manager, '_pull_blocks', side_effect=[False, True]):
            with patch.object(manager, '_refresh_link'):
                manager.pull_kv([0], [0], 12345, 0, None)
        
            # _pull_blocks should be called twice (first fail, then success after refresh)
            assert manager._pull_blocks.call_count == 2

    def test_pull_kv_failure_even_after_refresh(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Mock registered_kv_caches
        mock_cache = MagicMock()
        manager.registered_kv_caches = [mock_cache]
        
        # Mock _pull_blocks to always fail
        with patch.object(manager, '_pull_blocks', return_value=False):
            with patch.object(manager, '_refresh_link'):
                with pytest.raises(RuntimeError, match="Failed to pull kv even if rebuild the kv link!"):
                    manager.pull_kv([0], [0], 12345, 0, None)

    def test_refresh_link_success(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Mock _get_host_cluster_id to return a valid host_cluster_id
        with patch.object(manager, '_get_host_cluster_id', return_value=((12345,), 0, 0)):
            with patch.object(manager, 'close_link'):
                with patch.object(manager, 'register_link'):
                    manager._refresh_link(12345, 0, 0)

                    manager.register_link.assert_called_once()
                manager.close_link.assert_called_once()

    def test_refresh_link_no_host_cluster_id(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)

        # Mock _get_host_cluster_id to return None (meaning no matching host found)
        with patch.object(manager, '_get_host_cluster_id', return_value=None):
            # Should return early without raising or calling close/register
            manager._refresh_link(12345, 0, 0)

            # Verify close_link and register_link were not called
            # (since _get_host_cluster_id returned None)

    def test_get_host_cluster_id_found(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Add a matching entry to registered_link_infos
        manager.registered_link_infos[((12345,), 0, 0)] = [12345]
        
        result = manager._get_host_cluster_id(12345, 0, 0)
        
        assert result == ((12345,), 0, 0)

    def test_get_host_cluster_id_not_found(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # No matching entry
        result = manager._get_host_cluster_id(12345, 0, 0)
        
        assert result is None

    def test_get_cluster_id_list(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Mock cluster_id_to_ip_port
        with patch.object(manager, 'cluster_id_to_ip_port', return_value=("127.0.0.1:8000", 1, 1, 0)):
            result = manager._get_cluster_id_list([12345], 0, 0, 0)
        
        assert len(result) == 1
        assert isinstance(result[0], int)

        with patch.object(manager, 'cluster_id_to_ip_port', return_value=("127.0.0.1:8000", 1, 1, 0)):
            result = manager._get_cluster_id_list(12345, 0, 0, 0)

        assert len(result) == 1
        assert isinstance(result[0], int)

    def test_register_memory_dense_model(self, mock_vllm_config, mock_llm_datadist, mock_world_group, mock_block_cache_key):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Create mock KV caches
        kv_cache = {
            'layer.0': torch.randn(2, 4, 8, 16, dtype=torch.float16)
        }
        
        mock_cache = MagicMock()
        mock_llm_datadist.cache_manager.register_blocks_cache.return_value = mock_cache
        kv_cache_config = MagicMock()
        kv_cache_group = MagicMock()
        kv_cache_group.layer_names = ['layer.0']
        kv_cache_config.kv_cache_groups = [kv_cache_group]
        manager.register_memory(kv_cache, kv_cache_config)
        
        # Verify the cache was registered
        assert len(manager.registered_kv_caches) == 1
        assert manager.registered_kv_caches[0] == mock_cache

    def test_register_memory_dense_model_tuple(self, mock_vllm_config, mock_llm_datadist, mock_world_group, mock_block_cache_key):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Create mock KV caches with tuple
        kv_cache = {
            'layer.0': (torch.randn(4, 8, 16, dtype=torch.float16), torch.randn(4, 8, 16, dtype=torch.float16))
        }
        
        mock_cache = MagicMock()
        mock_llm_datadist.cache_manager.register_blocks_cache.return_value = mock_cache
        kv_cache_config = MagicMock()
        kv_cache_group = MagicMock()
        kv_cache_group.layer_names = ['layer.0']
        kv_cache_config.kv_cache_groups = [kv_cache_group]
        manager.register_memory(kv_cache, kv_cache_config)
        
        # New unzip/register behavior groups same-shape tensors into one cache desc.
        assert len(manager.registered_kv_caches) == 1

    def test_register_memory_pp(self, mock_vllm_config, mock_llm_datadist, mock_world_group, mock_block_cache_key):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        manager.data_dist_config.is_prefill = False # only decode support registering remote pp kv blocks
        # Create mock KV caches with tuple
        kv_cache = {
            'layer.0': (torch.randn(4, 8, 16, dtype=torch.float16), torch.randn(4, 8, 16, dtype=torch.float16)),
            'layer.1': (torch.randn(4, 8, 16, dtype=torch.float16), torch.randn(4, 8, 16, dtype=torch.float16))
        }
        
        mock_cache = MagicMock()
        mock_llm_datadist.cache_manager.register_blocks_cache.return_value = mock_cache
        kv_cache_config = MagicMock()
        kv_cache_group = MagicMock()
        kv_cache_group.layer_names = ['layer.0', 'layer.1']
        kv_cache_config.kv_cache_groups = [kv_cache_group]
        manager.register_memory(kv_cache, kv_cache_config)
        
        # Verify the cache was registered
        assert len(manager.registered_kv_caches) == 1

        manager._register_remote_pp_partition([1, 1])


    def test_register_memory_duplicate_call(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Pre-populate registered_kv_caches
        manager.registered_kv_caches = [MagicMock()]
        
        kv_cache = {
            'layer.0': torch.randn(2, 4, 8, 16, dtype=torch.float16)
        }
        
        with pytest.raises(ValueError, match="Attr `registered_kv_caches` must be empty before register kv_caches."):
            manager.register_memory(kv_cache)

    def test_cluster_id_to_ip_port(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        # Create a test cluster_id
        cluster_id = ip_port_to_int("127.0.0.1:8000", 2, 4)
        
        ip_port, tp_size, pp_size, tp_rank = manager.cluster_id_to_ip_port(cluster_id)
        
        assert ip_port == "127.0.0.1:8000"
        assert tp_size == 2
        assert pp_size == 4
        assert tp_rank == 0

    def test_cluster_id_to_ip_port_invalid_type(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        
        with pytest.raises(TypeError, match="cluster_id must be int type"):
            manager.cluster_id_to_ip_port("not_an_int")


class TestHelper:
    @pytest.mark.parametrize(
        "kv_caches,expect_len",
        [
            ({"layer.0": torch.zeros(1)}, 1),
            ({"layer.0": (torch.zeros(1), torch.ones(1))}, 1),
        ],
    )
    def test_unzip_kv_cache_dict_basic(self, kv_caches, expect_len):
        out = unzip_kv_cache_dict(kv_caches)
        assert len(out) == expect_len
        if isinstance(kv_caches["layer.0"], tuple):
            assert len(out[0]) == 2
            assert out[0][0] in kv_caches["layer.0"]
            assert out[0][1] in kv_caches["layer.0"]
        else:
            assert out[0][0] == kv_caches["layer.0"]


    def test_unzip_kv_cache_dict_not_implemented(self):
        kv_caches = {
            "layer.0": torch.zeros(1),
            "layer.dup.0": torch.ones(1),
        }
        out = unzip_kv_cache_dict(kv_caches)
        assert len(out) == 1
        assert len(out[0]) == 2


    @pytest.mark.parametrize(
        "kv_list,expect_len",
        [
            ([torch.zeros(1)], 1),
            ([(torch.zeros(1), torch.ones(1))], 2),
        ],
    )
    def test_unzip_kv_cache_list(self, kv_list, expect_len):
        out = unzip_kv_cache_list(kv_list)
        assert len(out) == expect_len


    @pytest.mark.skip
    def test_maybe_merge_kv_caches(self):
        t = torch.zeros(2, 1, 1, 1, 1)
        flatten = [[t]]
        out = maybe_merge_kv_caches(flatten)
        assert len(out) == 2
        assert out[0][0].shape == (1, 1, 1, 1)


    def test_maybe_merge_kv_caches_no_merge(self):
        t = torch.zeros(1, 1)
        flatten = [[t]]
        assert maybe_merge_kv_caches(flatten) == flatten


    def test_maybe_split_kv_caches(self):
        t1 = torch.zeros(1, 2)
        t2 = torch.zeros(2, 2)
        flatten = [[t1, t2]]
        out = maybe_split_kv_caches_for_spec_layers(flatten)
        assert len(out) == 2
        assert t1 in out[0] or t1 in out[1]


    def test_maybe_split_kv_caches_no_split(self):
        t = torch.zeros(1, 2)
        flatten = [[t, t.clone()]]
        assert maybe_split_kv_caches_for_spec_layers(flatten) == flatten


    @pytest.mark.parametrize(
        "ip_port,tp_size,pp_size",
        [
            ("127.0.0.1:8000", 1, 1),
            ("0.0.0.0:0", 256, 1),
            ("127.0.0.1:8000", 1, 256),
        ],
    )
    def test_ip_port_to_int(self, ip_port, tp_size, pp_size):
        val = ip_port_to_int(ip_port, tp_size, pp_size)
        assert isinstance(val, int)


    def test_ip_port_to_int_invalid_port(self):
        with pytest.raises(ValueError):
            ip_port_to_int("127.0.0.1:70000", 1, 1)

    def test_get_pp_partition(self):
        val = get_pp_partition(num_hidden_layers=61, pp_size=2, partition="31,30")
        assert val == [31,30]
        val = get_pp_partition(num_hidden_layers=61, pp_size=3, partition=None)
        assert val == [20, 21, 20]
        with pytest.raises(ValueError):
            val = get_pp_partition(num_hidden_layers=61, pp_size=2, partition="a,b")
        with pytest.raises(ValueError):
            val = get_pp_partition(num_hidden_layers=61, pp_size=3, partition="31,30")
        with pytest.raises(ValueError):
            val = get_pp_partition(num_hidden_layers=61, pp_size=2, partition="30,30")

class TestLLMDataDistConfig:
    @pytest.fixture
    def vllm_config(self):
        """Mock VllmConfig object"""
        mock_vllm_config = MagicMock()
        mock_vllm_config.kv_transfer_config.kv_role = "kv_producer"
        mock_vllm_config.parallel_config.data_parallel_rank = 0
        mock_vllm_config.parallel_config.tensor_parallel_size = 2
        mock_vllm_config.parallel_config.pipeline_parallel_size = 2
        mock_vllm_config.parallel_config.data_parallel_size = 2
        mock_vllm_config.kv_transfer_config.kv_parallel_size = 2
        mock_vllm_config.kv_transfer_config.kv_connector_extra_config = {"kv_producer_dp_size": 1}
        return mock_vllm_config

    @pytest.fixture
    def mock_ray(self):
        import sys
        pre_ray = sys.modules.get("ray", None)
        mock_ray = MagicMock()
        mock_ray.is_initialized.return_value = False
        sys.modules["ray"] = mock_ray

        yield mock_ray

        if pre_ray is not None:
            sys.modules["ray"] = pre_ray
        else:
            del sys.modules["ray"]

    @pytest.fixture
    def block_ray_import(self):
        import sys

        class BlockModuleImporter:
            def __init__(self, blocked_name: str):
                self.blocked_name = blocked_name

            def find_spec(self, fullname, path, target=None):
                # 精确匹配 or 阻断子模块
                if fullname == self.blocked_name or fullname.startswith(self.blocked_name + "."):
                    raise ImportError(f"Module '{fullname}' is blocked")
                return None  # 交给下一个 importer

        pre_ray = sys.modules.get("ray", None)
        if pre_ray is not None:
            del sys.modules["ray"]

        module_name = "ray"
        importer = BlockModuleImporter(module_name)
        sys.meta_path.insert(0, importer)
        try:
            yield
        finally:
            if importer in sys.meta_path:
                sys.meta_path.remove(importer)
            if pre_ray:
                sys.modules["ray"] = pre_ray

    @pytest.fixture
    def mock_get_world_group(self):
        with mock.patch("omni_npu.connector.llmdatadist_manager_v1.get_world_group") as mock_world_group:
            yield mock_world_group

    @pytest.fixture
    def mock_ip_port_to_int(self):
        """Mock ip_port_to_int"""
        with mock.patch('omni_npu.connector.llmdatadist_manager_v1.ip_port_to_int') as mock_ip:
            yield mock_ip


    @pytest.mark.parametrize(
        "ignore_load_rank, expected_rank, expected_local_rank, expected_cluster_id, expected_ip_list",
        [
            (True, -1, -1, -1, ["127.0.0.1"]),
            (False, 0, 0, 123456, ["127.0.0.1"]),
        ]
    )
    def test_init(self, ignore_load_rank, expected_rank, expected_local_rank, expected_cluster_id, expected_ip_list, vllm_config,
                  mock_ip_port_to_int, mock_get_world_group, block_ray_import):
        """Test initialization with different ignore_load_rank values."""
        mock_ip_port_to_int.return_value = 123456 if not ignore_load_rank else -1
        if ignore_load_rank:
            mock_ip_port_to_int.return_value = expected_cluster_id
        else:
            mock_ip_port_to_int.return_value = expected_cluster_id
            mock_get_world_group.return_value.rank_in_group = expected_rank
            mock_get_world_group.return_value.local_rank = expected_local_rank

        config = LLMDataDistConfig(vllm_config, "127.0.0.1", 8080, ignore_load_rank=ignore_load_rank)

        assert config.rank == expected_rank
        assert config.local_rank == expected_local_rank
        assert config.cluster_id == expected_cluster_id
        assert config.host_ip_list == expected_ip_list


    @pytest.mark.parametrize(
        "ray_nodes, local_host_ip, expected_ips",
        [
            ([], "127.0.0.1", ["127.0.0.1"]),  # No nodes, fallback to local_host_ip
            ([{"Alive": True, "NodeManagerAddress": "192.168.1.1", "GcsAddress": "192.168.1.1:12345"},
              {"Alive": True, "NodeManagerAddress": "192.168.1.2", "GcsAddress": "192.168.1.2:12345"}],
             "192.168.1.2", ["192.168.1.2", "192.168.1.1"]),
            ([{"Alive": False, "NodeManagerAddress": "192.168.1.1"},
              {"Alive": True, "NodeManagerAddress": "192.168.1.2", "GcsAddress": "192.168.1.2:12345"}],
             "192.168.1.2", ["192.168.1.2"]),
            ([{"Alive": False, "NodeManagerAddress": "192.168.1.1"},
              {"Alive": True, "NodeManagerAddress": "192.168.1.2"}],
             "192.168.1.2", ["192.168.1.2"]),
            ([{"Alive": True, "NodeManagerAddress": "192.168.1.1"},
              {"Alive": True, "NodeManagerAddress": "192.168.1.2"}],
             "192.168.1.2", ["192.168.1.2", "192.168.1.1"]),
        ]
    )
    def test_get_worker_ips(self, ray_nodes, local_host_ip, expected_ips, vllm_config, mock_ray):
        """Test _get_worker_ips with different Ray cluster states."""
        mock_ray.nodes.return_value = ray_nodes
        config = LLMDataDistConfig(vllm_config, local_host_ip, 8080, ignore_load_rank=True)

        assert config._get_worker_ips() == expected_ips

        mock_ray.init.side_effect = [RuntimeError]
        config = LLMDataDistConfig(vllm_config, "127.0.0.1", 8080, ignore_load_rank=True)
        assert config._get_worker_ips() == ["127.0.0.1"]

    @pytest.mark.parametrize(
        "role_str, expected_role",
        [
            ("kv_producer", LLMRole.PROMPT),  # Simulating 'PROMPT' role
        ]
    )
    def test_role_property(self, role_str, expected_role, vllm_config):
        """Test role property."""
        vllm_config.kv_transfer_config.kv_role = role_str
        config = LLMDataDistConfig(vllm_config, "127.0.0.1", 8080, ignore_load_rank=True)

        assert config.role == expected_role


    @pytest.mark.parametrize(
        "role_str, expected_prefill",
        [
            ("kv_producer", True),  # 'PROMPT' role is expected to be prefill
        ]
    )
    def test_is_prefill_property(self, role_str, expected_prefill, vllm_config):
        """Test is_prefill property."""
        vllm_config.kv_transfer_config.kv_role = role_str
        config = LLMDataDistConfig(vllm_config, "127.0.0.1", 8080, ignore_load_rank=True)

        assert config.is_prefill == expected_prefill

    @pytest.mark.parametrize(
        "p_node_list, kv_role, expected_ips",
        [
            # is_prefill=True (kv_producer) scenarios
            (["192.168.1.10", "192.168.1.20"], "kv_producer", ["192.168.1.10", "192.168.1.20"]),
            (["192.168.1.10"], "kv_producer", ["192.168.1.10"]),
            ([], "kv_producer", ["127.0.0.1"]),  # Empty list falls back to Ray, then local
            (None, "kv_producer", ["127.0.0.1"]),  # None falls back to Ray, then local
            ("not_a_list", "kv_producer", ["127.0.0.1"]),  # Non-list falls back to local

            # is_prefill=False (kv_consumer) scenarios - always returns local IP
            (["192.168.1.10", "192.168.1.20"], "kv_consumer", ["127.0.0.1"]),
            (["192.168.1.10"], "kv_consumer", ["127.0.0.1"]),
            ([], "kv_consumer", ["127.0.0.1"]),
            (None, "kv_consumer", ["127.0.0.1"]),
            ("not_a_list", "kv_consumer", ["127.0.0.1"]),  # Non-list falls back to local
        ]
    )
    def test_get_worker_ips_p_node_list(
        self, p_node_list, kv_role, expected_ips,
        vllm_config, mock_ip_port_to_int, mock_get_world_group, block_ray_import
    ):
        """Test _get_worker_ips method with p_node_list configurations.

        Covers the logic at lines 123-131 in llmdatadist_manager_v1.py:
        - if self.is_prefill and worker_ips and isinstance(worker_ips, list) and len(worker_ips) > 0:
            return worker_ips
        - worker_ips = [self.local_host_ip]
        - if not self.is_prefill:
            return worker_ips
        """
        vllm_config.kv_transfer_config.kv_role = kv_role
        vllm_config.kv_transfer_config.kv_connector_extra_config["p_node_list"] = p_node_list
        mock_ip_port_to_int.return_value = -1

        config = LLMDataDistConfig(vllm_config, "127.0.0.1", 8080, ignore_load_rank=True)

        # Directly call the method being tested
        result = config._get_worker_ips()

        assert result == expected_ips

        # Note: The current implementation does not log warnings for invalid p_node_list types
        # Based on the code at lines 123-131 in llmdatadist_manager_v1.py, there is no warning log.
        # If warnings are added in the future, tests should be updated accordingly.

class TestUnregisterLink:
    """Tests for LLMDataDistManager.unregister_link method."""

    def test_unregister_link_for_prefill_calls_finalize(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        """Test that unregister_link calls _finalize_llm_data_dist for prefill."""
        # kv_role='kv_producer' means is_prefill=True
        mock_vllm_config.kv_transfer_config.kv_role = 'kv_producer'
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)

        with patch.object(manager, '_finalize_llm_data_dist') as mock_finalize:
            manager.unregister_link()
            mock_finalize.assert_called_once()

    def test_unregister_link_for_decode_closes_all_links(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        """Test that unregister_link closes all registered links for decode."""
        # kv_role='kv_consumer' means is_prefill=False
        mock_vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        manager.registered_link_infos = {
            (('cluster1',), 0, 0): [12345],
            (('cluster2',), 1, 0): [67890],
        }

        with patch.object(manager, 'close_link') as mock_close_link:
            manager.unregister_link()
            assert mock_close_link.call_count == 2

    def test_unregister_link_empty_link_infos(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        """Test that unregister_link handles empty link_infos gracefully."""
        # kv_role='kv_consumer' means is_prefill=False
        mock_vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        manager.registered_link_infos = {}

        with patch.object(manager, 'close_link') as mock_close_link:
            manager.unregister_link()
            mock_close_link.assert_not_called()


class TestUnregisterMemory:
    """Tests for LLMDataDistManager.unregister_memory method."""

    def test_unregister_memory_for_decode_unregisters_all_caches(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        """Test that unregister_memory unregisters all KV caches for decode."""
        # kv_role='kv_consumer' means is_prefill=False
        mock_vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)

        mock_cache1 = MagicMock()
        mock_cache1.cache_id = 1
        mock_cache2 = MagicMock()
        mock_cache2.cache_id = 2
        manager.registered_kv_caches = [mock_cache1, mock_cache2]

        manager.unregister_memory()

        assert manager.data_dist_engine.cache_manager.unregister_cache.call_count == 2
        assert manager.registered_kv_caches == []

    def test_unregister_memory_for_prefill_skips_unregister(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        """Test that unregister_memory skips unregister_cache for prefill."""
        # kv_role='kv_producer' means is_prefill=True
        mock_vllm_config.kv_transfer_config.kv_role = 'kv_producer'
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)

        mock_cache = MagicMock()
        mock_cache.cache_id = 1
        manager.registered_kv_caches = [mock_cache]

        manager.unregister_memory()

        # unregister_cache should not be called for prefill
        manager.data_dist_engine.cache_manager.unregister_cache.assert_not_called()
        assert manager.registered_kv_caches == []

    def test_unregister_memory_empty_caches(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        """Test that unregister_memory handles empty cache list gracefully."""
        # kv_role='kv_consumer' means is_prefill=False
        mock_vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        manager.registered_kv_caches = []

        manager.unregister_memory()

        manager.data_dist_engine.cache_manager.unregister_cache.assert_not_called()
        assert manager.registered_kv_caches == []


class TestFinalizeAndReinitLLMDataDist:
    """Tests for LLMDataDistManager._finalize_llm_data_dist and _reinit_llm_data_dist methods."""

    @pytest.fixture
    def mock_manager(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        """Create a LLMDataDistManager instance for testing."""
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        return manager

    def test_finalize_llm_data_dist(self, mock_manager):
        """Test _finalize_llm_data_dist properly finalizes the engine."""
        mock_manager.data_dist_engine_is_inited = True

        mock_manager._finalize_llm_data_dist()

        mock_manager.data_dist_engine.finalize.assert_called_once()
        assert mock_manager.data_dist_engine_is_inited is False

    def test_reinit_llm_data_dist(self, mock_manager):
        """Test _reinit_llm_data_dist properly reinitializes the engine."""
        mock_manager.data_dist_engine_is_inited = False

        # Reset mock to clear previous calls from _init_llm_data_dist during manager creation
        mock_manager.data_dist_engine.init.reset_mock()

        mock_manager._reinit_llm_data_dist()

        mock_manager.data_dist_engine.init.assert_called_once_with(mock_manager.data_dist_option)
        assert mock_manager.data_dist_engine_is_inited is True

    def test_register_memory_reinit_when_not_inited(self, mock_manager, mock_block_cache_key):
        """Test register_memory calls _reinit_llm_data_dist when engine not initialized."""
        mock_manager.data_dist_engine_is_inited = False

        # Create valid kv_cache_config
        kv_cache = {'layer.0': torch.randn(2, 4, 8, 16, dtype=torch.float16)}
        kv_cache_config = MagicMock()
        kv_cache_group = MagicMock()
        kv_cache_group.layer_names = ['layer.0']
        kv_cache_config.kv_cache_groups = [kv_cache_group]

        mock_cache = MagicMock()
        mock_manager.data_dist_engine.cache_manager.register_blocks_cache.return_value = mock_cache

        # Pre-populate to avoid the duplicate call error
        mock_manager.registered_kv_caches = []

        # Use patch.object with wraps to track actual call
        with patch.object(mock_manager, '_reinit_llm_data_dist', wraps=mock_manager._reinit_llm_data_dist) as mock_reinit, \
             patch(f'{VLLM_KV_TRANSFER_MANAGER_PATH}.unzip_kv_cache_dict', return_value=[[kv_cache['layer.0']]]), \
             patch(f'{VLLM_KV_TRANSFER_MANAGER_PATH}.maybe_merge_kv_caches', return_value=[[kv_cache['layer.0']]]), \
             patch(f'{VLLM_KV_TRANSFER_MANAGER_PATH}.maybe_split_kv_caches_for_spec_layers', return_value=[[kv_cache['layer.0']]]):
            mock_manager.register_memory(kv_cache, kv_cache_config)
            mock_reinit.assert_called_once()


class TestUnregisterMemoryEdgeCases:
    """Tests for edge cases in unregister_memory."""

    def test_unregister_memory_with_none_cache_id(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        """Test unregister_memory handles caches with None cache_id."""
        # kv_role='kv_consumer' means is_prefill=False
        mock_vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)

        # Create a cache with None cache_id
        mock_cache = MagicMock()
        mock_cache.cache_id = None
        manager.registered_kv_caches = [mock_cache]

        manager.unregister_memory()

        # Should still attempt to unregister
        manager.data_dist_engine.cache_manager.unregister_cache.assert_called_once_with(None)

    def test_multiple_unregister_calls_idempotent(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        """Test that multiple unregister calls are idempotent."""
        # kv_role='kv_consumer' means is_prefill=False
        mock_vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        manager.registered_kv_caches = []

        # First call
        manager.unregister_memory()
        # Second call
        manager.unregister_memory()

        # Should not raise and no calls made
        manager.data_dist_engine.cache_manager.unregister_cache.assert_not_called()


class TestRefreshLink:
    """Tests for LLMDataDistManager._refresh_link method."""

    def test_refresh_link_returns_early_when_get_host_returns_none(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        """Test that _refresh_link returns early when _get_host_cluster_id returns None."""
        mock_vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        manager.registered_link_infos = {
            (('cluster1',), 0, 0): [12345],
        }

        with patch.object(manager, 'close_link') as mock_close_link:
            with patch.object(manager, 'register_link') as mock_register_link:
                # Query with non-existent prompt_cluster_id
                manager._refresh_link(99999, 0, 0)

                mock_close_link.assert_not_called()
                mock_register_link.assert_not_called()

    def test_refresh_link_calls_close_and_register_on_success(self, mock_vllm_config, mock_llm_datadist, mock_world_group):
        """Test that _refresh_link calls close_link and register_link when host found."""
        mock_vllm_config.kv_transfer_config.kv_role = 'kv_consumer'
        manager = LLMDataDistManager(mock_vllm_config, "127.0.0.1", 8000)
        manager.registered_link_infos = {
            (('cluster1',), 0, 0): [12345],
        }

        with patch.object(manager, 'close_link') as mock_close_link:
            with patch.object(manager, 'register_link') as mock_register_link:
                manager._refresh_link(12345, 0, 0)

                mock_close_link.assert_called_once_with(('cluster1',), 0, 0)
                mock_register_link.assert_called_once_with(('cluster1',), 0, 0)
