# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_hd_kv_manager.py
# Unit tests for HostDeviceKVCacheManager class

import pytest
import os
import sys
from unittest.mock import Mock, patch, MagicMock
from collections import defaultdict

# The conftest.py already sets up mock modules for vllm
# We just need to import torch and the module under test
import torch


class MockKVCacheSpec:
    """Mock KV cache spec for testing."""
    def __init__(self):
        self.block_size = 16
        self.num_kv_heads = 8
        self.head_size = 128
        self.dtype = torch.float16


class MockKVCacheGroup:
    """Mock KV cache group for testing."""
    def __init__(self):
        self.kv_cache_spec = MockKVCacheSpec()
        self.layer_names = [f"layer.{i}" for i in range(4)]


class MockKVCacheConfig:
    """Mock KV cache config for testing."""
    def __init__(self):
        self.kv_cache_groups = [MockKVCacheGroup()]
        self.num_blocks = 100


# Import the module under test after setting up mocks
from omni_cache.attention.kv_managers.hd_kv_manager import HostDeviceKVCacheManager


class TestHostDeviceKVCacheManagerInit:
    """Test suite for HostDeviceKVCacheManager initialization."""

    @pytest.fixture
    def mock_config(self):
        """Create mock configuration."""
        return MockKVCacheConfig()

    @patch("omni_cache.attention.kv_managers.hd_kv_manager.PrefillOmniCache")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.BlockPool")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.get_manager_for_kv_cache_spec")
    def test_init_prefill_role(
        self,
        mock_get_manager,
        mock_block_pool,
        mock_prefill_cache,
        mock_config
    ):
        """Test initialization with prefill role."""
        with patch.dict(os.environ, {"ROLE": "prefill", "TP_NNODES": "1", "ENABLE_DSA": "0"}, clear=True):
            # Setup mocks
            mock_prefill_cache.calc_cache_shape_for_prefill.return_value = (1, 50)
            mock_block_pool_instance = Mock()
            mock_block_pool.return_value = mock_block_pool_instance
            mock_manager_instance = Mock()
            mock_get_manager.return_value = mock_manager_instance

            # Create manager
            manager = HostDeviceKVCacheManager(
                kv_cache_config=mock_config,
                max_model_len=4096,
                enable_caching=True,
                caching_hash_algo="sha256",
                use_eagle=False,
                log_stats=False,
            )

            # Verify initialization
            assert manager.block_size == 16
            assert manager.num_gpu_blocks == 100
            assert manager.max_model_len == 4096
            assert manager.enable_caching is True
            assert manager.use_eagle is False
            assert manager.log_stats is False
            assert manager.tp_nnodes == 1

            # Verify block pool was created for prefill (1 pool)
            mock_block_pool.assert_called_once()
            assert len(manager.block_pools) == 1

            # Verify hybrid managers were created
            assert len(manager.hybrid_managers) == 1
            mock_get_manager.assert_called_once()

    @patch("omni_cache.attention.kv_managers.hd_kv_manager.DecodeOmniCache")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.BlockPool")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.get_manager_for_kv_cache_spec")
    def test_init_decode_role(
        self,
        mock_get_manager,
        mock_block_pool,
        mock_decode_cache,
        mock_config
    ):
        """Test initialization with decode role."""
        with patch.dict(os.environ, {"ROLE": "decode", "TP_NNODES": "1", "ENABLE_DSA": "0"}, clear=True):
            # Setup mocks
            mock_decode_cache.calc_cache_shape_for_decode.return_value = (1, 80)
            mock_block_pool_instance = Mock()
            mock_block_pool.return_value = mock_block_pool_instance
            mock_manager_instance = Mock()
            mock_get_manager.return_value = mock_manager_instance

            # Create manager
            manager = HostDeviceKVCacheManager(
                kv_cache_config=mock_config,
                max_model_len=4096,
                enable_caching=True,
            )

            # Verify block pool was created for decode (2 pools)
            assert mock_block_pool.call_count == 2
            assert len(manager.block_pools) == 2

            # Verify hybrid managers were created
            assert len(manager.hybrid_managers) == 2
            assert mock_get_manager.call_count == 2

    def test_init_multiple_kv_groups_raises(self, mock_config):
        """Test that multiple kv cache groups raises assertion error."""
        # Add a second kv cache group
        mock_config.kv_cache_groups.append(MockKVCacheGroup())

        with patch.dict(os.environ, {"TP_NNODES": "1", "ENABLE_DSA": "0"}, clear=True):
            with pytest.raises(AssertionError, match="does not support hybrid models"):
                HostDeviceKVCacheManager(
                    kv_cache_config=mock_config,
                    max_model_len=4096,
                )

    @patch("omni_cache.attention.kv_managers.hd_kv_manager.PrefillOmniCache")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.BlockPool")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.get_manager_for_kv_cache_spec")
    def test_init_with_dsa_enabled(
        self,
        mock_get_manager,
        mock_block_pool,
        mock_prefill_cache,
        mock_config
    ):
        """Test initialization with DSA enabled."""
        with patch.dict(os.environ, {"ROLE": "prefill", "TP_NNODES": "1", "ENABLE_DSA": "1"}, clear=True):
            mock_prefill_cache.calc_cache_shape_for_prefill.return_value = (1, 50)
            mock_block_pool.return_value = Mock()
            mock_get_manager.return_value = Mock()

            manager = HostDeviceKVCacheManager(
                kv_cache_config=mock_config,
                max_model_len=4096,
            )

            # Verify DSA head_size calculation was used
            # head_size = sum([512, 64, 128, 1]) - 1 = 704
            mock_prefill_cache.calc_cache_shape_for_prefill.assert_called_once()
            call_kwargs = mock_prefill_cache.calc_cache_shape_for_prefill.call_args.kwargs
            assert call_kwargs['head_size'] == 704

    @patch("omni_cache.attention.kv_managers.hd_kv_manager.PrefillOmniCache")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.BlockPool")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.get_manager_for_kv_cache_spec")
    def test_init_head_size_not_divisible_raises(
        self,
        mock_get_manager,
        mock_block_pool,
        mock_prefill_cache,
        mock_config
    ):
        """Test that head_size not divisible by tp_nnodes raises ValueError."""
        with patch.dict(os.environ, {"ROLE": "prefill", "TP_NNODES": "3", "ENABLE_DSA": "0"}, clear=True):
            with pytest.raises(ValueError, match="head_size .* is not divisible by tp_nnodes"):
                HostDeviceKVCacheManager(
                    kv_cache_config=mock_config,
                    max_model_len=4096,
                )


class TestHostDeviceKVCacheManagerAttributes:
    """Test suite for HostDeviceKVCacheManager attributes."""

    @pytest.fixture
    def mock_manager(self):
        """Create a mock initialized manager."""
        with patch.dict(os.environ, {"ROLE": "decode", "TP_NNODES": "1", "ENABLE_DSA": "0"}, clear=True):
            with patch("omni_cache.attention.kv_managers.hd_kv_manager.DecodeOmniCache") as mock_decode:
                with patch("omni_cache.attention.kv_managers.hd_kv_manager.BlockPool") as mock_block:
                    with patch("omni_cache.attention.kv_managers.hd_kv_manager.get_manager_for_kv_cache_spec") as mock_get:
                        mock_decode.calc_cache_shape_for_decode.return_value = (1, 80)
                        mock_block.return_value = Mock()
                        mock_get.return_value = Mock()

                        config = MockKVCacheConfig()
                        manager = HostDeviceKVCacheManager(
                            kv_cache_config=config,
                            max_model_len=4096,
                            enable_caching=True,
                            caching_hash_algo="sha256",
                            use_eagle=True,
                            log_stats=True,
                        )
                        return manager

    def test_caching_hash_fn_sha256(self, mock_manager):
        """Test that sha256 hash function is set correctly."""
        # The mock sha256 is set to hash function
        from vllm.utils.hashing import sha256
        assert mock_manager.caching_hash_fn is sha256

    def test_prefix_cache_stats_when_log_stats(self, mock_manager):
        """Test that prefix_cache_stats is created when log_stats is True."""
        assert mock_manager.prefix_cache_stats is not None

    def test_req_to_block_hashes_defaultdict(self, mock_manager):
        """Test that req_to_block_hashes is a defaultdict."""
        assert isinstance(mock_manager.req_to_block_hashes, defaultdict)
        # Test default value is empty list
        assert mock_manager.req_to_block_hashes["nonexistent_req"] == []


class TestHostDeviceKVCacheManagerCaching:
    """Test suite for HostDeviceKVCacheManager caching behavior."""

    @patch("omni_cache.attention.kv_managers.hd_kv_manager.DecodeOmniCache")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.BlockPool")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.get_manager_for_kv_cache_spec")
    def test_init_caching_disabled(
        self,
        mock_get_manager,
        mock_block_pool,
        mock_decode_cache,
    ):
        """Test initialization with caching disabled."""
        with patch.dict(os.environ, {"ROLE": "decode", "TP_NNODES": "1", "ENABLE_DSA": "0"}, clear=True):
            mock_decode_cache.calc_cache_shape_for_decode.return_value = (1, 80)
            mock_block_pool.return_value = Mock()
            mock_get_manager.return_value = Mock()

            config = MockKVCacheConfig()
            manager = HostDeviceKVCacheManager(
                kv_cache_config=config,
                max_model_len=4096,
                enable_caching=False,
            )

            assert manager.enable_caching is False

    @patch("omni_cache.attention.kv_managers.hd_kv_manager.DecodeOmniCache")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.BlockPool")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.get_manager_for_kv_cache_spec")
    def test_builtin_hash_algo(
        self,
        mock_get_manager,
        mock_block_pool,
        mock_decode_cache,
    ):
        """Test initialization with builtin hash algorithm."""
        with patch.dict(os.environ, {"ROLE": "decode", "TP_NNODES": "1", "ENABLE_DSA": "0"}, clear=True):
            mock_decode_cache.calc_cache_shape_for_decode.return_value = (1, 80)
            mock_block_pool.return_value = Mock()
            mock_get_manager.return_value = Mock()

            config = MockKVCacheConfig()
            manager = HostDeviceKVCacheManager(
                kv_cache_config=config,
                max_model_len=4096,
                caching_hash_algo="builtin",
            )

            assert manager.caching_hash_fn is hash


class TestHostDeviceKVCacheManagerDecodeBlocks:
    """Test suite for decode role block handling."""

    @patch("omni_cache.attention.kv_managers.hd_kv_manager.DecodeOmniCache")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.BlockPool")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.get_manager_for_kv_cache_spec")
    def test_decode_num_host_blocks_less_than_gpu(
        self,
        mock_get_manager,
        mock_block_pool,
        mock_decode_cache,
    ):
        """Test decode when num_host_blocks < num_gpu_blocks."""
        with patch.dict(os.environ, {"ROLE": "decode", "TP_NNODES": "1", "ENABLE_DSA": "0"}, clear=True):
            # Return fewer host blocks than gpu blocks
            mock_decode_cache.calc_cache_shape_for_decode.return_value = (1, 50)
            mock_block_pool.return_value = Mock()
            mock_get_manager.return_value = Mock()

            config = MockKVCacheConfig()
            config.num_blocks = 100  # gpu blocks

            manager = HostDeviceKVCacheManager(
                kv_cache_config=config,
                max_model_len=4096,
            )

            # num_host_blocks (50) < num_gpu_blocks (100), so num_gpu_blocks should be reduced
            assert manager.num_gpu_blocks == 50

    @patch("omni_cache.attention.kv_managers.hd_kv_manager.DecodeOmniCache")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.BlockPool")
    @patch("omni_cache.attention.kv_managers.hd_kv_manager.get_manager_for_kv_cache_spec")
    def test_decode_num_host_blocks_more_than_gpu(
        self,
        mock_get_manager,
        mock_block_pool,
        mock_decode_cache,
    ):
        """Test decode when num_host_blocks >= num_gpu_blocks."""
        with patch.dict(os.environ, {"ROLE": "decode", "TP_NNODES": "1", "ENABLE_DSA": "0"}, clear=True):
            # Return more host blocks than gpu blocks
            mock_decode_cache.calc_cache_shape_for_decode.return_value = (1, 150)
            mock_block_pool.return_value = Mock()
            mock_get_manager.return_value = Mock()

            config = MockKVCacheConfig()
            config.num_blocks = 100  # gpu blocks

            manager = HostDeviceKVCacheManager(
                kv_cache_config=config,
                max_model_len=4096,
            )

            # num_host_blocks (150) >= num_gpu_blocks (100), so num_host_blocks should be capped
            assert manager.num_gpu_blocks == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
