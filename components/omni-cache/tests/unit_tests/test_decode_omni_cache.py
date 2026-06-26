# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_decode_omni_cache.py
# Comprehensive unit tests for decode_omni_cache.py

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass

# Import module under test
from omni_cache.cache.decode import DecodeOmniCache


# ==================== Mock classes and fixtures ====================

@dataclass(frozen=True)
class MockAttentionSpec:
    """Mock attention specification for testing - extends AttentionSpec from conftest."""
    block_size: int = 128
    num_kv_heads: int = 4
    head_size: int = 64
    dtype: object = None
    use_mla: bool = False

    def __post_init__(self):
        if self.dtype is None:
            object.__setattr__(self, 'dtype', torch.float16)

    @property
    def page_size_bytes(self) -> int:
        element_size = 2 if self.dtype is None else (self.dtype.itemsize if hasattr(self.dtype, 'itemsize') else 2)
        return self.block_size * self.num_kv_heads * self.head_size * element_size


class MockKVCacheGroup:
    """Mock KV cache group configuration."""
    def __init__(self, layer_names=None, kv_cache_spec=None):
        self.layer_names = layer_names or ["layers.0", "layers.1"]
        self.kv_cache_spec = kv_cache_spec or MockAttentionSpec()


class MockKVCacheConfig:
    """Mock KV cache configuration."""
    def __init__(self, kv_cache_groups=None, num_blocks=100):
        self.kv_cache_groups = kv_cache_groups or [MockKVCacheGroup()]
        self.num_blocks = num_blocks
        mock_tensor = MagicMock()
        mock_tensor.size = 1905733632
        mock_tensor.shared_by = ['model.layers.0.self_attn.attn']
        self.kv_cache_tensors = [mock_tensor]


class MockVllmConfig:
    """Mock VLLM configuration."""
    def __init__(self, kv_role="kv_consumer"):
        self.kv_transfer_config = Mock(kv_role=kv_role)
        self.cache_config = Mock(block_size=128, gpu_memory_utilization=0.9)
        self.model_config = Mock(max_model_len=2048, dtype=torch.float16)
        self.model_config.use_mla = False  # Disable DSA detection
        self.model_config.hf_config = Mock(model_type="mock", num_hidden_layers=2, head_dim=128)
        self.scheduler_config = Mock(max_num_seqs=32)
        self.parallel_config = Mock(data_parallel_size=1)
        self.additional_config = {}
        # Ensure no index_topk attribute to disable DSA
        if hasattr(self.model_config.hf_config, 'index_topk'):
            delattr(self.model_config.hf_config, 'index_topk')


class MockNPUModelRunner:
    """Mock NPU model runner for testing."""
    def __init__(self):
        self.device = "npu:0"
        self.max_model_len = 2048
        self.max_num_reqs = 32
        self.dtype = torch.float16
        self.use_spec_decode = False
        self.graph_block_tables = torch.zeros(32, 128, dtype=torch.int32)
        self.speculative_config = Mock(num_speculative_tokens=5)
        self.vllm_config = Mock(
            scheduler_config=Mock(max_num_seqs=32),
            cache_config=Mock(gpu_memory_utilization=0.9)
        )

    def _allocate_kv_cache_tensors(self, kv_cache_config):
        """Mock allocate KV cache tensors."""
        return []

    def _reshape_kv_cache_tensors(self, kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes):
        """Mock reshape KV cache tensors."""
        return {}

    def _prepare_kernel_block_sizes(self, kv_cache_config):
        """Mock prepare kernel block sizes."""
        return []


class MockHostCache:
    """Mock host cache for testing."""
    def __init__(self):
        self.num_blocks = 100
        self.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

    def batch_layer_copy_to_npu(self, local_block_ids, npu_blocks, layer_indices=None):
        """Mock batch layer copy to NPU."""
        return [], [], [], []

    def memcpy_async(self, batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes):
        pass


@pytest.fixture
def mock_config():
    """Create a mock KV cache config."""
    spec = MockAttentionSpec()
    group = MockKVCacheGroup(layer_names=["layers.0", "layers.1"], kv_cache_spec=spec)
    config = MockKVCacheConfig(kv_cache_groups=[group], num_blocks=100)
    return config


@pytest.fixture
def mock_vllm_config():
    """Create a mock VLLM config."""
    return MockVllmConfig(kv_role="kv_consumer")


@pytest.fixture
def mock_runner():
    """Create a mock NPU model runner."""
    return MockNPUModelRunner()


# ==================== DecodeOmniCache H2D Operations Tests ====================

class TestDecodeOmniCacheH2D:
    """Test suite for H2D operations in DecodeOmniCache."""

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool._map_hugepage_memory")
    @patch("omni_cache.cache.decode.decode_omni_cache.isinstance", return_value=True)
    def test_build_h2d_ops_basic(self, mock_instance, mock_hugepage, mock_pool_tp,
                                  mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test basic build_h2d_ops functionality."""
        # Setup mocks
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_get_dp.return_value.world_size = 4
        mock_pool_tp.return_value.world_size = 16

        # Create cache instance
        cache = DecodeOmniCache(mock_config, mock_runner, vllm_config=mock_vllm_config)
        cache.host_cache = MockHostCache()
        cache.device_cache = {
            "layers.0": (torch.zeros(100, 128, 4, 64), torch.zeros(100, 128, 4, 64)),
            "layers.1": (torch.zeros(100, 128, 4, 64), torch.zeros(100, 128, 4, 64)),
        }
        # Set required attributes for build_h2d_ops
        cache.num_layers = 2

        # Test basic H2D ops
        local_block_ids = [[1, 2, 3], [4, 5]]
        result = cache.build_h2d_ops(local_block_ids, "test_request_id")

        # Result should be a tuple of 4 elements (possibly empty lists)
        assert result is not None
        assert len(result) == 4  # batch_device_mem, batch_device_max, batch_host_mem, batch_host_sizes

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool._map_hugepage_memory")
    @patch("omni_cache.cache.decode.decode_omni_cache.isinstance", return_value=True)
    def test_build_h2d_ops_hybrid(self, mock_instance, mock_hugepage, mock_pool_tp,
                                   mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test build_h2d_ops_hybrid with hybrid attention."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_get_dp.return_value.world_size = 4
        mock_pool_tp.return_value.world_size = 16

        # Setup hybrid attention config
        spec1 = MockAttentionSpec()
        spec2 = MockAttentionSpec()
        group1 = MockKVCacheGroup(layer_names=["layers.0"], kv_cache_spec=spec1)
        group2 = MockKVCacheGroup(layer_names=["layers.1"], kv_cache_spec=spec2)
        hybrid_config = MockKVCacheConfig(kv_cache_groups=[group1, group2], num_blocks=100)

        cache = DecodeOmniCache(hybrid_config, mock_runner, vllm_config=mock_vllm_config)
        cache.is_hybrid_attn = True
        cache.ENABLE_HOST_MAPPING = False
        cache.host_cache = MockHostCache()
        cache.device_cache = {"layers.0": torch.zeros(100, 128, 4, 64), "layers.1": torch.zeros(100, 128, 4, 64)}

        # local_block_ids[0] = host blocks, local_block_ids[1] = list of block ID lists per group
        local_block_ids = [[1, 2], [[3, 4], [5, 6]]]
        result = cache.build_h2d_ops_hybrid(local_block_ids)

        assert result is not None
        assert len(result) == 4

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool._map_hugepage_memory")
    @patch("omni_cache.cache.decode.decode_omni_cache.isinstance", return_value=True)
    def test_build_h2d_ops_omni_attn(self, mock_instance, mock_hugepage, mock_pool_tp,
                                      mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test build_h2d_ops_omni_attn with omni attention."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_get_dp.return_value.world_size = 4
        mock_pool_tp.return_value.world_size = 16

        cache = DecodeOmniCache(mock_config, mock_runner, vllm_config=mock_vllm_config)
        cache.enable_omni_attn = True
        cache.sink_blocks = 1
        cache.recent_blocks = 3
        cache.omni_attn_pattern = [0, 1]
        cache.grouped_layer_indices = [[0], [1]]
        cache.layer_indices = {"layers.0": 0, "layers.1": 1}
        cache.sorted_layer_names = ["layers.0", "layers.1"]
        cache.host_cache = MockHostCache()
        cache.device_cache = {
            "layers.0": (torch.zeros(100, 128, 4, 64), torch.zeros(100, 128, 4, 64)),
            "layers.1": (torch.zeros(100, 128, 4, 64), torch.zeros(100, 128, 4, 64)),
        }

        # Test with valid omni attention block IDs
        local_block_ids = [[1, 2, 3, 4], [5, 6, 7, 8]]
        result = cache.build_h2d_ops_omni_attn(local_block_ids)

        assert result is not None
        assert len(result) == 4

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool._map_hugepage_memory")
    @patch("omni_cache.cache.decode.decode_omni_cache.isinstance", return_value=True)
    def test_build_h2d_ops_omni_attn_wrong_length(self, mock_instance, mock_hugepage, mock_pool_tp,
                                                   mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test build_h2d_ops_omni_attn with wrong block IDs length."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_get_dp.return_value.world_size = 4
        mock_pool_tp.return_value.world_size = 16

        cache = DecodeOmniCache(mock_config, mock_runner, vllm_config=mock_vllm_config)
        cache.enable_omni_attn = True
        cache.sink_blocks = 1
        cache.recent_blocks = 3
        # Mock device_cache to avoid KeyError in ensure_device_cache_initialized
        cache.device_cache = {"layers.0": (torch.zeros(100, 128, 4, 64), torch.zeros(100, 128, 4, 64)),
                               "layers.1": (torch.zeros(100, 128, 4, 64), torch.zeros(100, 128, 4, 64))}
        cache.sorted_layer_names = ["layers.0", "layers.1"]
        cache.layer_indices = {"layers.0": 0, "layers.1": 1}
        cache.omni_attn_pattern = [0, 1]
        cache.grouped_layer_indices = [[0], [1]]
        cache.host_cache = MockHostCache()

        # Test with wrong number of block tables
        local_block_ids = [[1, 2, 3]]  # Only one group

        with pytest.raises(RuntimeError, match="Omni attention needs two block tables"):
            cache.build_h2d_ops_omni_attn(local_block_ids)


# ==================== DecodeOmniCache Cache Shape Tests ====================

class TestDecodeOmniCacheShape:
    """Test suite for cache shape calculations in DecodeOmniCache."""

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool._map_hugepage_memory")
    @patch("omni_cache.cache.decode.decode_omni_cache.isinstance", return_value=True)
    def test_calc_cache_shape_for_decode(self, mock_instance, mock_hugepage, mock_pool_tp,
                                          mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test calc_cache_shape_for_decode function."""
        from omni_cache.cache.transfer_engine import calc_cache_shape_for_decode

        shape, num_blocks = calc_cache_shape_for_decode(
            num_layers=2,
            block_size=128,
            head_size=64,
            dtype=torch.float16,
        )

        assert len(shape) == 5
        # shape[0] is local_dp_size (typically 1)
        assert shape[1] == 2   # num_layers
        assert shape[3] == 128 # block_size

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool._map_hugepage_memory")
    @patch("omni_cache.cache.decode.decode_omni_cache.isinstance", return_value=True)
    def test_calc_cache_shape(self, mock_instance, mock_hugepage, mock_pool_tp,
                               mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test instance method calc_cache_shape."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_get_dp.return_value.world_size = 4
        mock_pool_tp.return_value.world_size = 16

        cache = DecodeOmniCache(mock_config, mock_runner, vllm_config=mock_vllm_config)
        shape, num_blocks = cache.calc_cache_shape()

        assert len(shape) == 5
        assert shape[0] == cache.local_dp_size

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool._map_hugepage_memory")
    @patch("omni_cache.cache.decode.decode_omni_cache.isinstance", return_value=True)
    def test_calculate_kv_xsfer_params(self, mock_instance, mock_hugepage, mock_pool_tp,
                                        mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test calculate_kv_xsfer_params method."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_get_dp.return_value.world_size = 4
        mock_pool_tp.return_value.world_size = 16

        cache = DecodeOmniCache(mock_config, mock_runner, vllm_config=mock_vllm_config)
        block_len_dtype, dp_offset = cache.calculate_kv_xsfer_params()

        assert isinstance(block_len_dtype, int)
        assert isinstance(dp_offset, int)


# ==================== DecodeOmniCache Block Table Tests ====================

class TestDecodeOmniCacheBlockTable:
    """Test suite for block table operations in DecodeOmniCache."""

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool._map_hugepage_memory")
    @patch("omni_cache.cache.decode.decode_omni_cache.isinstance", return_value=True)
    def test_get_block_table_np(self, mock_instance, mock_hugepage, mock_pool_tp,
                                 mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test function get_block_table_np from static_utils."""
        from omni_cache.cache.decode.static_utils import get_block_table_np

        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_get_dp.return_value.world_size = 4
        mock_pool_tp.return_value.world_size = 16

        cache = DecodeOmniCache(mock_config, mock_runner, vllm_config=mock_vllm_config)
        cache.blk_table_buffers = {}
        cache.slot_mapping_buffers = {}
        cache.num_spec_token = 5

        # Create mock block table
        blk_table = Mock()
        # Return array with correct shape (2, 128) to match buffer shape
        blk_table.get_numpy_array = Mock(return_value=np.zeros((2, 128), dtype=np.int32))

        blk_table_np, slot_mapping_np = get_block_table_np(
            mock_runner, cache, blk_table, 0,
            torch.zeros(2, 128, dtype=torch.int32),
            2, 10
        )

        assert blk_table_np is not None
        assert slot_mapping_np is not None


# ==================== DecodeOmniCache Initialize Tests ====================

class TestDecodeOmniCacheInitialize:
    """Test suite for initialization methods in DecodeOmniCache."""

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool._map_hugepage_memory")
    @patch("omni_cache.cache.decode.decode_omni_cache.isinstance", return_value=True)
    @patch("omni_cache.cache.core.base.create_omni_cache")
    def test_initialize_decode_omni_cache(self, mock_create, mock_instance, mock_hugepage,
                                           mock_pool_tp, mock_get_dp, mock_get_tp):
        """Test static method initialize_decode_omni_cache."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_get_dp.return_value.world_size = 4
        mock_pool_tp.return_value.world_size = 16

        vllm_config = MockVllmConfig()
        model_runner = MockNPUModelRunner()

        # Mock the imports and torch.npu.mem_get_info
        with patch("torch.npu.mem_get_info", return_value=(0, 1000000000)):
            with patch("vllm.v1.core.kv_cache_utils.get_kv_cache_configs", return_value=[Mock()]):
                with patch("vllm.v1.kv_cache_interface.FullAttentionSpec"):
                    DecodeOmniCache.initialize_decode_omni_cache(vllm_config, model_runner)

        mock_create.assert_called_once()

# ==================== DecodeOmniCache Error Handling Tests ====================

class TestDecodeOmniCacheErrors:
    """Test suite for error handling in DecodeOmniCache."""

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool._map_hugepage_memory")
    @patch("omni_cache.cache.decode.decode_omni_cache.isinstance", return_value=True)
    def test_synchronize_d2h_not_implemented(self, mock_instance, mock_hugepage, mock_pool_tp,
                                              mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test that synchronize_d2h raises NotImplementedError."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_get_dp.return_value.world_size = 4
        mock_pool_tp.return_value.world_size = 16

        cache = DecodeOmniCache(mock_config, mock_runner, vllm_config=mock_vllm_config)

        with pytest.raises(NotImplementedError):
            cache.synchronize_d2h(
                attn_names=["layer.0"],
                attn_metadatas=["metadata"],
                kv_event=Mock(),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
