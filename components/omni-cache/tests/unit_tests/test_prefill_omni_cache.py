# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_prefill_omni_cache.py
# Comprehensive unit tests for prefill_omni_cache.py

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock, call
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass, field

# Import module under test
from omni_cache.cache.prefill.prefill_omni_cache import PrefillOmniCache


# ==================== Mock classes and fixtures ====================

def _get_attention_spec_cls():
    """Get the AttentionSpec class injected by conftest into vllm."""
    from vllm.v1.kv_cache_interface import AttentionSpec
    return AttentionSpec


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
        # resolve_kv_spec_state reads kv_cache_tensors[0].size
        # initialize_device_cache iterates kv_cache_tensors[i].shared_by
        _mock_tensor = Mock()
        # page_size * num_blocks: e.g. 128*64*2 * 100 = 1638400
        _mock_tensor.size = 1638400
        # shared_by lists the layer names that share this tensor
        _mock_tensor.shared_by = ["layers.0", "layers.1"]
        self.kv_cache_tensors = [_mock_tensor]


class MockVllmConfig:
    """Mock VLLM configuration."""
    def __init__(self, kv_role="kv_producer", num_speculative_tokens=0, is_hybrid=False):
        self.kv_transfer_config = Mock(kv_role=kv_role)
        self.cache_config = Mock(block_size=128)
        self.model_config = Mock(max_model_len=2048, use_mla=False)
        # Set up hf_config with proper attributes for hybrid detection
        if is_hybrid:
            self.model_config.hf_config = Mock(model_type="deepseek_v3", num_hidden_layers=2, compress_ratios=[1, 4])
        else:
            self.model_config.hf_config = Mock(model_type="mock", num_hidden_layers=2, compress_ratios=[1, 1])
        self.scheduler_config = Mock(
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            enable_chunked_prefill=False
        )
        self.parallel_config = Mock(data_parallel_size=1, data_parallel_rank=0)
        self.additional_config = {}
        # Add speculative_config for spec token tests
        if num_speculative_tokens > 0:
            self.speculative_config = Mock(num_speculative_tokens=num_speculative_tokens)
        else:
            self.speculative_config = None


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


class MockHostCache:
    """Mock host cache for testing."""
    def __init__(self):
        self.num_blocks = 100
        self.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]


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
    return MockVllmConfig(kv_role="kv_producer")


@pytest.fixture
def mock_runner():
    """Create a mock NPU model runner."""
    return MockNPUModelRunner()


# ==================== PrefillOmniCache Initialization Tests ====================

class TestPrefillOmniCacheInit:
    """Test suite for PrefillOmniCache initialization."""

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_init_basic(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test basic PrefillOmniCache initialization."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )

        assert isinstance(cache, PrefillOmniCache)
        assert cache.max_num_batched_tokens == 1024
        assert cache.max_num_seqs == 32
        assert cache.max_model_len == 2048

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_init_hybrid_attention(self, mock_pool, mock_get_dp, mock_get_tp, mock_runner):
        """Test PrefillOmniCache initialization with hybrid attention."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        # Setup hybrid attention config with different specs per group
        spec1 = MockAttentionSpec()
        spec2 = MockAttentionSpec()
        group1 = MockKVCacheGroup(layer_names=["layers.0"], kv_cache_spec=spec1)
        group2 = MockKVCacheGroup(layer_names=["layers.1"], kv_cache_spec=spec2)
        hybrid_config = MockKVCacheConfig(kv_cache_groups=[group1, group2], num_blocks=100)

        # Create a proper vllm_config mock that returns real values for hybrid detection
        hybrid_vllm_config = Mock()
        hybrid_vllm_config.kv_transfer_config = Mock(kv_role="kv_producer")
        hybrid_vllm_config.cache_config = Mock(block_size=128)
        hybrid_vllm_config.model_config = Mock(max_model_len=2048)
        # Create a real hf_config object with proper attributes (not Mock)
        class MockHFConfig:
            model_type = "deepseek_v3"
            num_hidden_layers = 2
            compress_ratios = [1, 4]  # Different values to trigger hybrid detection
        hybrid_vllm_config.model_config.hf_config = MockHFConfig()
        hybrid_vllm_config.scheduler_config = Mock(
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            enable_chunked_prefill=False
        )
        hybrid_vllm_config.parallel_config = Mock(data_parallel_size=1, data_parallel_rank=0)
        hybrid_vllm_config.speculative_config = None

        # Patch resolve_kv_spec_state to return a spec_state with is_hybrid_attn=True
        from omni_cache.cache.utils.support import KVSpecState
        mock_spec_state = KVSpecState(
            num_spec_token=0,
            is_hybrid_attn=True,
            num_layers=2,
            block_size=128,
            head_sizes=[256],
            head_size=256,
            dtype=torch.float16,
            block_num=100,
            block_bytes=16384,
        )
        with patch("omni_cache.cache.core.base.resolve_kv_spec_state", return_value=mock_spec_state):
            cache = PrefillOmniCache(
                hybrid_config,
                mock_runner,
                max_num_batched_tokens=1024,
                max_num_seqs=32,
                max_model_len=2048,
                vllm_config=hybrid_vllm_config,
            )

        # is_hybrid_attn is True when model_type contains "deepseek" and compress_ratios differ
        assert cache.is_hybrid_attn == True
        # batch_buffer_cpu should have buffers for each group
        assert len(cache.batch_buffer_cpu) == 2

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_init_with_spec_tokens(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner):
        """Test PrefillOmniCache initialization with speculative tokens."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        # Setup hybrid attention with speculative tokens
        spec1 = MockAttentionSpec()
        spec2 = MockAttentionSpec()
        group1 = MockKVCacheGroup(layer_names=["layers.0"], kv_cache_spec=spec1)
        group2 = MockKVCacheGroup(layer_names=["layers.1"], kv_cache_spec=spec2)
        hybrid_config = MockKVCacheConfig(kv_cache_groups=[group1, group2], num_blocks=100)

        # Create vllm_config with speculative tokens
        mock_vllm_config_with_spec = MockVllmConfig(kv_role="kv_producer", num_speculative_tokens=3)

        cache = PrefillOmniCache(
            hybrid_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config_with_spec,
        )

        assert cache.num_spec_token == 3


# ==================== PrefillOmniCache Cache Shape Tests ====================

class TestPrefillOmniCacheShape:
    """Test suite for cache shape calculations in PrefillOmniCache."""

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_calc_cache_shape(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test calc_cache_shape method."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )

        shape, num_blocks = cache.calc_cache_shape()

        assert len(shape) == 4
        assert shape[0] == 2  # num_layers
        # shape[2] = node_block_size = block_size / tp_nnodes = 128 / 2 = 64 (tp_world_size=16, LOCAL_DP_SIZE=8)
        assert shape[2] == 64  # node_block_size after TP division

    def test_calc_cache_shape_for_prefill(self):
        """Test calc_cache_shape_for_prefill function."""
        from omni_cache.cache.transfer_engine import calc_cache_shape_for_prefill

        shape, num_blocks = calc_cache_shape_for_prefill(
            num_layers=2,
            block_size=128,
            head_size=64,
            dtype=torch.float16,
        )

        assert len(shape) == 4
        assert shape[0] == 2   # num_layers
        assert shape[2] == 128 # block_size
        assert isinstance(num_blocks, int)

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_calculate_kv_xsfer_params(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test calculate_kv_xsfer_params method."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )

        block_len_dtype, dp_offset = cache.calculate_kv_xsfer_params()

        assert isinstance(block_len_dtype, int)
        assert isinstance(dp_offset, int)
        assert dp_offset == 0  # Since dp_world_size_local is 1


# ==================== PrefillOmniCache Device Cache Tests ====================

class TestPrefillOmniCacheDeviceCache:
    """Test suite for device cache operations in PrefillOmniCache."""

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_initialize_device_cache(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test initialize_device_cache method."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )

        # Mock runner methods
        mock_runner._allocate_kv_cache_tensors = Mock(return_value=[])
        mock_runner._prepare_kernel_block_sizes = Mock(return_value=[])
        mock_runner._reshape_kv_cache_tensors = Mock(return_value={"layer": torch.zeros(100, 128, 4, 64)})
        # Use CPU device for testing (npu:0 is not available in test env)
        mock_runner.device = "cpu"

        # Set num_layers high enough so scale_rate > 0 in initialize_device_cache
        # scale_rate = num_layers // (num_stages_layer_copy * 4)
        cache.num_layers = 8

        kv_caches = cache.initialize_device_cache(mock_config, mock_runner)

        assert kv_caches is not None

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_construct_device_cache(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test _construct_device_cache method."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )

        # Test with tensor
        kv_caches = torch.zeros(100, 128, 4, 64)
        device_cache = cache._construct_device_cache(kv_caches, num_stages=1)

        assert device_cache is not None

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_shard_kv_cache_by_dp_rank_single(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test _shard_kv_cache_by_dp_rank with single DP rank."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_get_dp.return_value.world_size = 1  # Single DP rank
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )

        # _init_dp_sharding is already called in __init__, so kvi_tensors is already transformed
        # The method _shard_kv_cache_by_dp_rank is a legacy wrapper that calls _init_dp_sharding
        # We verify the state after initialization
        assert len(cache.host_cache.kvi_tensors) >= 1
        # After _init_dp_sharding, kvi_tensors is a list of tuples (one per tensor component)
        # Each tuple contains one tensor per layer
        assert isinstance(cache.host_cache.kvi_tensors[0], tuple)


# ==================== PrefillOmniCache APC Tests ====================

class TestPrefillOmniCacheAPC:
    """Test suite for APC (Adaptive Prefix Caching) in PrefillOmniCache."""

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_get_prefill_prefix_copy_meta_disabled(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test get_prefill_prefix_copy_meta when chunked_prefill is disabled."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        mock_vllm_config.scheduler_config.enable_chunked_prefill = False

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )

        kv_lens = np.array([10, 20])
        query_lens_list = [5, 10]
        block_tables = np.array([[1, 2, 0], [3, 4, 5]])

        result = cache.get_prefill_prefix_copy_meta(
            mock_vllm_config,
            kv_lens,
            query_lens_list,
            block_tables,
        )

        assert result is None  # Because enable_chunked_prefill is False

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_get_prefill_prefix_copy_meta_no_remainder(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test get_prefill_prefix_copy_meta when remainder is 0."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        mock_vllm_config.scheduler_config.enable_chunked_prefill = True

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )
        cache.block_size = 128

        # When remainder is 0, should return None
        kv_lens = np.array([128, 256])  # Divisible by 128
        query_lens_list = [5, 10]
        block_tables = np.array([[1, 0, 0], [2, 3, 0]])

        result = cache.get_prefill_prefix_copy_meta(
            mock_vllm_config,
            kv_lens,
            query_lens_list,
            block_tables,
        )

        assert result is None

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_get_prefill_prefix_copy_meta_with_remainder(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test get_prefill_prefix_copy_meta with remainder."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        mock_vllm_config.scheduler_config.enable_chunked_prefill = True

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )
        cache.block_size = 128
        cache.device = "cpu"  # Use CPU device for testing

        # With remainder
        kv_lens = np.array([130, 200])  # Not divisible by 128
        query_lens_list = [5, 10]
        block_tables = np.array([[1, 2, 0], [3, 4, 5]])

        result = cache.get_prefill_prefix_copy_meta(
            mock_vllm_config,
            kv_lens,
            query_lens_list,
            block_tables,
        )

        # Should return a PrefixCopyMeta object
        assert result is not None


# ==================== PrefillOmniCache Token Indices Tests ====================

class TestPrefillOmniCacheTokenIndices:
    """Test suite for token indices in PrefillOmniCache."""

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_init_batch_token_indices(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test init_batch_token_indices method."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )
        cache.block_size = 128
        cache.tp_world_size = 16
        cache.tp_rank = 0
        cache.rank_block_size = 8
        cache.node_block_size = 128

        orig_slot_mapping = torch.tensor([0, 1, 129, 130, 256], dtype=torch.int64)

        # Should not raise error
        cache.init_batch_token_indices(orig_slot_mapping)

        assert cache.batch_token_indices is not None

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_init_batch_token_indices_hybrid(self, mock_pool, mock_get_dp, mock_get_tp, mock_runner):
        """Test init_batch_token_indices_hybrid method."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        # Setup hybrid attention with proper vllm_config
        spec1 = MockAttentionSpec()
        spec2 = MockAttentionSpec()
        group1 = MockKVCacheGroup(layer_names=["layers.0"], kv_cache_spec=spec1)
        group2 = MockKVCacheGroup(layer_names=["layers.1"], kv_cache_spec=spec2)
        hybrid_config = MockKVCacheConfig(kv_cache_groups=[group1, group2], num_blocks=100)

        # Create vllm_config with hybrid attention enabled
        hybrid_vllm_config = MockVllmConfig(kv_role="kv_producer", is_hybrid=True)

        cache = PrefillOmniCache(
            hybrid_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=hybrid_vllm_config,
        )

        # Set up additional attributes needed for hybrid token indices
        cache.tp_nnodes = 1
        cache.count_called = 0
        cache.num_called_builder = 2
        cache.num_attn_group = 2
        cache.batch_token_indices = [None, None]
        # batch_slots_cpu must be a list of tensors for hybrid mode
        cache.batch_slots_cpu = [
            torch.zeros(1024, dtype=torch.int32),
            torch.zeros(1024, dtype=torch.int32)
        ]

        orig_slot_mapping = torch.tensor([0, 1, 129, 130, 256], dtype=torch.int64)

        # Should not raise error
        cache.init_batch_token_indices_hybrid(orig_slot_mapping)


# ==================== PrefillOmniCache Helper Methods Tests ====================

class TestPrefillOmniCacheHelpers:
    """Test suite for helper methods in PrefillOmniCache."""

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_layer_name_to_group_and_layer_idx(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test _layer_name_to_group_and_layer_idx method."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )

        grp_idx, layer_idx = cache._layer_name_to_group_and_layer_idx("layers.0")

        assert grp_idx == 0
        assert layer_idx == 0

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_layer_name_to_group_and_layer_idx_invalid(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test _layer_name_to_group_and_layer_idx with invalid layer name."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )

        with pytest.raises(ValueError, match="Invalid layer name"):
            cache._layer_name_to_group_and_layer_idx("invalid_layer")

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_nd_to_nz(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test _nd_to_nz method."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )
        cache.block_size = 128
        cache.sum_total_len = 256
        cache.sum_query_len = 128
        cache.query_mask = torch.zeros(256, dtype=torch.bool)
        cache.query_mask[:128] = True
        cache._nz_size = 64

        tensor = torch.randn(128, 512)

        result = cache._nd_to_nz(tensor)

        assert result is not None


# ==================== PrefillOmniCache Volatile Metadata Tests ====================

class TestPrefillOmniCacheVolatileMetadata:
    """Test suite for volatile metadata in PrefillOmniCache."""

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_get_volatile_metadata(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test get_volatile_metadata method."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        # DSA is detected via vllm_config.model_config.use_mla and hf_config.index_topk
        # Setting use_mla=False means DSA is disabled
        mock_vllm_config.model_config.use_mla = False

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )
        cache.block_size = 128
        cache.tp_world_size = 16
        cache.tp_rank = 0
        cache.rank_block_size = 8
        cache.node_block_size = 128
        cache.arange = torch.arange(2048, dtype=torch.int64)
        cache.volatile_table = torch.arange(100, dtype=torch.int32).view(10, 10)

        query_lens_list = [5, 10]
        seq_lens_list = [15, 25]
        graph_pad_size = 0
        pad_slot_id = -1
        # Slot mapping needs to have enough elements for all query tokens
        # Total query_len = 5 + 10 = 15
        orig_slot_mapping = torch.arange(15, dtype=torch.int64) * 128

        result = cache.get_volatile_metadata(
            query_lens_list,
            seq_lens_list,
            graph_pad_size,
            pad_slot_id,
            orig_slot_mapping,
        )

        # When DSA is disabled, should return None, None
        assert result == (None, None)


# ==================== PrefillOmniCache Synchronization Tests ====================

class TestPrefillOmniCacheSynchronize:
    """Test suite for synchronization methods in PrefillOmniCache."""

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    def test_synchronize_h2d(self, mock_pool, mock_get_dp, mock_get_tp, mock_config, mock_runner, mock_vllm_config):
        """Test synchronize_h2d method."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(2, 100, 128, 4, 64)]

        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )
        cache.num_layers = 2
        cache.device = "cpu"  # Use cpu for testing
        cache.h2d_stream = Mock()
        cache.h2d_event = Mock()

        # Mock PrefixCopyMeta
        prefix_meta = Mock()
        prefix_meta.consecutive_blocks = [[(0, 2)], [(2, 4)]]

        # Mock the transfer_manager's sync_h2d_prefill method
        with patch.object(cache.transfer_manager, 'sync_h2d_prefill') as mock_sync:
            cache.synchronize_h2d(prefix_meta, ["layers.0.self_attn"])
            mock_sync.assert_called_once_with(prefix_meta, ["layers.0.self_attn"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
