# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_omni_cache_base.py

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass
from typing import List, Dict

# Try to import torch, skip tensor tests if not available
try:
    import torch
    # Check if torch is real (not a mock)
    _REAL_TORCH = hasattr(torch, '_C') or hasattr(torch, 'Tensor') and hasattr(torch.Tensor, '__torch_function__')
except ImportError:
    _REAL_TORCH = False

# Import module under test
from omni_cache.cache.core.base import (
    BaseOmniCache,
    create_omni_cache,
    PrefixCopyMeta,
    divide_or_raise,
)
from omni_cache.cache.prefill import PrefillOmniCache
from omni_cache.cache.decode import DecodeOmniCache
from omni_cache.cache.utils.ops import (
    _is_hybrid_attention_enabled,
    generate_full_block_slot,
    pad_inputs,
    pad_tensor,
)

# Skip decorator for tests requiring real torch
requires_torch = pytest.mark.skipif(not _REAL_TORCH, reason="Requires real PyTorch installation")


# ==================== Test helper functions ====================

def test_divide_or_raise():
    """Test divide_or_raise function for integer division with validation."""
    assert divide_or_raise(10, 2) == 5
    assert divide_or_raise(0, 5) == 0
    with pytest.raises(ValueError):
        divide_or_raise(10, 3)


@requires_torch
def test_pad_tensor():
    """Test tensor padding functionality."""
    tensor = torch.tensor([1, 2, 3])
    padded = pad_tensor(tensor, 2, 0)
    assert padded.shape == (5,)
    assert torch.all(padded[3:] == 0)

    tensor_2d = torch.tensor([[1, 2], [3, 4]])
    padded_2d = pad_tensor(tensor_2d, 1, -1)
    assert padded_2d.shape == (3, 2)
    assert torch.all(padded_2d[2] == -1)


@requires_torch
def test_generate_full_block_slot():
    """Test generation of full block slot mappings."""
    slot_mapping = torch.tensor([0, 1, 2, 3, 4, 5])
    query_lens = [2, 4]
    block_size = 3
    result = generate_full_block_slot(slot_mapping, query_lens, block_size)
    expected = torch.tensor([0, 1, 2, 0, 1, 2, 3, 4, 5])
    assert torch.all(result == expected)


@requires_torch
def test_pad_inputs():
    """Test padding of input tensors with query lengths and sparsity size."""
    input_tensor = torch.arange(10).float()
    query_lens = [3, 4, 3]
    sp_size = 4
    pad_value = -1
    result = pad_inputs(input_tensor, query_lens, sp_size, pad_value)
    assert result.shape[0] == 12
    expected = torch.tensor([0, 1, 2, -1, 3, 4, 5, 6, 7, 8, 9, - 1])  # 3 + 4 + 3 + padding(2) + padding(1) + padding(1)
    assert torch.all(result == expected)


@requires_torch
def test_utils():
    # 测试 divide_or_raise
    assert divide_or_raise(10, 2) == 5
    with pytest.raises(ValueError):
        divide_or_raise(10, 3)

    # 测试 generate_full_block_slot
    slot_mapping = torch.tensor([0, 16], dtype=torch.int64)  # 假设 block_size=16
    query_lens = [1, 1]
    res = generate_full_block_slot(slot_mapping, query_lens, 16)
    assert res.shape[0] == 32  # 2 blocks * 16 slots

    # 测试 pad_inputs
    input_tensor = torch.ones(5)
    res = pad_inputs(input_tensor, [5], sp_size=4, pad_value=0)
    assert res.shape[0] == 8  # 5 -> 8 (align to 4)

@dataclass
class MockAttentionSpec:
    """Mock attention specification for testing."""
    block_size: int = 128
    num_kv_heads: int = 4
    head_size: int = 64
    dtype: torch.dtype = torch.float16
    use_mla: bool = False
    page_size_bytes: int = None  # Optional override; computed if None

    def __post_init__(self):
        if self.page_size_bytes is None:
            element_size = 2
            if self.dtype is not None and hasattr(self.dtype, 'itemsize'):
                element_size = self.dtype.itemsize
            self.page_size_bytes = self.block_size * self.num_kv_heads * self.head_size * element_size


@dataclass
class MockKVCacheGroup:
    """Mock KV cache group configuration."""
    layer_names: List[str] = None
    kv_cache_spec: MockAttentionSpec = None


@dataclass
class MockKVCacheConfig:
    """Mock KV cache configuration."""
    kv_cache_groups: List[MockKVCacheGroup] = None
    tensors: Dict[str, any] = None #= {0:torch.randn(1024, 512)}
    num_blocks = 1024
    kv_cache_tensors = [MagicMock(size=1024 * 1024)]


class MockVllmConfig:
    """Mock VLLM configuration with concrete defaults used by tests."""

    def __init__(self):
        self.kv_transfer_config = Mock(kv_role="kv_producer")
        self.scheduler_config = Mock(
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            enable_chunked_prefill=False,
        )
        self.compilation_config = Mock(static_forward_context={})
        self.parallel_config = Mock(data_parallel_size=1, data_parallel_rank=0)
        self.model_config = Mock(max_model_len=2048)
        self.model_config.use_mla = False  # Disable DSA detection
        self.model_config.hf_config = Mock(model_type="mock", compress_ratios=[1, 1])
        # Ensure no index_topk attribute to disable DSA
        if hasattr(self.model_config.hf_config, 'index_topk'):
            delattr(self.model_config.hf_config, 'index_topk')
        self.speculative_config = None


class MockNPUModelRunner:
    """Mock NPU model runner for testing."""

    def __init__(self):
        self.device = "npu:0"
        self.max_model_len = 2048
        self.max_num_reqs = 32
        self.dtype = torch.float16
        self.model_config = Mock()
        self.enable_torchair_graph_mode = False
        self.attn_backends = [Mock()]
        self.graph_block_tables = torch.zeros(32, 128, dtype=torch.int32)
        self.use_spec_decode = False
        self.speculative_config = Mock(num_speculative_tokens=5)
        self.kv_caches = [None] * 10
        self.attn_backends[0].get_kv_cache_shape = Mock(return_value=(256, 128, 4, 64))
        self.attn_backends[0].init_kv_cache_each_layer = Mock(
            return_value=(torch.zeros(256, 128, 4, 64), torch.zeros(256, 128, 4, 64))
        )

    def _allocate_kv_cache_tensors(self, *args, **kwargs):
        """Mock method for allocating KV cache tensors."""
        return torch.zeros(256, 128, 4, 64)

    def _prepare_kernel_block_sizes(self, *args, **kwargs):
        """Mock method for preparing kernel block sizes."""
        return (1, 1)

    def _reshape_kv_cache_tensors(self, *args, **kwargs):
        """Mock method for reshaping KV cache tensors."""
        return {"layers.0": torch.zeros(100, 128, 4, 64), "layers.1": torch.zeros(100, 128, 4, 64)}


class TestPrefillOmniCache:
    @pytest.fixture
    def mock_config(self):
        spec = MockAttentionSpec()
        group = MockKVCacheGroup(layer_names=["layers.0", "layers.1"], kv_cache_spec=spec)
        config = MockKVCacheConfig(kv_cache_groups=[group], tensors={})
        return config

    @patch("omni_cache.cache.transfer_engine.manager.TransferManager.initialize_prefill")
    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    @patch("omni_cache.cache.memory.memory_pool.open_hugepage_file", return_value=-1)
    def test_init_with_exception(self, mock_open_hugepage, mock_pool, mock_get_dp, mock_get_tp, mock_init_prefill, mock_config):
        """Test prefill cache init under minimal mocked runtime."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 2
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(1, 1, 1, 1)]
        mock_runner = MockNPUModelRunner()
        mock_vllm_config = MockVllmConfig()
        cache = PrefillOmniCache(
            mock_config,
            mock_runner,
            max_num_batched_tokens=1024,
            max_num_seqs=32,
            max_model_len=2048,
            vllm_config=mock_vllm_config,
        )
        assert isinstance(cache, PrefillOmniCache)

    @patch("omni_cache.cache.transfer_engine.manager.TransferManager.initialize_prefill")
    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    @patch("omni_cache.cache.memory.memory_pool.open_hugepage_file", return_value=-1)
    def test_calc_cache_shape(self, mock_open_hugepage, mock_pool, mock_get_dp, mock_get_tp, mock_init_prefill, mock_config):
        """Test cache shape calculation for prefill mode."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_pool.return_value.ascend_cl_stream = None
        mock_pool.return_value.shared_tensor_npu = None
        mock_pool.return_value.kvi_tensors = [torch.zeros(1, 1, 1, 1)]
        mock_runner = MockNPUModelRunner()
        mock_vllm_config = MockVllmConfig()
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
        assert shape[3] == 256  # head_size * num_kv_heads / tp_world_size factor

    @pytest.fixture
    def setup_args(self):
        # 模拟 runner �?config
        runner = MagicMock()
        runner.device = "npu:0"
        runner.max_model_len = 100
        runner.max_num_reqs = 2

        # 模拟后端初始化返回元�?        runner.attn_backends[0].init_kv_cache_each_layer.return_value = (torch.randn(1), torch.randn(1))

        config = MagicMock()
        spec = MagicMock()
        spec.block_size = 16
        spec.head_size = 128
        spec.num_kv_heads = 1
        spec.dtype = torch.float16
        config.kv_cache_groups = [MagicMock(kv_cache_spec=spec, layer_names=["l1"])]
        return config, runner

# ==================== DecodeOmniCache tests ====================

class TestDecodeOmniCache:
    @pytest.fixture
    def mock_config(self):
        spec = MockAttentionSpec()
        spec.page_size_bytes = 1024
        spec.dtype = torch.bfloat16
        group = MockKVCacheGroup(layer_names=["layers.0", "layers.1"], kv_cache_spec=spec)
        config = MockKVCacheConfig(
            kv_cache_groups=[group],
            tensors={"layers.0": Mock(size=1024 * 1024), "layers.1": Mock(size=1024 * 1024)}
        )
        return config

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.KVCacheMemoryPool._map_hugepage_memory")
    @patch("omni_cache.cache.core.base.isinstance", return_value=True)
    def test_init(self, mock_instance, mock_hugepage, mock_get_tp_pool, mock_get_dp, mock_get_tp, mock_config):
        """Test cache shape calculation for decode mode."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_get_dp.return_value.world_size = 4
        mock_get_tp_pool.return_value.world_size = 16
        mock_runner = MockNPUModelRunner()
        mock_vllm_config = MockVllmConfig()
        cache = DecodeOmniCache(mock_config, mock_runner, vllm_config=mock_vllm_config)
        shape, num_blocks = cache.calc_cache_shape()
        assert len(shape) == 5
        assert shape[0] == cache.local_dp_size
        assert shape[1] == 2  # num_layers
        assert shape[3] == 128  # block_size
        assert shape[4] == 256

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.KVCacheMemoryPool._map_hugepage_memory")
    @patch("omni_cache.cache.core.base.isinstance", return_value=True)
    def test_init_with_dsa(self, mock_instance, mock_hugepage, mock_get_tp_pool, mock_get_dp, mock_get_tp, mock_config):
        """Test cache shape calculation for decode mode with DSA enabled."""
        mock_get_tp.return_value.rank = 0
        mock_get_tp.return_value.local_rank = 0
        mock_get_tp.return_value.world_size = 16
        mock_get_dp.return_value.local_rank = 0
        mock_get_dp.return_value.world_size = 4
        mock_get_tp_pool.return_value.world_size = 16
        mock_runner = MockNPUModelRunner()
        mock_vllm_config = MockVllmConfig()
        # Enable DSA by setting use_mla=True and adding required hf_config attributes
        mock_vllm_config.model_config.use_mla = True
        # Set proper hf_config attributes for DSA (used by _pangu_kv_head_dims_from_hf)
        mock_vllm_config.model_config.hf_config.index_topk = 8
        mock_vllm_config.model_config.hf_config.kv_lora_rank = 512
        mock_vllm_config.model_config.hf_config.qk_rope_head_dim = 64
        mock_vllm_config.model_config.hf_config.index_head_dim = 128
        cache = DecodeOmniCache(mock_config, mock_runner, vllm_config=mock_vllm_config)
        shape, num_blocks = cache.calc_cache_shape()
        assert len(shape) == 5
        assert shape[0] == cache.local_dp_size
        assert shape[1] == 2  # num_layers
        assert shape[3] == 128  # block_size
        assert shape[4] == 2816


# ==================== PrefixCopyMeta tests ====================

class TestPrefixCopyMeta:
    """Test suite for PrefixCopyMeta data class."""

    @patch("omni_cache.cache.core.base.get_world_group")
    def test_init(self, mock_world_group):
        mock_world_group.return_value = 1
        consecutive_blocks = [[(0, 2), (3, 5)], [(5, 7)]]
        query_lens = [5, 3]
        query_slots = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
        meta = PrefixCopyMeta(
            consecutive_blocks=consecutive_blocks,
            query_lens=query_lens,
            query_slots=query_slots,
        )
        assert meta.num_actual_tokens == 8
        assert meta.num_copy_blocks == (2 + 2 + 2)  # (2-0)+(5-3)+(7-5)

    def test_invalid_length(self):
        """Test validation of mismatched input lengths."""
        consecutive_blocks = [[(0, 2)]]
        query_lens = [1, 2]  # Length mismatch
        query_slots = torch.tensor([0])
        with pytest.raises(RuntimeError):
            PrefixCopyMeta(consecutive_blocks, query_lens, query_slots)


# ==================== Factory function tests ====================

class TestCreateOmniCache:
    """Test suite for omni cache factory function."""

    @patch("omni_cache.cache.prefill.prefill_omni_cache.PrefillOmniCache.initialize_device_cache")
    @patch("omni_cache.cache.transfer_engine.manager.TransferManager.initialize_prefill")
    @patch("omni_cache.cache.core.base.KVCacheMemoryPool")
    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.bind_kv_cache")
    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    def test_create_prefill(self, mock_pool_tp, mock_bind, mock_get_dp, mock_get_tp, mock_pool_cls, mock_init_prefill, mock_init_dev):
        """Test creation of PrefillOmniCache instance."""
        # initialize_device_cache is mocked out because it uses
        # runner.device = "npu:0" which isn't a valid torch device in tests.
        # Return an empty dict structure so ensure_device_cache_initialized
        # sets device_cache=[{}] and the OmniKvCacheAccessor loop skips.
        mock_init_dev.return_value = [{}]
        mock_pool_tp.return_value.world_size = 16
        mock_get_tp.return_value.world_size = 16
        mock_get_tp.return_value.rank = 0
        mock_get_dp.return_value.local_rank = 0
        mock_vllm_config = MockVllmConfig()
        mock_vllm_config.kv_transfer_config.kv_role = "kv_producer"
        mock_vllm_config.scheduler_config.max_num_batched_tokens = 1024
        mock_vllm_config.scheduler_config.max_num_seqs = 32
        mock_vllm_config.scheduler_config.max_model_len = 2048
        mock_vllm_config.compilation_config.static_forward_context = {"layers.0": Mock()}

        mock_runner = MockNPUModelRunner()
        spec = MockAttentionSpec()
        group = MockKVCacheGroup(layer_names=["layers.0"], kv_cache_spec=spec)
        config = MockKVCacheConfig(kv_cache_groups=[group], tensors={0: torch.randn(1024, 512)})

        # Setup the mock pool instance
        mock_pool_instance = Mock()
        mock_pool_instance.ascend_cl_stream = None
        mock_pool_instance.shared_tensor_npu = None
        mock_pool_instance.kvi_tensors = [torch.zeros(1, 1, 1, 1)]
        mock_pool_cls.return_value = mock_pool_instance

        cache = create_omni_cache(config, mock_vllm_config, mock_runner)
        assert isinstance(cache, PrefillOmniCache)

    @patch("omni_cache.cache.core.base.get_tp_group")
    @patch("omni_cache.cache.core.base.get_dp_group")
    @patch("omni_cache.cache.core.base.bind_kv_cache")
    @patch("omni_cache.cache.memory.memory_pool.get_tp_group")
    @patch("omni_cache.cache.memory.memory_pool.KVCacheMemoryPool._map_hugepage_memory")
    @patch("omni_cache.cache.core.base.isinstance", return_value=True)
    def test_create_decode(self, mock_instance, mock_hugepage, mock_pool_tp, mock_bind, mock_get_dp, mock_get_tp):
        """Test creation of DecodeOmniCache instance."""
        mock_pool_tp.return_value.world_size = 16
        mock_get_tp.return_value.world_size = 16
        mock_get_tp.return_value.rank = 0
        mock_get_dp.return_value.local_rank = 0
        mock_vllm_config = MockVllmConfig()
        mock_vllm_config.kv_transfer_config.kv_role = "kv_consumer"
        mock_vllm_config.compilation_config.static_forward_context = {}

        mock_runner = MockNPUModelRunner()
        spec = MockAttentionSpec()
        spec.page_size_bytes = 1024
        group = MockKVCacheGroup(layer_names=["layers.0"], kv_cache_spec=spec)
        config = MockKVCacheConfig(kv_cache_groups=[group], tensors={"layers.0": Mock(size=1024 * 1024)})

        cache = create_omni_cache(config, mock_vllm_config, mock_runner)
        assert isinstance(cache, DecodeOmniCache)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
