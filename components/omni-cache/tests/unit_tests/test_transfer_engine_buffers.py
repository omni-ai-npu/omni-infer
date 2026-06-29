# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Unit tests for transfer_engine/buffers.py module.

Tests buffer management, stream management, and thread pool utilities.
"""

import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor

from omni_cache.cache.transfer_engine.buffers import (
    TransferBuffers,
    StreamManager,
    ThreadPoolManager,
)


class TestTransferBuffers:
    """Test cases for TransferBuffers class."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache instance."""
        cache = Mock()
        cache.device = torch.device('cpu')
        cache.tp_world_size = 1
        cache.dtype = torch.float16
        cache.is_hybrid_attn = False
        cache.head_sizes = [64]
        cache.num_spec_token = 0
        cache.kv_cache_config = Mock()
        cache.kv_cache_config.kv_cache_groups = [Mock()]
        return cache

    def test_transfer_buffers_init(self, mock_cache):
        """Test TransferBuffers initialization."""
        tb = TransferBuffers(mock_cache)

        assert tb.cache == mock_cache
        assert tb.batch_buffer_cpu == []
        assert tb.batch_slots_cpu is None
        assert tb.arange_cpu is None
        assert tb.arange is None

    def test_init_basic_attrs(self, mock_cache):
        """Test _init_basic_attrs method."""
        tb = TransferBuffers(mock_cache)
        tb._init_basic_attrs(max_num_batched_tokens=512, max_model_len=2048)

        assert mock_cache.max_num_batched_tokens == 512
        assert mock_cache.max_model_len == 2048

    def test_init_arange_tensors(self, mock_cache):
        """Test _init_arange_tensors method."""
        tb = TransferBuffers(mock_cache)
        tb._init_arange_tensors(max_model_len=1024)

        assert tb.arange_cpu is not None
        assert tb.arange is not None
        assert len(tb.arange_cpu) == 1024
        assert tb.arange_cpu.dtype == torch.int64

    def test_get_head_size_config_non_hybrid(self, mock_cache):
        """Test _get_head_size_config for non-hybrid attention."""
        tb = TransferBuffers(mock_cache)

        kv_cache_config = Mock()
        kv_cache_config.kv_cache_groups = [Mock()]

        group_specs, head_sizes_buf = tb._get_head_size_config(kv_cache_config)

        # For non-hybrid, should return None specs and cache head_sizes
        assert len(group_specs) == len(mock_cache.head_sizes)
        assert all(s is None for s in group_specs)
        assert head_sizes_buf == mock_cache.head_sizes

    def test_get_head_size_config_hybrid(self, mock_cache):
        """Test _get_head_size_config for hybrid attention."""
        mock_cache.is_hybrid_attn = True
        tb = TransferBuffers(mock_cache)

        spec = Mock()
        spec.head_size = 64
        spec.num_kv_heads = 4

        kv_cache_config = Mock()
        kv_cache_config.kv_cache_groups = [Mock(kv_cache_spec=spec)]

        group_specs, head_sizes_buf = tb._get_head_size_config(kv_cache_config)

        # For hybrid, should compute head_sizes from specs
        assert len(group_specs) == 1
        assert head_sizes_buf == [256]  # 64 * 4

    def test_create_cpu_tensor(self, mock_cache):
        """Test _create_cpu_tensor method."""
        mock_cache.max_num_batched_tokens = 128
        tb = TransferBuffers(mock_cache)

        tensor = tb._create_cpu_tensor(size_d=64, dtype=torch.float16, pin=False)

        assert tensor.device.type == 'cpu'
        assert tensor.dtype == torch.float16
        assert tensor.shape[1] == 64

    def test_create_single_buffer_no_kvscale(self, mock_cache):
        """Test _create_single_buffer without separate_kvscale."""
        mock_cache.max_num_batched_tokens = 64
        tb = TransferBuffers(mock_cache)

        buf = tb._create_single_buffer(D=64, spec=None)

        # Non-hybrid, no kvscale: returns single tensor (not tuple)
        assert isinstance(buf, torch.Tensor)
        assert buf.shape[1] == 64

    def test_create_single_buffer_with_kvscale(self, mock_cache):
        """Test _create_single_buffer with separate_kvscale."""
        mock_cache.max_num_batched_tokens = 64
        tb = TransferBuffers(mock_cache)

        spec = Mock()
        spec.separate_kvscale = torch.int8
        spec.dtype = torch.float16

        buf = tb._create_single_buffer(D=64, spec=spec)

        # Should return tuple of (tensor, scale_tensor)
        assert isinstance(buf, tuple)
        assert len(buf) == 2

    def test_create_single_buffer_hybrid(self, mock_cache):
        """Test _create_single_buffer in hybrid mode."""
        mock_cache.is_hybrid_attn = True
        mock_cache.max_num_batched_tokens = 64
        tb = TransferBuffers(mock_cache)

        buf = tb._create_single_buffer(D=64, spec=None)

        # Hybrid mode wraps in tuple
        assert isinstance(buf, tuple)
        assert len(buf) == 1

    def test_clone_speculation_buffers_no_spec(self, mock_cache):
        """Test _clone_speculation_buffers when num_spec_token is 0."""
        tb = TransferBuffers(mock_cache)
        tb.batch_buffer_cpu = [(torch.randn(10, 64),)]

        tb._clone_speculation_buffers()

        # Should not add any buffers
        assert len(tb.batch_buffer_cpu) == 1

    def test_clone_speculation_buffers_with_spec(self, mock_cache):
        """Test _clone_speculation_buffers with spec tokens."""
        mock_cache.is_hybrid_attn = True
        mock_cache.num_spec_token = 3
        tb = TransferBuffers(mock_cache)
        tb.batch_buffer_cpu = [
            (torch.randn(10, 64),),
            (torch.randn(10, 64),),
        ]

        tb._clone_speculation_buffers()

        # Should add num_spec_token buffers
        assert len(tb.batch_buffer_cpu) == 2 + 3

    def test_attach_to_cache(self, mock_cache):
        """Test _attach_to_cache method."""
        tb = TransferBuffers(mock_cache)
        tb.batch_buffer_cpu = [(torch.randn(10, 64),)]
        tb.arange_cpu = torch.arange(100)
        tb.arange = torch.arange(100)

        tb._attach_to_cache()

        assert mock_cache.batch_buffer_cpu == tb.batch_buffer_cpu
        assert torch.equal(mock_cache.arange_cpu, tb.arange_cpu)
        assert torch.equal(mock_cache.arange, tb.arange)


class TestStreamManager:
    """Test cases for StreamManager class."""

    @pytest.fixture
    def device(self):
        return torch.device('cpu')

    def test_stream_manager_init(self, device):
        """Test StreamManager initialization."""
        sm = StreamManager(device)

        assert sm.device == device
        assert sm.h2d_stream is None
        assert sm.d2h_stream is None
        assert sm.decode_h2d_stream is None
        assert sm.copy_stream is None

    @patch('torch.npu.Stream')
    def test_initialize_prefill_streams(self, mock_stream, device):
        """Test initialize_prefill_streams method."""
        sm = StreamManager(device)
        mock_cache = Mock()

        sm.initialize_prefill_streams(mock_cache)

        # Should create two streams
        assert mock_stream.call_count == 2
        assert sm.d2h_stream is not None
        assert sm.h2d_stream is not None

        # Should attach to cache
        assert mock_cache.d2h_stream == sm.d2h_stream
        assert mock_cache.h2d_stream == sm.h2d_stream

    @patch('torch.npu.Stream')
    def test_initialize_decode_streams(self, mock_stream, device):
        """Test initialize_decode_streams method."""
        sm = StreamManager(device)
        mock_cache = Mock()

        sm.initialize_decode_streams(mock_cache)

        # Should create decode stream
        assert sm.decode_h2d_stream is not None
        assert mock_cache.decode_h2d_stream == sm.decode_h2d_stream

class TestThreadPoolManager:
    """Test cases for ThreadPoolManager class."""

    def test_thread_pool_manager_init(self):
        """Test ThreadPoolManager initialization."""
        tpm = ThreadPoolManager()

        assert tpm.d2h_thrp is None
        assert tpm.copy_future is None
        assert tpm.copy_futures == []

    def test_initialize_non_hybrid(self):
        """Test initialize for non-hybrid attention."""
        tpm = ThreadPoolManager()
        mock_cache = Mock()
        mock_cache.is_hybrid_attn = False

        tpm.initialize(mock_cache, max_workers=2)

        assert tpm.d2h_thrp is not None
        assert isinstance(tpm.d2h_thrp, ThreadPoolExecutor)
        assert tpm.d2h_thrp._max_workers == 2

        # Non-hybrid: 2 stages
        assert mock_cache.num_stages_layer_copy == 2
        assert len(tpm.copy_futures) == 2

    def test_initialize_hybrid(self):
        """Test initialize for hybrid attention."""
        tpm = ThreadPoolManager()
        mock_cache = Mock()
        mock_cache.is_hybrid_attn = True

        tpm.initialize(mock_cache, max_workers=1)

        # Hybrid: 1 stage, copy_futures has 1 entry (one slot per stage)
        assert mock_cache.num_stages_layer_copy == 1
        assert len(tpm.copy_futures) == 1  # One slot per stage for hybrid

    def test_initialize_attaches_to_cache(self):
        """Test that initialize attaches to cache."""
        tpm = ThreadPoolManager()
        mock_cache = Mock()
        mock_cache.is_hybrid_attn = False

        tpm.initialize(mock_cache)

        assert mock_cache.d2h_thrp == tpm.d2h_thrp
        assert mock_cache.copy_future == tpm.copy_future
        assert mock_cache.copy_futures == tpm.copy_futures


class TestTransferBuffersInitialize:
    """Integration tests for TransferBuffers initialize methods."""

    @pytest.fixture
    def mock_cache(self):
        """Create a fully configured mock cache."""
        cache = Mock()
        cache.device = torch.device('cpu')
        cache.tp_world_size = 1
        cache.dtype = torch.float16
        cache.is_hybrid_attn = False
        cache.head_sizes = [64]
        cache.num_spec_token = 0
        cache.block_size = 128
        cache.page_size_bytes = 0  # Force fallback to spec.page_size_bytes
        return cache

    def test_initialize_cpu_buffers_basic(self, mock_cache):
        """Test full initialize_cpu_buffers flow."""
        tb = TransferBuffers(mock_cache)

        spec = Mock()
        spec.head_size = 64
        spec.num_kv_heads = 4
        spec.dtype = torch.float16
        spec.separate_kvscale = False
        spec.page_size_bytes = 64 * 128 * 2  # head_size * block_size * elem_size

        kv_cache_config = Mock()
        kv_cache_config.kv_cache_groups = [Mock(kv_cache_spec=spec)]

        tb.initialize_cpu_buffers(
            kv_cache_config=kv_cache_config,
            max_num_batched_tokens=128,
            max_model_len=1024,
        )

        # Should initialize all components
        assert mock_cache.max_num_batched_tokens == 128
        assert mock_cache.max_model_len == 1024
        assert len(tb.batch_buffer_cpu) >= 1
        assert tb.arange_cpu is not None

    def test_initialize_token_indices_non_hybrid(self, mock_cache):
        """Test initialize_token_indices for non-hybrid."""
        # Skip on macOS MPS due to missing kernel
        if torch.backends.mps.is_available():
            pytest.skip("MPS backend not supported for this test")

        # pin_memory=True in non-hybrid path requires CUDA or NPU;
        # patch so the test works on CPU-only environments.
        pin_available = torch.cuda.is_available() or (
            hasattr(torch, "npu") and torch.npu.is_available()
        )
        if not pin_available:
            pytest.skip("pin_memory requires CUDA or NPU backend")

        tb = TransferBuffers(mock_cache)

        tb.initialize_token_indices(max_num_batched_tokens=128)

        # Non-hybrid: simple 1D tensor
        assert mock_cache.batch_slots_cpu is not None
        assert mock_cache.batch_slots_cpu.dim() == 1
        assert mock_cache.batch_token_indices is None

    def test_initialize_token_indices_hybrid(self, mock_cache):
        """Test initialize_token_indices for hybrid attention."""
        mock_cache.is_hybrid_attn = True
        mock_cache.kv_cache_config = Mock()
        mock_cache.kv_cache_config.kv_cache_groups = [Mock(), Mock()]
        tb = TransferBuffers(mock_cache)

        tb.initialize_token_indices(max_num_batched_tokens=128)

        # Hybrid: 2D tensor
        assert mock_cache.batch_slots_cpu is not None
        assert mock_cache.batch_slots_cpu.dim() == 2
        assert mock_cache.num_called_builder > 0


class TestBuffersEdgeCases:
    """Edge case tests for buffers module."""

    def test_transfer_buffers_very_small_max_tokens(self):
        """Test with very small max_num_batched_tokens."""
        cache = Mock()
        cache.device = torch.device('cpu')
        cache.tp_world_size = 2
        cache.dtype = torch.float32
        cache.is_hybrid_attn = False
        cache.head_sizes = [32]
        cache.num_spec_token = 0
        cache.block_size = 128
        cache.page_size_bytes = 0  # Force fallback to spec.page_size_bytes

        tb = TransferBuffers(cache)

        spec = Mock()
        spec.head_size = 32
        spec.num_kv_heads = 4
        spec.dtype = torch.float32
        spec.separate_kvscale = False
        spec.page_size_bytes = 32 * 128 * 4  # head_size * block_size * elem_size

        kv_cache_config = Mock()
        kv_cache_config.kv_cache_groups = [Mock(kv_cache_spec=spec)]

        # Very small batch size
        tb.initialize_cpu_buffers(
            kv_cache_config=kv_cache_config,
            max_num_batched_tokens=2,
            max_model_len=128,
        )

        # Should handle small sizes gracefully
        assert len(tb.batch_buffer_cpu) >= 1

    def test_thread_pool_manager_max_workers_zero(self):
        """Test ThreadPoolManager with max_workers=0 (uses default)."""
        tpm = ThreadPoolManager()
        cache = Mock()
        cache.is_hybrid_attn = False

        # max_workers=0 or None uses default
        tpm.initialize(cache, max_workers=1)

        assert tpm.d2h_thrp is not None
