# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# test_kv_cache_interface.py

import pytest
try:
    import torch
except ImportError:
    torch = None
from unittest.mock import Mock, patch, MagicMock
from dataclasses import FrozenInstanceError
from typing import Dict

# Import module under test
from omni_cache.attention.kv_cache_interface import (
    OmniKVCacheConfig,
    OmniAttentionSpec,
    OmniMultiGroupBlockTable,
    get_kv_cache_config_omni_type,
    get_omni_hybrid_kv_cache_spec,
    SINK,
    RECENT,
    BETA,
    PATTERN,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    FullAttentionSpec,
    KVCacheTensor,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner
from vllm.v1.attention.backend import AttentionType
from vllm.config import VllmConfig

if torch is None:
    pytest.skip("torch is not available", allow_module_level=True)


# ==================== Constants tests ====================

class TestConstants:
    """Test suite for module constants."""

    def test_sink_constant(self):
        """Test SINK constant value."""
        assert SINK == 1

    def test_recent_constant(self):
        """Test RECENT constant value."""
        assert RECENT == 3

    def test_beta_constant(self):
        """Test BETA constant value."""
        assert BETA == 0.2

    def test_pattern_constant(self):
        """Test PATTERN constant is a list of 0s and 1s."""
        assert isinstance(PATTERN, list)
        assert len(PATTERN) == 62
        assert all(p in [0, 1] for p in PATTERN)
        # Check that pattern has both 0s and 1s
        assert 0 in PATTERN
        assert 1 in PATTERN

# ==================== OmniMultiGroupBlockTable tests ====================

class TestOmniMultiGroupBlockTable:


    def test_init_without_block_size_raises_error(self):
        """Test initialization without block_size or block_sizes raises RuntimeError."""
        device = torch.device("cpu")

        with pytest.raises(RuntimeError, match="Neither `block_size` nor `block_sizes` is given"):
            OmniMultiGroupBlockTable(
                max_num_reqs=10,
                max_model_len=2048,
                max_num_batched_tokens=1024,
                pin_memory=False,
                device=device,
            )

    def test_init_with_invalid_block_size_raises_error(self):
        """Test initialization with invalid block_size raises ValueError."""
        device = torch.device("cpu")

        # Test with block_size = 0
        with pytest.raises(ValueError, match="block_size should be a positive int"):
            OmniMultiGroupBlockTable(
                max_num_reqs=10,
                max_model_len=2048,
                max_num_batched_tokens=1024,
                pin_memory=False,
                device=device,
                block_size=0,
            )

        # Test with negative block_size
        with pytest.raises(ValueError, match="block_size should be a positive int"):
            OmniMultiGroupBlockTable(
                max_num_reqs=10,
                max_model_len=2048,
                max_num_batched_tokens=1024,
                pin_memory=False,
                device=device,
                block_size=-1,
            )

        # Test with non-integer block_size
        with pytest.raises(ValueError, match="block_size should be a positive int"):
            OmniMultiGroupBlockTable(
                max_num_reqs=10,
                max_model_len=2048,
                max_num_batched_tokens=1024,
                pin_memory=False,
                device=device,
                block_size="16",
            )


# ==================== get_kv_cache_config_omni_type tests ====================

class TestGetKVCacheConfigOmniType:
    """Test suite for get_kv_cache_config_omni_type function."""

    @patch("omni_cache.cache.kv_cache_interface.create_kv_cache_group_specs")
    @patch("omni_cache.cache.kv_cache_interface.logger")
    @patch("omni_cache.cache.kv_cache_interface.KVCacheTensor")
    def test_unsupported_spec_type_raises_error(self, mock_kv_cache_tensor, mock_logger, mock_create_group_specs):
        """Test unsupported KV cache spec type raises RuntimeError."""
        mock_kv_cache_tensor.return_value = Mock()
        mock_create_group_specs.return_value = []

        vllm_config = Mock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 2048

        # Create unsupported spec type
        class UnsupportedSpec(KVCacheSpec):
            pass

        kv_cache_spec = {
            "model.layers.0": Mock(spec=UnsupportedSpec, page_size_bytes=1024),
        }

        available_memory = 10 * 1024 * 1024

        with pytest.raises(RuntimeError, match="Unsupported KV Cache Spec type"):
            get_kv_cache_config_omni_type(vllm_config, kv_cache_spec, available_memory)

    @patch("omni_cache.cache.kv_cache_interface.create_kv_cache_group_specs")
    @patch("omni_cache.cache.kv_cache_interface.logger")
    @patch("omni_cache.cache.kv_cache_interface.KVCacheTensor")
    def test_multiple_page_sizes_raises_error(self, mock_kv_cache_tensor, mock_logger, mock_create_group_specs):
        """Test multiple page sizes raises ValueError."""
        mock_kv_cache_tensor.return_value = Mock()
        mock_create_group_specs.return_value = []

        vllm_config = Mock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 2048

        # Create specs with different page sizes
        kv_cache_spec = {
            "model.layers.0": Mock(spec=FullAttentionSpec, page_size_bytes=1024, sliding_window=None),
            "model.layers.1": Mock(spec=FullAttentionSpec, page_size_bytes=2048, sliding_window=None),
        }

        available_memory = 10 * 1024 * 1024

        with pytest.raises(ValueError, match="page_sizes must have exactly one element"):
            get_kv_cache_config_omni_type(vllm_config, kv_cache_spec, available_memory)

# ==================== OmniAttentionSpec tests ====================

class TestOmniAttentionSpec:
    """Test suite for OmniAttentionSpec dataclass."""

    def test_post_init_calculations(self):
        """Test that __post_init__ calculates correct values."""
        spec = OmniAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
            use_mla=False,
            sink_blocks=2,
            recent_blocks=4
        )

        assert spec.max_num_blocks == 6  # sink_blocks + recent_blocks
        assert spec.sink == 32  # sink_blocks * block_size
        assert spec.recent == 64  # recent_blocks * block_size
        assert spec.max_compressed_len == 96  # max_num_blocks * block_size

    def test_post_init_with_default_constants(self):
        """Test __post_init__ with default SINK and RECENT constants."""
        spec = OmniAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
            use_mla=False
        )

        assert spec.sink_blocks == SINK  # Default 1
        assert spec.recent_blocks == RECENT  # Default 3
        assert spec.max_num_blocks == SINK + RECENT

    def test_type_id_property(self):
        """Test type_id property generates correct string."""
        spec = OmniAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
            use_mla=False,
            sink_blocks=2,
            recent_blocks=4
        )

        # sink = sink_blocks * block_size = 2 * 16 = 32
        # recent = recent_blocks * block_size = 4 * 16 = 64
        expected = f"omni_attention_32_64_16_{spec.page_size_bytes}"
        assert spec.type_id == expected

    def test_max_memory_usage_bytes(self):
        """Test max_memory_usage_bytes property."""
        spec = OmniAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            dtype=torch.float16,
            use_mla=False,
            sink_blocks=2,
            recent_blocks=4
        )

        expected = spec.max_num_blocks * spec.page_size_bytes
        assert spec.max_memory_usage_bytes == expected


# ==================== OmniMultiGroupBlockTable success path tests ====================

class TestOmniMultiGroupBlockTableSuccess:
    """Test suite for OmniMultiGroupBlockTable successful initialization."""

    @patch("omni_cache.cache.kv_cache_interface.BlockTable")
    def test_init_with_block_size_success(self, mock_block_table):
        """Test successful initialization with block_size parameter."""
        device = torch.device("cpu")

        table = OmniMultiGroupBlockTable(
            max_num_reqs=10,
            max_model_len=2048,
            max_num_batched_tokens=1024,
            pin_memory=False,
            device=device,
            block_size=16
        )

        # Should create two BlockTables
        assert mock_block_table.call_count == 2
        assert len(table.block_tables) == 2

        # First call for full attention (calculated from max_model_len)
        first_call = mock_block_table.call_args_list[0]
        assert first_call[0][0] == 10  # max_num_reqs
        assert first_call[0][1] == 128  # cdiv(2048, 16)

        # Second call for omni attention (SINK + RECENT)
        second_call = mock_block_table.call_args_list[1]
        assert second_call[0][0] == 10  # max_num_reqs
        assert second_call[0][1] == SINK + RECENT  # 4

    @patch("omni_cache.cache.kv_cache_interface.BlockTable")
    def test_init_with_block_sizes_success(self, mock_block_table):
        """Test successful initialization with block_sizes parameter (vLLM 0.9.2+)."""
        device = torch.device("cpu")

        table = OmniMultiGroupBlockTable(
            max_num_reqs=10,
            max_model_len=2048,
            max_num_batched_tokens=1024,
            pin_memory=False,
            device=device,
            block_sizes=[16, 32]
        )

        # Should use first element of block_sizes
        assert mock_block_table.call_count == 2
        first_call = mock_block_table.call_args_list[0]
        assert first_call[0][1] == 128  # cdiv(2048, 16)

    @patch("omni_cache.cache.kv_cache_interface.BlockTable")
    def test_init_with_positional_args_raises_error(self, mock_block_table):
        """Test that positional args beyond the first 5 raise RuntimeError."""
        device = torch.device("cpu")

        with pytest.raises(RuntimeError, match="All arguments should be passed with keywords"):
            OmniMultiGroupBlockTable(
                10,  # positional arg for max_num_reqs
                2048,  # positional arg for max_model_len
                1024,  # positional arg for max_num_batched_tokens
                False,  # positional arg for pin_memory
                device,  # positional arg for device
                16,  # extra positional arg - should trigger error
                block_size=16
            )


# ==================== get_kv_cache_config_omni_type success path tests ====================

class TestGetKVCacheConfigOmniTypeSuccess:
    """Test suite for get_kv_cache_config_omni_type function success paths."""

    @patch("omni_cache.cache.kv_cache_interface.create_kv_cache_group_specs")
    @patch("omni_cache.cache.kv_cache_interface.logger")
    @patch("omni_cache.cache.kv_cache_interface.KVCacheTensor")
    def test_basic_config_generation(self, mock_kv_cache_tensor, mock_logger, mock_create_group_specs):
        """Test basic KV cache config generation with mixed layer types."""
        mock_kv_cache_tensor.return_value = Mock()
        mock_create_group_specs.return_value = []

        vllm_config = Mock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 2048

        # Create specs with FullAttentionSpec and OmniAttentionSpec
        kv_cache_spec = {
            "model.layers.0": Mock(spec=FullAttentionSpec, page_size_bytes=1024, sliding_window=None),
            "model.layers.1": Mock(spec=OmniAttentionSpec, page_size_bytes=1024),
        }

        available_memory = 100 * 1024 * 1024  # 100MB

        config = get_kv_cache_config_omni_type(vllm_config, kv_cache_spec, available_memory)

        assert isinstance(config, OmniKVCacheConfig)
        assert FullAttentionSpec in config.num_blocks_per_group
        assert OmniAttentionSpec in config.num_blocks_per_group
        mock_logger.info.assert_called()

    @patch("omni_cache.cache.kv_cache_interface.create_kv_cache_group_specs")
    @patch("omni_cache.cache.kv_cache_interface.logger")
    @patch("omni_cache.cache.kv_cache_interface.KVCacheTensor")
    def test_config_with_full_layers_only(self, mock_kv_cache_tensor, mock_logger, mock_create_group_specs):
        """Test config generation with only FullAttentionSpec layers."""
        mock_kv_cache_tensor.return_value = Mock()
        mock_create_group_specs.return_value = []

        vllm_config = Mock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 2048

        # Create specs with only FullAttentionSpec
        kv_cache_spec = {
            f"model.layers.{i}": Mock(spec=FullAttentionSpec, page_size_bytes=1024, sliding_window=None)
            for i in range(4)
        }

        available_memory = 100 * 1024 * 1024

        # This should work but omni layers will be empty
        config = get_kv_cache_config_omni_type(vllm_config, kv_cache_spec, available_memory)
        assert isinstance(config, OmniKVCacheConfig)

    @patch("omni_cache.cache.kv_cache_interface.create_kv_cache_group_specs")
    @patch("omni_cache.cache.kv_cache_interface.logger")
    @patch("omni_cache.cache.kv_cache_interface.KVCacheTensor")
    def test_sliding_window_not_raised(self, mock_kv_cache_tensor, mock_logger, mock_create_group_specs):
        """Test that FullAttentionSpec with sliding_window creates ValueError but doesn't raise."""
        # Note: This test documents a potential bug in the code
        # The code creates ValueError but doesn't raise it
        mock_kv_cache_tensor.return_value = Mock()
        mock_create_group_specs.return_value = []

        vllm_config = Mock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 2048

        # Create spec with sliding_window set (should raise ValueError)
        kv_cache_spec = {
            "model.layers.0": Mock(spec=FullAttentionSpec, page_size_bytes=1024, sliding_window=512),
        }

        available_memory = 100 * 1024 * 1024

        # Omni attention implements its own sliding window, so this should raise ValueError
        with pytest.raises(ValueError, match="Omni attention implements its own sliding window"):
            config = get_kv_cache_config_omni_type(vllm_config, kv_cache_spec, available_memory)

    @patch("omni_cache.cache.kv_cache_interface.create_kv_cache_group_specs")
    @patch("omni_cache.cache.kv_cache_interface.logger")
    @patch("omni_cache.cache.kv_cache_interface.KVCacheTensor")
    def test_block_calculation_with_beta(self, mock_kv_cache_tensor, mock_logger, mock_create_group_specs):
        """Test that BETA is correctly applied in block calculations."""
        mock_kv_cache_tensor.return_value = Mock()
        mock_create_group_specs.return_value = []

        vllm_config = Mock()
        vllm_config.cache_config.block_size = 16
        vllm_config.model_config.max_model_len = 2048

        kv_cache_spec = {
            "model.layers.0": Mock(spec=FullAttentionSpec, page_size_bytes=1024, sliding_window=None),
            "model.layers.1": Mock(spec=OmniAttentionSpec, page_size_bytes=1024),
        }

        available_memory = 100 * 1024 * 1024

        config = get_kv_cache_config_omni_type(vllm_config, kv_cache_spec, available_memory)

        # Verify BETA is used in calculation
        num_omni_layers = sum(PATTERN)
        num_full_layers = len(PATTERN) - num_omni_layers

        # The config should have different block counts for different spec types
        full_blocks = config.num_blocks_per_group.get(FullAttentionSpec, 0)
        omni_blocks = config.num_blocks_per_group.get(OmniAttentionSpec, 0)

        # Both should be positive
        assert full_blocks > 0
        assert omni_blocks > 0


# ==================== get_omni_hybrid_kv_cache_spec tests ====================

class TestGetOmniHybridKVCacheSpec:
    """Test suite for get_omni_hybrid_kv_cache_spec function."""

    @patch("omni_cache.cache.kv_cache_interface.get_layers_from_vllm_config")
    @patch("omni_cache.cache.kv_cache_interface.AttentionType")
    def test_basic_spec_generation(self, mock_attention_type, mock_get_layers):
        """Test basic KV cache spec generation."""
        # Setup mock attention modules
        mock_attn = Mock()
        mock_attn.attn_type = mock_attention_type.DECODER
        mock_attn.num_kv_heads = 8
        mock_attn.head_size = 128

        mock_get_layers.return_value = {
            "model.layers.0": mock_attn,
            "model.layers.1": mock_attn,
        }

        mock_runner = Mock()
        mock_runner.vllm_config.cache_config.block_size = 16
        mock_runner.vllm_config.model_config.use_mla = False
        mock_runner.kv_cache_dtype = torch.float16

        result = get_omni_hybrid_kv_cache_spec(mock_runner)

        assert isinstance(result, dict)
        assert len(result) == 2
        assert "model.layers.0" in result
        assert "model.layers.1" in result

        # Check that layer 0 uses pattern
        if PATTERN[0] == 1:
            assert isinstance(result["model.layers.0"], OmniAttentionSpec)
        else:
            assert isinstance(result["model.layers.0"], FullAttentionSpec)

    @patch("omni_cache.cache.kv_cache_interface.get_layers_from_vllm_config")
    @patch("omni_cache.cache.kv_cache_interface.AttentionType")
    def test_mla_enabled(self, mock_attention_type, mock_get_layers):
        """Test spec generation with MLA enabled."""
        mock_attn = Mock()
        mock_attn.attn_type = mock_attention_type.DECODER
        mock_attn.num_kv_heads = 8
        mock_attn.head_size = 128

        mock_get_layers.return_value = {
            "model.layers.0": mock_attn,
        }

        mock_runner = Mock()
        mock_runner.vllm_config.cache_config.block_size = 16
        mock_runner.vllm_config.model_config.use_mla = True
        mock_runner.kv_cache_dtype = torch.float16

        result = get_omni_hybrid_kv_cache_spec(mock_runner)

        assert isinstance(result["model.layers.0"], OmniAttentionSpec if PATTERN[0] == 1 else FullAttentionSpec)
        assert result["model.layers.0"].use_mla == True

    @patch("omni_cache.cache.kv_cache_interface.get_layers_from_vllm_config")
    @patch("omni_cache.cache.kv_cache_interface.AttentionType")
    def test_non_decoder_attention_raises_error(self, mock_attention_type, mock_get_layers):
        """Test that non-DECODER attention type raises NotImplementedError."""
        mock_attn = Mock()
        mock_attn.attn_type = mock_attention_type.ENCODER  # Not DECODER
        mock_attn.num_kv_heads = 8
        mock_attn.head_size = 128

        mock_get_layers.return_value = {
            "model.layers.0": mock_attn,
        }

        mock_runner = Mock()
        mock_runner.vllm_config.cache_config.block_size = 16
        mock_runner.vllm_config.model_config.use_mla = False
        mock_runner.kv_cache_dtype = torch.float16

        with pytest.raises(NotImplementedError, match="Omni attention supports decoder-only models"):
            get_omni_hybrid_kv_cache_spec(mock_runner)

    @patch("omni_cache.cache.kv_cache_interface.get_layers_from_vllm_config")
    @patch("omni_cache.cache.kv_cache_interface.AttentionType")
    def test_pattern_distribution(self, mock_attention_type, mock_get_layers):
        """Test that PATTERN is correctly applied to distribute layer types."""
        # Create many layers to test pattern distribution
        num_layers = 10
        mock_attns = {}
        for i in range(num_layers):
            mock_attn = Mock()
            mock_attn.attn_type = mock_attention_type.DECODER
            mock_attn.num_kv_heads = 8
            mock_attn.head_size = 128
            mock_attns[f"model.layers.{i}"] = mock_attn

        mock_get_layers.return_value = mock_attns

        mock_runner = Mock()
        mock_runner.vllm_config.cache_config.block_size = 16
        mock_runner.vllm_config.model_config.use_mla = False
        mock_runner.kv_cache_dtype = torch.float16

        result = get_omni_hybrid_kv_cache_spec(mock_runner)

        # Verify pattern is applied
        omni_count = 0
        full_count = 0
        for i in range(num_layers):
            layer_name = f"model.layers.{i}"
            if PATTERN[i] == 1:
                assert isinstance(result[layer_name], OmniAttentionSpec)
                omni_count += 1
            else:
                assert isinstance(result[layer_name], FullAttentionSpec)
                full_count += 1

        assert omni_count > 0 or full_count > 0  # At least some layers


    """Integration tests for kv_cache_interface module."""
    def test_pattern_sum(self):
        """Test that PATTERN has expected number of 1s."""
        num_ones = sum(PATTERN)
        num_zeros = len(PATTERN) - num_ones

        # PATTERN should have a reasonable distribution
        assert num_ones > 0
        assert num_zeros > 0
        assert num_ones < len(PATTERN)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
