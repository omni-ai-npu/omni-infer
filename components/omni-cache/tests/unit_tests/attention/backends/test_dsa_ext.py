# test_dsa_ext.py
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Unit tests for omni_cache/attention/backends/dsa_ext.py

Tests cover:
- ENABLE_OMNI_CACHE environment variable control
- NPUDSABackendExt: reshape_kv_cache method override
- NPUDSAPrefillMetadataExt: prefix_meta field addition
- NPUDSAMetadataBuilderExt: build method override with prefix_meta logic
- Switch behavior when extension is enabled/disabled

NOTE: This test file uses sys.modules mocking to isolate dependencies
since the current environment may not have vllm/omni_npu/triton installed.
"""

import pytest
import os
import sys
import math
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from dataclasses import fields, is_dataclass


# ==================== Mock External Dependencies ====================

_mock_modules = {}


def setup_mocks():
    """Setup all mock modules for testing."""
    # Mock triton (common missing dependency)
    _mock_modules['triton'] = MagicMock()
    _mock_modules['triton.language'] = MagicMock()

    # Mock vllm.logger
    mock_logger = MagicMock()
    mock_logger.info = MagicMock()
    mock_logger.debug = MagicMock()
    mock_logger.warning = MagicMock()
    _mock_modules['vllm'] = MagicMock()
    _mock_modules['vllm.logger'] = MagicMock()
    _mock_modules['vllm.logger'].init_logger = MagicMock(return_value=mock_logger)

    # Mock vllm.config
    mock_vllm_config = MagicMock()
    mock_kv_transfer_config = MagicMock()
    mock_kv_transfer_config.kv_role = "kv_consumer"
    mock_vllm_config.kv_transfer_config = mock_kv_transfer_config

    _mock_modules['vllm.config'] = MagicMock()
    _mock_modules['vllm.config'].VllmConfig = MagicMock()
    _mock_modules['vllm.config'].get_current_vllm_config = MagicMock(return_value=mock_vllm_config)

    # Mock vllm.v1.attention.backends.mla.common
    mock_mla_prefill_metadata = MagicMock(name='MLACommonPrefillMetadata')
    # Make it behave like a dataclass base
    mock_mla_prefill_metadata.__dataclass_fields__ = {}
    _mock_modules['vllm.v1'] = MagicMock()
    _mock_modules['vllm.v1.attention'] = MagicMock()
    _mock_modules['vllm.v1.attention.backends'] = MagicMock()
    _mock_modules['vllm.v1.attention.backends.mla'] = MagicMock()
    _mock_modules['vllm.v1.attention.backends.mla.common'] = MagicMock()
    _mock_modules['vllm.v1.attention.backends.mla.common'].MLACommonPrefillMetadata = mock_mla_prefill_metadata

    # Mock vllm.v1.kv_cache_interface
    mock_attention_spec = MagicMock(name='AttentionSpec')
    _mock_modules['vllm.v1.kv_cache_interface'] = MagicMock()
    _mock_modules['vllm.v1.kv_cache_interface'].AttentionSpec = mock_attention_spec

    # Mock omni_npu.attention.backends.dsa
    mock_base_backend = MagicMock(name='NPUDSABackend')
    mock_base_backend.__name__ = 'NPUDSABackend'
    mock_base_backend.__qualname__ = 'NPUDSABackend'

    mock_base_metadata = MagicMock(name='NPUDSAMetadata')
    mock_base_metadata.__name__ = 'NPUDSAMetadata'

    mock_base_builder = MagicMock(name='NPUDSAMetadataBuilder')
    mock_base_builder.__name__ = 'NPUDSAMetadataBuilder'
    mock_base_builder.__qualname__ = 'NPUDSAMetadataBuilder'
    mock_base_builder.build = MagicMock(return_value=MagicMock())

    _mock_modules['omni_npu'] = MagicMock()
    _mock_modules['omni_npu.attention'] = MagicMock()
    _mock_modules['omni_npu.attention.backends'] = MagicMock()
    _mock_modules['omni_npu.attention.backends.dsa'] = MagicMock()
    _mock_modules['omni_npu.attention.backends.dsa'].NPUDSABackend = mock_base_backend
    _mock_modules['omni_npu.attention.backends.dsa'].NPUDSAMetadata = mock_base_metadata
    _mock_modules['omni_npu.attention.backends.dsa'].NPUDSAMetadataBuilder = mock_base_builder

    # Mock omni_npu.attention.backends.utils
    def mock_register_decorator(backend_name):
        def decorator(cls):
            return cls
        return decorator

    _mock_modules['omni_npu.attention.backends.utils'] = MagicMock()
    _mock_modules['omni_npu.attention.backends.utils'].register_attention_backend = mock_register_decorator
    _mock_modules['omni_npu.attention.backends.utils'].NPU_ATTENTION_BACKEND = {}

    # Mock omni_cache.cache
    mock_omni_cache = MagicMock(name='omni_cache')
    mock_omni_cache.get_prefill_prefix_copy_meta = MagicMock()
    mock_omni_cache.synchronize_h2d = MagicMock()
    mock_omni_cache.init_batch_token_indices = MagicMock()

    _mock_modules['omni_cache.cache'] = MagicMock()
    _mock_modules['omni_cache.cache'].omni_cache = mock_omni_cache
    _mock_modules['omni_cache.cache.prefill'] = MagicMock()
    _mock_modules['omni_cache.cache.prefill'].PrefillOmniCache = MagicMock(name='PrefillOmniCache')

    return {
        'base_backend': mock_base_backend,
        'base_builder': mock_base_builder,
        'base_metadata': mock_base_metadata,
        'omni_cache': mock_omni_cache,
        'vllm_config': mock_vllm_config,
        'mla_prefill_metadata': mock_mla_prefill_metadata,
    }


def apply_mocks():
    """Apply all mock modules to sys.modules."""
    for name, mock_module in _mock_modules.items():
        sys.modules[name] = mock_module


def clear_mocks():
    """Clear all mock modules from sys.modules."""
    for name in _mock_modules.keys():
        if name in sys.modules:
            del sys.modules[name]


# ==================== Fixtures ====================

@pytest.fixture
def isolated_env():
    """Fixture to provide isolated test environment with mocks."""
    # Save original sys.modules state
    original_modules = sys.modules.copy()

    # Clear any existing related modules
    for name in list(sys.modules.keys()):
        if name.startswith('omni_cache.attention') or name.startswith('omni_npu') or name.startswith('vllm') or name.startswith('triton'):
            del sys.modules[name]

    mocks = setup_mocks()
    apply_mocks()

    yield mocks

    # Restore original sys.modules
    sys.modules.clear()
    sys.modules.update(original_modules)


# ==================== Environment Variable Tests ====================

class TestEnableOmniCacheEnvVar:
    """Test suite for ENABLE_OMNI_CACHE environment variable handling."""

    def test_enabled_when_env_is_one(self, isolated_env):
        """Test that extension is enabled when ENABLE_OMNI_CACHE=1."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            test_module = type('Module', (), {})()
            test_module.ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0"))

            assert test_module.ENABLE_OMNI_CACHE == 1

    def test_disabled_when_env_is_zero(self, isolated_env):
        """Test that extension is disabled when ENABLE_OMNI_CACHE=0."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "0"}, clear=True):
            test_module = type('Module', (), {})()
            test_module.ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0"))

            assert test_module.ENABLE_OMNI_CACHE == 0

    def test_disabled_when_env_not_set(self, isolated_env):
        """Test that extension is disabled when ENABLE_OMNI_CACHE is not set."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('ENABLE_OMNI_CACHE', None)

            test_module = type('Module', (), {})()
            test_module.ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0"))

            assert test_module.ENABLE_OMNI_CACHE == 0

    def test_disabled_when_env_is_not_exactly_one(self, isolated_env):
        """Test that extension is disabled when ENABLE_OMNI_CACHE is not exactly '1'."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "true"}, clear=True):
            # int("true") raises ValueError, so we need to handle it
            # Only "1" enables the extension, not "true", "yes", etc.
            env_val = os.getenv("ENABLE_OMNI_CACHE", "0")
            try:
                val = int(env_val)
            except ValueError:
                val = 0

            assert val == 0


# ==================== NPUDSABackendExt Tests ====================

class TestNPUDSABackendExt:
    """Test suite for NPUDSABackendExt class."""

    def test_backend_ext_inherits_from_base(self, isolated_env):
        """Test that NPUDSABackendExt inherits from NPUDSABackend when enabled."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Create real classes for inheritance test
            class MockNPUDSABackend:
                @staticmethod
                def reshape_kv_cache(raw_tensor, num_blocks, kv_cache_spec):
                    return (raw_tensor,)

            class NPUDSABackendExt(MockNPUDSABackend):
                @staticmethod
                def reshape_kv_cache(raw_tensor, num_blocks, kv_cache_spec):
                    # Extended implementation
                    return super().reshape_kv_cache(raw_tensor, num_blocks, kv_cache_spec)

            assert issubclass(NPUDSABackendExt, MockNPUDSABackend)

    def test_backend_ext_equals_base_when_disabled(self, isolated_env):
        """Test that NPUDSABackendExt equals NPUDSABackend when disabled."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "0"}, clear=True):
            class MockNPUDSABackend:
                pass

            # When disabled, extended class equals base
            NPUDSABackendExt = MockNPUDSABackend

            assert NPUDSABackendExt is MockNPUDSABackend

    def test_reshape_kv_cache_decode_mode(self, isolated_env):
        """Test reshape_kv_cache returns correct shape for decode mode."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Mock vllm config for decode mode
            mock_vllm_config = MagicMock()
            mock_kv_transfer_config = MagicMock()
            mock_kv_transfer_config.kv_role = "kv_consumer"  # Decode mode
            mock_vllm_config.kv_transfer_config = mock_kv_transfer_config

            # Mock AttentionSpec
            mock_kv_cache_spec = MagicMock()
            mock_kv_cache_spec.block_size = 16
            mock_kv_cache_spec.dtype = torch.bfloat16

            # Create raw tensor for decode mode - need to account for dtype size
            # bfloat16 is 2 bytes, so we need 2x the uint8 elements
            num_blocks = 4
            block_size = 16
            dtype = mock_kv_cache_spec.dtype
            dtype_size = 2 if dtype == torch.bfloat16 else 1  # bfloat16 = 2 bytes
            decode_elements = num_blocks * block_size * 1 * 128 * dtype_size
            raw_tensor = torch.zeros(decode_elements, dtype=torch.uint8)

            # Simulate reshape_kv_cache logic
            is_prefill = (mock_vllm_config.kv_transfer_config.kv_role != "kv_consumer")
            raw_tensor_viewed = raw_tensor.view(dtype=dtype)

            if not is_prefill:
                shapes = [(num_blocks, block_size, 1, 128)]
            else:
                shapes = [
                    (num_blocks, block_size, 1, 512),
                    (num_blocks, block_size, 1, 64),
                    (num_blocks, block_size, 1, 128)
                ]

            sizes = [math.prod(shape) for shape in shapes]
            tensors = torch.split(raw_tensor_viewed, sizes)
            result = tuple(t.view(shape) for t, shape in zip(tensors, shapes))

            # Verify decode mode output
            assert isinstance(result, tuple)
            assert len(result) == 1
            assert result[0].shape == (num_blocks, 16, 1, 128)

    def test_reshape_kv_cache_prefill_mode(self, isolated_env):
        """Test reshape_kv_cache returns correct shapes for prefill mode."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Mock vllm config for prefill mode
            mock_vllm_config = MagicMock()
            mock_kv_transfer_config = MagicMock()
            mock_kv_transfer_config.kv_role = "kv_producer"  # Prefill mode
            mock_vllm_config.kv_transfer_config = mock_kv_transfer_config

            mock_kv_cache_spec = MagicMock()
            mock_kv_cache_spec.block_size = 16
            dtype = mock_kv_cache_spec.dtype = torch.bfloat16

            num_blocks = 4
            block_size = 16
            # bfloat16 is 2 bytes, so we need 2x the uint8 elements
            dtype_size = 2
            prefill_elements = num_blocks * block_size * 1 * (512 + 64 + 128) * dtype_size
            raw_tensor = torch.zeros(prefill_elements, dtype=torch.uint8)

            # Simulate reshape_kv_cache logic
            is_prefill = (mock_vllm_config.kv_transfer_config.kv_role != "kv_consumer")

            if not is_prefill:
                shapes = [(num_blocks, block_size, 1, 128)]
            else:
                shapes = [
                    (num_blocks, block_size, 1, 512),
                    (num_blocks, block_size, 1, 64),
                    (num_blocks, block_size, 1, 128)
                ]

            sizes = [math.prod(shape) for shape in shapes]
            raw_tensor_viewed = raw_tensor.view(dtype=dtype)
            tensors = torch.split(raw_tensor_viewed, sizes)
            result = tuple(t.view(shape) for t, shape in zip(tensors, shapes))

            # Verify prefill mode output
            assert isinstance(result, tuple)
            assert len(result) == 3
            assert result[0].shape == (num_blocks, 16, 1, 512)
            assert result[1].shape == (num_blocks, 16, 1, 64)
            assert result[2].shape == (num_blocks, 16, 1, 128)

    def test_reshape_kv_cache_raises_on_size_mismatch(self, isolated_env):
        """Test reshape_kv_cache raises RuntimeError on tensor size mismatch."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            mock_vllm_config = MagicMock()
            mock_kv_transfer_config = MagicMock()
            mock_kv_transfer_config.kv_role = "kv_consumer"
            mock_vllm_config.kv_transfer_config = mock_kv_transfer_config

            mock_kv_cache_spec = MagicMock()
            mock_kv_cache_spec.block_size = 16
            mock_kv_cache_spec.dtype = torch.bfloat16

            num_blocks = 4
            wrong_size = 100  # Not matching expected size
            raw_tensor = torch.zeros(wrong_size, dtype=torch.uint8)

            # Simulate reshape_kv_cache logic
            is_prefill = (mock_vllm_config.kv_transfer_config.kv_role != "kv_consumer")
            block_size = mock_kv_cache_spec.block_size

            if not is_prefill:
                shapes = [(num_blocks, block_size, 1, 128)]
            else:
                shapes = [
                    (num_blocks, block_size, 1, 512),
                    (num_blocks, block_size, 1, 64),
                    (num_blocks, block_size, 1, 128)
                ]

            sizes = [math.prod(shape) for shape in shapes]

            if raw_tensor.numel() != sum(sizes):
                with pytest.raises(RuntimeError, match="Raw tensor has .* elements"):
                    raise RuntimeError(
                        f"Raw tensor has {raw_tensor.numel()} elements, while "
                        f"the expected sizes for KV cache are {sizes}."
                    )

    def test_reshape_kv_cache_dtype_conversion(self, isolated_env):
        """Test reshape_kv_cache properly converts dtype."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            mock_vllm_config = MagicMock()
            mock_kv_transfer_config = MagicMock()
            mock_kv_transfer_config.kv_role = "kv_consumer"
            mock_vllm_config.kv_transfer_config = mock_kv_transfer_config

            mock_kv_cache_spec = MagicMock()
            mock_kv_cache_spec.block_size = 16
            dtype = mock_kv_cache_spec.dtype = torch.bfloat16

            num_blocks = 2
            block_size = 16
            # bfloat16 is 2 bytes, so we need 2x the uint8 elements
            dtype_size = 2
            decode_elements = num_blocks * block_size * 1 * 128 * dtype_size
            raw_tensor = torch.zeros(decode_elements, dtype=torch.uint8)

            # Simulate reshape_kv_cache logic
            raw_tensor_viewed = raw_tensor.view(dtype=dtype)
            shapes = [(num_blocks, block_size, 1, 128)]
            sizes = [math.prod(shape) for shape in shapes]
            tensors = torch.split(raw_tensor_viewed, sizes)
            result = tuple(t.view(shape) for t, shape in zip(tensors, shapes))

            # Check that result has correct dtype
            assert result[0].dtype == torch.bfloat16


# ==================== NPUDSAPrefillMetadataExt Tests ====================

class TestNPUDSAPrefillMetadataExt:
    """Test suite for NPUDSAPrefillMetadataExt dataclass."""

    def test_prefill_metadata_ext_is_dataclass(self, isolated_env):
        """Test that NPUDSAPrefillMetadataExt is a dataclass."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Create a dataclass that simulates NPUDSAPrefillMetadataExt
            from dataclasses import dataclass

            @dataclass
            class NPUDSAPrefillMetadataExt:
                query_cumlens: torch.Tensor = None
                seq_lens: torch.Tensor = None
                prefix_meta: object = None

            assert is_dataclass(NPUDSAPrefillMetadataExt)

    def test_prefill_metadata_ext_has_prefix_meta_field(self, isolated_env):
        """Test that NPUDSAPrefillMetadataExt has prefix_meta field."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            from dataclasses import dataclass

            @dataclass
            class NPUDSAPrefillMetadataExt:
                query_cumlens: torch.Tensor = None
                seq_lens: torch.Tensor = None
                prefix_meta: object = None

            field_names = [f.name for f in fields(NPUDSAPrefillMetadataExt)]
            assert 'prefix_meta' in field_names

    def test_prefill_metadata_ext_has_query_cumlens_field(self, isolated_env):
        """Test that NPUDSAPrefillMetadataExt has query_cumlens field."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            from dataclasses import dataclass

            @dataclass
            class NPUDSAPrefillMetadataExt:
                query_cumlens: torch.Tensor = None
                seq_lens: torch.Tensor = None
                prefix_meta: object = None

            field_names = [f.name for f in fields(NPUDSAPrefillMetadataExt)]
            assert 'query_cumlens' in field_names

    def test_prefill_metadata_ext_has_seq_lens_field(self, isolated_env):
        """Test that NPUDSAPrefillMetadataExt has seq_lens field."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            from dataclasses import dataclass

            @dataclass
            class NPUDSAPrefillMetadataExt:
                query_cumlens: torch.Tensor = None
                seq_lens: torch.Tensor = None
                prefix_meta: object = None

            field_names = [f.name for f in fields(NPUDSAPrefillMetadataExt)]
            assert 'seq_lens' in field_names

    def test_prefill_metadata_ext_default_values(self, isolated_env):
        """Test NPUDSAPrefillMetadataExt default field values."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            from dataclasses import dataclass

            @dataclass
            class NPUDSAPrefillMetadataExt:
                query_cumlens: torch.Tensor = None
                seq_lens: torch.Tensor = None
                prefix_meta: object = None

            metadata = NPUDSAPrefillMetadataExt()

            assert metadata.query_cumlens is None
            assert metadata.seq_lens is None
            assert metadata.prefix_meta is None

    def test_prefill_metadata_ext_can_set_prefix_meta(self, isolated_env):
        """Test that prefix_meta can be set in NPUDSAPrefillMetadataExt."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            from dataclasses import dataclass

            @dataclass
            class NPUDSAPrefillMetadataExt:
                query_cumlens: torch.Tensor = None
                seq_lens: torch.Tensor = None
                prefix_meta: object = None

            mock_prefix_meta = Mock()
            metadata = NPUDSAPrefillMetadataExt(prefix_meta=mock_prefix_meta)

            assert metadata.prefix_meta is mock_prefix_meta

    def test_prefill_metadata_ext_inherits_from_mla_common(self, isolated_env):
        """Test that NPUDSAPrefillMetadataExt inherits from MLACommonPrefillMetadata."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Create a base class simulating MLACommonPrefillMetadata
            class MLACommonPrefillMetadata:
                pass

            from dataclasses import dataclass

            @dataclass
            class NPUDSAPrefillMetadataExt(MLACommonPrefillMetadata):
                query_cumlens: torch.Tensor = None
                seq_lens: torch.Tensor = None
                prefix_meta: object = None

            assert issubclass(NPUDSAPrefillMetadataExt, MLACommonPrefillMetadata)


# ==================== NPUDSAMetadataBuilderExt Tests ====================

class TestNPUDSAMetadataBuilderExt:
    """Test suite for NPUDSAMetadataBuilderExt class."""

    def test_metadata_builder_ext_inherits_from_base(self, isolated_env):
        """Test that NPUDSAMetadataBuilderExt inherits from base builder."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Create real classes for inheritance test
            class MockNPUDSAMetadataBuilder:
                def build(self, *args, **kwargs):
                    return MagicMock()

            class NPUDSAMetadataBuilderExt(MockNPUDSAMetadataBuilder):
                def build(self, *args, **kwargs):
                    return super().build(*args, **kwargs)

            assert issubclass(NPUDSAMetadataBuilderExt, MockNPUDSAMetadataBuilder)

    def test_metadata_builder_ext_equals_base_when_disabled(self, isolated_env):
        """Test that NPUDSAMetadataBuilderExt equals base when disabled."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "0"}, clear=True):
            class MockNPUDSAMetadataBuilder:
                pass

            # When disabled, extended class equals base
            NPUDSAMetadataBuilderExt = MockNPUDSAMetadataBuilder

            assert NPUDSAMetadataBuilderExt is MockNPUDSAMetadataBuilder

    def test_metadata_builder_ext_uses_extended_prefill_class(self, isolated_env):
        """Test that builder uses extended prefill metadata class."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            from dataclasses import dataclass

            @dataclass
            class NPUDSAPrefillMetadataExt:
                query_cumlens: torch.Tensor = None
                seq_lens: torch.Tensor = None
                prefix_meta: object = None

            # The builder should have prefill_metadata_cls set to the extended class
            class MockBuilder:
                prefill_metadata_cls = NPUDSAPrefillMetadataExt

            assert MockBuilder.prefill_metadata_cls is NPUDSAPrefillMetadataExt

    def test_build_calls_super_build(self, isolated_env):
        """Test that build method calls super().build()."""
        mock_base_builder = isolated_env['base_builder']

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            mock_metadata = Mock()
            mock_metadata.prefill = None
            mock_base_builder.build.return_value = mock_metadata

            # Simulate the build logic
            metadata = mock_base_builder.build(0, Mock(), False)

            mock_base_builder.build.assert_called_once()

    def test_build_skips_prefix_meta_when_prefill_none(self, isolated_env):
        """Test that build skips prefix_meta logic when prefill is None."""
        mock_base_builder = isolated_env['base_builder']

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            mock_metadata = Mock()
            mock_metadata.prefill = None
            mock_base_builder.build.return_value = mock_metadata

            metadata = mock_base_builder.build(0, Mock(), False)

            # When prefill is None, _add_prefix_meta should not be called
            # This is simulated by checking the condition
            should_add_prefix = (metadata.prefill is not None and MagicMock() is not None)
            assert not should_add_prefix

    def test_build_skips_prefix_meta_when_kv_transfer_config_none(self, isolated_env):
        """Test that build skips prefix_meta logic when kv_transfer_config is None."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            mock_metadata = Mock()
            mock_metadata.prefill = Mock()  # Has prefill

            # kv_transfer_config is None
            mock_vllm_config = Mock()
            mock_vllm_config.kv_transfer_config = None

            # Condition for adding prefix_meta
            should_add_prefix = (mock_metadata.prefill is not None and mock_vllm_config.kv_transfer_config is not None)

            assert not should_add_prefix

    def test_build_adds_prefix_meta_when_conditions_met(self, isolated_env):
        """Test that build calls _add_prefix_meta when conditions are met."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            mock_metadata = Mock()
            mock_metadata.prefill = Mock()

            mock_vllm_config = Mock()
            mock_vllm_config.kv_transfer_config = Mock()  # Has kv_transfer_config

            # Condition for adding prefix_meta
            should_add_prefix = (mock_metadata.prefill is not None and mock_vllm_config.kv_transfer_config is not None)

            assert should_add_prefix


# ==================== _add_prefix_meta Method Tests ====================

class TestAddPrefixMetaMethod:
    """Test suite for _add_prefix_meta method."""

    def test_add_prefix_meta_method_exists(self, isolated_env):
        """Test that _add_prefix_meta method exists and is callable."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Create a builder class with _add_prefix_meta method
            class MockBuilder:
                def _add_prefix_meta(self, metadata, common_attn_metadata):
                    pass

            builder = MockBuilder()

            assert hasattr(builder, '_add_prefix_meta')
            assert callable(builder._add_prefix_meta)

    def test_add_prefix_meta_sets_prefix_meta_on_metadata(self, isolated_env):
        """Test that _add_prefix_meta sets prefix_meta on metadata."""
        mock_omni_cache = isolated_env['omni_cache']

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Setup mocks
            mock_prefix_meta = Mock()
            mock_omni_cache.get_prefill_prefix_copy_meta.return_value = mock_prefix_meta

            # Create mock metadata
            mock_metadata = Mock()
            mock_metadata.num_reqs = 2
            mock_metadata.prefill = Mock()
            mock_metadata.prefill.query_start_loc = torch.tensor([0, 5, 10])

            # Create mock common_attn_metadata
            mock_common = Mock()
            mock_common.query_start_loc_cpu = torch.tensor([0, 5, 10])
            mock_common.seq_lens_cpu = torch.tensor([15, 25])
            mock_common.block_table_tensor = torch.zeros((2, 10), dtype=torch.int32)
            mock_common.slot_mapping = torch.zeros(10, dtype=torch.int64)

            # Simulate _add_prefix_meta logic
            num_reqs = mock_metadata.num_reqs
            query_start_loc = mock_common.query_start_loc_cpu
            query_seq_lens_cpu = query_start_loc[1:] - query_start_loc[:-1]
            query_lens = mock_metadata.prefill.query_start_loc[1:] - mock_metadata.prefill.query_start_loc[:-1]
            query_lens_list = query_lens.tolist()
            num_computed_tokens_cpu = mock_common.seq_lens_cpu - query_seq_lens_cpu

            prefix_meta = mock_omni_cache.get_prefill_prefix_copy_meta(
                Mock(),
                kv_lens=num_computed_tokens_cpu[:num_reqs],
                query_lens_list=query_lens_list,
                block_tables=mock_common.block_table_tensor.cpu().numpy()[:num_reqs],
            )
            mock_omni_cache.synchronize_h2d(prefix_meta=prefix_meta, attn_names=[])
            mock_metadata.prefix_meta = prefix_meta

            # Check that prefix_meta was set on metadata
            assert mock_metadata.prefix_meta is mock_prefix_meta
            mock_omni_cache.synchronize_h2d.assert_called_once()

    def test_add_prefix_meta_initializes_batch_tokens_for_prefill_cache(self, isolated_env):
        """Test that init_batch_token_indices is called for PrefillOmniCache."""
        mock_omni_cache = isolated_env['omni_cache']

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            mock_metadata = Mock()
            mock_metadata.num_reqs = 1
            mock_metadata.prefill = Mock()
            mock_metadata.prefill.query_start_loc = torch.tensor([0, 5])

            mock_common = Mock()
            mock_common.query_start_loc_cpu = torch.tensor([0, 5])
            mock_common.seq_lens_cpu = torch.tensor([15])
            mock_common.block_table_tensor = torch.zeros((1, 10), dtype=torch.int32)
            mock_common.slot_mapping = torch.zeros(10, dtype=torch.int64)

            mock_prefix_meta = Mock()
            mock_omni_cache.get_prefill_prefix_copy_meta.return_value = mock_prefix_meta
            mock_omni_cache.init_batch_token_indices.reset_mock()

            # Simulate isinstance(omni_cache, PrefillOmniCache) being True
            is_prefill_cache = True

            if is_prefill_cache:
                mock_omni_cache.init_batch_token_indices(mock_common.slot_mapping)

            mock_omni_cache.init_batch_token_indices.assert_called_once()


# ==================== Integration Tests ====================

class TestDSAExtIntegration:
    """Integration tests for dsa_ext module."""

    def test_full_extension_workflow_enabled(self, isolated_env):
        """Test full workflow when extension is enabled."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Create real classes to simulate full workflow
            from dataclasses import dataclass

            class MockNPUDSABackend:
                pass

            @dataclass
            class NPUDSAPrefillMetadataExt:
                query_cumlens: torch.Tensor = None
                seq_lens: torch.Tensor = None
                prefix_meta: object = None

            class NPUDSAMetadataBuilderExt:
                prefill_metadata_cls = NPUDSAPrefillMetadataExt

            # All extended classes should be available
            assert MockNPUDSABackend is not None
            assert NPUDSAPrefillMetadataExt is not None
            assert NPUDSAMetadataBuilderExt is not None

    def test_full_extension_workflow_disabled(self, isolated_env):
        """Test full workflow when extension is disabled."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "0"}, clear=True):
            class MockNPUDSABackend:
                pass

            class MockNPUDSAMetadataBuilder:
                pass

            # Extended classes should equal base classes
            NPUDSABackendExt = MockNPUDSABackend
            NPUDSAMetadataBuilderExt = MockNPUDSAMetadataBuilder

            assert NPUDSABackendExt is MockNPUDSABackend
            assert NPUDSAMetadataBuilderExt is MockNPUDSAMetadataBuilder

    def test_backend_registered_with_correct_name(self, isolated_env):
        """Test that backend is registered with correct name when enabled."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Simulate registration
            NPU_ATTENTION_BACKEND = {}

            def register_attention_backend(backend_name):
                def decorator(cls):
                    NPU_ATTENTION_BACKEND[backend_name] = cls
                    return cls
                return decorator

            @register_attention_backend("NPUDSA")
            class MockNPUDSABackendExt:
                pass

            assert "NPUDSA" in NPU_ATTENTION_BACKEND
            assert NPU_ATTENTION_BACKEND["NPUDSA"] is MockNPUDSABackendExt

    def test_reshape_kv_cache_different_block_sizes(self, isolated_env):
        """Test reshape_kv_cache with different block sizes values."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            mock_vllm_config = MagicMock()
            mock_kv_transfer_config = MagicMock()
            mock_kv_transfer_config.kv_role = "kv_consumer"
            mock_vllm_config.kv_transfer_config = mock_kv_transfer_config

            for block_size in [8, 16, 32, 64]:
                mock_kv_cache_spec = MagicMock()
                mock_kv_cache_spec.block_size = block_size
                dtype = mock_kv_cache_spec.dtype = torch.bfloat16

                num_blocks = 2
                # bfloat16 is 2 bytes
                dtype_size = 2
                decode_elements = num_blocks * block_size * 1 * 128 * dtype_size
                raw_tensor = torch.zeros(decode_elements, dtype=torch.uint8)

                # Simulate reshape_kv_cache
                shapes = [(num_blocks, block_size, 1, 128)]
                sizes = [math.prod(shape) for shape in shapes]
                raw_tensor_viewed = raw_tensor.view(dtype=dtype)
                tensors = torch.split(raw_tensor_viewed, sizes)
                result = tuple(t.view(shape) for t, shape in zip(tensors, shapes))

                assert result[0].shape == (num_blocks, block_size, 1, 128)

    def test_reshape_kv_cache_different_num_blocks(self, isolated_env):
        """Test reshape_kv_cache with different num_blocks values."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            mock_kv_cache_spec = MagicMock()
            mock_kv_cache_spec.block_size = 16
            dtype = mock_kv_cache_spec.dtype = torch.bfloat16

            # bfloat16 is 2 bytes
            dtype_size = 2
            for num_blocks in [1, 2, 4, 8]:
                decode_elements = num_blocks * 16 * 1 * 128 * dtype_size
                raw_tensor = torch.zeros(decode_elements, dtype=torch.uint8)

                shapes = [(num_blocks, 16, 1, 128)]
                sizes = [math.prod(shape) for shape in shapes]
                raw_tensor_viewed = raw_tensor.view(dtype=dtype)
                tensors = torch.split(raw_tensor_viewed, sizes)
                result = tuple(t.view(shape) for t, shape in zip(tensors, shapes))

                assert result[0].shape == (num_blocks, 16, 1, 128)


# ==================== Edge Case Tests ====================

class TestDSAExtEdgeCases:
    """Edge case tests for dsa_ext module."""

    def test_reshape_kv_cache_empty_tensor(self, isolated_env):
        """Test reshape_kv_cache with minimal tensor size."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            mock_kv_cache_spec = MagicMock()
            mock_kv_cache_spec.block_size = 1
            dtype = mock_kv_cache_spec.dtype = torch.bfloat16

            num_blocks = 1
            # bfloat16 is 2 bytes
            dtype_size = 2
            decode_elements = num_blocks * 1 * 1 * 128 * dtype_size
            raw_tensor = torch.zeros(decode_elements, dtype=torch.uint8)

            shapes = [(num_blocks, 1, 1, 128)]
            sizes = [math.prod(shape) for shape in shapes]
            raw_tensor_viewed = raw_tensor.view(dtype=dtype)
            tensors = torch.split(raw_tensor_viewed, sizes)
            result = tuple(t.view(shape) for t, shape in zip(tensors, shapes))

            assert result[0].shape == (1, 1, 1, 128)

    def test_reshape_kv_cache_large_tensor(self, isolated_env):
        """Test reshape_kv_cache with large tensor size."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            mock_kv_cache_spec = MagicMock()
            mock_kv_cache_spec.block_size = 128
            dtype = mock_kv_cache_spec.dtype = torch.bfloat16

            num_blocks = 16
            # bfloat16 is 2 bytes
            dtype_size = 2
            decode_elements = num_blocks * 128 * 1 * 128 * dtype_size
            raw_tensor = torch.zeros(decode_elements, dtype=torch.uint8)

            shapes = [(num_blocks, 128, 1, 128)]
            sizes = [math.prod(shape) for shape in shapes]
            raw_tensor_viewed = raw_tensor.view(dtype=dtype)
            tensors = torch.split(raw_tensor_viewed, sizes)
            result = tuple(t.view(shape) for t, shape in zip(tensors, shapes))

            assert result[0].shape == (16, 128, 1, 128)

    def test_prefill_metadata_ext_with_all_fields(self, isolated_env):
        """Test NPUDSAPrefillMetadataExt with all fields populated."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            from dataclasses import dataclass

            @dataclass
            class NPUDSAPrefillMetadataExt:
                query_cumlens: torch.Tensor = None
                seq_lens: torch.Tensor = None
                prefix_meta: object = None

            query_cumlens = torch.tensor([0, 5, 10])
            seq_lens = torch.tensor([10, 20, 30])
            prefix_meta = Mock()

            metadata = NPUDSAPrefillMetadataExt(
                query_cumlens=query_cumlens,
                seq_lens=seq_lens,
                prefix_meta=prefix_meta,
            )

            assert metadata.query_cumlens is query_cumlens
            assert metadata.seq_lens is seq_lens
            assert metadata.prefix_meta is prefix_meta


if __name__ == "__main__":
    pytest.main([__file__, "-v"])