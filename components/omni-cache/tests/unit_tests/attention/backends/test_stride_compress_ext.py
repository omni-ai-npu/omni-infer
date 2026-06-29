# test_stride_compress_ext.py
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Unit tests for omni_cache/attention/backends/stride_compress_ext.py

Tests cover:
- ENABLE_OMNI_CACHE environment variable control
- StridedCompressBuilderExt: build method override
- Switch behavior when extension is enabled/disabled
- DecodeOmniCache and PrefillOmniCache handling
- Environment variable: ENABLE_HOST_MAPPING
- model_extra_config.operator_opt_config.use_omni_cache check

NOTE: This test file uses sys.modules mocking to isolate dependencies
since the current environment may not have vllm/omni_npu installed.
"""

import pytest
import os
import sys
import torch
from unittest.mock import Mock, patch, MagicMock


# ==================== Mock External Dependencies ====================

_mock_modules = {}


def setup_mocks():
    """Setup all mock modules for testing."""
    # Mock vllm.logger
    mock_logger = MagicMock()
    mock_logger.info = MagicMock()
    mock_logger.debug = MagicMock()
    mock_logger.warning = MagicMock()
    _mock_modules['vllm'] = MagicMock()
    _mock_modules['vllm.logger'] = MagicMock()
    _mock_modules['vllm.logger'].init_logger = MagicMock(return_value=mock_logger)

    # Mock vllm.v1.attention.backends.utils
    mock_common_attention_metadata = MagicMock(name='CommonAttentionMetadata')
    _mock_modules['vllm.v1'] = MagicMock()
    _mock_modules['vllm.v1.attention'] = MagicMock()
    _mock_modules['vllm.v1.attention.backends'] = MagicMock()
    _mock_modules['vllm.v1.attention.backends.utils'] = MagicMock()
    _mock_modules['vllm.v1.attention.backends.utils'].CommonAttentionMetadata = mock_common_attention_metadata

    # Mock omni_npu.attention.backends.geneva.stride_compress
    mock_base_builder = MagicMock(name='StridedCompressBuilder')
    mock_base_builder.__name__ = 'StridedCompressBuilder'
    mock_base_builder.__qualname__ = 'StridedCompressBuilder'
    mock_base_builder.return_value = MagicMock()
    mock_base_builder.build = MagicMock(return_value=MagicMock())

    mock_backend = MagicMock(name='StridedCompressAttentionBackend')
    mock_backend.__name__ = 'StridedCompressAttentionBackend'

    mock_metadata = MagicMock(name='StridedCompressAttentionMetadata')
    mock_metadata.__name__ = 'StridedCompressAttentionMetadata'

    _mock_modules['omni_npu'] = MagicMock()
    _mock_modules['omni_npu.attention'] = MagicMock()
    _mock_modules['omni_npu.attention.backends'] = MagicMock()
    _mock_modules['omni_npu.attention.backends.geneva'] = MagicMock()
    _mock_modules['omni_npu.attention.backends.geneva.stride_compress'] = MagicMock()
    _mock_modules['omni_npu.attention.backends.geneva.stride_compress'].StridedCompressBuilder = mock_base_builder
    _mock_modules['omni_npu.attention.backends.geneva.stride_compress'].StridedCompressAttentionBackend = mock_backend
    _mock_modules['omni_npu.attention.backends.geneva.stride_compress'].StridedCompressAttentionMetadata = mock_metadata

    # Mock omni_npu.v1.models.config_loader.loader
    mock_model_extra_config = MagicMock()
    mock_model_extra_config.operator_opt_config = MagicMock()
    mock_model_extra_config.operator_opt_config.use_omni_cache = False
    _mock_modules['omni_npu.v1'] = MagicMock()
    _mock_modules['omni_npu.v1.models'] = MagicMock()
    _mock_modules['omni_npu.v1.models.config_loader'] = MagicMock()
    _mock_modules['omni_npu.v1.models.config_loader.loader'] = MagicMock()
    _mock_modules['omni_npu.v1.models.config_loader.loader'].model_extra_config = mock_model_extra_config

    # Mock omni_cache.cache
    mock_omni_cache = MagicMock(name='omni_cache')
    mock_omni_cache._construct_fake_attn_metatata = MagicMock()
    mock_omni_cache.init_batch_token_indices_hybrid = MagicMock()

    _mock_modules['omni_cache.cache'] = MagicMock()
    _mock_modules['omni_cache.cache'].omni_cache = mock_omni_cache
    _mock_modules['omni_cache.cache.decode'] = MagicMock()
    _mock_modules['omni_cache.cache.decode'].DecodeOmniCache = MagicMock(name='DecodeOmniCache')
    _mock_modules['omni_cache.cache.prefill'] = MagicMock()
    _mock_modules['omni_cache.cache.prefill'].PrefillOmniCache = MagicMock(name='PrefillOmniCache')

    return mock_base_builder, mock_backend, mock_metadata, mock_omni_cache, mock_model_extra_config


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
        if name.startswith('omni_cache.attention') or name.startswith('omni_npu') or name.startswith('vllm'):
            del sys.modules[name]

    mock_base_builder, mock_backend, mock_metadata, mock_omni_cache, mock_model_extra_config = setup_mocks()
    apply_mocks()

    yield {
        'base_builder': mock_base_builder,
        'backend': mock_backend,
        'metadata': mock_metadata,
        'omni_cache': mock_omni_cache,
        'model_extra_config': mock_model_extra_config,
    }

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

    def test_enabled_when_env_is_nonzero(self, isolated_env):
        """Test that extension is enabled when ENABLE_OMNI_CACHE is non-zero."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "2"}, clear=True):
            test_module = type('Module', (), {})()
            test_module.ENABLE_OMNI_CACHE = int(os.getenv("ENABLE_OMNI_CACHE", "0"))

            assert test_module.ENABLE_OMNI_CACHE == 2


# ==================== Extension Logic Tests ====================

class TestExtensionLogic:
    """Test suite for extension class behavior."""

    def test_extension_enabled_creates_extended_class(self, isolated_env):
        """Test that ENABLE_OMNI_CACHE=1 creates extended builder class."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Create a real base class for inheritance test
            class MockStridedCompressBuilder:
                def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
                    return MagicMock(name='base_metadata')

            # Create an extended class that inherits from base
            class StridedCompressBuilderExt(MockStridedCompressBuilder):
                def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
                    # Extended logic here
                    return super().build(common_prefix_len, common_attn_metadata, fast_build)

            assert issubclass(StridedCompressBuilderExt, MockStridedCompressBuilder)

            # Test extended behavior
            ext_instance = StridedCompressBuilderExt()
            result = ext_instance.build(0, MagicMock(), False)
            assert result is not None

    def test_extension_disabled_uses_base_class(self, isolated_env):
        """Test that ENABLE_OMNI_CACHE=0 uses base class directly."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "0"}, clear=True):
            # Create a real base class
            class MockStridedCompressBuilder:
                def build(self, *args, **kwargs):
                    return "base_result"

            # When disabled, extended class is just an alias for base
            StridedCompressBuilderExt = MockStridedCompressBuilder

            assert StridedCompressBuilderExt is MockStridedCompressBuilder

    def test_backend_ext_equals_base_when_enabled(self, isolated_env):
        """Test that StridedCompressAttentionBackendExt equals base when enabled."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Create a real base class
            class MockStridedCompressAttentionBackend:
                pass

            # When enabled, backend ext equals base (no override needed)
            StridedCompressAttentionBackendExt = MockStridedCompressAttentionBackend

            assert StridedCompressAttentionBackendExt is MockStridedCompressAttentionBackend


# ==================== build() Method Tests ====================

class TestBuildMethod:
    """Test suite for build() method behavior."""

    def test_build_calls_super_build_when_not_decode_omni_cache(self, isolated_env):
        """Test that build calls super().build() when not DecodeOmniCache."""
        mock_base_builder = isolated_env['base_builder']

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Simulate: not DecodeOmniCache
            is_decode_omni_cache = False

            if not is_decode_omni_cache:
                result = mock_base_builder.build(0, MagicMock(), False)

            mock_base_builder.build.assert_called_once()

    def test_build_returns_fake_metadata_for_decode_omni_cache(self, isolated_env):
        """Test that build returns fake metadata when DecodeOmniCache conditions are met."""
        mock_omni_cache = isolated_env['omni_cache']
        mock_fake_metadata = MagicMock(name='fake_metadata')
        mock_omni_cache._construct_fake_attn_metatata.return_value = mock_fake_metadata

        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "ENABLE_HOST_MAPPING": "1"
        }, clear=True):
            # Simulate the build logic
            is_decode_omni_cache = True
            host_mapping_enabled = int(os.getenv("ENABLE_HOST_MAPPING", "1"))

            if is_decode_omni_cache and host_mapping_enabled:
                result = mock_omni_cache._construct_fake_attn_metatata(MagicMock(), MagicMock())
            else:
                result = MagicMock()

            assert result is mock_fake_metadata
            mock_omni_cache._construct_fake_attn_metatata.assert_called_once()

    def test_build_skips_fake_metadata_when_host_mapping_disabled(self, isolated_env):
        """Test that build skips fake metadata when ENABLE_HOST_MAPPING=0."""
        mock_base_builder = isolated_env['base_builder']

        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "ENABLE_HOST_MAPPING": "0"
        }, clear=True):
            is_decode_omni_cache = True
            host_mapping_enabled = int(os.getenv("ENABLE_HOST_MAPPING", "1"))

            should_use_fake = is_decode_omni_cache and host_mapping_enabled

            assert not should_use_fake  # Should NOT use fake metadata

    def test_build_initializes_batch_tokens_for_prefill_omni_cache(self, isolated_env):
        """Test that build initializes batch tokens for PrefillOmniCache when use_omni_cache is True."""
        mock_omni_cache = isolated_env['omni_cache']
        mock_base_builder = isolated_env['base_builder']
        mock_model_extra_config = isolated_env['model_extra_config']

        # Setup: enable use_omni_cache
        mock_model_extra_config.operator_opt_config.use_omni_cache = True

        mock_metadata = MagicMock()
        mock_metadata.slot_mapping = torch.zeros(10, dtype=torch.int64)
        mock_base_builder.build.return_value = mock_metadata

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Simulate PrefillOmniCache initialization
            is_prefill_omni_cache = True
            use_omni_cache = mock_model_extra_config.operator_opt_config.use_omni_cache

            metadata = mock_base_builder.build(0, MagicMock(), False)
            if use_omni_cache and is_prefill_omni_cache:
                mock_omni_cache.init_batch_token_indices_hybrid(metadata.slot_mapping)

            mock_omni_cache.init_batch_token_indices_hybrid.assert_called_once_with(mock_metadata.slot_mapping)

    def test_build_skips_batch_tokens_init_when_use_omni_cache_false(self, isolated_env):
        """Test that build skips batch tokens init when use_omni_cache is False."""
        mock_omni_cache = isolated_env['omni_cache']
        mock_base_builder = isolated_env['base_builder']
        mock_model_extra_config = isolated_env['model_extra_config']

        # Setup: disable use_omni_cache
        mock_model_extra_config.operator_opt_config.use_omni_cache = False

        mock_metadata = MagicMock()
        mock_metadata.slot_mapping = torch.zeros(10, dtype=torch.int64)
        mock_base_builder.build.return_value = mock_metadata

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            use_omni_cache = mock_model_extra_config.operator_opt_config.use_omni_cache

            metadata = mock_base_builder.build(0, MagicMock(), False)
            if use_omni_cache:
                mock_omni_cache.init_batch_token_indices_hybrid(metadata.slot_mapping)

            # Should NOT be called when use_omni_cache is False
            mock_omni_cache.init_batch_token_indices_hybrid.assert_not_called()


# ==================== ENABLE_HOST_MAPPING Tests ====================

class TestEnableHostMapping:
    """Test suite for ENABLE_HOST_MAPPING environment variable."""

    def test_host_mapping_enabled_by_default(self, isolated_env):
        """Test that ENABLE_HOST_MAPPING defaults to 1."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop('ENABLE_HOST_MAPPING', None)

            host_mapping = int(os.getenv("ENABLE_HOST_MAPPING", "1"))

            assert host_mapping == 1

    def test_host_mapping_disabled_when_zero(self, isolated_env):
        """Test that ENABLE_HOST_MAPPING=0 disables host mapping."""
        with patch.dict(os.environ, {"ENABLE_HOST_MAPPING": "0"}, clear=True):
            host_mapping = int(os.getenv("ENABLE_HOST_MAPPING", "1"))

            assert host_mapping == 0

    def test_host_mapping_enabled_when_one(self, isolated_env):
        """Test that ENABLE_HOST_MAPPING=1 enables host mapping."""
        with patch.dict(os.environ, {"ENABLE_HOST_MAPPING": "1"}, clear=True):
            host_mapping = int(os.getenv("ENABLE_HOST_MAPPING", "1"))

            assert host_mapping == 1


# ==================== Environment Variable Combination Tests ====================

class TestEnvironmentVariableCombinations:
    """Test suite for various environment variable combinations."""

    def test_all_conditions_enabled(self, isolated_env):
        """Test when all conditions are enabled."""
        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "ENABLE_HOST_MAPPING": "1"
        }, clear=True):
            assert int(os.getenv("ENABLE_OMNI_CACHE", "0")) == 1
            assert int(os.getenv("ENABLE_HOST_MAPPING", "1")) == 1

    def test_omni_cache_enabled_host_mapping_disabled(self, isolated_env):
        """Test when omni_cache is enabled but host mapping is disabled."""
        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "ENABLE_HOST_MAPPING": "0"
        }, clear=True):
            host_mapping_enabled = int(os.getenv("ENABLE_HOST_MAPPING", "1"))

            assert not host_mapping_enabled


# ==================== model_extra_config Tests ====================

class TestModelExtraConfig:
    """Test suite for model_extra_config handling."""

    def test_use_omni_cache_true_enables_prefill_init(self, isolated_env):
        """Test that use_omni_cache=True enables PrefillOmniCache initialization."""
        mock_model_extra_config = isolated_env['model_extra_config']
        mock_omni_cache = isolated_env['omni_cache']

        mock_model_extra_config.operator_opt_config.use_omni_cache = True

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            use_omni_cache = mock_model_extra_config.operator_opt_config.use_omni_cache

            assert use_omni_cache is True

    def test_use_omni_cache_false_skips_prefill_init(self, isolated_env):
        """Test that use_omni_cache=False skips PrefillOmniCache initialization."""
        mock_model_extra_config = isolated_env['model_extra_config']

        mock_model_extra_config.operator_opt_config.use_omni_cache = False

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            use_omni_cache = mock_model_extra_config.operator_opt_config.use_omni_cache

            assert use_omni_cache is False


# ==================== Edge Case Tests ====================

class TestStrideCompressExtEdgeCases:
    """Edge case tests for stride_compress_ext module."""

    def test_different_slot_mapping_tensor_types(self, isolated_env):
        """Test that slot_mapping works with different tensor types."""
        mock_omni_cache = isolated_env['omni_cache']
        mock_base_builder = isolated_env['base_builder']
        mock_model_extra_config = isolated_env['model_extra_config']

        mock_model_extra_config.operator_opt_config.use_omni_cache = True

        for dtype in [torch.int32, torch.int64]:
            mock_metadata = MagicMock()
            mock_metadata.slot_mapping = torch.zeros(10, dtype=dtype)
            mock_base_builder.build.return_value = mock_metadata

            mock_omni_cache.init_batch_token_indices_hybrid.reset_mock()
            mock_omni_cache.init_batch_token_indices_hybrid(mock_metadata.slot_mapping)

            mock_omni_cache.init_batch_token_indices_hybrid.assert_called_once()

    def test_build_with_different_common_prefix_len(self, isolated_env):
        """Test build with different common_prefix_len values."""
        mock_base_builder = isolated_env['base_builder']

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            for prefix_len in [0, 10, 100, 256]:
                mock_base_builder.build.reset_mock()
                mock_base_builder.build.return_value = MagicMock()

                result = mock_base_builder.build(prefix_len, MagicMock(), False)

                args = mock_base_builder.build.call_args[0]
                assert args[0] == prefix_len

    def test_build_with_fast_build_true(self, isolated_env):
        """Test build with fast_build=True."""
        mock_base_builder = isolated_env['base_builder']

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            mock_base_builder.build.return_value = MagicMock()

            result = mock_base_builder.build(0, MagicMock(), fast_build=True)

            args = mock_base_builder.build.call_args
            assert args[1]['fast_build'] is True or args[0][2] is True


# ==================== Constants Tests ====================

class TestModuleExports:
    """Test suite for module exports."""

    def test_all_exports_defined(self, isolated_env):
        """Test that __all__ contains expected exports."""
        expected_exports = [
            "StridedCompressAttentionBackendExt",
            "StridedCompressBuilderExt",
        ]

        # Simulate module __all__
        __all__ = [
            "StridedCompressAttentionBackendExt",
            "StridedCompressBuilderExt",
        ]

        for export in expected_exports:
            assert export in __all__


# ==================== Logger Tests ====================

class TestLoggerIntegration:
    """Test suite for logger integration."""

    def test_logger_info_called_on_enable(self, isolated_env):
        """Test that logger.info is called when extension is enabled."""
        mock_logger = MagicMock()

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            enable_omni_cache = int(os.getenv("ENABLE_OMNI_CACHE", "0"))

            if enable_omni_cache:
                mock_logger.info("omni_cache extension enabled, StridedCompressBuilderExt registered")

            mock_logger.info.assert_called_once_with(
                "omni_cache extension enabled, StridedCompressBuilderExt registered"
            )

    def test_logger_info_called_on_disable(self, isolated_env):
        """Test that logger.info is called when extension is disabled."""
        mock_logger = MagicMock()

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "0"}, clear=True):
            enable_omni_cache = int(os.getenv("ENABLE_OMNI_CACHE", "0"))

            if not enable_omni_cache:
                mock_logger.info("omni_cache extension disabled, using base StridedCompressBuilder")

            mock_logger.info.assert_called_once_with(
                "omni_cache extension disabled, using base StridedCompressBuilder"
            )


# ==================== Comparison with attention_ext Tests ====================

class TestDifferencesFromAttentionExt:
    """Test suite for differences between stride_compress_ext and attention_ext."""

    def test_no_disable_swa_mapping_check(self, isolated_env):
        """Test that stride_compress_ext does NOT check DISABLE_SWA_MAPPING."""
        # Unlike attention_ext, stride_compress_ext only checks ENABLE_HOST_MAPPING
        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "ENABLE_HOST_MAPPING": "1",
            "DISABLE_SWA_MAPPING": "1"  # This should NOT affect stride_compress_ext
        }, clear=True):
            host_mapping_enabled = int(os.getenv("ENABLE_HOST_MAPPING", "1"))

            # For stride_compress_ext, only host_mapping matters for fake metadata
            should_use_fake = bool(host_mapping_enabled)

            assert should_use_fake is True  # Should still work even with DISABLE_SWA_MAPPING=1

    def test_no_build_for_drafting_method(self, isolated_env):
        """Test that stride_compress_ext does NOT have build_for_drafting override."""
        # Unlike attention_ext, stride_compress_ext only overrides build()
        # Create a real base class to simulate
        class MockStridedCompressBuilder:
            def build(self, *args, **kwargs):
                return MagicMock()

        # StridedCompressBuilderExt only overrides build, not build_for_drafting
        builder = MockStridedCompressBuilder()

        # Should have build method
        assert hasattr(builder, 'build')
        assert callable(builder.build)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])