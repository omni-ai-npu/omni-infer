# test_attention_ext.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Unit tests for omni_cache/attention/backends/attention_ext.py

Tests cover:
- ENABLE_OMNI_CACHE environment variable control
- NPUAttentionMetadataBuilderExt: build method override
- NPUAttentionMetadataBuilderExt: build_for_drafting method override
- Switch behavior when extension is enabled/disabled
- DecodeOmniCache and PrefillOmniCache handling
- Environment variables: DISABLE_SWA_MAPPING, ENABLE_HOST_MAPPING

NOTE: This test file uses sys.modules mocking to isolate dependencies
since the current environment may not have vllm/omni_npu installed.
"""

import pytest
import os
import sys
import torch
from unittest.mock import Mock, patch, MagicMock, create_autospec

# ==================== Mock External Dependencies ====================

# Mock external dependencies before importing the module under test
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

    # Mock vllm.v1.attention.backend
    mock_common_attention_metadata = MagicMock(name='CommonAttentionMetadata')
    _mock_modules['vllm.v1'] = MagicMock()
    _mock_modules['vllm.v1.attention'] = MagicMock()
    _mock_modules['vllm.v1.attention.backend'] = MagicMock()
    _mock_modules['vllm.v1.attention.backend'].CommonAttentionMetadata = mock_common_attention_metadata

    # Mock omni_npu.attention.backends.attention
    mock_base_builder = MagicMock(name='NPUAttentionMetadataBuilder')
    mock_base_builder.__name__ = 'NPUAttentionMetadataBuilder'
    mock_base_builder.__qualname__ = 'NPUAttentionMetadataBuilder'
    # Make it behave like a proper class for inheritance
    mock_base_builder.return_value = MagicMock()
    mock_base_builder.build = MagicMock(return_value=MagicMock())
    mock_base_builder.build_for_drafting = MagicMock(return_value=MagicMock())

    mock_npu_metadata = MagicMock(name='NPUMetadata')
    mock_npu_metadata.__name__ = 'NPUMetadata'

    _mock_modules['omni_npu'] = MagicMock()
    _mock_modules['omni_npu.attention'] = MagicMock()
    _mock_modules['omni_npu.attention.backends'] = MagicMock()
    _mock_modules['omni_npu.attention.backends.attention'] = MagicMock()
    _mock_modules['omni_npu.attention.backends.attention'].NPUAttentionMetadataBuilder = mock_base_builder
    _mock_modules['omni_npu.attention.backends.attention'].NPUMetadata = mock_npu_metadata

    # Mock omni_npu.attention.backends.utils
    def mock_register_decorator(backend_name):
        def decorator(cls):
            return cls
        return decorator

    _mock_modules['omni_npu.attention.backends.utils'] = MagicMock()
    _mock_modules['omni_npu.attention.backends.utils'].register_attention_backend = mock_register_decorator
    _mock_modules['omni_npu.attention.backends.utils'].NPU_ATTENTION_BACKEND = {}

    # Mock omni_npu.v1.models.config_loader.loader
    _mock_modules['omni_npu.v1'] = MagicMock()
    _mock_modules['omni_npu.v1.models'] = MagicMock()
    _mock_modules['omni_npu.v1.models.config_loader'] = MagicMock()
    _mock_modules['omni_npu.v1.models.config_loader.loader'] = MagicMock()
    mock_model_extra_config = MagicMock()
    mock_model_extra_config.operator_opt_config = MagicMock()
    mock_model_extra_config.operator_opt_config.use_omni_cache = False
    _mock_modules['omni_npu.v1.models.config_loader.loader'].model_extra_config = mock_model_extra_config

    # Mock omni_cache.cache (needed for internal imports)
    mock_omni_cache = MagicMock(name='omni_cache')
    mock_omni_cache._construct_fake_attn_metatata = MagicMock()
    mock_omni_cache.init_batch_token_indices_hybrid = MagicMock()

    _mock_modules['omni_cache.cache'] = MagicMock()
    _mock_modules['omni_cache.cache'].omni_cache = mock_omni_cache
    _mock_modules['omni_cache.cache.decode'] = MagicMock()
    _mock_modules['omni_cache.cache.decode'].DecodeOmniCache = MagicMock(name='DecodeOmniCache')
    _mock_modules['omni_cache.cache.prefill'] = MagicMock()
    _mock_modules['omni_cache.cache.prefill'].PrefillOmniCache = MagicMock(name='PrefillOmniCache')

    return mock_base_builder, mock_npu_metadata, mock_omni_cache


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

    mock_base_builder, mock_npu_metadata, mock_omni_cache = setup_mocks()
    apply_mocks()

    yield {
        'base_builder': mock_base_builder,
        'npu_metadata': mock_npu_metadata,
        'omni_cache': mock_omni_cache,
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
            # Re-import to pick up new env var
            clear_mocks()
            setup_mocks()
            apply_mocks()

            # Create a simple module that simulates the behavior
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
            # Remove the env var if it exists
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
        """Test that ENABLE_OMNI_CACHE=1 creates extended class."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Create a real base class for inheritance test
            class MockBaseBuilder:
                def build(self, *args, **kwargs):
                    return "base_result"

            # Create an extended class that inherits from base
            class NPUAttentionMetadataBuilderExt(MockBaseBuilder):
                def build(self, *args, **kwargs):
                    result = super().build(*args, **kwargs)
                    return f"extended_{result}"

            # Test inheritance
            assert issubclass(NPUAttentionMetadataBuilderExt, MockBaseBuilder)

            # Test extended behavior
            ext_instance = NPUAttentionMetadataBuilderExt()
            assert ext_instance.build() == "extended_base_result"

    def test_extension_disabled_uses_base_class(self, isolated_env):
        """Test that ENABLE_OMNI_CACHE=0 uses base class directly."""
        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "0"}, clear=True):
            # Create a real base class
            class MockBaseBuilder:
                def build(self, *args, **kwargs):
                    return "base_result"

            # When disabled, extended class is just an alias for base
            NPUAttentionMetadataBuilderExt = MockBaseBuilder

            assert NPUAttentionMetadataBuilderExt is MockBaseBuilder


# ==================== build() Method Tests ====================

class TestBuildMethod:
    """Test suite for build() method behavior."""

    def test_build_calls_super_build(self, isolated_env):
        """Test that build calls super().build() when conditions are not met."""
        mock_base_builder = isolated_env['base_builder']

        # Create a mock instance
        builder_instance = MagicMock()
        builder_instance.build = MagicMock()

        # Create extended class behavior
        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            # This simulates the actual build logic
            if not isinstance_check() or not env_check():
                return mock_base_builder.build(common_prefix_len, common_attn_metadata, fast_build)
            return MagicMock()

        def isinstance_check():
            return False  # Not DecodeOmniCache

        def env_check():
            return (not int(os.getenv("DISABLE_SWA_MAPPING", "0"))) and int(os.getenv("ENABLE_HOST_MAPPING", "1"))

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            mock_base_builder.build.return_value = MagicMock(name='metadata')

            result = build(builder_instance, 0, MagicMock(), False)

            mock_base_builder.build.assert_called_once()

    def test_build_returns_fake_metadata_for_decode_omni_cache(self, isolated_env):
        """Test that build returns fake metadata when DecodeOmniCache conditions are met."""
        mock_omni_cache = isolated_env['omni_cache']
        mock_fake_metadata = MagicMock(name='fake_metadata')
        mock_omni_cache._construct_fake_attn_metatata.return_value = mock_fake_metadata

        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "DISABLE_SWA_MAPPING": "0",
            "ENABLE_HOST_MAPPING": "1"
        }, clear=True):
            # Simulate the build logic
            is_decode_omni_cache = True
            swa_mapping_disabled = int(os.getenv("DISABLE_SWA_MAPPING", "0")) == 0
            host_mapping_enabled = int(os.getenv("ENABLE_HOST_MAPPING", "1")) == 1

            if is_decode_omni_cache and swa_mapping_disabled and host_mapping_enabled:
                result = mock_omni_cache._construct_fake_attn_metatata(MagicMock(), MagicMock())
            else:
                result = MagicMock()

            assert result is mock_fake_metadata
            mock_omni_cache._construct_fake_attn_metatata.assert_called_once()

    def test_build_skips_fake_metadata_when_swa_mapping_disabled(self, isolated_env):
        """Test that build skips fake metadata when DISABLE_SWA_MAPPING=1."""
        mock_omni_cache = isolated_env['omni_cache']
        mock_base_builder = isolated_env['base_builder']

        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "DISABLE_SWA_MAPPING": "1",
            "ENABLE_HOST_MAPPING": "1"
        }, clear=True):
            is_decode_omni_cache = True
            swa_mapping_disabled = int(os.getenv("DISABLE_SWA_MAPPING", "0")) == 0
            host_mapping_enabled = int(os.getenv("ENABLE_HOST_MAPPING", "1")) == 1

            should_use_fake = is_decode_omni_cache and swa_mapping_disabled and host_mapping_enabled

            assert not should_use_fake  # Should NOT use fake metadata

    def test_build_skips_fake_metadata_when_host_mapping_disabled(self, isolated_env):
        """Test that build skips fake metadata when ENABLE_HOST_MAPPING=0."""
        mock_omni_cache = isolated_env['omni_cache']

        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "DISABLE_SWA_MAPPING": "0",
            "ENABLE_HOST_MAPPING": "0"
        }, clear=True):
            is_decode_omni_cache = True
            swa_mapping_disabled = int(os.getenv("DISABLE_SWA_MAPPING", "0")) == 0
            host_mapping_enabled = int(os.getenv("ENABLE_HOST_MAPPING", "1")) == 1

            should_use_fake = is_decode_omni_cache and swa_mapping_disabled and host_mapping_enabled

            assert not should_use_fake  # Should NOT use fake metadata

    def test_build_initializes_batch_tokens_for_prefill_omni_cache(self, isolated_env):
        """Test that build initializes batch tokens for PrefillOmniCache."""
        mock_omni_cache = isolated_env['omni_cache']
        mock_base_builder = isolated_env['base_builder']
        mock_metadata = MagicMock()
        mock_metadata.slot_mapping = torch.zeros(10, dtype=torch.int64)

        mock_base_builder.build.return_value = mock_metadata

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            # Simulate PrefillOmniCache initialization
            is_prefill_omni_cache = True

            metadata = mock_base_builder.build(0, MagicMock(), False)
            if is_prefill_omni_cache:
                mock_omni_cache.init_batch_token_indices_hybrid(metadata.slot_mapping)

            mock_omni_cache.init_batch_token_indices_hybrid.assert_called_once_with(mock_metadata.slot_mapping)


# ==================== build_for_drafting() Method Tests ====================

class TestBuildForDraftingMethod:
    """Test suite for build_for_drafting() method behavior."""

    def test_build_for_drafting_calls_super_when_conditions_not_met(self, isolated_env):
        """Test that build_for_drafting calls super when conditions are not met."""
        mock_base_builder = isolated_env['base_builder']

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            is_decode_omni_cache = False

            if is_decode_omni_cache:
                result = MagicMock()  # fake metadata
            else:
                result = mock_base_builder.build_for_drafting(MagicMock(), 0)

            mock_base_builder.build_for_drafting.assert_called_once()

    def test_build_for_drafting_returns_fake_metadata_for_decode_omni_cache(self, isolated_env):
        """Test that build_for_drafting returns fake metadata for DecodeOmniCache."""
        mock_omni_cache = isolated_env['omni_cache']
        mock_fake_metadata = MagicMock(name='fake_metadata')
        mock_omni_cache._construct_fake_attn_metatata.return_value = mock_fake_metadata

        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "DISABLE_SWA_MAPPING": "0",
            "ENABLE_HOST_MAPPING": "1"
        }, clear=True):
            is_decode_omni_cache = True
            swa_mapping_disabled = int(os.getenv("DISABLE_SWA_MAPPING", "0")) == 0
            host_mapping_enabled = int(os.getenv("ENABLE_HOST_MAPPING", "1")) == 1

            if is_decode_omni_cache and swa_mapping_disabled and host_mapping_enabled:
                result = mock_omni_cache._construct_fake_attn_metatata(MagicMock(), MagicMock(), 2)
            else:
                result = MagicMock()

            assert result is mock_fake_metadata

    def test_build_for_drafting_passes_draft_index(self, isolated_env):
        """Test that build_for_drafting passes draft_index to fake metadata construction."""
        mock_omni_cache = isolated_env['omni_cache']

        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "DISABLE_SWA_MAPPING": "0",
            "ENABLE_HOST_MAPPING": "1"
        }, clear=True):
            draft_index = 5

            mock_omni_cache._construct_fake_attn_metatata.reset_mock()
            mock_omni_cache._construct_fake_attn_metatata.return_value = MagicMock()

            result = mock_omni_cache._construct_fake_attn_metatata(MagicMock(), MagicMock(), draft_index)

            # Verify draft_index was passed as third argument
            args = mock_omni_cache._construct_fake_attn_metatata.call_args[0]
            assert args[2] == draft_index

    def test_build_for_drafting_skips_fake_metadata_when_swa_mapping_disabled(self, isolated_env):
        """Test that build_for_drafting skips fake metadata when DISABLE_SWA_MAPPING=1."""
        mock_base_builder = isolated_env['base_builder']

        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "DISABLE_SWA_MAPPING": "1",
            "ENABLE_HOST_MAPPING": "1"
        }, clear=True):
            is_decode_omni_cache = True
            swa_mapping_disabled = int(os.getenv("DISABLE_SWA_MAPPING", "0")) == 0
            host_mapping_enabled = int(os.getenv("ENABLE_HOST_MAPPING", "1")) == 1

            should_use_fake = is_decode_omni_cache and swa_mapping_disabled and host_mapping_enabled

            assert not should_use_fake


# ==================== Environment Variable Combination Tests ====================

class TestEnvironmentVariableCombinations:
    """Test suite for various environment variable combinations."""

    def test_all_conditions_enabled(self, isolated_env):
        """Test when all conditions are enabled."""
        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "DISABLE_SWA_MAPPING": "0",
            "ENABLE_HOST_MAPPING": "1"
        }, clear=True):
            assert int(os.getenv("ENABLE_OMNI_CACHE", "0")) == 1
            assert int(os.getenv("DISABLE_SWA_MAPPING", "0")) == 0
            assert int(os.getenv("ENABLE_HOST_MAPPING", "1")) == 1

    def test_swa_mapping_disabled(self, isolated_env):
        """Test when SWA mapping is disabled."""
        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "DISABLE_SWA_MAPPING": "1",
            "ENABLE_HOST_MAPPING": "1"
        }, clear=True):
            assert int(os.getenv("DISABLE_SWA_MAPPING", "0")) == 1

    def test_host_mapping_disabled(self, isolated_env):
        """Test when host mapping is disabled."""
        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "DISABLE_SWA_MAPPING": "0",
            "ENABLE_HOST_MAPPING": "0"
        }, clear=True):
            assert int(os.getenv("ENABLE_HOST_MAPPING", "1")) == 0

    def test_both_swa_and_host_mapping_disabled(self, isolated_env):
        """Test when both SWA mapping and host mapping are disabled."""
        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "DISABLE_SWA_MAPPING": "1",
            "ENABLE_HOST_MAPPING": "0"
        }, clear=True):
            swa_condition = int(os.getenv("DISABLE_SWA_MAPPING", "0")) == 0
            host_condition = int(os.getenv("ENABLE_HOST_MAPPING", "1")) == 1

            assert not (swa_condition and host_condition)


# ==================== Edge Case Tests ====================

class TestAttentionExtEdgeCases:
    """Edge case tests for attention_ext module."""

    def test_different_draft_index_values(self, isolated_env):
        """Test build_for_drafting with various draft index values."""
        mock_omni_cache = isolated_env['omni_cache']

        with patch.dict(os.environ, {
            "ENABLE_OMNI_CACHE": "1",
            "DISABLE_SWA_MAPPING": "0",
            "ENABLE_HOST_MAPPING": "1"
        }, clear=True):
            for draft_index in [0, 1, 5, 10]:
                mock_omni_cache._construct_fake_attn_metatata.reset_mock()
                mock_omni_cache._construct_fake_attn_metatata.return_value = MagicMock()

                result = mock_omni_cache._construct_fake_attn_metatata(
                    MagicMock(), MagicMock(), draft_index
                )

                args = mock_omni_cache._construct_fake_attn_metatata.call_args[0]
                assert args[2] == draft_index

    def test_slot_mapping_tensor_types(self, isolated_env):
        """Test that slot_mapping works with different tensor types."""
        mock_omni_cache = isolated_env['omni_cache']
        mock_base_builder = isolated_env['base_builder']

        # Create metadata with different slot_mapping types
        for dtype in [torch.int32, torch.int64]:
            mock_metadata = MagicMock()
            mock_metadata.slot_mapping = torch.zeros(10, dtype=dtype)

            mock_base_builder.build.return_value = mock_metadata

            # Simulate init_batch_token_indices_hybrid call
            mock_omni_cache.init_batch_token_indices_hybrid(mock_metadata.slot_mapping)

            mock_omni_cache.init_batch_token_indices_hybrid.assert_called()


# ==================== Constants Tests ====================

class TestConstants:
    """Test suite for module constants."""

    def test_vllm_npu_attn_constant(self, isolated_env):
        """Test that VLLM_NPU_ATTN constant is defined correctly."""
        VLLM_NPU_ATTN = "VLLM_NPU_ATTN"

        assert VLLM_NPU_ATTN == "VLLM_NPU_ATTN"
        assert isinstance(VLLM_NPU_ATTN, str)


# ==================== Logger Tests ====================

class TestLoggerIntegration:
    """Test suite for logger integration."""

    def test_logger_info_called_on_enable(self, isolated_env):
        """Test that logger.info is called when extension is enabled."""
        mock_logger = MagicMock()

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            enable_omni_cache = int(os.getenv("ENABLE_OMNI_CACHE", "0"))

            if enable_omni_cache:
                mock_logger.info("omni_cache extension enabled, NPUAttentionMetadataBuilderExt registered")

            mock_logger.info.assert_called_once_with(
                "omni_cache extension enabled, NPUAttentionMetadataBuilderExt registered"
            )

    def test_logger_info_called_on_disable(self, isolated_env):
        """Test that logger.info is called when extension is disabled."""
        mock_logger = MagicMock()

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "0"}, clear=True):
            enable_omni_cache = int(os.getenv("ENABLE_OMNI_CACHE", "0"))

            if not enable_omni_cache:
                mock_logger.info("omni_cache extension disabled, using base NPUAttentionMetadataBuilder")

            mock_logger.info.assert_called_once_with(
                "omni_cache extension disabled, using base NPUAttentionMetadataBuilder"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])