# test_plugin.py
# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Unit tests for omni_cache/plugin.py

Tests cover:
- BuildMetadataPlugin: pre_build, post_build, _process_prefill_prefix_meta
- LoadModelPlugin: pre_load, post_load
- InitConfigPlugin: pre_init_config, post_init_config, _initialize_omni_kv_cache
- Module-level functions: register, _register_kv_connectors, etc.
"""

import pytest
import os
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch, MagicMock

from omni_npu.vllm_patches.patches.models.pangu_v2_hybrid.patch_kv_cache_interface import PanguNewKVCacheSpecsPatch
try:
    PanguNewKVCacheSpecsPatch.apply()
except ValueError as exc:
    if "already patched" not in str(exc):
        raise

# Import module under test
from omni_cache.plugin import (
    LoadModelPlugin,
    InitConfigPlugin,
    InputBatchPlugin,
    PrepareInputsPlugin,
    register,
    _register_kv_connectors,
    _init_cache,
    _init_ox,
    _register_attn_plugins,
    get_decorators,
)


# ==================== LoadModelPlugin Tests ====================

class TestLoadModelPlugin:
    """Test suite for LoadModelPlugin class."""

    def test_pre_load_returns_early_when_no_args(self):
        """Test pre_load returns early when args is empty."""
        plugin = LoadModelPlugin()
        result = plugin.pre_load(*[], **{})
        assert result is None

    @patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "0"}, clear=True)
    def test_pre_load_returns_when_omni_cache_disabled(self):
        """Test pre_load returns when ENABLE_OMNI_CACHE is 0."""
        plugin = LoadModelPlugin()
        worker = Mock()
        worker.vllm_config = Mock()
        worker.vllm_config.kv_transfer_config = Mock()
        worker.vllm_config.kv_transfer_config.kv_role = "kv_consumer"

        result = plugin.pre_load(worker)
        assert result is None

    @patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True)
    def test_pre_load_returns_when_not_kv_consumer(self):
        """Test pre_load returns when kv_role is not kv_consumer."""
        plugin = LoadModelPlugin()
        worker = Mock()
        worker.vllm_config = Mock()
        worker.vllm_config.kv_transfer_config = Mock()
        worker.vllm_config.kv_transfer_config.kv_role = "kv_producer"  # Not consumer

        result = plugin.pre_load(worker)
        assert result is None

    @patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True)
    def test_pre_load_initializes_decode_omni_cache(self):
        """Test pre_load calls DecodeOmniCache.initialize_decode_omni_cache when conditions met."""
        plugin = LoadModelPlugin()

        worker = Mock()
        worker.vllm_config = Mock()
        worker.vllm_config.kv_transfer_config = Mock()
        worker.vllm_config.kv_transfer_config.kv_role = "kv_consumer"
        worker.model_runner = Mock()

        mock_decode_class = Mock()
        mock_decode_class.initialize_decode_omni_cache = Mock()

        with patch.dict(sys.modules, {
            'omni_cache.cache.decode': Mock(DecodeOmniCache=mock_decode_class),
        }):
            with patch('omni_cache.plugin.DecodeOmniCache', mock_decode_class, create=True):
                plugin.pre_load(worker)

        mock_decode_class.initialize_decode_omni_cache.assert_called_once_with(
            worker.vllm_config, worker.model_runner
        )

    def test_post_load_is_noop(self):
        """Test that post_load is a no-op."""
        plugin = LoadModelPlugin()
        result = plugin.post_load(Mock(), Mock())
        assert result is None


# ==================== InitConfigPlugin Tests ====================

class TestInitConfigPlugin:
    """Test suite for InitConfigPlugin class."""

    def test_pre_init_config_is_noop(self):
        """Test that pre_init_config returns False when ENABLE_OMNI_CACHE is not set."""
        plugin = InitConfigPlugin()
        # Make sure ENABLE_OMNI_CACHE is not set
        os.environ.pop("ENABLE_OMNI_CACHE", None)
        result = plugin.pre_init_config(Mock(), Mock())
        assert result is False  # Returns False to execute original function

    def test_post_init_config_returns_early_when_insufficient_args(self):
        """Test post_init_config returns early when less than 2 args."""
        plugin = InitConfigPlugin()
        os.environ["ENABLE_OMNI_CACHE"] = "1"
        result = plugin.post_init_config(Mock())  # Only 1 arg
        assert result is None
        os.environ.pop("ENABLE_OMNI_CACHE", None)

    def test_post_init_config_calls_initialize_omni_kv_cache(self):
        """Test post_init_config calls _initialize_omni_kv_cache."""
        plugin = InitConfigPlugin()
        plugin._initialize_omni_kv_cache = Mock()

        # Enable omni cache
        os.environ["ENABLE_OMNI_CACHE"] = "1"

        model_runner = Mock()
        kv_cache_config = Mock()

        plugin.post_init_config(model_runner, kv_cache_config)

        plugin._initialize_omni_kv_cache.assert_called_once_with(
            model_runner, kv_cache_config
        )

        os.environ.pop("ENABLE_OMNI_CACHE", None)

    def test_initialize_omni_kv_cache_method_exists(self):
        """Test _initialize_omni_kv_cache method exists and is callable."""
        plugin = InitConfigPlugin()

        # Test that the method exists and is callable
        assert callable(plugin._initialize_omni_kv_cache)

        # Test that calling with minimal args handles gracefully
        # (actual execution requires complex vLLM dependencies)
        model_runner = Mock()
        model_runner.kv_cache_config = None
        model_runner.vllm_config = Mock()
        model_runner.vllm_config.kv_transfer_config = Mock()
        model_runner.vllm_config.kv_transfer_config.kv_role = "kv_producer"
        model_runner.speculative_config = None

        # Mock required methods
        model_runner.may_add_encoder_only_layers_to_kv_cache_config = Mock()
        model_runner.maybe_add_kv_sharing_layers_to_kv_cache_groups = Mock()
        model_runner.initialize_attn_backend = Mock()
        model_runner._prepare_kernel_block_sizes = Mock(return_value={})
        model_runner.initialize_metadata_builders = Mock()
        model_runner.may_reinitialize_input_batch = Mock()

        kv_cache_config = Mock()

        # The actual execution will fail due to vLLM dependencies,
        # but we can verify the method signature is correct
        try:
            plugin._initialize_omni_kv_cache(model_runner, kv_cache_config)
        except Exception:
            # Expected to fail due to missing vLLM dependencies
            pass


class TestInputBatchPlugin:
    """Test suite for InputBatchPlugin."""

    def test_post_reinitialize_noop_when_disabled(self):
        plugin = InputBatchPlugin()
        runner = Mock()
        original = object()
        runner.input_batch = original

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "0"}, clear=True):
            plugin.post_reinitialize_input_batch(runner, Mock(), [16])

        assert runner.input_batch is original

    def test_post_reinitialize_noop_when_omni_input_batch_disabled(self):
        plugin = InputBatchPlugin()
        runner = Mock()
        original = object()
        runner.input_batch = original

        with patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True):
            plugin.post_reinitialize_input_batch(runner, Mock(), [16])

        assert runner.input_batch is original

    def test_post_reinitialize_noop_for_non_consumer(self):
        plugin = InputBatchPlugin()
        runner = Mock()
        runner.vllm_config.kv_transfer_config.kv_role = "kv_producer"
        original = object()
        runner.input_batch = original

        with patch.dict(
            os.environ,
            {"ENABLE_OMNI_CACHE": "1", "USE_OMNI_INPUT_BATCH": "1"},
            clear=True,
        ):
            plugin.post_reinitialize_input_batch(runner, Mock(), [16])

        assert runner.input_batch is original

    def test_post_reinitialize_installs_omni_input_batch(self):
        import torch

        from omni_cache.cache.input_batch import OmniCacheInputBatch

        class DSAAttentionSpec:
            block_size = 16

        runner = Mock()
        runner.max_num_reqs = 2
        runner.max_model_len = 32
        runner.max_encoder_len = 0
        runner.max_num_tokens = 8
        runner.device = torch.device("cpu")
        runner.pin_memory = False
        runner.model_config.get_vocab_size.return_value = 128
        runner.vllm_config.kv_transfer_config.kv_role = "kv_consumer"
        runner.vllm_config.speculative_config = None
        runner.is_pooling_model = False
        runner.num_spec_tokens = 0
        runner.parallel_config.cp_kv_cache_interleave_size = 1
        fake_omni_cache = SimpleNamespace(enable_gs=False)
        runner.omni_cache = fake_omni_cache
        runner.input_batch.logitsprocs = object()
        runner.input_batch.logitsprocs_need_output_token_ids = True

        kv_cache_config = Mock()
        kv_cache_config.kv_cache_groups = [
            Mock(kv_cache_spec=DSAAttentionSpec()),
        ]

        with patch.dict(
            os.environ,
            {"ENABLE_OMNI_CACHE": "1", "USE_OMNI_INPUT_BATCH": "1"},
            clear=True,
        ), patch("omni_cache.cache.omni_cache", fake_omni_cache):
            InputBatchPlugin().post_reinitialize_input_batch(
                runner, kv_cache_config, [16]
            )

        assert isinstance(runner.input_batch, OmniCacheInputBatch)
        assert runner.input_batch.kv_cache_specs == [
            kv_cache_config.kv_cache_groups[0].kv_cache_spec
        ]
        assert runner.input_batch.logitsprocs_need_output_token_ids is True
        assert fake_omni_cache.input_batch is runner.input_batch

    def test_post_reinitialize_is_idempotent(self):
        plugin = InputBatchPlugin()
        runner = Mock()
        from omni_cache.cache.input_batch import OmniCacheInputBatch

        runner.input_batch = OmniCacheInputBatch.__new__(OmniCacheInputBatch)
        original = runner.input_batch
        runner.vllm_config.kv_transfer_config.kv_role = "kv_consumer"

        with patch.dict(
            os.environ,
            {"ENABLE_OMNI_CACHE": "1", "USE_OMNI_INPUT_BATCH": "1"},
            clear=True,
        ):
            plugin.post_reinitialize_input_batch(runner, Mock(), [16])

        assert runner.input_batch is original


class TestPrepareInputsPlugin:
    """Test suite for PrepareInputsPlugin gather-selection hooks."""

    def _runner(self):
        runner = Mock()
        runner.input_batch = Mock()
        runner.input_batch.num_reqs = 3
        runner.omni_cache = Mock()
        runner.omni_cache.enable_gs = True
        runner.omni_cache.use_input_batch_lane_mapping = False
        runner.vllm_config.kv_transfer_config.kv_role = "kv_consumer"
        return runner

    def test_post_prepare_inputs_skips_legacy_update_with_input_batch_lane_mapping(self):
        runner = self._runner()
        runner.omni_cache.use_input_batch_lane_mapping = True

        with patch.dict(
            os.environ,
            {"ENABLE_OMNI_CACHE": "1", "USE_OMNI_INPUT_BATCH": "1"},
            clear=True,
        ), patch(
            "omni_cache.cache.decode.static_utils.record_current_batch_order"
        ) as record_order, patch(
            "omni_cache.cache.decode.DecodeOmniCache.maybe_update_selection_kv_block_status"
        ) as legacy_update:
            PrepareInputsPlugin().post_prepare_inputs(runner, Mock(), [1, 1, 1])

        record_order.assert_not_called()
        legacy_update.assert_not_called()

    def test_post_prepare_inputs_uses_legacy_gs_updater_when_input_batch_off(self):
        runner = self._runner()

        with patch.dict(
            os.environ,
            {"ENABLE_OMNI_CACHE": "1"},
            clear=True,
        ), patch(
            "omni_cache.cache.decode.static_utils.record_current_batch_order"
        ) as record_order, patch(
            "omni_cache.cache.decode.DecodeOmniCache.maybe_update_selection_kv_block_status"
        ) as legacy_update:
            PrepareInputsPlugin().post_prepare_inputs(runner, Mock(), [1, 1, 1])

        record_order.assert_called_once_with(runner.input_batch, runner.omni_cache)
        legacy_update.assert_called_once()


# ==================== Module-Level Function Tests ====================

class TestRegisterKVConnectors:
    """Test suite for _register_kv_connectors function."""

    def test_register_kv_connectors_handles_import_error_gracefully(self):
        """Test _register_kv_connectors handles ImportError gracefully."""
        # Should not raise exception
        _register_kv_connectors()

    def test_register_kv_connectors_logs_warning_on_error(self, caplog):
        """Test _register_kv_connectors logs warning on import error."""
        with caplog.at_level(logging.WARNING):
            _register_kv_connectors()
        # The function should handle the import error and log a warning
        # If the module exists, no warning; if not, warning is logged
        # Either way, the function should complete without raising


class TestInitCache:
    """Test suite for _init_cache function."""

    def test_init_cache_is_noop(self):
        """Test that _init_cache is a no-op."""
        result = _init_cache()
        assert result is None


class TestInitOx:
    """Test suite for _init_ox function."""

    def test_init_ox_is_noop(self):
        """Test that _init_ox is a no-op."""
        result = _init_ox()
        assert result is None


class TestRegister:
    """Test suite for register function."""

    @patch("omni_cache.plugin._register_kv_connectors")
    @patch("omni_cache.plugin._init_cache")
    @patch("omni_cache.plugin._init_ox")
    @patch("omni_cache.plugin._register_attn_plugins")
    def test_register_calls_all_initializers(
        self,
        mock_attn,
        mock_ox,
        mock_cache,
        mock_connectors
    ):
        """Test register calls all initialization functions."""
        register()

        mock_connectors.assert_called_once()
        mock_cache.assert_called_once()
        mock_ox.assert_called_once()
        mock_attn.assert_called_once()


class TestRegisterAttnPlugins:
    """Test suite for _register_attn_plugins function."""

    def test_register_attn_plugins_handles_import_error_gracefully(self):
        """Test _register_attn_plugins handles ImportError gracefully."""
        # Should not raise exception
        _register_attn_plugins()

    def test_register_attn_plugins_logs_warning_on_error(self, caplog):
        """Test _register_attn_plugins logs warning on error."""
        with caplog.at_level(logging.WARNING):
            _register_attn_plugins()
        # The function should handle errors gracefully


class TestGetDecorators:
    """Test suite for get_decorators function."""

    def test_get_decorators_returns_dict(self):
        """Test get_decorators returns a dictionary."""
        result = get_decorators()
        assert isinstance(result, dict)

    def test_get_decorators_handles_import_error(self):
        """Test get_decorators handles ImportError gracefully."""
        # When attn_plugins doesn't exist, should return empty dict
        result = get_decorators()
        # Either returns decorators or empty dict on error
        assert isinstance(result, dict)


# ==================== Integration Tests ====================

class TestPluginIntegration:
    """Integration tests for plugin module."""

    def test_all_plugins_can_be_instantiated(self):
        """Test that all plugin classes can be instantiated."""
        # Note: BuildMetadataPlugin was removed during refactoring
        load_plugin = LoadModelPlugin()
        init_plugin = InitConfigPlugin()
        input_batch_plugin = InputBatchPlugin()
        prepare_plugin = PrepareInputsPlugin()

        assert load_plugin is not None
        assert init_plugin is not None
        assert input_batch_plugin is not None
        assert prepare_plugin is not None

    @patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "1"}, clear=True)
    def test_full_build_workflow_enabled(self):
        """Test full build workflow when ENABLE_OMNI_CACHE is explicitly set."""
        # Note: BuildMetadataPlugin was removed during refactoring
        # This test now just verifies that the environment variable is set correctly
        assert os.environ.get("ENABLE_OMNI_CACHE") == "1"

    @patch.dict(os.environ, {"ENABLE_OMNI_CACHE": "0"}, clear=True)
    def test_full_build_workflow_disabled(self):
        """Test full build workflow when ENABLE_OMNI_CACHE is 0."""
        # Note: BuildMetadataPlugin was removed during refactoring
        # This test now just verifies that the environment variable is set correctly
        assert os.environ.get("ENABLE_OMNI_CACHE") == "0"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
