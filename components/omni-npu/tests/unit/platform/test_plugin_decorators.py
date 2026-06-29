# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""
Unit tests for omni_npu.plugin_decorators module (plugin-based decorators).
Covers missing lines: 38-42, 46-51, 53, 60-65, 67, 112-118, 121-129, 131, 134-136, 139-145, 147, 149
"""

import ast
import pytest
from pathlib import Path
from unittest.mock import Mock

from omni_npu.plugin_decorators import (
    _cached_eps,
    create_plugin_decorator,
    create_conditional_plugin_decorator,
    load_model_decorator,
    init_config_decorator,
    prepare_inputs_decorator,
    reinitialize_input_batch_decorator,
    attn_decorator,
)


class TestCreatePluginDecorator:
    """Tests for create_plugin_decorator factory function."""

    def setup_method(self):
        """Clear cached entry points before each test."""
        _cached_eps.clear()

    def test_plugin_decorator_calls_pre_hook(self, monkeypatch):
        """Test that pre-hook is called before the decorated function (covers lines 46-48)."""
        # Create a mock plugin with pre method
        mock_plugin = Mock()
        mock_plugin.pre_method = Mock()

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.pre_post":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        # Create decorator
        decorator = create_plugin_decorator(
            "test.pre_post",
            "pre_method",
            "post_method"
        )

        @decorator
        def test_func(arg1, arg2):
            return "result"

        result = test_func("a", "b")

        # Verify pre-hook was called
        mock_plugin.pre_method.assert_called_once_with("a", "b")
        assert result == "result"

    def test_plugin_decorator_pre_hook_returns_dict_updates_kwargs(self, monkeypatch):
        """Test that when pre-hook returns a dict, it is merged into kwargs (covers lines 49-50)."""
        mock_plugin = Mock()
        mock_plugin.pre_method = Mock(return_value={"extra": "injected"})

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.pre_dict":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = create_plugin_decorator(
            "test.pre_dict",
            "pre_method",
            "post_method"
        )

        received_kwargs = {}

        @decorator
        def test_func(*args, **kwargs):
            received_kwargs.update(kwargs)
            return "result"

        result = test_func("a", key="val")

        assert result == "result"
        assert received_kwargs["key"] == "val"
        assert received_kwargs["extra"] == "injected"

    def test_plugin_decorator_calls_post_hook(self, monkeypatch):
        """Test that post-hook is called after the decorated function (covers lines 60-62)."""
        mock_plugin = Mock()
        mock_plugin.post_method = Mock()

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.post_only":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = create_plugin_decorator(
            "test.post_only",
            "pre_method",
            "post_method"
        )

        @decorator
        def test_func():
            return "done"

        test_func()

        mock_plugin.post_method.assert_called_once()

    def test_plugin_decorator_pre_hook_raises_exception(self, monkeypatch):
        """Test exception handling in pre-hook (covers lines 49-51)."""
        mock_plugin = Mock()
        mock_plugin.pre_method = Mock(side_effect=RuntimeError("Pre-hook failed"))

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.pre_exception":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = create_plugin_decorator(
            "test.pre_exception",
            "pre_method",
            "post_method"
        )

        @decorator
        def test_func():
            return "should not reach"

        with pytest.raises(RuntimeError, match="Pre-hook failed"):
            test_func()

    def test_plugin_decorator_post_hook_raises_exception(self, monkeypatch):
        """Test exception handling in post-hook (covers lines 63-65)."""
        mock_plugin = Mock()
        mock_plugin.pre_method = Mock()
        mock_plugin.post_method = Mock(side_effect=RuntimeError("Post-hook failed"))

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.post_exception":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = create_plugin_decorator(
            "test.post_exception",
            "pre_method",
            "post_method"
        )

        @decorator
        def test_func():
            return "done"

        with pytest.raises(RuntimeError, match="Post-hook failed"):
            test_func()

    def test_plugin_decorator_plugin_missing_pre_method(self, monkeypatch):
        """Test that plugin missing pre_method is silently skipped (no warning logged)."""
        mock_plugin = Mock(spec=[])  # No pre_method attribute
        mock_plugin.post_method = Mock()

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.missing_pre":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        mock_logger = Mock()
        monkeypatch.setattr("omni_npu.plugin_decorators.logger", mock_logger)

        decorator = create_plugin_decorator(
            "test.missing_pre",
            "pre_method",
            "post_method"
        )

        @decorator
        def test_func():
            return "done"

        result = test_func()

        # Function should still execute and return correct result
        assert result == "done"
        # Post-hook should still be called
        mock_plugin.post_method.assert_called_once()
        # No warning should be logged for missing pre_method (silently skipped)
        warning_calls = [call for call in mock_logger.warning.call_args_list]
        assert not any("missing method: pre_method" in str(call) for call in warning_calls)

    def test_plugin_decorator_plugin_missing_post_method(self, monkeypatch):
        """Test that plugin missing post_method is silently skipped (no warning logged)."""
        mock_plugin = Mock(spec=["pre_method"])
        mock_plugin.pre_method = Mock()

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.missing_post":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        mock_logger = Mock()
        monkeypatch.setattr("omni_npu.plugin_decorators.logger", mock_logger)

        decorator = create_plugin_decorator(
            "test.missing_post",
            "pre_method",
            "post_method"
        )

        @decorator
        def test_func():
            return "done"

        result = test_func()

        # Function should still execute and return correct result
        assert result == "done"
        # Pre-hook should still be called
        mock_plugin.pre_method.assert_called_once()
        # No warning should be logged for missing post_method (silently skipped)
        warning_calls = [call for call in mock_logger.warning.call_args_list]
        assert not any("missing method: post_method" in str(call) for call in warning_calls)

    def test_plugin_decorator_failed_to_load_plugin(self, monkeypatch):
        """Test handling of failed plugin loading (covers lines 38-42)."""
        mock_ep = Mock()
        mock_ep.load = Mock(side_effect=ImportError("Plugin not found"))
        mock_ep.name = "failing_plugin"

        def mock_entry_points(group):
            if group == "test.load_fail":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        mock_logger = Mock()
        monkeypatch.setattr("omni_npu.plugin_decorators.logger", mock_logger)

        decorator = create_plugin_decorator(
            "test.load_fail",
            "pre_method",
            "post_method"
        )

        @decorator
        def test_func():
            return "done"

        result = test_func()

        # Function should still execute
        assert result == "done"
        # Warning should be logged (now "Failed to pre-load plugin" instead of "Failed to load plugin")
        mock_logger.warning.assert_called()
        assert "Failed to pre-load plugin" in mock_logger.warning.call_args[0][0]

    def test_plugin_decorator_no_plugins(self, monkeypatch):
        """Test decorator with no plugins registered."""
        def mock_entry_points(group):
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = create_plugin_decorator(
            "test.no_plugins",
            "pre_method",
            "post_method"
        )

        @decorator
        def test_func(x, y):
            return x + y

        result = test_func(1, 2)
        assert result == 3


class TestCreateConditionalPluginDecorator:
    """Tests for create_conditional_plugin_decorator factory function."""

    def setup_method(self):
        """Clear cached entry points before each test."""
        _cached_eps.clear()

    def test_conditional_decorator_executes_original_when_not_skipped(self, monkeypatch):
        """Test that original function executes when pre_hook returns False (covers lines 134-136)."""
        mock_plugin = Mock()
        mock_plugin.pre_check = Mock(return_value=False)
        mock_plugin.post_check = Mock()

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.conditional":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = create_conditional_plugin_decorator(
            "test.conditional",
            "pre_check",
            "post_check"
        )

        @decorator
        def test_func():
            return "original_result"

        result = test_func()
        assert result == "original_result"

    def test_conditional_decorator_skips_original_when_pre_returns_true(self, monkeypatch):
        """Test that original function is skipped when pre_hook returns True (covers lines 125-126, 134-136)."""
        mock_plugin = Mock()
        mock_plugin.pre_check = Mock(return_value=True)
        mock_plugin.post_check = Mock()

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.skip_original":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = create_conditional_plugin_decorator(
            "test.skip_original",
            "pre_check",
            "post_check"
        )

        func_called = []

        @decorator
        def test_func():
            func_called.append(True)
            return "should_not_return"

        result = test_func()

        # Original function should not be called
        assert len(func_called) == 0
        # Result should be None since original was skipped
        assert result is None
        # Post hook should still be called
        mock_plugin.post_check.assert_called_once()

    def test_conditional_decorator_pre_hook_raises_exception(self, monkeypatch):
        """Test exception handling in conditional pre-hook (covers lines 127-129)."""
        mock_plugin = Mock()
        mock_plugin.pre_check = Mock(side_effect=ValueError("Pre-check error"))

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.conditional_pre_exception":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = create_conditional_plugin_decorator(
            "test.conditional_pre_exception",
            "pre_check",
            "post_check"
        )

        @decorator
        def test_func():
            return "done"

        with pytest.raises(ValueError, match="Pre-check error"):
            test_func()

    def test_conditional_decorator_post_hook_raises_exception(self, monkeypatch):
        """Test exception handling in conditional post-hook (covers lines 143-145)."""
        mock_plugin = Mock()
        mock_plugin.pre_check = Mock(return_value=False)
        mock_plugin.post_check = Mock(side_effect=RuntimeError("Post-check error"))

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.conditional_post_exception":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = create_conditional_plugin_decorator(
            "test.conditional_post_exception",
            "pre_check",
            "post_check"
        )

        @decorator
        def test_func():
            return "done"

        with pytest.raises(RuntimeError, match="Post-check error"):
            test_func()

    def test_conditional_decorator_missing_pre_method(self, monkeypatch):
        """Test warning when plugin missing pre_method in conditional decorator (covers line 131)."""
        mock_plugin = Mock(spec=[])  # No pre_check method

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.conditional_missing_pre":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        mock_logger = Mock()
        monkeypatch.setattr("omni_npu.plugin_decorators.logger", mock_logger)

        decorator = create_conditional_plugin_decorator(
            "test.conditional_missing_pre",
            "pre_check",
            "post_check"
        )

        @decorator
        def test_func():
            return "done"

        test_func()

        # Verify warning was logged
        warning_calls = [call for call in mock_logger.warning.call_args_list]
        assert any("missing method: pre_check" in str(call) for call in warning_calls)

    def test_conditional_decorator_missing_post_method(self, monkeypatch):
        """Test warning when plugin missing post_method in conditional decorator (covers line 147)."""
        mock_plugin = Mock(spec=["pre_check"])
        mock_plugin.pre_check = Mock(return_value=False)

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.conditional_missing_post":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        mock_logger = Mock()
        monkeypatch.setattr("omni_npu.plugin_decorators.logger", mock_logger)

        decorator = create_conditional_plugin_decorator(
            "test.conditional_missing_post",
            "pre_check",
            "post_check"
        )

        @decorator
        def test_func():
            return "done"

        test_func()

        # Verify warning was logged
        warning_calls = [call for call in mock_logger.warning.call_args_list]
        assert any("missing method: post_check" in str(call) for call in warning_calls)

    def test_conditional_decorator_failed_to_load_plugin(self, monkeypatch):
        """Test handling of failed plugin loading in conditional decorator (covers lines 112-118)."""
        mock_ep = Mock()
        mock_ep.load = Mock(side_effect=ImportError("Plugin load failed"))
        mock_ep.name = "failing_plugin"

        def mock_entry_points(group):
            if group == "test.conditional_load_fail":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        mock_logger = Mock()
        monkeypatch.setattr("omni_npu.plugin_decorators.logger", mock_logger)

        decorator = create_conditional_plugin_decorator(
            "test.conditional_load_fail",
            "pre_check",
            "post_check"
        )

        @decorator
        def test_func():
            return "done"

        result = test_func()

        # Function should still execute
        assert result == "done"
        # Warning should be logged
        mock_logger.warning.assert_called()

    def test_conditional_decorator_return_value(self, monkeypatch):
        """Test return value handling in conditional decorator (covers line 149)."""
        mock_plugin = Mock()
        mock_plugin.pre_check = Mock(return_value=False)
        mock_plugin.post_check = Mock()

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.return_value":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = create_conditional_plugin_decorator(
            "test.return_value",
            "pre_check",
            "post_check"
        )

        @decorator
        def test_func():
            return "expected_result"

        result = test_func()
        assert result == "expected_result"

    def test_conditional_decorator_return_none_when_skipped(self, monkeypatch):
        """Test that None is returned when original function is skipped."""
        mock_plugin = Mock()
        mock_plugin.pre_check = Mock(return_value=True)
        mock_plugin.post_check = Mock()

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        def mock_entry_points(group):
            if group == "test.skip_return":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = create_conditional_plugin_decorator(
            "test.skip_return",
            "pre_check",
            "post_check"
        )

        @decorator
        def test_func():
            return "not_returned"

        result = test_func()
        assert result is None


class TestPredefinedDecorators:
    """Tests for predefined decorator instances."""

    def test_load_model_decorator_exists(self):
        """Test that load_model_decorator is properly defined."""
        assert callable(load_model_decorator)

    def test_init_config_decorator_exists(self):
        """Test that init_config_decorator is properly defined."""
        assert callable(init_config_decorator)

    def test_prepare_inputs_decorator_exists(self):
        """Test that prepare_inputs_decorator is properly defined."""
        assert callable(prepare_inputs_decorator)

    def test_reinitialize_input_batch_decorator_exists(self):
        """Test that reinitialize_input_batch_decorator is properly defined."""
        assert callable(reinitialize_input_batch_decorator)

    def test_load_model_decorator_can_decorate(self, monkeypatch):
        """Test that load_model_decorator can be applied to a function."""
        def mock_entry_points(group):
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )
        _cached_eps.clear()

        @load_model_decorator
        def sample_load_function():
            return "model_loaded"

        result = sample_load_function()
        assert result == "model_loaded"

    def test_init_config_decorator_can_decorate(self, monkeypatch):
        """Test that init_config_decorator can be applied to a function."""
        def mock_entry_points(group):
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )
        _cached_eps.clear()

        @init_config_decorator
        def sample_init_function():
            return "config_initialized"

        result = sample_init_function()
        assert result == "config_initialized"

    def test_prepare_inputs_decorator_can_decorate(self, monkeypatch):
        """Test that prepare_inputs_decorator can be applied to a function."""
        def mock_entry_points(group):
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )
        _cached_eps.clear()

        @prepare_inputs_decorator
        def sample_prepare_function():
            return "inputs_prepared"

        result = sample_prepare_function()
        assert result == "inputs_prepared"

    def test_reinitialize_input_batch_decorator_can_decorate(self, monkeypatch):
        """Test that reinitialize_input_batch_decorator delegates with no plugins."""
        def mock_entry_points(group):
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )
        _cached_eps.clear()

        @reinitialize_input_batch_decorator
        def sample_reinitialize_function(kv_cache_config, kernel_block_sizes):
            return kv_cache_config, kernel_block_sizes

        result = sample_reinitialize_function("kv-config", [16, 32])
        assert result == ("kv-config", [16, 32])
        assert "omni.reinitialize_input_batch_decorators" in _cached_eps

    def test_reinitialize_input_batch_decorator_uses_expected_hooks(
        self, monkeypatch
    ):
        """Test entry point group and hook names for input-batch reinit plugins."""
        mock_plugin = Mock()
        mock_plugin.pre_reinitialize_input_batch = Mock()
        mock_plugin.post_reinitialize_input_batch = Mock()

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "input_batch_plugin"

        captured_groups = []

        def mock_entry_points(group):
            captured_groups.append(group)
            if group == "omni.reinitialize_input_batch_decorators":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )
        _cached_eps.clear()

        @reinitialize_input_batch_decorator
        def sample_reinitialize_function(model_runner, kv_cache_config,
                                         kernel_block_sizes):
            return "base-called"

        result = sample_reinitialize_function("runner", "kv-config", [16])

        assert result == "base-called"
        assert captured_groups == ["omni.reinitialize_input_batch_decorators"]
        mock_plugin.pre_reinitialize_input_batch.assert_called_once_with(
            "runner", "kv-config", [16]
        )
        mock_plugin.post_reinitialize_input_batch.assert_called_once_with(
            "runner", "kv-config", [16], result="base-called"
        )

    def test_npu_model_runner_reinitialize_input_batch_is_decorated(self):
        """Test that NPUModelRunner exposes the reinitialize input batch hook."""
        repo_root = Path(__file__).resolve().parents[3]
        runner_path = (
            repo_root / "src" / "omni_npu" / "worker" / "npu_model_runner.py"
        )
        text = runner_path.read_text()

        assert "@reinitialize_input_batch_decorator" in text
        assert "def may_reinitialize_input_batch(" in text
        assert (
            "super().may_reinitialize_input_batch(kv_cache_config, "
            "kernel_block_sizes)"
        ) in text

    def test_npu_model_runner_reinitialize_input_batch_signature_contract(self):
        """Test the positional hook contract consumed by reinit plugins."""
        repo_root = Path(__file__).resolve().parents[3]
        runner_path = (
            repo_root / "src" / "omni_npu" / "worker" / "npu_model_runner.py"
        )
        tree = ast.parse(runner_path.read_text())

        runner_cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NPUModelRunner"
        )
        method = next(
            node for node in runner_cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "may_reinitialize_input_batch"
        )

        assert [arg.arg for arg in method.args.args] == [
            "self",
            "kv_cache_config",
            "kernel_block_sizes",
        ]
        assert ast.unparse(method.args.args[1].annotation) == "KVCacheConfig"
        assert ast.unparse(method.args.args[2].annotation) == "list[int]"
        assert method.args.posonlyargs == []
        assert method.args.kwonlyargs == []
        assert method.args.vararg is None
        assert method.args.kwarg is None
        assert method.args.defaults == []
        assert any(
            ast.unparse(decorator) == "reinitialize_input_batch_decorator"
            for decorator in method.decorator_list
        )


class TestCachedEntryPoints:
    """Tests for entry point caching behavior."""

    def test_entry_points_are_cached(self, monkeypatch):
        """Test that entry points are cached after first load."""
        _cached_eps.clear()

        call_count = [0]

        def mock_entry_points(group):
            call_count[0] += 1
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = create_plugin_decorator(
            "test.cache",
            "pre",
            "post"
        )

        @decorator
        def func1():
            return 1

        @decorator
        def func2():
            return 2

        func1()
        func2()

        # entry_points should only be called once due to caching
        assert call_count[0] == 1

        _cached_eps.clear()


class TestAttnDecorator:
    """Tests for the unified attn_decorator function."""

    def setup_method(self):
        """Clear cached entry points before each test."""
        _cached_eps.clear()

    def test_attn_decorator_creates_correct_entry_point_group(self, monkeypatch):
        """Test that attn_decorator constructs the correct entry point group from type."""
        mock_plugin = Mock()
        mock_plugin.pre_attn = Mock()
        mock_plugin.post_attn = Mock()

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "test_plugin"

        captured_groups = []

        def mock_entry_points(group):
            captured_groups.append(group)
            if group == "omni.test_type_attn_decorators":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = attn_decorator(type='test_type')

        @decorator
        def test_func():
            return "result"

        test_func()

        # Verify the entry point group was constructed correctly
        assert "omni.test_type_attn_decorators" in captured_groups
        mock_plugin.pre_attn.assert_called_once()
        mock_plugin.post_attn.assert_called_once()

    def test_attn_decorator_with_dsa_type(self, monkeypatch):
        """Test attn_decorator with type='dsa'."""
        mock_plugin = Mock()
        mock_plugin.pre_attn = Mock()
        mock_plugin.post_attn = Mock()

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "dsa_plugin"

        def mock_entry_points(group):
            if group == "omni.dsa_attn_decorators":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = attn_decorator(type='dsa')

        @decorator
        def _apply_attention(arg1, kwarg1=None):
            return "attention_result"

        result = _apply_attention("input", kwarg1="value")

        assert result == "attention_result"
        mock_plugin.pre_attn.assert_called_once_with("input", kwarg1="value")
        mock_plugin.post_attn.assert_called_once()

    def test_attn_decorator_with_mla_type(self, monkeypatch):
        """Test attn_decorator with type='mla'."""
        mock_plugin = Mock()
        mock_plugin.pre_attn = Mock()
        mock_plugin.post_attn = Mock()

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "mla_plugin"

        def mock_entry_points(group):
            if group == "omni.mla_attn_decorators":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = attn_decorator(type='mla')

        @decorator
        def _apply_attention(self, q, k, v):
            return "mla_result"

        result = _apply_attention("self", "q", "k", "v")

        assert result == "mla_result"
        mock_plugin.pre_attn.assert_called_once_with("self", "q", "k", "v")

    def test_attn_decorator_with_mome_type(self, monkeypatch):
        """Test attn_decorator with type='mome'."""
        mock_plugin = Mock()
        mock_plugin.pre_attn = Mock()
        mock_plugin.post_attn = Mock()

        mock_ep = Mock()
        mock_ep.load = Mock(return_value=lambda: mock_plugin)
        mock_ep.name = "mome_plugin"

        def mock_entry_points(group):
            if group == "omni.mome_attn_decorators":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = attn_decorator(type='mome')

        @decorator
        def apply_mome_conv(x):
            return "mome_result"

        result = apply_mome_conv("input")

        assert result == "mome_result"
        mock_plugin.pre_attn.assert_called_once_with("input")
        mock_plugin.post_attn.assert_called_once()

    def test_attn_decorator_no_plugins_registered(self, monkeypatch):
        """Test attn_decorator when no plugins are registered for the type."""
        def mock_entry_points(group):
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        decorator = attn_decorator(type='unknown')

        @decorator
        def test_func(x):
            return x * 2

        result = test_func(5)
        assert result == 10

    def test_attn_decorator_failed_to_load_plugin(self, monkeypatch):
        """Test attn_decorator handles plugin loading failure gracefully."""
        mock_ep = Mock()
        mock_ep.load = Mock(side_effect=ImportError("Plugin not found"))
        mock_ep.name = "failing_plugin"

        def mock_entry_points(group):
            if group == "omni.fail_attn_decorators":
                return [mock_ep]
            return []

        monkeypatch.setattr(
            "omni_npu.plugin_decorators.entry_points",
            mock_entry_points
        )

        mock_logger = Mock()
        monkeypatch.setattr("omni_npu.plugin_decorators.logger", mock_logger)

        decorator = attn_decorator(type='fail')

        @decorator
        def test_func():
            return "done"

        result = test_func()

        assert result == "done"
        mock_logger.warning.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
